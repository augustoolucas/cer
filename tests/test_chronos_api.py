"""Real Chronos-2 encoder & CPU training integration tests (no HF weight download)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chronos_rul

try:
    import torch
    import torch.nn as nn
    from chronos.chronos2 import Chronos2Model, Chronos2Pipeline
    from chronos.chronos2.config import Chronos2CoreConfig, Chronos2ForecastingConfig
    _HAS_CHRONOS = True
except ImportError:
    _HAS_CHRONOS = False


def _build_tiny_real_chronos2_pipeline() -> "Chronos2Pipeline":
    """Instantiate a real Chronos2Pipeline with d_model=768, patch=16, REG=True,
    and reduced layers (num_layers=1) with random weights for fast CPU API tests.
    """
    chronos_cfg = Chronos2ForecastingConfig(
        context_length=512,
        output_patch_size=16,
        input_patch_size=16,
        input_patch_stride=16,
        quantiles=[0.1, 0.5, 0.9],
        use_reg_token=True,
        max_output_patches=1,
        time_encoding_scale=512,
    )
    core_cfg = Chronos2CoreConfig(
        d_model=768,
        d_kv=64,
        d_ff=256,
        num_layers=1,
        num_heads=4,
        chronos_config=chronos_cfg.__dict__,
    )
    model = Chronos2Model(core_cfg).to(device="cpu", dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Chronos2Pipeline(model=model)


@unittest.skipUnless(_HAS_CHRONOS, "torch and chronos-forecasting==2.3.2 required")
class TestChronosRealAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(123)
        cls.pipeline = _build_tiny_real_chronos2_pipeline()

    def test_schema_validation_and_last_context_patch_extraction(self) -> None:
        schema = chronos_rul.validate_chronos_pipeline_schema(self.pipeline)
        self.assertEqual(schema["d_model"], 768)
        self.assertEqual(schema["input_patch_size"], 16)
        self.assertTrue(schema["use_reg_token"])

        rng = np.random.default_rng(0)
        # L=32 -> 2 context patches -> embed returns (8, 2 + 2 = 4, 768)
        # L=20 -> ceil(20/16)=2 context patches -> embed returns (8, 4, 768)
        # L=45 -> ceil(45/16)=3 context patches -> embed returns (8, 5, 768)
        grids = [
            rng.standard_normal((8, 32)).astype(np.float32),
            rng.standard_normal((8, 20)).astype(np.float32),
            rng.standard_normal((8, 45)).astype(np.float32),
        ]
        # Introduce NaNs in leading prefix to verify masked context works
        grids[1][:, :5] = np.nan

        extracted = chronos_rul.extract_last_context_patch_embeddings(
            self.pipeline, grids, batch_size_variates=64
        )
        self.assertEqual(extracted.shape, (3, 8, 768))
        self.assertTrue(np.isfinite(extracted).all())

        # Verify extracted[:, :, :] equals patch index (P - 1), NOT REG (-2) or output (-1)
        raw_embeds, _ = self.pipeline.embed([grids[2]], batch_size=64, context_length=512)
        self.assertEqual(tuple(raw_embeds[0].shape), (8, 5, 768))
        np.testing.assert_allclose(extracted[2], raw_embeds[0][:, 2, :].numpy(), atol=1e-6)
        self.assertFalse(np.allclose(extracted[2], raw_embeds[0][:, -1, :].numpy(), atol=1e-4))

    def test_group_isolation_across_endpoints(self) -> None:
        rng = np.random.default_rng(1)
        target_grid = rng.standard_normal((8, 32)).astype(np.float32)
        other_a = rng.standard_normal((8, 32)).astype(np.float32)
        other_b = other_a * 50.0 + 100.0

        solo = chronos_rul.extract_last_context_patch_embeddings(
            self.pipeline, [target_grid], batch_size_variates=64
        )
        batched_a = chronos_rul.extract_last_context_patch_embeddings(
            self.pipeline, [target_grid, other_a], batch_size_variates=64
        )
        batched_b = chronos_rul.extract_last_context_patch_embeddings(
            self.pipeline, [target_grid, other_b], batch_size_variates=64
        )
        np.testing.assert_allclose(solo[0], batched_a[0], atol=1e-5)
        np.testing.assert_allclose(solo[0], batched_b[0], atol=1e-5)

    def test_cpu_head_training_and_weights_only_checkpoint(self) -> None:
        rng = np.random.default_rng(42)
        X_tr = rng.standard_normal((40, 8, 768)).astype(np.float32)
        y_tr = rng.uniform(0.0, 1.0, size=40).astype(np.float32)
        X_va = rng.standard_normal((12, 8, 768)).astype(np.float32)
        y_va = rng.uniform(0.0, 1.0, size=12).astype(np.float32)

        class TinyHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(nn.Linear(768, 16), nn.ReLU(), nn.Linear(16, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x.mean(dim=1))

        res = chronos_rul.train_head_epochs(
            TinyHead(), X_tr, y_tr, X_val=X_va, y_val=y_va,
            optimizer_name="adam", lr=1e-3, batch_size=16, max_epochs=5, patience=3, seed=42
        )
        self.assertGreaterEqual(res["best_epoch"], 1)
        self.assertLessEqual(res["best_epoch"], 5)
        preds = chronos_rul.predict_head(res["head"], X_va)
        self.assertEqual(preds.shape, (12,))
        self.assertTrue(np.isfinite(preds).all())

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "head.pt"
            chronos_rul.save_head_checkpoint(ckpt, res["state_dict"])
            loaded_sd = chronos_rul.load_head_checkpoint(ckpt)
            self.assertEqual(set(loaded_sd.keys()), set(res["state_dict"].keys()))

    def test_elghoussani_heads_shapes_params_and_gradients(self) -> None:
        x = torch.randn(6, 8, 768)
        for arch, expected_params in (
            ("mlp2", 768 * 128 + 128 + 128 * 1 + 1),
            ("linear_relu", 768 * 1 + 1),
            ("mlp4", 768 * 128 + 128 + 2 * (128 * 128 + 128) + 128 + 1),
        ):
            head = chronos_rul.build_elghoussani_head(arch)
            self.assertEqual(chronos_rul.count_parameters(head), expected_params)
            out = head(x)
            self.assertEqual(tuple(out.shape), (6, 1))
            self.assertTrue((out >= 0).all())
            out.sum().backward()
            self.assertTrue(any(p.grad is not None for p in head.parameters()))

    def test_elghoussani_train_and_outer_rul_invariance(self) -> None:
        gerador_csv = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"
        if not gerador_csv.is_file():
            self.skipTest("gerador CSV not available")
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            bundle = chronos_rul.load_dataset_bundle("gerador", gerador_csv)
            outer_folds = chronos_rul.build_nested_splits(bundle["profile"])
            rng = np.random.default_rng(99)
            fake_emb = rng.standard_normal((len(bundle["endpoints"]), 8, 768)).astype(np.float32)

            ctxs = [5, 32]
            for outer in outer_folds:
                for ctx in ctxs:
                    ctx_spec = {
                        "kind": "fixed", "context_steps": ctx,
                        "step_hours": chronos_rul.GRID_STEP_HOURS,
                        "max_gap_hours": chronos_rul.MAX_INTERP_GAP_HOURS,
                    }
                    for train_ids in [i["inner_train_ids"] for i in outer["inner_folds"]] + [outer["outer_train_ids"]]:
                        prep = chronos_rul.CausalPreprocessor("clip_p1_p99_zscore").fit(
                            bundle["clean"], train_ids, bundle["sensor_columns"]
                        )
                        vid = chronos_rul.compute_view_identity(
                            dataset_name="gerador",
                            sensor_columns=bundle["sensor_columns"],
                            sensor_data_sha256=bundle["sensor_data_sha256"],
                            endpoints_sha256=bundle["endpoints_sha256"],
                            preprocessor_dict=prep.to_dict(),
                            context_spec=ctx_spec,
                        )
                        meta = chronos_rul._build_view_metadata(
                            bundle, prep, ctx_spec, extractor_env={"torch": "fake-gpu"}
                        )
                        chronos_rul.save_embedding_cache(cache_dir, vid, fake_emb, meta)

            cfg_base = {
                "dataset": "gerador",
                "input_csv": str(gerador_csv),
                "output_dir": str(tmp_path / "out1"),
                "cache_dir": str(cache_dir),
                "context_candidates": ctxs,
                "epochs": 2,
                "batch_size": 64,
                "seeds": [42, 43],
                "controls": ["linear_relu"],
            }
            res1 = chronos_rul.run_article_pipeline("elghoussani", cfg_base, "train")
            self.assertEqual(res1["status"], "trained")
            self.assertTrue(Path(res1["zip_path"]).is_file())

            # Perturb RUL ONLY on outer group A (Falha 1, 2)
            raw_df = pd.read_csv(gerador_csv)
            raw_df.loc[raw_df["Falha"].isin([1, 2]), "RUL"] = 0.05
            perturbed_csv = tmp_path / "perturbed.csv"
            raw_df.to_csv(perturbed_csv, index=False)

            cfg_pert = dict(cfg_base, input_csv=str(perturbed_csv), output_dir=str(tmp_path / "out2"))
            res2 = chronos_rul.run_article_pipeline("elghoussani", cfg_pert, "train")
            self.assertEqual(res1["selections"]["fold_A"], res2["selections"]["fold_A"])

    def test_abdouni_heads_shared_compression_and_gradients(self) -> None:
        emb = torch.randn(5, 8, 768)
        wide = torch.randn(5, 8)

        head_fusion = chronos_rul.build_abdouni_head("fusion")
        # Verify compressor is shared across all 8 channels (8*768 -> 8*8, not separate per channel)
        self.assertEqual(tuple(head_fusion.compressor[0].weight.shape), (8, 768))
        out_f = head_fusion(emb, wide)
        self.assertEqual(tuple(out_f.shape), (5, 1))
        out_f.sum().backward()
        self.assertTrue(all(p.grad is not None for p in head_fusion.parameters()))

        head_deep = chronos_rul.build_abdouni_head("deep")
        self.assertEqual(tuple(head_deep(emb).shape), (5, 1))

        head_wide = chronos_rul.build_abdouni_head("wide")
        self.assertEqual(tuple(head_wide(emb, wide).shape), (5, 1))

    def test_abdouni_train_ablations_and_outer_rul_invariance(self) -> None:
        gerador_csv = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"
        if not gerador_csv.is_file():
            self.skipTest("gerador CSV not available")
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            bundle = chronos_rul.load_dataset_bundle("gerador", gerador_csv)
            outer_folds = chronos_rul.build_nested_splits(bundle["profile"])
            rng = np.random.default_rng(123)
            fake_emb = rng.standard_normal((len(bundle["endpoints"]), 8, 768)).astype(np.float32)

            ctx_spec = {
                "kind": "full_prefix",
                "context_steps": None,
                "step_hours": chronos_rul.GRID_STEP_HOURS,
                "max_gap_hours": chronos_rul.MAX_INTERP_GAP_HOURS,
                "max_context_steps": chronos_rul.CHRONOS_MAX_CONTEXT_STEPS,
            }
            for outer in outer_folds:
                for train_ids in [i["inner_train_ids"] for i in outer["inner_folds"]] + [outer["outer_train_ids"]]:
                    prep = chronos_rul.CausalPreprocessor("zscore").fit(
                        bundle["clean"], train_ids, bundle["sensor_columns"]
                    )
                    vid = chronos_rul.compute_view_identity(
                        dataset_name="gerador",
                        sensor_columns=bundle["sensor_columns"],
                        sensor_data_sha256=bundle["sensor_data_sha256"],
                        endpoints_sha256=bundle["endpoints_sha256"],
                        preprocessor_dict=prep.to_dict(),
                        context_spec=ctx_spec,
                    )
                    meta = chronos_rul._build_view_metadata(
                        bundle, prep, ctx_spec, extractor_env={"torch": "fake-gpu"}
                    )
                    chronos_rul.save_embedding_cache(cache_dir, vid, fake_emb, meta)

            cfg_base = {
                "dataset": "gerador",
                "input_csv": str(gerador_csv),
                "output_dir": str(tmp_path / "out1"),
                "cache_dir": str(cache_dir),
                "max_epochs": 4,
                "patience": 2,
                "batch_size": 1024,
                "seeds": [42, 43],
                "variants": ["fusion", "deep", "wide"],
            }
            res1 = chronos_rul.run_article_pipeline("abdouni", cfg_base, "train")
            self.assertEqual(res1["status"], "trained")
            self.assertEqual(set(res1["metrics_summary"].keys()), {"fusion", "deep", "wide"})
            self.assertTrue(Path(res1["zip_path"]).is_file())

            # Perturb RUL ONLY on outer group A (Falha 1, 2)
            raw_df = pd.read_csv(gerador_csv)
            raw_df.loc[raw_df["Falha"].isin([1, 2]), "RUL"] = 0.05
            perturbed_csv = tmp_path / "perturbed.csv"
            raw_df.to_csv(perturbed_csv, index=False)

            cfg_pert = dict(cfg_base, input_csv=str(perturbed_csv), output_dir=str(tmp_path / "out2"))
            res2 = chronos_rul.run_article_pipeline("abdouni", cfg_pert, "train")
            self.assertEqual(res1["selections"]["fold_A"], res2["selections"]["fold_A"])

    def test_elghoussani_isolation_and_fold_seed_resumption(self) -> None:
        gerador_csv = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"
        if not gerador_csv.is_file():
            self.skipTest("gerador CSV not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            bundle = chronos_rul.load_dataset_bundle("gerador", gerador_csv)
            outer_folds = chronos_rul.build_nested_splits(bundle["profile"])
            rng = np.random.default_rng(7)
            fake_emb = rng.standard_normal((len(bundle["endpoints"]), 8, 768)).astype(np.float32)

            ctx_spec = {
                "kind": "fixed",
                "context_steps": 5,
                "step_hours": chronos_rul.GRID_STEP_HOURS,
                "max_gap_hours": chronos_rul.MAX_INTERP_GAP_HOURS,
            }
            for outer in outer_folds:
                for train_ids in [i["inner_train_ids"] for i in outer["inner_folds"]] + [outer["outer_train_ids"]]:
                    prep = chronos_rul.CausalPreprocessor("clip_p1_p99_zscore").fit(
                        bundle["clean"], train_ids, bundle["sensor_columns"]
                    )
                    vid = chronos_rul.compute_view_identity(
                        dataset_name="gerador",
                        sensor_columns=bundle["sensor_columns"],
                        sensor_data_sha256=bundle["sensor_data_sha256"],
                        endpoints_sha256=bundle["endpoints_sha256"],
                        preprocessor_dict=prep.to_dict(),
                        context_spec=ctx_spec,
                    )
                    meta = chronos_rul._build_view_metadata(
                        bundle, prep, ctx_spec, extractor_env={"torch": "fake-gpu"}
                    )
                    chronos_rul.save_embedding_cache(cache_dir, vid, fake_emb, meta)

            shared_out = tmp_path / "shared_out"
            cfg_a = {
                "dataset": "gerador",
                "input_csv": str(gerador_csv),
                "output_dir": str(shared_out),
                "cache_dir": str(cache_dir),
                "context_candidates": [5],
                "epochs": 2,
                "lr": 1e-3,
                "batch_size": 64,
                "seeds": [42, 43],
                "controls": ["linear_relu"],
            }
            res_a = chronos_rul.run_article_pipeline("elghoussani", cfg_a, "train")
            cfg_b = dict(cfg_a, lr=5e-4)
            res_b = chronos_rul.run_article_pipeline("elghoussani", cfg_b, "train")
            self.assertNotEqual(res_a["run_identity"], res_b["run_identity"])
            self.assertNotEqual(res_a["run_dir"], res_b["run_dir"])
            self.assertTrue((Path(res_a["run_dir"]) / "config.json").is_file())
            self.assertTrue((Path(res_a["run_dir"]) / "manifest.json").is_file())
            self.assertTrue((Path(res_b["run_dir"]) / "config.json").is_file())

            # Full resumption: 0 calls to train_head_epochs
            with mock.patch.object(
                chronos_rul, "train_head_epochs", wraps=chronos_rul.train_head_epochs
            ) as train_spy:
                res_resume = chronos_rul.run_article_pipeline("elghoussani", cfg_a, "train")
            self.assertEqual(train_spy.call_count, 0)
            self.assertGreater(res_resume["resumed_steps"], 0)
            self.assertEqual(res_resume["trained_steps"], 0)
            self.assertEqual(res_resume["metrics_summary"], res_a["metrics_summary"])

            # Partial resumption: remove one outer checkpoint + resume file -> trains only 1 step
            run_a_dir = Path(res_a["run_dir"])
            ckpt = run_a_dir / "checkpoints" / "linear_relu_fold_A_seed43.pt"
            resume_file = run_a_dir / "resume" / "outer_linear_relu_fold_A_seed43.json"
            ckpt.unlink()
            resume_file.unlink()
            with mock.patch.object(
                chronos_rul, "train_head_epochs", wraps=chronos_rul.train_head_epochs
            ) as train_spy_partial:
                res_partial = chronos_rul.run_article_pipeline("elghoussani", cfg_a, "train")
            self.assertEqual(train_spy_partial.call_count, 1)
            self.assertEqual(res_partial["trained_steps"], 1)
            self.assertEqual(res_partial["resumed_steps"], res_a["trained_steps"] - 1)
            self.assertTrue(ckpt.is_file())
            self.assertEqual(res_partial["metrics_summary"], res_a["metrics_summary"])

    def test_abdouni_fold_seed_resumption(self) -> None:
        gerador_csv = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"
        if not gerador_csv.is_file():
            self.skipTest("gerador CSV not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            bundle = chronos_rul.load_dataset_bundle("gerador", gerador_csv)
            outer_folds = chronos_rul.build_nested_splits(bundle["profile"])
            rng = np.random.default_rng(11)
            fake_emb = rng.standard_normal((len(bundle["endpoints"]), 8, 768)).astype(np.float32)

            ctx_spec = {
                "kind": "full_prefix",
                "context_steps": None,
                "step_hours": chronos_rul.GRID_STEP_HOURS,
                "max_gap_hours": chronos_rul.MAX_INTERP_GAP_HOURS,
                "max_context_steps": chronos_rul.CHRONOS_MAX_CONTEXT_STEPS,
            }
            for outer in outer_folds:
                for train_ids in [i["inner_train_ids"] for i in outer["inner_folds"]] + [outer["outer_train_ids"]]:
                    prep = chronos_rul.CausalPreprocessor("zscore").fit(
                        bundle["clean"], train_ids, bundle["sensor_columns"]
                    )
                    vid = chronos_rul.compute_view_identity(
                        dataset_name="gerador",
                        sensor_columns=bundle["sensor_columns"],
                        sensor_data_sha256=bundle["sensor_data_sha256"],
                        endpoints_sha256=bundle["endpoints_sha256"],
                        preprocessor_dict=prep.to_dict(),
                        context_spec=ctx_spec,
                    )
                    meta = chronos_rul._build_view_metadata(
                        bundle, prep, ctx_spec, extractor_env={"torch": "fake-gpu"}
                    )
                    chronos_rul.save_embedding_cache(cache_dir, vid, fake_emb, meta)

            cfg = {
                "dataset": "gerador",
                "input_csv": str(gerador_csv),
                "output_dir": str(tmp_path / "out"),
                "cache_dir": str(cache_dir),
                "max_epochs": 3,
                "patience": 2,
                "batch_size": 1024,
                "seeds": [42, 43],
                "variants": ["fusion"],
            }
            res1 = chronos_rul.run_article_pipeline("abdouni", cfg, "train")

            with mock.patch.object(
                chronos_rul, "train_head_epochs", wraps=chronos_rul.train_head_epochs
            ) as train_spy:
                res2 = chronos_rul.run_article_pipeline("abdouni", cfg, "train")
            self.assertEqual(train_spy.call_count, 0)
            self.assertEqual(res2["trained_steps"], 0)
            self.assertEqual(res2["resumed_steps"], res1["trained_steps"])
            self.assertEqual(res2["metrics_summary"], res1["metrics_summary"])

            run_dir = Path(res1["run_dir"])
            ckpt = run_dir / "checkpoints" / "fusion_fold_A_seed43.pt"
            resume_file = run_dir / "resume" / "outer_fusion_fold_A_seed43.json"
            ckpt.unlink()
            resume_file.unlink()
            with mock.patch.object(
                chronos_rul, "train_head_epochs", wraps=chronos_rul.train_head_epochs
            ) as train_spy_partial:
                res3 = chronos_rul.run_article_pipeline("abdouni", cfg, "train")
            self.assertEqual(train_spy_partial.call_count, 1)
            self.assertEqual(res3["trained_steps"], 1)
            self.assertEqual(res3["resumed_steps"], res1["trained_steps"] - 1)
            self.assertEqual(res3["metrics_summary"], res1["metrics_summary"])

    def test_portable_extract_zip_trains_in_isolated_cpu_dir(self) -> None:
        gerador_csv = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"
        if not gerador_csv.is_file():
            self.skipTest("gerador CSV not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ext_out = tmp_path / "extract_workspace" / "outputs"
            ext_cache = tmp_path / "extract_workspace" / "cache"
            ext_cfg = {
                "dataset": "gerador",
                "input_csv": str(gerador_csv),
                "output_dir": str(ext_out),
                "cache_dir": str(ext_cache),
                "context_candidates": [5],
                "epochs": 1,
                "batch_size": 128,
                "batch_size_variates": 512,
                "seeds": [42],
                "controls": ["linear_relu"],
            }
            with mock.patch.object(chronos_rul, "load_chronos_pipeline", return_value=self.pipeline):
                ext_res = chronos_rul.run_article_pipeline("elghoussani", ext_cfg, "extract")

            zip_path = Path(ext_res["zip_path"])
            self.assertTrue(zip_path.is_file())
            cpu_dir = tmp_path / "isolated_cpu_dir"
            cpu_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("config.json", names)
                self.assertIn("manifest.json", names)
                self.assertIn("views.json", names)
                cache_entries = [n for n in names if n.startswith("cache/") and n.endswith(".npz")]
                self.assertGreater(len(cache_entries), 0)
                zf.extractall(cpu_dir)

            # Remove original extraction directory completely to prove portability
            shutil.rmtree(tmp_path / "extract_workspace")

            cpu_cfg = json.loads((cpu_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cpu_cfg["cache_dir"], "cache")

            cpu_out = tmp_path / "cpu_outputs"
            with mock.patch.object(
                chronos_rul,
                "load_chronos_pipeline",
                side_effect=AssertionError("CPU train should not invoke load_chronos_pipeline"),
            ):
                rc = chronos_rul.main([
                    "--article", "elghoussani",
                    "--config", str(cpu_dir / "config.json"),
                    "--mode", "train",
                    "--output-dir", str(cpu_out),
                ])
            self.assertEqual(rc, 0)
            train_manifests = list(cpu_out.glob("**/manifest.json"))
            self.assertEqual(len(train_manifests), 1)
            trained_manifest = json.loads(train_manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(trained_manifest["status"], "trained")
            self.assertEqual(trained_manifest["run_identity"], ext_res["run_identity"])
            self.assertGreater(trained_manifest["trained_steps"], 0)
            self.assertFalse((tmp_path / "extract_workspace").exists())

    def test_training_code_fingerprint_invalidates_resumption_and_preserves_cache(self) -> None:
        gerador_csv = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"
        if not gerador_csv.is_file():
            self.skipTest("gerador CSV not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            bundle = chronos_rul.load_dataset_bundle("gerador", gerador_csv)
            outer_folds = chronos_rul.build_nested_splits(bundle["profile"])
            rng = np.random.default_rng(2026)
            fake_emb = rng.standard_normal((len(bundle["endpoints"]), 8, 768)).astype(np.float32)

            ctx_spec = {
                "kind": "fixed",
                "context_steps": 5,
                "step_hours": chronos_rul.GRID_STEP_HOURS,
                "max_gap_hours": chronos_rul.MAX_INTERP_GAP_HOURS,
            }
            ext_sha_before = chronos_rul.compute_extractor_code_sha256()
            view_ids_before = []
            for outer in outer_folds:
                for train_ids in [i["inner_train_ids"] for i in outer["inner_folds"]] + [outer["outer_train_ids"]]:
                    prep = chronos_rul.CausalPreprocessor("clip_p1_p99_zscore").fit(
                        bundle["clean"], train_ids, bundle["sensor_columns"]
                    )
                    vid = chronos_rul.compute_view_identity(
                        dataset_name="gerador",
                        sensor_columns=bundle["sensor_columns"],
                        sensor_data_sha256=bundle["sensor_data_sha256"],
                        endpoints_sha256=bundle["endpoints_sha256"],
                        preprocessor_dict=prep.to_dict(),
                        context_spec=ctx_spec,
                    )
                    view_ids_before.append(vid)
                    meta = chronos_rul._build_view_metadata(
                        bundle, prep, ctx_spec, extractor_env={"torch": "fake-gpu", "cuda_available": True}
                    )
                    chronos_rul.save_embedding_cache(cache_dir, vid, fake_emb, meta)

            # GPU vs CPU operational fields must NOT change run_identity or training fingerprint
            eff_gpu = chronos_rul.effective_article_config("elghoussani", {
                "dataset": "gerador", "input_csv": "/gpu/path.csv", "output_dir": "/gpu/out",
                "cache_dir": "/gpu/cache", "device": "cuda", "batch_size_variates": 256,
                "context_candidates": [5], "epochs": 1, "batch_size": 128, "seeds": [42], "controls": [],
            })
            eff_cpu = chronos_rul.effective_article_config("elghoussani", {
                "dataset": "gerador", "input_csv": "/cpu/path.csv", "output_dir": "/cpu/out",
                "cache_dir": "/cpu/cache", "device": "cpu", "batch_size_variates": 16,
                "context_candidates": [5], "epochs": 1, "batch_size": 128, "seeds": [42], "controls": [],
            })
            self.assertEqual(
                chronos_rul.compute_run_identity("elghoussani", bundle, eff_gpu),
                chronos_rul.compute_run_identity("elghoussani", bundle, eff_cpu),
            )

            cfg_v1 = {
                "dataset": "gerador",
                "input_csv": str(gerador_csv),
                "output_dir": str(tmp_path / "out"),
                "cache_dir": str(cache_dir),
                "run_id": "exp_named",
                "device": "cpu",
                "context_candidates": [5],
                "epochs": 1,
                "batch_size": 128,
                "seeds": [42],
                "controls": [],
            }
            res_v1 = chronos_rul.run_article_pipeline("elghoussani", cfg_v1, "train")
            train_sha_v1 = res_v1["training_code_sha256"]
            self.assertGreater(res_v1["trained_steps"], 0)
            self.assertEqual(res_v1["resumed_steps"], 0)

            # Equal fingerprint & identity permits 0 new fits on same named run_id
            with mock.patch.object(
                chronos_rul, "train_head_epochs", wraps=chronos_rul.train_head_epochs
            ) as train_spy_same:
                res_v1_resumed = chronos_rul.run_article_pipeline("elghoussani", cfg_v1, "train")
            self.assertEqual(train_spy_same.call_count, 0)
            self.assertEqual(res_v1_resumed["trained_steps"], 0)
            self.assertEqual(res_v1_resumed["resumed_steps"], res_v1["trained_steps"])

            run1_dir = Path(res_v1["run_dir"])
            snap_v1 = {
                p.relative_to(run1_dir).as_posix(): p.read_bytes()
                for p in sorted(run1_dir.rglob("*")) if p.is_file()
            }

            # Same shapes, altered activation (ReLU -> Tanh)
            def alt_build_elghoussani_head(
                arch: str = "mlp2",
                *,
                d_model: int = chronos_rul.CHRONOS_D_MODEL,
                hidden_dim: int = 128,
                dropout: float = 0.2,
            ):
                class AltHead(nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.arch = arch
                        self.body = nn.Sequential(
                            nn.Linear(d_model, hidden_dim),
                            nn.Tanh(),
                            nn.Dropout(p=float(dropout)),
                            nn.Linear(hidden_dim, 1),
                            nn.Tanh(),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.body(x.mean(dim=1))

                return AltHead()

            with mock.patch.object(chronos_rul, "build_elghoussani_head", new=alt_build_elghoussani_head):
                ext_sha_after = chronos_rul.compute_extractor_code_sha256()
                train_sha_v2 = chronos_rul.compute_training_code_sha256("elghoussani")
                self.assertEqual(ext_sha_after, ext_sha_before)
                self.assertNotEqual(train_sha_v2, train_sha_v1)

                # Named destination from old training code must refuse without altering bytes
                with self.assertRaisesRegex(RuntimeError, "Conflito de identidade"):
                    chronos_rul.run_article_pipeline("elghoussani", cfg_v1, "train")
                snap_v1_after = {
                    p.relative_to(run1_dir).as_posix(): p.read_bytes()
                    for p in sorted(run1_dir.rglob("*")) if p.is_file()
                }
                self.assertEqual(snap_v1_after, snap_v1)

                # Different named / auto destination reuses compatible GPU embedding cache and retrains cleanly
                cache_snap = {
                    p.name: p.read_bytes() for p in sorted(cache_dir.iterdir()) if p.is_file()
                }
                cfg_v2 = dict(cfg_v1, run_id="exp_named_v2")
                res_v2 = chronos_rul.run_article_pipeline("elghoussani", cfg_v2, "train")
                self.assertEqual(
                    {p.name: p.read_bytes() for p in sorted(cache_dir.iterdir()) if p.is_file()},
                    cache_snap,
                )
                self.assertNotEqual(res_v2["run_identity"], res_v1["run_identity"])
                self.assertEqual(res_v2["training_code_sha256"], train_sha_v2)
                self.assertEqual(res_v2["resumed_steps"], 0)
                self.assertEqual(res_v2["trained_steps"], res_v1["trained_steps"])

            # Stale training_code_sha256 in resume metadata must prevent reusing old checkpoint/score
            stale_outer_resume = run1_dir / "resume" / "outer_mlp2_fold_A_seed42.json"
            stale_data = json.loads(stale_outer_resume.read_text(encoding="utf-8"))
            stale_data["training_code_sha256"] = "0" * 64
            stale_outer_resume.write_text(json.dumps(stale_data), encoding="utf-8")
            with mock.patch.object(
                chronos_rul, "train_head_epochs", wraps=chronos_rul.train_head_epochs
            ) as train_spy_stale:
                res_stale = chronos_rul.run_article_pipeline("elghoussani", cfg_v1, "train")
            self.assertEqual(train_spy_stale.call_count, 1)
            self.assertEqual(res_stale["trained_steps"], 1)
            self.assertEqual(res_stale["resumed_steps"], res_v1["trained_steps"] - 1)

            # View IDs remain unchanged
            for vid_orig, outer in zip(view_ids_before[:1], outer_folds[:1]):
                prep = chronos_rul.CausalPreprocessor("clip_p1_p99_zscore").fit(
                    bundle["clean"], outer["inner_folds"][0]["inner_train_ids"], bundle["sensor_columns"]
                )
                self.assertEqual(
                    vid_orig,
                    chronos_rul.compute_view_identity(
                        dataset_name="gerador",
                        sensor_columns=bundle["sensor_columns"],
                        sensor_data_sha256=bundle["sensor_data_sha256"],
                        endpoints_sha256=bundle["endpoints_sha256"],
                        preprocessor_dict=prep.to_dict(),
                        context_spec=ctx_spec,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
