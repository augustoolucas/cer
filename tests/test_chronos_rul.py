"""Fast unit tests for chronos_rul core (no Torch or Chronos weights required)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cer_rul
import chronos_rul

GERADOR_CSV = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"
GEARBOX_CSV = ROOT / "data" / "Features_gearbox_score_todos_RUL-porcentagem-data.csv"


class TestLightImportAndCLI(unittest.TestCase):
    def test_torch_not_imported_on_module_load(self) -> None:
        code = (
            "import sys, chronos_rul; "
            "assert 'torch' not in sys.modules; "
            "assert 'chronos' not in sys.modules"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_cli_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "chronos_rul.py"), "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--article", proc.stdout)
        self.assertIn("--mode", proc.stdout)


class TestDataAndNestedSplits(unittest.TestCase):
    @unittest.skipUnless(GERADOR_CSV.is_file(), "gerador CSV not available")
    def test_gerador_bundle_and_nested_splits(self) -> None:
        bundle = chronos_rul.load_dataset_bundle("gerador", GERADOR_CSV)
        self.assertEqual(len(bundle["clean"]), 1910)
        self.assertEqual(len(bundle["endpoints"]), 574)
        self.assertEqual(len(bundle["sensor_columns"]), 8)

        outer_folds = chronos_rul.build_nested_splits(bundle["profile"])
        self.assertEqual(len(outer_folds), 4)
        for outer in outer_folds:
            self.assertEqual(len(outer["inner_folds"]), 3)
            for inner in outer["inner_folds"]:
                self.assertTrue(set(outer["test_ids"]).isdisjoint(inner["val_ids"]))
                self.assertTrue(set(outer["test_ids"]).isdisjoint(inner["inner_train_ids"]))
                self.assertTrue(set(inner["val_ids"]).isdisjoint(inner["inner_train_ids"]))

    @unittest.skipUnless(GEARBOX_CSV.is_file(), "gearbox CSV not available")
    def test_gearbox_bundle_and_nested_splits(self) -> None:
        bundle = chronos_rul.load_dataset_bundle("gearbox", GEARBOX_CSV)
        self.assertEqual(len(bundle["clean"]), 2946)
        self.assertEqual(len(bundle["endpoints"]), 822)
        outer_folds = chronos_rul.build_nested_splits(bundle["profile"])
        self.assertEqual(len(outer_folds), 3)
        for outer in outer_folds:
            self.assertEqual(len(outer["inner_folds"]), 2)


class TestCausalGridAndMasking(unittest.TestCase):
    def _synthetic_fault_df(self) -> pd.DataFrame:
        # Irregular timestamps: 0h, 24h (gap 24h <= 48h -> 12h interpolated),
        # 96h (gap 72h > 48h -> intermediate 12h steps must remain NaN!).
        dates = pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-01-05 00:00:00",
            "2024-01-06 00:00:00",  # future relative to 2024-01-05
        ])
        sensors = [f"s{i}" for i in range(8)]
        data = {
            "Date": dates,
            "Falha": [1, 1, 1, 1],
            "RUL": [1.0, 0.7, 0.2, 0.0],
        }
        for i, col in enumerate(sensors):
            data[col] = [10.0 + i, 20.0 + i, 50.0 + i, 999.0 + i]
        return pd.DataFrame(data), sensors

    def test_interpolation_gap_limit_and_future_causality(self) -> None:
        df, sensors = self._synthetic_fault_df()
        t_end = pd.Timestamp("2024-01-05 00:00:00")
        grid = chronos_rul.build_causal_12h_grid(df, t_end, sensors, context_steps=None)
        # Span 96h -> 9 steps (0, 12, 24, 36, 48, 60, 72, 84, 96)
        self.assertEqual(grid.shape, (8, 9))
        # Step 0h exact -> 10.0
        self.assertAlmostEqual(float(grid[0, 0]), 10.0, places=5)
        # Step 12h interpolated between 0h (10.0) and 24h (20.0) -> 15.0
        self.assertAlmostEqual(float(grid[0, 1]), 15.0, places=5)
        # Step 24h exact -> 20.0
        self.assertAlmostEqual(float(grid[0, 2]), 20.0, places=5)
        # Steps 36h..84h lie inside 72h gap (>48h) -> must be NaN, never 0.0
        self.assertTrue(np.isnan(grid[:, 3:8]).all())
        # Endpoint 96h exact -> 50.0, and future 999.0 never leaked
        self.assertAlmostEqual(float(grid[0, -1]), 50.0, places=5)

        # Perturb future row (> t_end) and verify exact equality
        df_future_perturbed = df.copy()
        df_future_perturbed.loc[3, "s0"] = -12345.0
        grid2 = chronos_rul.build_causal_12h_grid(df_future_perturbed, t_end, sensors, context_steps=None)
        np.testing.assert_array_equal(np.isnan(grid), np.isnan(grid2))
        np.testing.assert_allclose(grid[~np.isnan(grid)], grid2[~np.isnan(grid2)])

    def test_leading_unavailable_prefix_stays_nan(self) -> None:
        df, sensors = self._synthetic_fault_df()
        t_end = pd.Timestamp("2024-01-02 00:00:00")
        grid = chronos_rul.build_causal_12h_grid(df, t_end, sensors, context_steps=10)
        self.assertEqual(grid.shape, (8, 10))
        # Only last 3 steps (0h, 12h, 24h) have observations; first 7 must be NaN
        self.assertTrue(np.isnan(grid[:, :7]).all())
        self.assertTrue(np.isfinite(grid[:, 7:]).all())

    def test_refuses_context_exceeding_8192(self) -> None:
        sensors = [f"s{i}" for i in range(8)]
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2020-01-01 00:00:00", "2032-01-01 00:00:00"]),
            "Falha": [1, 1],
            "RUL": [1.0, 0.0],
            **{s: [1.0, 2.0] for s in sensors},
        })
        with self.assertRaisesRegex(ValueError, "8192"):
            chronos_rul.build_causal_12h_grid(
                df, pd.Timestamp("2032-01-01 00:00:00"), sensors, context_steps=None
            )


class TestPreprocessorAndCache(unittest.TestCase):
    def test_train_only_preprocessor_preserves_nan(self) -> None:
        sensors = [f"s{i}" for i in range(8)]
        clean = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "Falha": [1, 1, 2, 2],
            "RUL": [1.0, 0.0, 1.0, 0.0],
            **{s: [1.0, 3.0, 1000.0, 2000.0] for s in sensors},
        })
        prep = chronos_rul.CausalPreprocessor("zscore").fit(clean, [1], sensors)
        # Stats must come ONLY from Falha=1 (mean=2.0, std=1.0), ignoring Falha=2 (1000/2000)
        self.assertAlmostEqual(float(prep.mean[0]), 2.0, places=6)
        self.assertAlmostEqual(float(prep.std[0]), 1.0, places=6)

        grid = np.full((8, 4), np.nan, dtype=np.float32)
        grid[:, -1] = 2.0
        out = prep.transform(grid)
        self.assertTrue(np.isnan(out[:, :-1]).all())
        np.testing.assert_allclose(out[:, -1], 0.0, atol=1e-6)

    def test_cache_roundtrip_and_corruption_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(7)
            emb = rng.standard_normal((5, 8, chronos_rul.CHRONOS_D_MODEL)).astype(np.float32)
            code_sha = chronos_rul.compute_extractor_code_sha256()
            meta = {
                "schema_version": chronos_rul.SCHEMA_VERSION,
                "extractor_version": chronos_rul.EXTRACTOR_VERSION,
                "extractor_code_sha256": code_sha,
                "chronos_package_version": chronos_rul.CHRONOS_PACKAGE_VERSION,
                "dataset": "gerador",
                "sensor_columns": [f"s{i}" for i in range(8)],
                "sensor_data_sha256": "abc",
                "endpoints_sha256": "def",
                "train_ids": [1, 2],
                "preprocessor": {"mode": "zscore", "train_ids": [1, 2]},
                "context_spec": {"kind": "fixed", "context_steps": 32},
                "model_id": chronos_rul.CHRONOS_MODEL_ID,
                "model_revision": chronos_rul.CHRONOS_MODEL_REVISION,
                "patch_size": chronos_rul.CHRONOS_PATCH_SIZE,
                "d_model": chronos_rul.CHRONOS_D_MODEL,
                "use_reg_token": True,
                "extractor_dtype": "float32",
                "extractor_env": {"torch": "2.15.0+cu121", "cuda_available": True},
            }
            view_id = "testview001"
            chronos_rul.save_embedding_cache(tmp, view_id, emb, meta)

            # GPU -> CPU transfer: different consumer env (no extractor_env check) must succeed
            consumer_expected = dict(meta)
            consumer_expected.pop("extractor_env", None)
            loaded, loaded_meta = chronos_rul.load_embedding_cache(tmp, view_id, consumer_expected)
            np.testing.assert_allclose(loaded, emb)
            self.assertEqual(loaded_meta["extractor_env"]["torch"], "2.15.0+cu121")

            # Mismatched train_ids must be rejected
            with self.assertRaisesRegex(ValueError, "train_ids"):
                chronos_rul.load_embedding_cache(tmp, view_id, dict(consumer_expected, train_ids=[1, 2, 3]))

            # Regression #3: incompatible extractor version / code sha / package version must invalidate cache
            with self.assertRaisesRegex(ValueError, "extractor_version"):
                chronos_rul.load_embedding_cache(tmp, view_id, dict(consumer_expected, extractor_version="9.9.9"))
            with self.assertRaisesRegex(ValueError, "extractor_code_sha256"):
                chronos_rul.load_embedding_cache(tmp, view_id, dict(consumer_expected, extractor_code_sha256="deadbeef"))
            with self.assertRaisesRegex(ValueError, "chronos_package_version"):
                chronos_rul.load_embedding_cache(tmp, view_id, dict(consumer_expected, chronos_package_version="2.4.0"))

            # View identity itself must change if extractor version or code changes
            vid_a = chronos_rul.compute_view_identity(
                dataset_name="gerador", sensor_columns=meta["sensor_columns"],
                sensor_data_sha256="abc", endpoints_sha256="def",
                preprocessor_dict=meta["preprocessor"], context_spec=meta["context_spec"],
                extractor_version="1.0.0", extractor_code_sha256=code_sha,
            )
            vid_b = chronos_rul.compute_view_identity(
                dataset_name="gerador", sensor_columns=meta["sensor_columns"],
                sensor_data_sha256="abc", endpoints_sha256="def",
                preprocessor_dict=meta["preprocessor"], context_spec=meta["context_spec"],
                extractor_version="1.0.1", extractor_code_sha256=code_sha,
            )
            vid_c = chronos_rul.compute_view_identity(
                dataset_name="gerador", sensor_columns=meta["sensor_columns"],
                sensor_data_sha256="abc", endpoints_sha256="def",
                preprocessor_dict=meta["preprocessor"], context_spec=meta["context_spec"],
                extractor_version="1.0.0", extractor_code_sha256="changed_code",
            )
            self.assertNotEqual(vid_a, vid_b)
            self.assertNotEqual(vid_a, vid_c)

            # Corrupt npz payload must be rejected
            npz_path, _ = chronos_rul.cache_paths(tmp, view_id)
            chronos_rul.atomic_save_npz(npz_path, embeddings=emb + 1.0)
            with self.assertRaisesRegex(ValueError, "checksum"):
                chronos_rul.load_embedding_cache(tmp, view_id, consumer_expected)

    @unittest.skipUnless(GERADOR_CSV.is_file(), "gerador CSV not available")
    def test_different_configs_do_not_overwrite_and_unmounted_drive_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            res1 = chronos_rul.run_smoke_audit("elghoussani", {
                "dataset": "gerador", "input_csv": str(GERADOR_CSV),
                "output_dir": str(tmp_path), "lr": 1e-3,
            })
            res2 = chronos_rul.run_smoke_audit("elghoussani", {
                "dataset": "gerador", "input_csv": str(GERADOR_CSV),
                "output_dir": str(tmp_path), "lr": 5e-4,
            })
            self.assertNotEqual(res1["run_identity"], res2["run_identity"])
            self.assertNotEqual(res1["manifest_path"], res2["manifest_path"])
            self.assertTrue(Path(res1["manifest_path"]).is_file())
            self.assertTrue(Path(res2["manifest_path"]).is_file())
            cfg1 = json.loads((Path(res1["manifest_path"]).parent / "config.json").read_text(encoding="utf-8"))
            cfg2 = json.loads((Path(res2["manifest_path"]).parent / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg1["lr"], 1e-3)
            self.assertEqual(cfg2["lr"], 5e-4)

            # Regression #5: unmounted Drive path must fail without creating fake Drive directory
            fake_drive_root = tmp_path / "fake_drive"
            fake_out = fake_drive_root / "MyDrive" / "cer_out"
            with self.assertRaisesRegex(RuntimeError, "Google Drive"):
                chronos_rul.run_smoke_audit("elghoussani", {
                    "dataset": "gerador", "input_csv": str(GERADOR_CSV),
                    "output_dir": str(fake_out),
                    "drive_mount_root": str(fake_drive_root),
                })
            self.assertFalse(fake_drive_root.exists())

    @unittest.skipUnless(GERADOR_CSV.is_file(), "gerador CSV not available")
    def test_named_destination_collisions_and_legacy_dirs_refused_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Case 1: config.run_id collision + same-config resumption
            res_id1 = chronos_rul.run_smoke_audit("elghoussani", {
                "dataset": "gerador",
                "input_csv": str(GERADOR_CSV),
                "output_dir": str(tmp_path / "out_run_id"),
                "run_id": "named_experiment",
                "lr": 1e-3,
            })
            dir_id1 = Path(res_id1["run_dir"])
            snap_id1 = {
                p.relative_to(dir_id1).as_posix(): p.read_bytes()
                for p in sorted(dir_id1.rglob("*")) if p.is_file()
            }
            zip_id1_bytes = Path(res_id1["zip_path"]).read_bytes()

            with self.assertRaisesRegex(RuntimeError, "Conflito de identidade"):
                chronos_rul.run_smoke_audit("elghoussani", {
                    "dataset": "gerador",
                    "input_csv": str(GERADOR_CSV),
                    "output_dir": str(tmp_path / "out_run_id"),
                    "run_id": "named_experiment",
                    "lr": 1e-2,
                })
            snap_id1_after = {
                p.relative_to(dir_id1).as_posix(): p.read_bytes()
                for p in sorted(dir_id1.rglob("*")) if p.is_file()
            }
            self.assertEqual(snap_id1, snap_id1_after)
            self.assertEqual(zip_id1_bytes, Path(res_id1["zip_path"]).read_bytes())

            # Same config on same run_id is allowed
            res_id1_again = chronos_rul.run_smoke_audit("elghoussani", {
                "dataset": "gerador",
                "input_csv": str(GERADOR_CSV),
                "output_dir": str(tmp_path / "out_run_id"),
                "run_id": "named_experiment",
                "lr": 1e-3,
            })
            self.assertEqual(res_id1_again["run_identity"], res_id1["run_identity"])

            # Case 2: config.run_dir collision + same-config resumption
            cfg_run_dir = tmp_path / "explicit_cfg_run_dir"
            res_cfg1 = chronos_rul.run_smoke_audit("elghoussani", {
                "dataset": "gerador",
                "input_csv": str(GERADOR_CSV),
                "run_dir": str(cfg_run_dir),
                "lr": 1e-3,
            })
            manifest_bytes = (cfg_run_dir / "manifest.json").read_bytes()
            config_bytes = (cfg_run_dir / "config.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Conflito de identidade"):
                chronos_rul.run_smoke_audit("elghoussani", {
                    "dataset": "gerador",
                    "input_csv": str(GERADOR_CSV),
                    "run_dir": str(cfg_run_dir),
                    "lr": 1e-2,
                })
            self.assertEqual((cfg_run_dir / "manifest.json").read_bytes(), manifest_bytes)
            self.assertEqual((cfg_run_dir / "config.json").read_bytes(), config_bytes)
            res_cfg1_again = chronos_rul.run_smoke_audit("elghoussani", {
                "dataset": "gerador",
                "input_csv": str(GERADOR_CSV),
                "run_dir": str(cfg_run_dir),
                "lr": 1e-3,
            })
            self.assertEqual(res_cfg1_again["run_identity"], res_cfg1["run_identity"])

            # Case 3: explicit run_dir argument of run_smoke_audit
            arg_run_dir = tmp_path / "explicit_arg_run_dir"
            res_arg1 = chronos_rul.run_smoke_audit(
                "elghoussani",
                {"dataset": "gerador", "input_csv": str(GERADOR_CSV), "lr": 1e-3},
                run_dir=arg_run_dir,
            )
            arg_cfg_bytes = (arg_run_dir / "config.json").read_bytes()
            arg_man_bytes = (arg_run_dir / "manifest.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Conflito de identidade"):
                chronos_rul.run_smoke_audit(
                    "elghoussani",
                    {"dataset": "gerador", "input_csv": str(GERADOR_CSV), "lr": 1e-2},
                    run_dir=arg_run_dir,
                )
            self.assertEqual((arg_run_dir / "config.json").read_bytes(), arg_cfg_bytes)
            self.assertEqual((arg_run_dir / "manifest.json").read_bytes(), arg_man_bytes)
            res_arg1_again = chronos_rul.run_smoke_audit(
                "elghoussani",
                {"dataset": "gerador", "input_csv": str(GERADOR_CSV), "lr": 1e-3},
                run_dir=arg_run_dir,
            )
            self.assertEqual(res_arg1_again["run_identity"], res_arg1["run_identity"])

            # Case 4: legacy non-empty folder without verifiable run_identity refuses safely
            legacy_dir = tmp_path / "legacy_non_empty"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "old_artifact.txt").write_bytes(b"keep-me")
            (legacy_dir / "manifest.json").write_text('{"status": "old"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "run_identity"):
                chronos_rul.run_smoke_audit(
                    "elghoussani",
                    {"dataset": "gerador", "input_csv": str(GERADOR_CSV)},
                    run_dir=legacy_dir,
                )
            self.assertEqual((legacy_dir / "old_artifact.txt").read_bytes(), b"keep-me")
            self.assertEqual((legacy_dir / "manifest.json").read_text(encoding="utf-8"), '{"status": "old"}')

    @unittest.skipUnless(GERADOR_CSV.is_file(), "gerador CSV not available")
    def test_drive_canonicalization_relative_absolute_and_symlink_under_temp_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd_path = Path(tmp)
            temp_drive_root = cwd_path / "drive"
            symlink_alias = cwd_path / "drive_alias"
            symlink_alias.symlink_to(temp_drive_root)
            orig_cwd = Path.cwd()
            try:
                os.chdir(cwd_path)
                # Relative path against cwd -> temp_drive_root
                with self.assertRaisesRegex(RuntimeError, "Google Drive"):
                    chronos_rul.run_smoke_audit("elghoussani", {
                        "dataset": "gerador",
                        "input_csv": str(GERADOR_CSV),
                        "output_dir": "drive/MyDrive/out",
                        "drive_mount_root": str(temp_drive_root),
                    })
                self.assertFalse(temp_drive_root.exists())

                # Absolute path -> temp_drive_root
                with self.assertRaisesRegex(RuntimeError, "Google Drive"):
                    chronos_rul.run_smoke_audit("elghoussani", {
                        "dataset": "gerador",
                        "input_csv": str(GERADOR_CSV),
                        "output_dir": str(temp_drive_root / "MyDrive" / "abs_out"),
                        "drive_mount_root": str(temp_drive_root),
                    })
                self.assertFalse(temp_drive_root.exists())

                # Symlink alias -> temp_drive_root (via output_dir and run_dir arg)
                with self.assertRaisesRegex(RuntimeError, "Google Drive"):
                    chronos_rul.run_smoke_audit("elghoussani", {
                        "dataset": "gerador",
                        "input_csv": str(GERADOR_CSV),
                        "output_dir": "drive_alias/MyDrive/sym_out",
                        "drive_mount_root": "drive",
                    })
                with self.assertRaisesRegex(RuntimeError, "Google Drive"):
                    chronos_rul.run_smoke_audit(
                        "elghoussani",
                        {
                            "dataset": "gerador",
                            "input_csv": str(GERADOR_CSV),
                            "output_dir": "local_ok",
                            "drive_mount_root": str(temp_drive_root),
                        },
                        run_dir="drive_alias/MyDrive/sym_run",
                    )
                self.assertFalse(temp_drive_root.exists())

                # Normal local path works cleanly
                ok_res = chronos_rul.run_smoke_audit("elghoussani", {
                    "dataset": "gerador",
                    "input_csv": str(GERADOR_CSV),
                    "output_dir": "local_outputs",
                    "drive_mount_root": str(temp_drive_root),
                })
                self.assertEqual(ok_res["status"], "smoke")
                self.assertFalse(temp_drive_root.exists())
            finally:
                os.chdir(orig_cwd)


class TestHierarchicalMetricsAndSmoke(unittest.TestCase):
    def test_hierarchical_macro_vs_pooled(self) -> None:
        # Group A has 2 faults (1 with 1 row err=0.1, 2 with 3 rows err=0.3) -> group A macro = 0.2
        # Group B has 1 fault (3 with 1 row err=0.4) -> group B macro = 0.4
        # Overall macro = (0.2 + 0.4) / 2 = 0.3; pooled MAE = (0.1 + 3*0.3 + 0.4) / 5 = 0.28
        df = pd.DataFrame({
            "Falha": [1, 2, 2, 2, 3],
            "equipment": ["A", "A", "A", "A", "B"],
            "phase": ["saudavel", "saudavel", "pos_FPT", "pos_FPT", "pos_FPT"],
            "y_true": [0.5, 0.5, 0.5, 0.05, 0.5],
            "pred_raw": [0.6, 0.8, 0.8, 0.35, 0.9],
        })
        m = chronos_rul.compute_hierarchical_metrics(df)
        self.assertAlmostEqual(m["macro"]["mae"], 0.3, places=6)
        self.assertAlmostEqual(m["pooled"]["mae"], 0.28, places=6)
        self.assertIsNotNone(m["near_zero"])

    @unittest.skipUnless(GERADOR_CSV.is_file(), "gerador CSV not available")
    def test_cli_smoke_gerador(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps({
                "dataset": "gerador",
                "input_csv": str(GERADOR_CSV),
                "output_dir": tmp,
            }), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable, str(ROOT / "chronos_rul.py"),
                    "--article", "elghoussani",
                    "--config", str(cfg_path),
                    "--mode", "smoke",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = json.loads(proc.stdout)
            self.assertEqual(out["status"], "smoke")
            self.assertEqual(out["clean_keys"], 1910)
            self.assertEqual(out["endpoint_count"], 574)
            self.assertTrue(Path(out["zip_path"]).is_file())


class TestPortableZipExport(unittest.TestCase):
    def test_dotted_run_names_preserve_distinct_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dir_v1 = tmp_path / "experiment.v1"
            dir_v2 = tmp_path / "experiment.v2"
            dir_v1.mkdir()
            dir_v2.mkdir()
            payload_v1 = json.dumps({"run_id": "experiment.v1", "lr": 1e-3})
            payload_v2 = json.dumps({"run_id": "experiment.v2", "lr": 1e-2})
            (dir_v1 / "manifest.json").write_text(payload_v1, encoding="utf-8")
            (dir_v2 / "manifest.json").write_text(payload_v2, encoding="utf-8")

            zip_v1 = chronos_rul.create_portable_zip(dir_v1)
            zip_v2 = chronos_rul.create_portable_zip(dir_v2)

            self.assertNotEqual(zip_v1, zip_v2)
            self.assertEqual(zip_v1.name, "experiment.v1.zip")
            self.assertEqual(zip_v2.name, "experiment.v2.zip")
            self.assertTrue(zip_v1.is_file())
            self.assertTrue(zip_v2.is_file())

            with zipfile.ZipFile(zip_v1, "r") as zf1:
                self.assertEqual(zf1.read("manifest.json").decode("utf-8"), payload_v1)
            with zipfile.ZipFile(zip_v2, "r") as zf2:
                self.assertEqual(zf2.read("manifest.json").decode("utf-8"), payload_v2)


if __name__ == "__main__":
    unittest.main()
