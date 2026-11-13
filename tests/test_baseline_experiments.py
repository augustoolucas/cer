"""Fast synthetic tests for baseline_experiments (no real-data fitting).

The end-to-end pilot is run through the CLI and verified separately; these tests
cover protocol invariants, causality, train-only statistics, the fit runner and
the cache/budget/resource rules with small synthetic inputs.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baseline_experiments as b  # noqa: E402

GROUP_SETS = b.load_config()["feature_group_sets"]


def _recipe(family="ridge", params=None, weights=None, **features):
    cfg = b.load_config()
    overrides = {f"features.{k}": v for k, v in features.items()}
    if weights is not None:
        overrides["weights"] = weights
    return b.normalize_recipe(family, params or {}, overrides, cfg["default_recipe"])


def _synthetic_dataset(groups=("A", "B", "C"), records=14, seed=0):
    """Small synthetic Dataset whose feature table goes through the shared cer_rul code."""
    rng = np.random.default_rng(seed)
    sensors = b.cer.SENSOR_COLUMNS
    clean_rows, endpoint_rows, meta_rows = [], [], []
    falha = 1
    for group in groups:
        for _component in range(2):
            baseline = rng.uniform(0.2, 0.8, size=len(sensors))
            for index in range(records):
                date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=index)
                values = baseline + 0.01 * rng.standard_normal(len(sensors))
                clean_rows.append({"Date": date, "Falha": falha,
                                   **dict(zip(sensors, values))})
                if index >= 3:
                    endpoint_id = f"F{falha:02d}-r{index:04d}"
                    y = float(np.clip(1 - index / records, 0, 1))
                    endpoint_rows.append({"endpoint_id": endpoint_id, "Falha": falha,
                                          "Date": date, "record_index": index})
                    meta_rows.append({"endpoint_id": endpoint_id, "Falha": falha,
                                      "equipment": group, "Date": date,
                                      "phase": "saudavel" if index < records // 2
                                      else "pos_FPT", "y": y})
            falha += 1
    clean = pd.DataFrame(clean_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    meta = pd.DataFrame(meta_rows)
    profile = {"measurement_by_id": {fid: "gerador" for fid in range(1, falha)},
               "sensor_columns": sensors}
    return b.Dataset(name="gerador", profile=profile, clean=clean, endpoints=endpoints,
                     meta=meta, csv_path=Path("synthetic.csv"), csv_sha256="0",
                     endpoints_sha256="0", outer_order=tuple(groups),
                     sensor_columns=tuple(sensors))


def _synthetic_meta(groups=("B", "C", "D"), per_component=10, seed=0):
    """Synthetic endpoint metadata and a feature table with causal-looking columns."""
    rng = np.random.default_rng(seed)
    rows, table_rows = [], []
    falha = 1
    for group in groups:
        for _component in range(2):
            for i in range(per_component):
                endpoint_id = f"F{falha:02d}-r{i:04d}"
                level = rng.uniform(0, 1)
                y = float(np.clip(1 - i / per_component + 0.02 * rng.standard_normal(), 0, 1))
                rows.append({"endpoint_id": endpoint_id, "Falha": falha, "equipment": group,
                             "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                             "phase": "saudavel" if i < per_component // 2 else "pos_FPT",
                             "y": y})
                table_rows.append({
                    "cur__s0": level + 0.1 * y, "cur__s1": level - 0.2 * y,
                    "dmean__s0": level, "dstd__s0": 0.05 + 0.01 * rng.standard_normal(),
                    "dslope__s0": -0.1 * y,
                    "nobs__s0": 10.0 + i, "missbins__s0": 0.25,
                    "window__records": float(i + 1),
                    "timing__prior_gap_h": 24.0,
                    "time__elapsed_days": float(i),
                    "rel10__s0": level - 1.0, "rel10__s1": level - 1.0,
                    "rob_median__s0": level, "rob_iqr__s0": 0.1,
                })
            falha += 1
    meta = pd.DataFrame(rows)
    table = pd.DataFrame(table_rows, index=pd.Index(meta["endpoint_id"], name="endpoint_id"))
    return meta, table


def _small_config(families=("mean", "ridge")):
    cfg = b.effective_config(b.load_config(), pilot=False)
    cfg["families"] = {name: cfg["families"][name] for name in families}
    cfg["stages"].update({"screen_trials": 2, "tune_top_k": 1, "tune_extra_trials": 2,
                          "tune_min_total_trials": 2, "tune_patience_trials": 2,
                          "batch_size": 2})
    return cfg


def _runner(tmp, *, available=8 * b.GIB, max_fits=10000, timeout_s=60.0, resources=None):
    cfg = b.load_config()
    resources = dict(resources or cfg["resources"])
    policy = b.ResourcePolicy(resources, available_bytes=lambda: available)
    budget = b.FitBudget(max_fits=max_fits)
    cache = b.FitCache(Path(tmp) / "fits")
    cache.directory.mkdir(parents=True, exist_ok=True)
    return b.FitRunner(cache, budget, policy, timeout_s=timeout_s, poll_s=0.01,
                       ledger_path=Path(tmp) / "ledger.json", pause_poll_s=0.01,
                       max_pause_s=0.2)


def _task(X=None, y=None, family="ridge", params=None, seed=1, key_salt="x"):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 3)) if X is None else X
    y = X[:, 0] * 0.1 + 0.5 if y is None else y
    eval_ids = tuple(f"e{i}" for i in range(5))
    key = b.sha256_text(f"{family}|{params}|{seed}|{key_salt}")
    return b.FitTask(key=key, family=family, params=params or {"alpha": 1.0}, seed=seed,
                     X_train=X, y_train=y, weights=None, X_eval=X[:5], eval_ids=eval_ids,
                     timeout_s=60.0)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid_and_pilot_restricts_families(self):
        cfg = b.load_config()
        eff = b.effective_config(cfg, pilot=True)
        self.assertEqual(set(eff["families"]), set(cfg["pilot"]["fixed_candidates"]))
        self.assertEqual(eff["budgets"]["max_fits_per_dataset"], 200)
        self.assertFalse(eff["tuning_enabled"])
        self.assertTrue(b.effective_config(cfg, pilot=False)["tuning_enabled"])

    def test_missing_section_is_rejected(self):
        cfg = b.load_config()
        del cfg["stages"]
        with self.assertRaises(b.ConfigError):
            b.validate_config(cfg)

    def test_bad_space_definitions_are_rejected(self):
        cfg = b.load_config()
        cfg["families"]["ridge"]["space"]["alpha"] = {"type": "float", "low": 0.0,
                                                      "high": 1.0, "log": True}
        with self.assertRaises(b.ConfigError):
            b.validate_config(cfg)
        cfg = b.load_config()
        cfg["families"]["ridge"]["space"]["alpha"] = {"type": "float", "low": 2.0, "high": 1.0}
        with self.assertRaises(b.ConfigError):
            b.validate_config(cfg)

    def test_unknown_family_and_pilot_param_are_rejected(self):
        cfg = b.load_config()
        cfg["families"]["xgboost"] = {"kind": "model", "space": {}}
        with self.assertRaises(b.ConfigError):
            b.validate_config(cfg)
        cfg = b.load_config()
        cfg["pilot"]["fixed_candidates"]["ridge"]["params"]["beta"] = 1.0
        with self.assertRaises(b.ConfigError):
            b.validate_config(cfg)

    def test_pilot_is_refused_for_gearbox_before_any_data_load(self):
        with self.assertRaises(b.ConfigError):
            b.Run("gearbox", pilot=True, output_root=ROOT / "results" / "nope")


class RecipeTests(unittest.TestCase):
    def test_invalid_combinations_are_rejected(self):
        with self.assertRaises(b.InvalidRecipe):
            b.validate_recipe(_recipe("knn", weights="component"), "gerador", GROUP_SETS)
        with self.assertRaises(b.InvalidRecipe):
            b.validate_recipe(_recipe("mean", weights="group_equal"), "gerador", GROUP_SETS)
        with self.assertRaises(b.InvalidRecipe):
            b.validate_recipe(_recipe(measurement_onehot=True), "gerador", GROUP_SETS)
        recipe = _recipe("ridge")
        recipe["postprocess"]["ema_alpha"] = 0.3
        with self.assertRaises(b.InvalidRecipe):
            b.validate_recipe(recipe, "gerador", GROUP_SETS)
        with self.assertRaises(b.InvalidRecipe):
            b.validate_recipe(_recipe("xgboost"), "gerador", GROUP_SETS)

    def test_relative_and_variance_need_sensors_group(self):
        columns = ["cur__s0", "cur__s1", "time__elapsed_days", "nobs__s0", "rel10__s0"]
        with self.assertRaises(b.InvalidRecipe):
            b.select_feature_columns(columns, _recipe(feature_groups="cadence+elapsed",
                                                      relative_reference=10)["features"],
                                     GROUP_SETS, "s2")
        with self.assertRaises(b.InvalidRecipe):
            b.select_feature_columns(columns, _recipe(feature_groups="cadence+elapsed",
                                                      drop_variance=True)["features"],
                                     GROUP_SETS, "s2")

    def test_feature_group_semantics(self):
        columns = ["cur__s0", "dmean__s0", "cur__s2", "rob_median__s2", "nobs__s0",
                   "timing__prior_gap_h", "time__elapsed_days", "rel10__s0", "rel30__s0"]

        def pick(**kwargs):
            return b.select_feature_columns(columns, _recipe(**kwargs)["features"],
                                            GROUP_SETS, "s2")

        self.assertEqual(pick(feature_groups="sensors", representation="summary"),
                         ["cur__s0", "dmean__s0", "cur__s2"])
        self.assertEqual(pick(feature_groups="cadence+elapsed", representation="current"),
                         ["nobs__s0", "timing__prior_gap_h", "time__elapsed_days"])
        self.assertEqual(pick(feature_groups="sensors+elapsed", representation="robust",
                              relative_reference=10, drop_variance=True),
                         ["cur__s0", "time__elapsed_days", "rel10__s0"])
        drop = pick(feature_groups="sensors", representation="robust", drop_variance=True)
        self.assertNotIn("cur__s2", drop)
        self.assertNotIn("rob_median__s2", drop)

    def test_complexity_counts_stages(self):
        plain = _recipe("ridge")
        more = copy.deepcopy(plain)
        more["weights"] = "component"
        more["postprocess"]["ema_alpha"] = 0.5
        self.assertLess(b.recipe_complexity(plain, 10), b.recipe_complexity(more, 10))

    def test_seed_and_recipe_id_are_deterministic(self):
        self.assertEqual(b.derive_seed("ns", "a", 1), b.derive_seed("ns", "a", 1))
        self.assertNotEqual(b.derive_seed("ns", "a", 1), b.derive_seed("ns", "a", 2))
        self.assertEqual(b.recipe_id(_recipe()), b.recipe_id(_recipe()))

    def test_unknown_constant_and_weight_rules_in_factory(self):
        with self.assertRaises(b.InvalidRecipe):
            b.build_estimator("unknown", {}, 0)
        model = b.build_estimator("mean", {}, 0)
        with self.assertRaises(b.InvalidRecipe):
            model.fit(np.zeros((3, 1)), np.zeros(3), sample_weight=np.ones(3))


class MetricAndScoreTests(unittest.TestCase):
    def test_inner_score_is_equal_weight_over_groups_then_components(self):
        # Group X has components 1 (error 0.1) and 2 (error 0.3) -> 0.2 ; group Y -> 0.5.
        oof = pd.DataFrame({
            "endpoint_id": ["a", "b", "c", "d", "e"],
            "Falha": [1, 1, 2, 2, 3],
            "equipment": ["X", "X", "X", "X", "Y"],
            "Date": pd.to_datetime(["2024-01-01"] * 5),
            "phase": ["saudavel"] * 5,
            "y": [0.5, 0.5, 0.5, 0.5, 0.5],
            "pred_post": [0.6, 0.4, 0.8, 0.2, 1.0],
        })
        score = b.inner_score(oof)
        self.assertAlmostEqual(score["group_scores_pp"]["X"], 20.0)
        self.assertAlmostEqual(score["group_scores_pp"]["Y"], 50.0)
        self.assertAlmostEqual(score["score_pp"], 35.0)

    def test_outer_metrics_macro_over_components(self):
        frame = pd.DataFrame({
            "Falha": [1, 1, 2], "equipment": ["A", "A", "A"],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "phase": ["saudavel"] * 3, "y": [0.0, 0.0, 1.0],
            "pred_raw": [0.1, 0.1, 0.7], "pred_post": [0.1, 0.1, 0.7]})
        metrics = b.outer_metrics(frame)
        self.assertAlmostEqual(metrics["component_equal_mae_pp"], (10.0 + 30.0) / 2)
        self.assertAlmostEqual(metrics["mae_pp"], (10 + 10 + 30) / 3)
        self.assertEqual(metrics["raw_outside_unit_interval"], 0)

    def test_tie_rule_prefers_complexity_then_rmse_then_id(self):
        candidates = [
            {"candidate_id": "z", "score_pp": 20.00, "rmse_pp": 5.0, "complexity": 10},
            {"candidate_id": "a", "score_pp": 20.05, "rmse_pp": 9.0, "complexity": 3},
            {"candidate_id": "m", "score_pp": 20.11, "rmse_pp": 1.0, "complexity": 1},
        ]
        chosen = b.select_best(candidates, 0.1)
        self.assertEqual(chosen["candidate_id"], "a")
        equal = [
            {"candidate_id": "q", "score_pp": 1.0, "rmse_pp": 2.0, "complexity": 4},
            {"candidate_id": "p", "score_pp": 1.0, "rmse_pp": 2.0, "complexity": 4},
        ]
        self.assertEqual(b.select_best(equal, 0.1)["candidate_id"], "p")
        self.assertIsNone(b.select_best([{"score_pp": None}], 0.1))

    def test_ensemble_of_two_members_mean_equals_median(self):
        base = pd.DataFrame({"endpoint_id": ["a", "b"], "Falha": [1, 1],
                             "equipment": ["X", "X"], "Date": pd.to_datetime(["2024-01-01"] * 2),
                             "phase": ["saudavel"] * 2, "y": [0.1, 0.2],
                             "pred_post": [0.1, 0.9]})
        other = base.assign(pred_post=[0.5, 0.3])
        mean = b.ensemble_predictions([base, other], "mean")
        median = b.ensemble_predictions([base, other], "median")
        np.testing.assert_allclose(mean["pred_post"], median["pred_post"])


class PostprocessTests(unittest.TestCase):
    def test_clip_ema_running_min_are_causal(self):
        frame = pd.DataFrame({"Falha": [1, 1, 1, 1], "Date": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "pred_raw": [1.3, 0.2, 0.9, -0.5]})
        out = b.postprocess_predictions(frame, clip01=True, ema_alpha=0.5, running_min=True)
        clipped = np.array([1.0, 0.2, 0.9, 0.0])
        ema = [clipped[0]]
        for value in clipped[1:]:
            ema.append(0.5 * value + 0.5 * ema[-1])
        np.testing.assert_allclose(out["pred_post"], np.minimum.accumulate(ema))

    def test_future_rows_never_change_past_predictions(self):
        base = pd.DataFrame({"Falha": [1, 1, 2], "Date": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-01"]), "pred_raw": [0.4, 0.8, 0.6]})
        future = pd.concat([base, pd.DataFrame({"Falha": [1, 2], "Date": pd.to_datetime(
            ["2024-01-03", "2024-01-02"]), "pred_raw": [0.0, 1.0]})], ignore_index=True)
        options = dict(clip01=True, ema_alpha=0.25, running_min=True)
        small = b.postprocess_predictions(base, **options)["pred_post"].to_numpy()
        large = b.postprocess_predictions(future, **options)["pred_post"].to_numpy()
        np.testing.assert_allclose(small, large[:3])

    def test_invalid_ema_alpha_rejected(self):
        frame = pd.DataFrame({"Falha": [1], "Date": pd.to_datetime(["2024-01-01"]),
                              "pred_raw": [0.5]})
        with self.assertRaises(b.InvalidRecipe):
            b.postprocess_predictions(frame, clip01=False, ema_alpha=0.1, running_min=False)


class PreprocessingTests(unittest.TestCase):
    def test_statistics_come_from_training_rows_only(self):
        train = pd.DataFrame({"a": np.arange(100, dtype=float), "const": 1.0})
        eval_ = pd.DataFrame({"a": [1e6, -1e6], "const": 1.0})
        x_train, x_eval, names = b.prepare_matrices(
            train, eval_, {"method": "none", "missing_indicators": False,
                           "clip_quantiles": True}, "legacy")
        self.assertEqual(names, ["a"])
        high = np.quantile(np.arange(100, dtype=float), 0.99)
        self.assertAlmostEqual(float(x_eval[0, 0]), high)
        low = np.quantile(np.arange(100, dtype=float), 0.01)
        self.assertAlmostEqual(float(x_eval[1, 0]), low)

    def test_eval_extremes_do_not_change_fitted_transform_of_train(self):
        rng = np.random.default_rng(5)
        train = pd.DataFrame({"a": rng.standard_normal(60), "b": rng.standard_normal(60)})
        preprocess = {"method": "standard", "missing_indicators": True, "clip_quantiles": True}
        first, _, _ = b.prepare_matrices(train, train.iloc[:4] * 0 + 1,
                                         preprocess, "legacy")
        second, _, _ = b.prepare_matrices(train, train.iloc[:4] * 1000,
                                          preprocess, "legacy")
        np.testing.assert_allclose(first, second)

    def test_constant_and_dedup_selection_is_train_only(self):
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "dup": [1.0, 2.0, 3.0],
                              "const": [5.0, 5.0, 5.0], "nan_only": [np.nan] * 3})
        evaluation = train.copy()
        kept_legacy, _, names_legacy = b.prepare_matrices(
            train, evaluation, {"method": "none", "missing_indicators": False,
                                "clip_quantiles": False}, "legacy")
        kept_dedup, _, names_dedup = b.prepare_matrices(
            train, evaluation, {"method": "none", "missing_indicators": False,
                                "clip_quantiles": False}, "dedup")
        self.assertEqual(names_legacy, ["x", "dup"])
        self.assertEqual(names_dedup, ["x"])
        self.assertEqual(kept_dedup.shape, (3, 1))

    def test_no_informative_training_column_is_invalid(self):
        train = pd.DataFrame({"c": [1.0, 1.0]})
        with self.assertRaises(b.InvalidRecipe):
            b.prepare_matrices(train, train, {"method": "none", "missing_indicators": False,
                                              "clip_quantiles": False}, "legacy")

    def test_group_equal_weights_have_mean_one_and_equal_groups(self):
        meta = pd.DataFrame({"equipment": ["A", "A", "A", "B"], "Falha": [1, 1, 2, 3],
                             "phase": ["saudavel"] * 4})
        weights = b.training_weights("group_equal", meta)
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertAlmostEqual(weights[:3].sum(), weights[3])


class CausalFeatureTests(unittest.TestCase):
    def _group(self, n=40, seed=7):
        rng = np.random.default_rng(seed)
        dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            np.cumsum(rng.uniform(1, 9, n)), unit="h")
        frame = pd.DataFrame({"Date": dates, "Falha": 1, "RUL": 0.5})
        for key, column in zip(("s0", "s1"), ("g_0_RMS", "g_1_Meanfreq")):
            frame[column] = rng.uniform(0.5, 2.0, n)
        return frame.reset_index(drop=True), ("g_0_RMS", "g_1_Meanfreq")

    def test_extra_features_ignore_future_records(self):
        group, columns = self._group()
        keys = ("s0", "s1")
        eps = {"floor": 1e-8, "relative": 1e-6}
        endpoint = pd.Series({"record_index": 32, "Date": group["Date"].iloc[32]})
        full = b._extra_feature_row(endpoint, group, columns, keys, eps, 12, 16)
        truncated = b._extra_feature_row(endpoint, group.iloc[:33].reset_index(drop=True),
                                         columns, keys, eps, 12, 16)
        self.assertEqual(full.keys(), truncated.keys())
        for name in full:
            self.assertAlmostEqual(full[name], truncated[name], places=12, msg=name)

    def test_relative_reference_uses_first_records_only(self):
        group, columns = self._group()
        eps = {"floor": 1e-8, "relative": 1e-6}
        endpoint = pd.Series({"record_index": 35, "Date": group["Date"].iloc[35]})
        row = b._extra_feature_row(endpoint, group, columns, ("s0", "s1"), eps, 12, 16)
        reference = group.iloc[:10][list(columns)].to_numpy().mean(axis=0)[0]
        current = group.iloc[35]["g_0_RMS"]
        expected = (current - reference) / (abs(reference) + max(1e-8, 1e-6 * abs(reference)))
        self.assertAlmostEqual(row["rel10__s0"], expected)
        self.assertAlmostEqual(row["time__elapsed_days"],
                               (group["Date"].iloc[35] - group["Date"].iloc[0]).total_seconds()
                               / 86400.0)

    def test_robust_stats_on_constant_series_are_degenerate(self):
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        values = np.full((3, 1), 2.0)
        stats = b._robust_sensor_stats(dates, values, pd.Timestamp("2024-01-03"), 24, 4, ("s0",))
        self.assertAlmostEqual(stats["rob_iqr__s0"], 0.0)
        self.assertAlmostEqual(stats["rob_delta__s0"], 0.0)
        self.assertAlmostEqual(stats["rob_median__s0"], 2.0)


class GearboxHelperTests(unittest.TestCase):
    def test_event_position_and_component_mapping(self):
        self.assertEqual(b.gearbox_event_of(1), 1)
        self.assertEqual(b.gearbox_event_of(7), 2)
        self.assertEqual(b.gearbox_event_of(18), 3)
        self.assertEqual(b.gearbox_position_of(7), 1)
        self.assertEqual(b.gearbox_component_of(2), "HSS")
        self.assertEqual(b.gearbox_component_of(3), "IMS")
        self.assertEqual(b.gearbox_component_of(6), "PLANETARIA")
        with self.assertRaises(ValueError):
            b.gearbox_event_of(19)

    def _meta(self, dates_by_falha, target=0.5):
        rows = []
        for falha in range(1, 7):
            for index, date in enumerate(dates_by_falha.get(falha, [])):
                rows.append({"endpoint_id": f"F{falha:02d}-{index}", "Falha": falha,
                             "equipment": "E1", "Date": pd.Timestamp(date), "phase": "saudavel",
                             "y": target})
        return pd.DataFrame(rows)

    def _features(self, meta):
        table = pd.DataFrame({"cur__s0": np.arange(len(meta), dtype=float)},
                             index=meta["endpoint_id"].to_numpy())
        return table

    def test_joint_rows_are_strictly_aligned_and_never_filled(self):
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        meta = self._meta({f: dates for f in range(1, 7)})
        meta = meta[~((meta["Falha"] == 6) & (meta["Date"] == pd.Timestamp("2024-01-02")))]
        joint = b.build_joint_event_rows(meta, self._features(meta), ["cur__s0"])
        self.assertEqual(len(joint), 2)
        self.assertEqual(joint.attrs["dropped_unaligned"], 1)  # 01-02 missing for Falha 6
        self.assertEqual(len([c for c in joint.columns if c.startswith("m")]), 6)
        self.assertIn("m1__cur__s0", joint.columns)

    def test_joint_rejects_target_mismatch_and_incomplete_event(self):
        dates = ["2024-01-01"]
        meta = self._meta({f: dates for f in range(1, 7)})
        meta.loc[meta["Falha"] == 3, "y"] = 0.9
        with self.assertRaises(ValueError):
            b.build_joint_event_rows(meta, self._features(meta), ["cur__s0"])
        incomplete = self._meta({f: dates for f in range(1, 6)})
        with self.assertRaises(ValueError):
            b.build_joint_event_rows(incomplete, self._features(incomplete), ["cur__s0"])

    def test_joint_event_track_fits_inner_outer_without_event_leakage(self):
        rows, values = [], []
        names = ["cur__s0", "cur__s1"]
        for event in range(1, 4):
            for position in range(1, 7):
                falha = (event - 1) * 6 + position
                for day in range(4):
                    endpoint = f"F{falha:02d}-{day}"
                    rows.append({"endpoint_id": endpoint, "Falha": falha,
                                 "equipment": f"E{event}",
                                 "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                                 "phase": "saudavel", "y": 0.9 - 0.1 * day})
                    values.append({"cur__s0": float(position + day),
                                   "cur__s1": float(event + position * day)})
        meta = pd.DataFrame(rows)
        features = pd.DataFrame(values, index=pd.Index(meta.endpoint_id, name="endpoint_id"))
        joint_meta, joint_features = b.make_joint_event_track(meta, features, names)
        self.assertEqual(len(joint_meta), 12)
        cfg = _small_config(("ridge",))
        cfg["feature_group_sets"] = b.load_config()["feature_group_sets"]
        cfg["tuning_enabled"] = False
        train = joint_meta[joint_meta.equipment != "E3"].reset_index(drop=True)
        recipe = b.normalize_recipe("ridge", {"alpha": 1.0}, {"strategy": "joint_event"},
                                    cfg["default_recipe"])
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gearbox", outer_group="E3", train_meta=train,
                                   tables=lambda _w, _c: joint_features, runner=_runner(tmp),
                                   config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidate"),
                                   identity={}, variance_key="s4")
            inner = search._evaluate([(0, "joint", recipe)])[0]
            self.assertEqual(inner.status, "ok", inner.reasons)
            self.assertEqual(set(inner.oof.equipment), {"E1", "E2"})
            tasks = b.build_strategy_fold_tasks(
                joint_meta, joint_features, (joint_meta.equipment != "E3").to_numpy(),
                (joint_meta.equipment == "E3").to_numpy(), recipe,
                cfg["feature_group_sets"], lambda expert: b.derive_seed("joint", expert),
                {}, 60, "gearbox", "s4")
            self.assertEqual(len(tasks), 1)
            self.assertTrue(all(joint_meta.set_index("endpoint_id").loc[x, "equipment"] != "E3"
                                for x in tasks[0][1].train_ids))
            record = _runner(tmp).run([tasks[0][1]], purpose="final")[tasks[0][1].key]
            self.assertEqual(len(record.predictions), 4)

    def test_measurement_aggregation_mean_median_and_count_check(self):
        predictions = pd.DataFrame({
            "event": [1] * 6, "Date": pd.to_datetime(["2024-01-01"] * 6),
            "pred_post": [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]})
        mean = b.aggregate_measurement_predictions(predictions, "mean")
        median = b.aggregate_measurement_predictions(predictions, "median")
        self.assertAlmostEqual(mean["pred_post"].iloc[0], np.mean([0, .1, .2, .3, .4, 1]))
        self.assertAlmostEqual(median["pred_post"].iloc[0], 0.25)
        with self.assertRaises(ValueError):
            b.aggregate_measurement_predictions(predictions.iloc[:5], "mean")

    def test_all_four_series_strategies_run_inner_fits_without_event_leakage(self):
        rows, features = [], []
        for falha in range(1, 19):
            event = b.gearbox_event_of(falha)
            for ordinal in range(5):
                endpoint = f"F{falha:02d}-{ordinal}"
                rows.append({"endpoint_id": endpoint, "Falha": falha,
                             "equipment": f"E{event}",
                             "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=ordinal),
                             "phase": "saudavel", "y": 0.9 - ordinal * 0.1})
                feature = {"cur__s0": float(ordinal + falha / 20),
                           "cur__s1": float((ordinal + 1) * falha / 20),
                           "nobs__s0": float(ordinal + 1), "missbins__s0": 0.0}
                features.append(feature)
        # Build fixed metadata one-hot columns for all six measurement positions.
        meta = pd.DataFrame(rows)
        table = pd.DataFrame(features, index=pd.Index(meta.endpoint_id, name="endpoint_id"))
        for position, name in enumerate(("HSS-Radial", "HSS-Axial", "IMS", "LSS",
                                         "Planetária 0°", "Planetária 90°"), start=1):
            table[f"meas__{name}"] = [float(b.gearbox_position_of(fid) == position)
                                      for fid in meta.Falha]
        train = meta.loc[meta.equipment != "E3"].reset_index(drop=True)
        config = _small_config(("ridge",))
        config["feature_group_sets"] = {"sensors": ["sensors"],
                                         "sensors+cadence": ["sensors", "cadence"],
                                         "sensors+elapsed": ["sensors", "elapsed"],
                                         "cadence+elapsed": ["cadence", "elapsed"]}
        config["datasets"]["gearbox"] = {"outer_order": ["E1", "E2", "E3"]}
        config["tuning_enabled"] = False
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp, max_fits=1000)
            search = b.OuterSearch(dataset_name="gearbox", outer_group="E3", train_meta=train,
                                   tables=lambda _w, _c: table, runner=runner, config=config,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s4")
            outputs = {}
            for strategy in ("pooled_anonymous", "measurement_onehot",
                             "component_experts", "measurement_experts"):
                recipe = b.normalize_recipe("ridge", {"alpha": 1.0},
                                             {"strategy": strategy}, config["default_recipe"])
                results = search._evaluate([(100 + len(outputs), "synthetic", recipe)])
                result = results[0]
                self.assertEqual(result.status, "ok", result.reasons)
                self.assertEqual(set(result.oof.endpoint_id), set(train.endpoint_id))
                self.assertEqual(len(result.oof), len(train))
                outputs[strategy] = result
                self.assertEqual(runner.budget.charged > 0, True)
            self.assertGreater(len(outputs["component_experts"].fit_keys),
                               len(outputs["pooled_anonymous"].fit_keys))
            self.assertGreater(len(outputs["measurement_experts"].fit_keys),
                               len(outputs["component_experts"].fit_keys))
            self.assertNotEqual(outputs["measurement_onehot"].fit_keys,
                                outputs["pooled_anonymous"].fit_keys)
            train_mask, test_mask = (meta.equipment != "E3").to_numpy(), \
                (meta.equipment == "E3").to_numpy()
            falha_by_endpoint = meta.set_index("endpoint_id").Falha.to_dict()
            expected_order = meta.loc[test_mask, "endpoint_id"].tolist()
            for strategy in outputs:
                recipe = b.normalize_recipe("ridge", {"alpha": 1.0},
                                            {"strategy": strategy}, config["default_recipe"])
                tasks = b.build_strategy_fold_tasks(
                    meta, table, train_mask, test_mask, recipe, config["feature_group_sets"],
                    lambda expert: b.derive_seed("outer", strategy, expert), {}, 60, "gearbox",
                    "s4")
                self.assertTrue(all(not {falha_by_endpoint[x] for x in task.train_ids}
                                    & set(range(13, 19)) for _, task in tasks))
                records = runner.run([task for _, task in tasks], purpose="final")
                predicted = {}
                for _expert, task in tasks:
                    predicted.update(zip(task.eval_ids, records[task.key].predictions))
                self.assertEqual(set(predicted), set(expected_order))
                primary_order_predictions = [predicted[x] for x in expected_order]
                self.assertEqual(len(primary_order_predictions), len(expected_order))


def _space_midpoint(spec):
    kind = spec["type"]
    if kind == "const":
        return spec["value"]
    if kind == "categorical":
        return spec["choices"][0]
    if kind == "int":
        return int(spec["low"])
    if spec.get("log"):
        return float(np.sqrt(spec["low"] * spec["high"]))
    return float((spec["low"] + spec["high"]) / 2)


class FamilyFitSmokeTests(unittest.TestCase):
    def test_every_configured_family_fits_tiny_data(self):
        """Each family runs through the real fit runner (forked workers, not in-process)."""
        cfg = b.load_config()
        rng = np.random.default_rng(11)
        X = rng.standard_normal((40, 4))
        y = np.clip(0.5 + 0.2 * X[:, 0] + 0.05 * rng.standard_normal(40), 0, 1)
        tasks = []
        for family, entry in cfg["families"].items():
            params = {name: _space_midpoint(spec) for name, spec in entry["space"].items()}
            if family == "knn":
                params["n_neighbors"] = 3
            weights = np.ones(30) if family in b.WEIGHT_CAPABLE else None
            tasks.append(b.FitTask(key=b.sha256_text(family), family=family,
                                   params=b.normalize_params(family, params), seed=3,
                                   X_train=X[:30], y_train=y[:30], weights=weights,
                                   X_eval=X[30:], eval_ids=tuple(f"e{i}" for i in range(10)),
                                   timeout_s=60.0))
        with tempfile.TemporaryDirectory() as tmp:
            records = _runner(tmp).run(tasks)
        for task in tasks:
            record = records[task.key]
            if record.status != "ok":
                self.assertTrue(str(record.reason).startswith("convergence"),
                                f"{task.family}: unexpected failure {record.reason}")
            else:
                self.assertEqual(len(record.predictions), 10, task.family)


class FitRunnerTests(unittest.TestCase):
    def test_success_then_cache_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            task = _task()
            first = runner.run([task])
            self.assertEqual(first[task.key].status, "ok")
            charged = runner.budget.charged
            second_runner = _runner(tmp)
            second = second_runner.run([task])
            self.assertEqual(second[task.key].source, "cache")
            self.assertEqual(charged, 1)
            self.assertEqual(second_runner.budget.charged, 0)  # cache hits are not charged
            self.assertEqual(second_runner.stats.fits_started, 0)
            np.testing.assert_allclose(first[task.key].predictions,
                                       second[task.key].predictions)
            self.assertEqual(list(Path(tmp, "fits").glob(".*.tmp-*")), [])

    def test_duplicate_tasks_in_one_batch_fit_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            task = _task()
            runner.run([task, task])
            self.assertEqual(runner.stats.fits_started, 1)

    def test_convergence_is_a_recorded_failure_not_a_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            rng = np.random.default_rng(1)
            X = rng.standard_normal((80, 6))
            y = rng.standard_normal(80)
            task = b.FitTask(
                key=b.sha256_text("conv"), family="elasticnet",
                params={"alpha": 1e-6, "l1_ratio": 0.5, "max_iter": 1}, seed=0,
                X_train=X, y_train=y, weights=None, X_eval=X[:4], eval_ids=("a", "b", "c", "d"),
                timeout_s=60.0)
            record = runner.run([task])[task.key]
            self.assertEqual(record.status, "failed")
            self.assertIn("convergence", record.reason)
            self.assertIsNone(record.predictions)
            self.assertTrue(Path(tmp, "fits", f"{task.key}.json").exists())  # deterministic

    def test_timeout_is_transient_and_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp, timeout_s=0.3)
            task = _task(key_salt="timeout")

            def slow(_task):
                time.sleep(5)
                return {"status": "ok", "reason": None, "predictions": [0.0] * 5,
                        "elapsed_s": 0.0}

            with mock.patch.object(b, "fit_task_locally", slow):
                record = runner.run([task])[task.key]
            self.assertEqual(record.status, "failed")
            self.assertTrue(record.reason.startswith("timeout"))
            self.assertEqual(runner.stats.timeouts, 1)
            self.assertFalse((Path(tmp) / "fits" / f"{task.key}.json").exists())

    def test_worker_death_is_transient_and_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            task = _task(key_salt="death")

            def dying(_task):
                os._exit(3)

            with mock.patch.object(b, "fit_task_locally", dying):
                record = runner.run([task])[task.key]
            self.assertEqual(record.reason, "worker_exit_3")
            self.assertEqual(runner.stats.worker_failures, 1)
            self.assertFalse((Path(tmp) / "fits" / f"{task.key}.json").exists())

    def test_budget_is_checked_before_any_worker_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp, max_fits=1)
            tasks = [_task(key_salt="a"), _task(key_salt="b")]
            with self.assertRaises(b.BudgetExceeded):
                runner.run(tasks)
            self.assertEqual(runner.stats.fits_started, 0)

    def test_corrupt_cache_is_rejected_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            task = _task(key_salt="corrupt")
            runner.run([task])
            path = Path(tmp) / "fits" / f"{task.key}.json"
            payload = json.loads(path.read_text())
            payload["predictions"][0] = 999.0
            path.write_text(json.dumps(payload))
            with self.assertRaises(b.CorruptCacheError):
                _runner(tmp).run([task])

    def test_truncated_cache_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            task = _task(key_salt="truncated")
            runner.run([task])
            path = Path(tmp) / "fits" / f"{task.key}.json"
            path.write_text(path.read_text()[:20])
            with self.assertRaises(b.CorruptCacheError):
                _runner(tmp).run([task])

    def test_resource_policy_limits_and_growth(self):
        resources = dict(b.load_config()["resources"])
        policy = b.ResourcePolicy(resources, available_bytes=lambda: 3.4 * b.GIB)
        # estimate = 256 MiB floor; share 0.65*3.4/0.25 -> 8, headroom 1.9 GiB -> 7
        self.assertEqual(policy.limit(3.4 * b.GIB), 7)
        self.assertEqual(policy.limit(1.0 * b.GIB), 0)
        policy.grow()  # no observation yet: calibration keeps one worker
        self.assertEqual(policy.target, 1)
        policy.observe_peak(200)
        for _ in range(20):
            policy.grow()
        self.assertEqual(policy.target, resources["max_workers"])
        self.assertFalse(policy.can_start(1.0 * b.GIB, 0))

    def test_resource_pause_then_error_when_headroom_never_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp, available=0.2 * b.GIB)
            with self.assertRaises(b.ResourceError):
                runner.run([_task(key_salt="lowmem")])
            self.assertEqual(runner.stats.fits_started, 0)  # nothing was started
            self.assertIn("pause_low_memory", [d["action"] for d in runner.policy.decisions])

    def test_worker_estimate_is_safety_times_observed_uss(self):
        policy = b.ResourcePolicy(dict(b.load_config()["resources"]),
                                  available_bytes=lambda: 3 * b.GIB)
        self.assertEqual(policy.estimate_bytes(), 256 * b.MIB)  # floor before observation
        policy.observe_peak(100)  # a tiny estimator must not lower the floor
        policy.observe_peak(50)
        self.assertEqual(policy.estimate_bytes(), 256 * b.MIB)
        policy.observe_peak(2000)
        self.assertEqual(policy.estimate_bytes(), int(1.3 * 2000 * b.MIB))

    def test_limit_respects_headroom_and_share(self):
        resources = dict(b.load_config()["resources"])
        resources["worker_floor_mb"] = 100
        policy = b.ResourcePolicy(resources, available_bytes=lambda: 2 * b.GIB)
        # estimate = 100 MiB floor; headroom: (2 GiB - 1.5 GiB) -> 5
        self.assertEqual(policy.limit(2 * b.GIB), 5)
        self.assertEqual(policy.limit(1.4 * b.GIB), 0)


class MemoryEstimateTests(unittest.TestCase):
    """Observed peaks must never be underestimated; calibration is explicit."""

    def test_observed_peak_above_any_old_cap_is_still_inflated_by_safety(self):
        policy = b.ResourcePolicy(dict(b.load_config()["resources"]),
                                  available_bytes=lambda: 8 * b.GIB)
        policy.observe_peak(5000)
        self.assertEqual(policy.estimate_bytes(), int(1.3 * 5000 * b.MIB))

    def test_pause_limit_is_practical_and_floor_is_explicit(self):
        resources = b.load_config()["resources"]
        self.assertLessEqual(resources["max_pause_s"], 30)
        self.assertEqual(resources["worker_floor_mb"], 256)
        self.assertNotIn("worker_ceiling_mb", resources)

    def test_real_host_preflight_allows_at_least_one_worker_or_reports_zero(self):
        policy = b.ResourcePolicy(dict(b.load_config()["resources"]))
        limit = policy.limit(policy.available())
        self.assertGreaterEqual(limit, 0)
        self.assertLessEqual(limit, policy._resources["max_workers"])

class CacheKeyTests(unittest.TestCase):
    def test_fit_key_changes_with_identity_params_and_seed(self):
        base = dict(identity={"csv": "a"}, family="ridge", params={"alpha": 1.0},
                    feature_columns=["c"], feature_spec={}, preprocess={}, weights="none",
                    train_ids=["x"], eval_ids=["y"], seed=1)
        key = b.fit_key(**base)
        self.assertEqual(key, b.fit_key(**base))
        for change in ({"seed": 2}, {"params": {"alpha": 2.0}}, {"train_ids": ["x", "z"]},
                       {"identity": {"csv": "b"}}):
            self.assertNotEqual(key, b.fit_key(**{**base, **change}))


class SearchTests(unittest.TestCase):
    def test_outer_group_in_search_rows_is_refused(self):
        meta, table = _synthetic_meta()
        leaked = pd.concat([meta, meta.iloc[:2].assign(equipment="A")], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=leaked,
                              tables=lambda w, c: table, runner=_runner(tmp), candidates=b.CandidateStore(Path(tmp) / "candidates"),
                              config=_small_config(), identity={}, variance_key="s2").run()

    def test_search_is_deterministic_and_uses_only_training_rows(self):
        meta, table = _synthetic_meta()
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config()
        identity = {"dataset": "synthetic", "csv_sha256": "0", "endpoints_sha256": "0",
                    "code_sha256": "0"}
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                search = b.OuterSearch(dataset_name="gerador", outer_group="A",
                                       train_meta=train, tables=lambda w, c: table,
                                       runner=_runner(tmp), candidates=b.CandidateStore(Path(tmp) / "candidates"), config=cfg, identity=identity, variance_key="s2")
                result = search.run()
                results.append((result["chosen"]["candidate_id"], result["chosen"]["score_pp"]))
                self.assertTrue(all(row["fit_keys"] for row in result["trials"]
                                    if row["status"] == "ok"))
        self.assertEqual(results[0], results[1])

    def test_invalid_combination_is_recorded_not_fitted(self):
        meta, table = _synthetic_meta()
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("ridge",))
        cfg["recipe_space"]["features.feature_groups"] = {
            "type": "categorical", "choices": ["cadence+elapsed"]}
        cfg["recipe_space"]["features.drop_variance"] = {
            "type": "categorical", "choices": [True]}
        cfg["legacy_quick_control"]["recipes"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                   tables=lambda w, c: table, runner=runner, config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            with self.assertRaises(ValueError):  # every combination is invalid
                search.run()
            self.assertEqual(runner.stats.fits_started, 0)
            self.assertTrue(all(state.invalid == state.proposed
                                for state in search.states.values()))

    def test_search_uses_inner_groups_only_and_returns_known_kinds(self):
        meta, table = _synthetic_meta()
        train = meta[meta["equipment"] != "A"][["endpoint_id", "Falha", "equipment", "Date",
                                                "phase", "y"]]
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   config=_small_config(), identity={}, variance_key="s2")
            self.assertNotIn("A", search.inner_groups)
            result = search.run()
            self.assertIn(result["chosen"]["kind"], ("family", "ensemble"))


    def test_outer_test_labels_cannot_alter_the_search(self):
        """Permuting the held-out group's targets leaves the search output unchanged."""
        meta, table = _synthetic_meta(groups=("A", "B", "C", "D"))
        cols = ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]
        permuted = meta.copy()
        test_rows = permuted["equipment"] == "A"
        permuted.loc[test_rows, "y"] = np.roll(permuted.loc[test_rows, "y"].to_numpy(), 3)
        cfg = _small_config()
        outcomes = []
        for frame in (meta, permuted):
            train = frame.loc[frame["equipment"] != "A", cols]
            with tempfile.TemporaryDirectory() as tmp:
                result = b.OuterSearch(dataset_name="gerador", outer_group="A",
                                       train_meta=train, tables=lambda w, c: table,
                                       runner=_runner(tmp), candidates=b.CandidateStore(Path(tmp) / "candidates"), config=cfg, identity={}, variance_key="s2").run()
            outcomes.append(b.canonical_json([result["chosen"], result["family_best"]]))
        self.assertEqual(outcomes[0], outcomes[1])


class SearchRuleTests(unittest.TestCase):
    def _candidate(self, trial, score):
        recipe = b.normalize_recipe("ridge", {"alpha": 1.0}, {}, b.load_config()["default_recipe"])
        return b.CandidateResult(b.recipe_id(recipe), "ridge", recipe, trial, "tune", "ok", [],
                                 score, 1.0, 1, None, [])

    def test_tuning_stops_after_patience_without_gain(self):
        meta, table = _synthetic_meta()
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("ridge",))
        cfg["stages"]["batch_size"] = 1
        scores = iter([10.0] * 20)
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            with mock.patch.object(search, "_evaluate",
                                   side_effect=lambda items: [self._candidate(n, next(scores))
                                                              for n, _p, _r in items]):
                state = search._run_family("ridge", max_total=20, phase="tune",
                                           min_total=2, patience=2)
        # First candidate is an improvement; then 2 proposals without gain stop the search.
        self.assertEqual(state.proposed, 3)
        self.assertTrue(any("stopped" in note for note in state.notes))

    def test_expansion_is_blocked_above_failure_fraction(self):
        search = b.OuterSearch.__new__(b.OuterSearch)
        search.config = b.load_config()
        blocked = b.FamilyState(family="ridge", proposed=8, failed=3, invalid=0)
        allowed = b.FamilyState(family="ridge", proposed=8, failed=2, invalid=0)
        too_few = b.FamilyState(family="ridge", proposed=3, failed=3, invalid=0)
        self.assertTrue(search._expansion_blocked(blocked))
        self.assertFalse(search._expansion_blocked(allowed))
        self.assertFalse(search._expansion_blocked(too_few))

    def test_ablation_hypotheses_include_postprocessing_and_use_stable_seeds(self):
        meta, table = _synthetic_meta()
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("ridge",))
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            recipe = b.normalize_recipe("ridge", {"alpha": 1.0}, {}, cfg["default_recipe"])
            oof = pd.DataFrame({"endpoint_id": ["x"], "Falha": [1], "equipment": ["B"],
                                "Date": pd.to_datetime(["2024-01-01"]), "phase": ["x"],
                                "y": [0.5], "pred_raw": [0.5], "pred_post": [0.5]})
            candidate = b.CandidateResult(b.recipe_id(recipe), "ridge", recipe, 0, "screen",
                                          "ok", [], 1.0, 1.0, 1, oof, [])
            search.states["ridge"] = b.FamilyState(family="ridge", best=candidate)
            proposed = []
            search._evaluate = lambda items, **kw: (proposed.extend(items) or [])
            search._ablate_best()
            axes = {key for _trial, _phase, recipe in proposed
                    for key, value in recipe["postprocess"].items()
                    if value != cfg["default_recipe"]["postprocess"][key]}
            self.assertTrue({"clip01", "ema_alpha", "running_min"} <= axes)
            self.assertLessEqual(len(proposed), 12)

            # Different hyperparameter trial numbers must share the same fit seed/cache key.
            search._evaluate = b.OuterSearch._evaluate.__get__(search)
            search._evaluate([(1, "screen", recipe)])
            hits_before = search.runner.stats.cache_hits
            search._evaluate([(2, "screen", recipe)])
            self.assertGreater(search.runner.stats.cache_hits, hits_before)

    def test_declared_legacy_controls_reproduce_exact_original_column_lists(self):
        dataset = _synthetic_dataset(groups=("A", "B", "C"), records=14)
        table = b.build_feature_table(dataset, 12, 32, {"protocol": {
            "relative_eps": {"floor": 1e-8, "relative": 1e-6}}})
        train = dataset.meta.loc[
            dataset.meta["equipment"] != "C",
            ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("mean", "ridge"))
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="C", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            captured, arrays = [], {}

            def fake_evaluate(items, seed_offset=0):
                results = []
                for trial, phase, recipe in items:
                    columns = b.select_feature_columns(
                        list(table.columns), recipe["features"], cfg["feature_group_sets"],
                        "s2")
                    arrays[recipe["features"]["representation"]] = (
                        columns, table[columns].to_numpy(float))
                    captured.append(recipe)
                    results.append(b.CandidateResult(
                        b.recipe_id(recipe), recipe["family"], recipe, trial, phase, "ok",
                        [], 1.0, 1.0, 1, None, []))
                return results

            search._evaluate = fake_evaluate
            declared = [
                {"name": "mean_summary_none", "family": "mean", "params": {},
                 "representation": "summary", "weights": "none"},
                {"name": "ridge_current_component", "family": "ridge",
                 "params": {"alpha": 10.0}, "representation": "current",
                 "weights": "component"},
            ]
            cfg["legacy_quick_control"]["recipes"] = declared
            search._legacy_controls()

        self.assertEqual(len(captured), 2)
        self.assertEqual(arrays["summary"][0], b.cer.feature_columns(12, "summary", False))
        self.assertEqual(arrays["current"][0], b.cer.feature_columns(12, "current", False))
        self.assertEqual(arrays["summary"][1].shape[1], 51)
        self.assertEqual(arrays["current"][1].shape[1], 27)
        for recipe in captured:
            self.assertFalse(recipe["postprocess"]["clip01"])
            self.assertEqual((recipe["features"]["width_hours"],
                              recipe["features"]["context_bins"]), (12, 32))
            self.assertTrue(recipe["preprocess"]["missing_indicators"])
            self.assertTrue(recipe["features"]["legacy_control_columns"])
            self.assertNotIn("timing__window_days", arrays[
                recipe["features"]["representation"]][0])

    def test_legacy_control_marks_an_unavailable_schema_instead_of_faking_a_score(self):
        meta, table = _synthetic_meta()  # deliberately lacks the historical columns
        cfg = _small_config(("ridge",))
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            search._legacy_controls()
        self.assertEqual(search._legacy_results[0]["status"], "unavailable")
        self.assertNotIn("score_pp", search._legacy_results[0])

    def test_ablation_retune_is_triggered_only_at_gain_threshold_and_locks_recipe(self):
        search = b.OuterSearch.__new__(b.OuterSearch)
        search.config = b.load_config()
        search.states = {"ridge": b.FamilyState(family="ridge", best=mock.Mock(recipe={"fixed": 1}))}
        search._ablation_gain = ("ridge", 0.5)
        calls = []
        search._run_family = lambda *args, **kwargs: calls.append((args, kwargs))
        search._retune_ablation_winner()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][1], 16)
        self.assertEqual(calls[0][1]["fixed_recipe"], {"fixed": 1})
        search._ablation_gain = ("ridge", 0.49)
        calls.clear()
        search._retune_ablation_winner()
        self.assertFalse(calls)

    def test_three_seed_confirmation_fits_offsets_and_saves_spread(self):
        meta, table = _synthetic_meta()
        cfg = _small_config(("extra_trees",))
        cfg["default_recipe"]["features"]["feature_groups"] = "sensors"
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            recipe = b.normalize_recipe("extra_trees", {
                "n_estimators": 4, "max_depth": 3, "min_samples_leaf": 2,
                "max_features": "sqrt", "n_jobs": 1}, {}, cfg["default_recipe"])
            base = search._evaluate([(0, "screen", recipe)])[0]
            search.states["extra_trees"] = b.FamilyState(family="extra_trees", best=base)
            search._confirm_seeds()
            confirmed = search.states["extra_trees"].best
            self.assertEqual(confirmed.seed_offsets, (0, 1, 2))
            self.assertEqual(len(confirmed.stage_metrics["seed_scores_pp"]), 3)
            self.assertGreaterEqual(confirmed.stage_metrics["seed_score_std_pp"], 0)
            loaded = search.candidates.load(confirmed.key, search.inner_groups)
            self.assertEqual(loaded.seed_offsets, (0, 1, 2))
            self.assertEqual(len(confirmed.fit_keys), 9)  # 3 offsets x 3 inner groups
            averaged = b.inner_score(loaded.oof)["score_pp"]
            self.assertAlmostEqual(loaded.score_pp, averaged, places=12)
            self.assertAlmostEqual(loaded.stage_metrics["averaged_oof_score_pp"], averaged,
                                   places=12)
            self.assertIn("seed_score_mean_pp", loaded.stage_metrics)
            self.assertEqual(len(loaded.stage_metrics["seed_scores_pp"]), 3)

    def test_seed_mean_helper_matches_outer_averaging(self):
        per_seed = [np.array([0.2, 0.8]), np.array([0.4, 0.6]), np.array([0.3, 0.7])]
        np.testing.assert_allclose(b.mean_seed_predictions(per_seed), [0.3, 0.7])
        with self.assertRaises(ValueError):
            b.mean_seed_predictions([np.array([1.0]), np.array([1.0, 2.0])])


class DeclaredRangeTests(unittest.TestCase):
    def test_numeric_ranges_match_the_protocol(self):
        space = b.load_config()["families"]
        self.assertEqual(space["svr"]["space"]["epsilon"]["low"], 0.005)
        self.assertEqual(space["svr"]["space"]["epsilon"]["high"], 0.2)
        for family in ("hist_gb", "catboost"):
            rate = space[family]["space"]["learning_rate"]
            self.assertEqual((rate["low"], rate["high"]), (0.02, 0.2), family)
        self.assertEqual(space["catboost"]["space"]["l2_leaf_reg"]["high"], 30)
        self.assertEqual(space["huber"]["space"]["epsilon"]["high"], 2.0)


class PlateauAndEnsembleRuleTests(unittest.TestCase):
    def _candidate(self, trial, score):
        recipe = b.normalize_recipe("ridge", {"alpha": 1.0}, {},
                                    b.load_config()["default_recipe"])
        return b.CandidateResult(b.recipe_id(recipe), "ridge", recipe, trial, "tune", "ok", [],
                                 score, 1.0, 1, None, [])

    def test_small_gains_accumulate_against_the_anchor(self):
        meta, table = _synthetic_meta()
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("ridge",))
        cfg["stages"]["batch_size"] = 1
        scores = iter([10.0, 9.7, 9.4] + [9.4] * 20)  # 0.3 steps: 0.6 cumulative at trial 3
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "c"), identity={}, variance_key="s2")
            with mock.patch.object(search, "_evaluate", side_effect=lambda items: [
                    self._candidate(n, next(scores)) for n, _p, _r in items]):
                state = search._run_family("ridge", max_total=20, phase="tune",
                                           min_total=2, patience=2)
        self.assertEqual(state.anchor_score, 9.4)
        self.assertEqual(state.proposed, 5)  # last significant gain at 3, patience 2 -> stop

    def test_ensembles_use_learned_families_and_skip_duplicate_two_member_median(self):
        search = b.OuterSearch.__new__(b.OuterSearch)
        search.config = _small_config(("mean", "median", "ridge", "huber", "extra_trees"))
        search.config["stages"]["ensemble_sizes"] = [2, 3]
        search.inner_groups = ["B"]
        search.trials = []
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])

        def candidate(family, score, preds):
            recipe = b.normalize_recipe(family, {}, {}, search.config["default_recipe"])
            oof = pd.DataFrame({"endpoint_id": ["a", "b"], "Falha": [1, 1], "equipment": "B",
                                "Date": dates, "phase": "saudavel", "y": [0.5, 0.5],
                                "pred_post": preds})
            return b.CandidateResult(b.recipe_id(recipe), family, recipe, 0, "tune", "ok", [],
                                     score, 1.0, 3, oof, [])

        search.states = OrderedDict(
            (name, b.FamilyState(family=name, best=candidate(name, score, preds)))
            for name, score, preds in (
                ("mean", 1.0, [0.1, 0.1]), ("median", 2.0, [0.2, 0.2]),
                ("ridge", 5.0, [0.3, 0.4]), ("huber", 5.5, [0.35, 0.45]),
                ("extra_trees", 6.0, [0.6, 0.7])))
        result = search._select()
        for ensemble in result["ensembles"]:
            self.assertFalse(set(ensemble["member_families"]) & b.CONSTANT_FAMILIES)
        two = [e for e in result["ensembles"] if len(e["members"]) == 2]
        three = [e for e in result["ensembles"] if len(e["members"]) == 3]
        self.assertEqual([e["how"] for e in two], ["mean"])
        self.assertEqual(sorted(e["how"] for e in three), ["mean", "median"])


class FitIdentityAndErrorTests(unittest.TestCase):
    def test_postprocess_changes_reuse_the_same_fit(self):
        meta, table = _synthetic_meta()
        features = table.loc[meta["endpoint_id"].to_numpy()].set_axis(meta.index)
        base = b.load_config()["default_recipe"]
        plain = b.normalize_recipe("ridge", {"alpha": 1.0}, {}, base)
        changed = b.normalize_recipe("ridge", {"alpha": 1.0}, {
            "postprocess.clip01": False, "postprocess.ema_alpha": 0.5,
            "postprocess.running_min": True}, base)
        train_mask = (meta["equipment"] != "D").to_numpy()
        eval_mask = (meta["equipment"] == "D").to_numpy()
        identity = {"dataset": "synthetic"}
        t1 = b.build_fold_task(meta, features, train_mask, eval_mask, plain,
                               b.load_config()["feature_group_sets"], 7, identity, 60,
                               variance_key="s2")
        t2 = b.build_fold_task(meta, features, train_mask, eval_mask, changed,
                               b.load_config()["feature_group_sets"], 7, identity, 60,
                               variance_key="s2")
        self.assertEqual(t1.key, t2.key)
        with tempfile.TemporaryDirectory() as tmp:
            _runner(tmp).run([t1])
            second = _runner(tmp)
            record = second.run([t2])[t2.key]
        self.assertEqual(record.source, "cache")
        self.assertEqual(second.stats.fits_started, 0)

    def test_model_errors_fail_the_candidate_and_other_errors_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _task(key_salt="modelerr")
            with mock.patch.object(b, "build_estimator", side_effect=ValueError("bad params")):
                record = _runner(tmp).run([task])[task.key]
            self.assertEqual(record.status, "failed")
            self.assertTrue(record.reason.startswith("ValueError"))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(b, "build_estimator", side_effect=KeyError("beta")):
                with self.assertRaises(b.WorkerFatalError):
                    _runner(tmp).run([_task(key_salt="fatal")])

    def test_budget_reservation_protects_final_refits(self):
        budget = b.FitBudget(max_fits=10, reserved_for_final=3)
        budget.check(7, "search")
        with self.assertRaises(b.BudgetExceeded):
            budget.check(8, "search")
        budget.check(10, "final")
        self.assertEqual(budget.remaining("search"), 7)


def _rewrite_payload(path, mutate):
    """Edit a stored payload and re-hash it, so only the semantic check can reject it."""
    data = json.loads(Path(path).read_text())
    data.pop("payload_sha256")
    mutate(data)
    data["payload_sha256"] = b.sha256_text(b.canonical_json(data))
    Path(path).write_text(json.dumps(data))


class CandidatePayloadTests(unittest.TestCase):
    GROUPS = ["B", "C"]

    def _result(self):
        recipe = b.normalize_recipe("ridge", {"alpha": 1.0}, {}, b.load_config()["default_recipe"])
        rng = np.random.default_rng(2)
        raw = rng.uniform(-0.2, 1.2, 6)
        oof = pd.DataFrame({
            "endpoint_id": [f"e{i}" for i in range(6)], "Falha": [1, 1, 1, 2, 2, 2],
            "equipment": ["B"] * 3 + ["C"] * 3,
            "Date": pd.to_datetime(["2024-01-0%d" % (i + 1) for i in range(6)]),
            "phase": "saudavel", "y": rng.uniform(0, 1, 6), "pred_raw": raw,
            "pred_post": np.clip(raw, 0, 1)})
        score = b.inner_score(oof)
        key = b.candidate_key(recipe, 0, "screen", ["k1"], self.GROUPS)
        return b.CandidateResult(b.recipe_id(recipe), "ridge", recipe, 0, "screen", "ok", [],
                                 score["score_pp"], score["rmse_pp"], 5, oof, ["k1"], key)

    def test_round_trip_and_interrupted_commit(self):
        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            store = b.CandidateStore(tmp)
            store.save(result.key, result, self.GROUPS)
            loaded = store.load(result.key, self.GROUPS)
            self.assertAlmostEqual(loaded.score_pp, result.score_pp)
            np.testing.assert_allclose(loaded.oof["pred_post"], result.oof["pred_post"])
        with tempfile.TemporaryDirectory() as tmp:
            store = b.CandidateStore(tmp)
            with mock.patch.object(b.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    store.save(result.key, result, self.GROUPS)
            self.assertIsNone(store.load(result.key, self.GROUPS))  # no final file, not success

    def test_missing_predictions_mismatched_score_and_recipe_are_rejected(self):
        result = self._result()
        cases = {
            "missing prediction": lambda d: d["oof"][0].update(pred_raw=None),
            "mismatched score": lambda d: d.update(score_pp=d["score_pp"] + 1.0),
            "recipe changed": lambda d: d["recipe"]["postprocess"].update(clip01=False),
        }
        for label, mutate in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                store = b.CandidateStore(tmp)
                store.save(result.key, result, self.GROUPS)
                _rewrite_payload(Path(tmp) / f"{result.key}.json", mutate)
                with self.assertRaises(b.CorruptCacheError):
                    store.load(result.key, self.GROUPS)

    def test_unhashed_tampering_is_rejected(self):
        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            store = b.CandidateStore(tmp)
            store.save(result.key, result, self.GROUPS)
            path = Path(tmp) / f"{result.key}.json"
            path.write_text(path.read_text().replace('"score_pp": ', '"score_pp": 1e9, "x": ', 1))
            with self.assertRaises(b.CorruptCacheError):
                store.load(result.key, self.GROUPS)


class FeatureCacheTests(unittest.TestCase):
    IDENTITY = {"dataset": "x", "csv_sha256": "v", "endpoints_sha256": "e", "code_sha256": "c"}

    def test_grid_is_built_once_then_loaded_and_corruption_is_rejected(self):
        frame = pd.DataFrame({"cur__s0": [1.0, 2.0]},
                             index=pd.Index(["a", "b"], name="endpoint_id"))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(b, "build_feature_table", return_value=frame) as build:
            first = b.FeatureCache(None, {}, self.IDENTITY, tmp)
            first.table(12, 32)
            second = b.FeatureCache(None, {}, self.IDENTITY, tmp)
            loaded = second.table(12, 32)
            build.assert_called_once()
            self.assertEqual((second.builds, second.disk_hits), (0, 1))
            pd.testing.assert_frame_equal(loaded, frame)
            (Path(tmp) / "w12_b32.pkl").write_bytes(b"junk")
            with self.assertRaises(b.CorruptCacheError):
                b.FeatureCache(None, {}, self.IDENTITY, tmp).table(12, 32)

    def test_grid_from_another_code_version_is_rejected(self):
        frame = pd.DataFrame({"cur__s0": [1.0]}, index=pd.Index(["a"], name="endpoint_id"))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(b, "build_feature_table", return_value=frame):
                b.FeatureCache(None, {}, self.IDENTITY, tmp).table(12, 32)
            other = {**self.IDENTITY, "code_sha256": "different"}
            with self.assertRaises(b.CorruptCacheError):
                b.FeatureCache(None, {}, other, tmp).table(12, 32)


class ResumeTests(unittest.TestCase):
    """A real interruption in the middle of an outer search, then a resumed run."""

    class InterruptingRunner:
        def __init__(self, inner, fail_on_call):
            self._inner = inner
            self._fail_on_call = fail_on_call
            self._calls = 0
            self.budget = inner.budget
            self.stats = inner.stats

        def run(self, tasks, purpose="search"):
            self._calls += 1
            if self._calls == self._fail_on_call:
                raise RuntimeError("simulated interruption before this batch")
            return self._inner.run(tasks, purpose)

    def _search(self, directory, runner, train, table, cfg):
        return b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                             tables=lambda w, c: table, runner=runner, config=cfg,
                             candidates=b.CandidateStore(Path(directory) / "candidates"),
                             identity={"dataset": "synthetic"}, variance_key="s2")

    def test_interrupted_search_resumes_to_the_same_decisions(self):
        meta, table = _synthetic_meta()
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("mean", "ridge"))
        with tempfile.TemporaryDirectory() as reference_dir:
            reference_runner = _runner(reference_dir)
            reference = self._search(reference_dir, reference_runner, train, table, cfg).run()
        with tempfile.TemporaryDirectory() as tmp:
            interrupted = self.InterruptingRunner(_runner(tmp), fail_on_call=2)
            with self.assertRaises(RuntimeError):
                self._search(tmp, interrupted, train, table, cfg).run()
            resumed_runner = _runner(tmp)
            resumed = self._search(tmp, resumed_runner, train, table, cfg).run()
            self.assertLess(resumed_runner.stats.fits_started, reference_runner.stats.fits_started)
        self.assertEqual(resumed["chosen"]["candidate_id"], reference["chosen"]["candidate_id"])
        self.assertEqual([t["candidate_id"] for t in resumed["trials"]],
                         [t["candidate_id"] for t in reference["trials"]])
        self.assertEqual([t["score_pp"] for t in resumed["trials"]],
                         [t["score_pp"] for t in reference["trials"]])

    def test_sqlite_batch_resumes_mid_candidate_commit_without_refitting(self):
        meta, table = _synthetic_meta()
        train = meta[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("ridge",))
        cfg["stages"].update({"screen_trials": 2, "batch_size": 2})
        cfg["tuning_enabled"] = False
        cfg["legacy_quick_control"]["recipes"] = {}
        with tempfile.TemporaryDirectory() as uninterrupted, tempfile.TemporaryDirectory() as resumed:
            ref_runner = _runner(uninterrupted)
            ref = self._search(uninterrupted, ref_runner, train, table, cfg)
            ref.study_dir = Path(uninterrupted) / "outer"
            ref.journal = b.ProposalJournal(ref.study_dir, ref.journal.identity)
            expected = ref.run()

            run_dir = Path(resumed)
            runner = _runner(resumed)
            search = self._search(resumed, runner, train, table, cfg)
            search.study_dir = run_dir / "outer"
            search.journal = b.ProposalJournal(search.study_dir, search.journal.identity)
            original_save = search.candidates.save
            calls = 0

            def interrupt_after_first(key, result, groups):
                nonlocal calls
                calls += 1
                original_save(key, result, groups)
                if calls == 1:
                    raise RuntimeError("crash after first candidate payload")

            with mock.patch.object(search.candidates, "save", side_effect=interrupt_after_first):
                with self.assertRaisesRegex(RuntimeError, "crash after first"):
                    search.run()
            self.assertTrue((search.study_dir / "studies.sqlite3").exists())
            pending = next(search.study_dir.glob("*.pending.json"))
            payload = json.loads(pending.read_text())
            self.assertEqual(len(payload["batch"]), 2)

            resumed_runner = _runner(resumed)
            again = self._search(resumed, resumed_runner, train, table, cfg)
            again.study_dir = run_dir / "outer"
            again.journal = b.ProposalJournal(again.study_dir, again.journal.identity)
            actual = again.run()
            self.assertEqual(actual["chosen"]["candidate_id"], expected["chosen"]["candidate_id"])
            self.assertEqual(actual["chosen"]["recipe"], expected["chosen"]["recipe"])
            self.assertEqual([x["candidate_id"] for x in actual["trials"]],
                             [x["candidate_id"] for x in expected["trials"]])
            self.assertEqual([x["score_pp"] for x in actual["trials"]],
                             [x["score_pp"] for x in expected["trials"]])
            for left, right in zip(actual["trials"], expected["trials"]):
                left_payload = b.read_json(again.candidates._path(left["candidate_key"]))
                right_payload = b.read_json(ref.candidates._path(right["candidate_key"]))
                self.assertEqual(left_payload["oof"], right_payload["oof"])
            self.assertEqual(resumed_runner.stats.fits_started, 0)
            self.assertGreater(resumed_runner.stats.cache_hits, 0)

    def test_corrupt_sampler_snapshot_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = b.ProposalJournal(tmp, {"run": "identity"})
            sampler = b.optuna.samplers.TPESampler(seed=7)
            journal.save("ridge", [{"number": 3, "recipe": {"x": 1}, "reason": None}],
                         sampler)
            sampler_path = Path(tmp) / "ridge.sampler.pkl"
            sampler_path.write_bytes(b"corrupt")
            with self.assertRaises(b.CorruptCacheError):
                journal.load("ridge")


class GearboxStrategyStageTests(unittest.TestCase):
    """Bounded strategy screening: caps, pooled reference and top-2 gating."""

    def _search(self, tmp):
        meta, table = _synthetic_meta(groups=("E1", "E2", "E3"), per_component=6)
        for name in ("HSS-Radial", "HSS-Axial", "IMS", "LSS",
                     "Planetaria 0", "Planetaria 90"):
            table[f"meas__{name}"] = [
                float(b.gearbox_component_of(fid) == name) for fid in meta["Falha"]]
        train = meta.loc[meta["equipment"] != "E3",
                         ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(tuple(b.SPECIALIST_FAMILIES))
        cfg["tuning_enabled"] = False
        search = b.OuterSearch(dataset_name="gearbox", outer_group="E3", train_meta=train,
                               tables=lambda w, c: table, runner=_runner(tmp, max_fits=5000),
                               config=cfg,
                               candidates=b.CandidateStore(Path(tmp) / "candidates"),
                               identity={}, variance_key="s4")
        return search

    def test_screening_caps_expansion_and_gates_on_pooled_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            search = self._search(tmp)
            scores = {"pooled_anonymous": 50.0, "measurement_onehot": 49.8,
                      "component_experts": 49.0, "measurement_experts": 48.0}
            calls = []

            def fake_evaluate(items, seed_offset=0):
                results = []
                for trial, phase, recipe in items:
                    calls.append((phase, recipe["strategy"], recipe["family"]))
                    results.append(b.CandidateResult(
                        b.recipe_id(recipe), recipe["family"], recipe, trial, phase,
                        "ok", [], scores[recipe["strategy"]], 1.0, 1, None, []))
                return results

            search._evaluate = fake_evaluate
            reference = b.CandidateResult("ref", "ridge", {}, 0, "screen", "ok", [], 50.0,
                                          1.0, 1, None, [])
            search.states["ridge"] = b.FamilyState(family="ridge", best=reference)
            search._screen_gearbox_strategies()

            base = [c for c in calls if c[0] == "strategy_screen"]
            expanded = [c for c in calls if c[0] == "strategy_expand"]
            for strategy in b.SERIES_STRATEGIES:
                self.assertLessEqual(len([c for c in base if c[1] == strategy]), 6, strategy)
            eligible = {row["strategy"] for row in search.strategy_results
                        if row["strategy"] != "pooled_anonymous"
                        and row["gain_vs_pooled_pp"] >= 0.5}
            self.assertEqual(eligible, {"component_experts", "measurement_experts"})
            self.assertEqual({c[1] for c in expanded}, eligible)
            for strategy in eligible:
                self.assertLessEqual(len([c for c in expanded if c[1] == strategy]), 24)
            self.assertNotIn("measurement_onehot", {c[1] for c in expanded})

    def test_screening_notes_record_the_gate_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            search = self._search(tmp)
            search._evaluate = lambda items, seed_offset=0: [
                b.CandidateResult(b.recipe_id(r), r["family"], r, t, p, "ok", [],
                                  49.0, 1.0, 1, None, []) for t, p, r in items]
            search.states["ridge"] = b.FamilyState(
                family="ridge", best=b.CandidateResult("ref", "ridge", {}, 0, "screen",
                                                       "ok", [], 60.0, 1.0, 1, None, []))
            search._screen_gearbox_strategies()
            self.assertIn("pooled reference", search._strategy_note)
            self.assertTrue(all(row["best_family"] in b.SPECIALIST_FAMILIES
                                for row in search.strategy_results))


class EventTrackStageTests(unittest.TestCase):
    def _track(self, *, missing_position=None, target_shift=None):
        rows, values = [], []
        for event in range(1, 4):
            for position in range(1, 7):
                falha = (event - 1) * 6 + position
                for day in range(4):
                    if missing_position == (falha, day):
                        continue
                    endpoint = f"F{falha:02d}-{day}"
                    target = 0.9 - 0.1 * day
                    if target_shift == (falha, day):
                        target = 0.5
                    rows.append({"endpoint_id": endpoint, "Falha": falha,
                                 "equipment": f"E{event}",
                                 "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                                 "phase": "saudavel", "y": target})
                    values.append({"cur__s0": float(position + day),
                                   "cur__s1": float(event + position * day)})
        meta = pd.DataFrame(rows)
        table = pd.DataFrame(values, index=pd.Index(meta.endpoint_id, name="endpoint_id"))
        return meta, table

    def test_joint_bounded_search_runs_inner_and_selection(self):
        meta, table = self._track()
        e_meta, e_features = b.make_joint_event_track(
            meta, table, ["cur__s0", "cur__s1"])
        self.assertEqual(len(e_meta), 12)
        cfg = _small_config(tuple(b.SPECIALIST_FAMILIES))
        cfg["tuning_enabled"] = False
        train = e_meta[e_meta.equipment != "E3"].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(
                dataset_name="gearbox", outer_group="E3", train_meta=train,
                tables=lambda _w, _c: e_features, runner=_runner(tmp, max_fits=2000),
                config=cfg, candidates=b.CandidateStore(Path(tmp) / "candidates"),
                identity={}, forced_strategy="joint_event", variance_key="s4")
            result = search.run_bounded(families=b.SPECIALIST_FAMILIES, per_family=1,
                                        extra_per_family=1, extra_top=1)
            self.assertIsNotNone(result["chosen"])
            self.assertTrue(result["chosen"]["candidate_id"])
            screened = [t for t in result["trials"] if t["phase"] == "bounded_screen"]
            self.assertLessEqual(len(screened), 3)  # one recipe per family
            self.assertEqual({t["family"] for t in screened}, set(b.SPECIALIST_FAMILIES))

    def test_aggregation_selects_mean_or_median_from_inner_oof(self):
        meta, table = self._track()
        e_meta, _ = b.make_joint_event_track(meta, table, ["cur__s0", "cur__s1"])
        rows = []
        for falha in range(1, 7):
            event = b.gearbox_event_of(falha)
            for day in range(4):
                rows.append({"endpoint_id": f"F{falha:02d}-{day}", "Falha": falha,
                             "equipment": f"E{event}",
                             "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                             "phase": "saudavel", "y": 0.9 - 0.1 * day,
                             "pred_raw": 0.9 - 0.1 * day + (0.4 if falha == 6 else 0.0),
                             "pred_post": 0.9 - 0.1 * day + (0.4 if falha == 6 else 0.0)})
        oof = pd.DataFrame(rows)
        recipe = b.normalize_recipe("ridge", {"alpha": 1.0}, {"strategy": "joint_event"},
                                    b.load_config()["default_recipe"])
        candidate = b.CandidateResult(b.recipe_id(recipe), "ridge", recipe, 0, "screen",
                                      "ok", [], 1.0, 1.0, 1, oof, [])
        search = mock.Mock()
        search.states = {"ridge": b.FamilyState(family="ridge", best=candidate)}
        run = b.Run.__new__(b.Run)
        run.dataset_name = "gearbox"
        aggregation = run._event_aggregation_selection(
            search, {"chosen": {"candidate_id": candidate.candidate_id, "kind": "family"}})
        self.assertIn(aggregation["how"], ("mean", "median"))
        self.assertEqual(set(aggregation["scores_pp"]), {"mean", "median"})
        self.assertIsNotNone(aggregation["score_pp"])

    def test_aggregation_rejects_incomplete_event_timestamps(self):
        meta, table = self._track()
        e_meta, _ = b.make_joint_event_track(meta, table, ["cur__s0", "cur__s1"])
        frame = meta[meta.equipment == "E1"].copy()
        frame["pred_raw"] = frame["y"]
        frame["pred_post"] = frame["y"]
        frame = frame[frame.Falha != 6]
        with self.assertRaises(ValueError):
            b.aggregate_series_frame_to_events(frame, "mean")

    def test_joint_rejects_missing_measurement_and_target_mismatch(self):
        meta, table = self._track(missing_position=(3, 1))
        features = table.loc[meta.endpoint_id]
        joint = b.build_joint_event_rows(meta, features, ["cur__s0", "cur__s1"])
        self.assertGreater(joint.attrs["dropped_unaligned"], 0)
        meta_bad, table_bad = self._track(target_shift=(3, 1))
        with self.assertRaises(ValueError):
            b.build_joint_event_rows(meta_bad, table_bad.loc[meta_bad.endpoint_id],
                                     ["cur__s0", "cur__s1"])

class EventTrackRunIntegrationTests(unittest.TestCase):
    """Outer event-track evaluation: joint refit plus six-prediction aggregation."""

    def _series_frame(self, meta, group="E3"):
        rows = meta.loc[meta.equipment == group].copy()
        rows["pred_raw"] = rows["y"] + 0.05
        rows["pred_post"] = np.clip(rows["pred_raw"], 0, 1)
        return rows[["endpoint_id", "Falha", "equipment", "Date", "phase", "y",
                     "pred_raw", "pred_post"]]

    def test_event_evaluation_writes_joint_and_aggregation_candidates(self):
        meta_rows, values = [], []
        for event in range(1, 4):
            for position in range(1, 7):
                falha = (event - 1) * 6 + position
                for day in range(4):
                    meta_rows.append({"endpoint_id": f"F{falha:02d}-{day}", "Falha": falha,
                                      "equipment": f"E{event}",
                                      "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                                      "phase": "saudavel", "y": 0.9 - 0.1 * day})
                    values.append({"cur__s0": float(position + day),
                                    "cur__s1": float(event + position * day)})
        meta = pd.DataFrame(meta_rows)
        table = pd.DataFrame(values, index=pd.Index(meta.endpoint_id, name="endpoint_id"))
        e_meta, e_features = b.make_joint_event_track(meta, table, ["cur__s0", "cur__s1"])
        cfg = _small_config(tuple(b.SPECIALIST_FAMILIES))
        cfg["tuning_enabled"] = False
        train = e_meta[e_meta.equipment != "E3"].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp, max_fits=3000)
            search = b.OuterSearch(
                dataset_name="gearbox", outer_group="E3", train_meta=train,
                tables=lambda _w, _c: e_features, runner=runner, config=cfg,
                candidates=b.CandidateStore(Path(tmp) / "cand"), identity={},
                forced_strategy="joint_event", variance_key="s4")
            joint = search.run_bounded(families=b.SPECIALIST_FAMILIES, per_family=1,
                                       extra_per_family=1, extra_top=1)
            source_id = "aggregation-source"
            selection = {
                "chosen": {"candidate_id": joint["chosen"]["candidate_id"], "kind": "family"},
                "family_best": joint["family_best"], "ensembles": joint["ensembles"],
            }
            event_dir = Path(tmp) / "outer" / "E3" / "event"
            event_dir.mkdir(parents=True, exist_ok=True)
            b.write_json_atomic(event_dir / "selection.json", {
                "track": "event", "frozen": True, "outer_group": "E3",
                "availability": "one target per event timestamp",
                "search_train_groups": ["E1", "E2"], "inner_groups": ["E1", "E2"],
                "identity": {"track": "event"},
                "joint": joint,
                "aggregation": {"source_candidate_ids": [source_id], "how": "mean",
                                "score_pp": 1.0, "scores_pp": {"mean": 1.0, "median": 2.0},
                                "problems": []},
            })
            run = b.Run.__new__(b.Run)
            run.dataset_name = "gearbox"
            run.run_dir = Path(tmp)
            run.dataset = mock.Mock(meta=meta)
            run.config = cfg
            run.identity = {}
            run.runner = runner
            run.variance_key = "s4"
            run.event_track = lambda: e_meta
            # genuine (width, bins) grid table: a broken recipe->grid adapter must fail
            run.event_table = lambda width_hours, context_bins: e_features
            run._evaluate_event_track("E3", selection, {source_id: self._series_frame(meta)})
            evaluation = b.read_json(event_dir / "evaluation.json")
            self.assertEqual(evaluation["track"], "event")
            self.assertIn("aggregation", evaluation["candidates"])
            self.assertEqual(evaluation["candidates"]["aggregation"]["kind"], "aggregation")
            self.assertGreaterEqual(len(evaluation["candidates"]), 2)
            frame = pd.read_csv(event_dir / "predictions.csv")
            aggregation = frame.loc[frame.candidate_id == "aggregation"]
            test_rows = e_meta[e_meta.equipment == "E3"]
            self.assertEqual(set(aggregation.endpoint_id), set(test_rows.endpoint_id))
            self.assertEqual(len(aggregation), 4)

class CorrectionRound1Tests(unittest.TestCase):
    """Regressions for the review defects: variance keys, causality, resume phases."""

    def test_variance_key_derives_from_each_dataset_profile(self):
        self.assertEqual(b.variance_sensor_key(b.cer.SENSOR_COLUMNS), "s2")
        self.assertEqual(b.variance_sensor_key(b.cer.GEARBOX_SENSOR_COLUMNS), "s4")
        gearbox_columns = list(b.cer.GEARBOX_SENSOR_COLUMNS)
        self.assertEqual(gearbox_columns[int("s4"[1:])], "g_0_Variance")
        self.assertEqual(gearbox_columns[int("s2"[1:])], "g_0_Peak_peak")

    def test_drop_variance_removes_the_right_sensor_in_every_joint_block(self):
        base = ["cur__s2", "cur__s4", "nobs__s4", "missbins__s4"]
        joint = [f"m{index}__{column}" for index in range(1, 7) for column in base]
        recipe = _recipe(feature_groups="sensors")
        recipe["features"]["drop_variance"] = True
        gearbox = b.select_feature_columns(joint, recipe["features"], GROUP_SETS, "s4")
        generator = b.select_feature_columns(joint, recipe["features"], GROUP_SETS, "s2")
        for index in range(1, 7):
            self.assertNotIn(f"m{index}__cur__s4", gearbox)
            self.assertIn(f"m{index}__cur__s2", gearbox)
            self.assertNotIn(f"m{index}__cur__s2", generator)
            self.assertIn(f"m{index}__cur__s4", generator)
        self.assertEqual(len(gearbox), 6)  # one surviving sensor column per block

    def test_future_records_do_not_change_past_relative_reference(self):
        dataset = _synthetic_dataset(groups=("A", "B"), records=8)
        table = b.build_feature_table(dataset, 12, 32, {"protocol": {
            "relative_eps": {"floor": 1e-8, "relative": 1e-6}}})
        # endpoint recorded at index 3; mutate later records only
        endpoint_id = table.index[(table.index.str.contains("-r0003"))][0]
        changed = dataset.clean.copy()
        later = changed.index > changed.groupby("Falha").cumcount().where(
            changed["Falha"] == 1).bfill().notna()
        mask = changed["Falha"] == 1
        rows = changed[mask].sort_values("Date").index[4:]
        changed.loc[rows, list(b.cer.SENSOR_COLUMNS)] = 99.0
        mutated = b.Dataset(**{**dataset.__dict__, "clean": changed})
        mutated_table = b.build_feature_table(mutated, 12, 32, {"protocol": {
            "relative_eps": {"floor": 1e-8, "relative": 1e-6}}})
        pd.testing.assert_series_equal(table.loc[endpoint_id],
                                       mutated_table.loc[endpoint_id])

    def test_final_refit_reserve_covers_worst_case_specialist_seed_refits(self):
        config = b.load_config()
        reserve = b.final_refit_reserve(3, config)
        families = len(config["families"])
        experts = config["budgets"]["experts_per_fit_max"]
        seeds = config["budgets"]["seed_refits_max"]
        worst = 3 * (families * experts * seeds
                     + len(b.SPECIALIST_FAMILIES) + len(config["stages"]["ensemble_sizes"]))
        self.assertGreaterEqual(reserve, worst)
        self.assertLess(reserve, config["budgets"]["max_fits_per_dataset"])

    def test_median_combination_differs_from_mean_for_three_members(self):
        base = pd.DataFrame({"endpoint_id": ["a", "b"], "Falha": [1, 1],
                             "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                             "phase": ["x", "x"], "y": [0.5, 0.5], "pred_raw": [0, 0],
                             "pred_post": [0, 0]})
        frames = {}
        for name, value in (("m1", 0.1), ("m2", 0.2), ("m3", 0.9)):
            frame = base.copy()
            frame["pred_raw"] = value
            frame["pred_post"] = value
            frames[name] = frame
        mean = b.Run._combine_series_frames(["m1", "m2", "m3"], frames, "mean")
        median = b.Run._combine_series_frames(["m1", "m2", "m3"], frames, "median")
        np.testing.assert_allclose(mean["pred_post"], [0.4, 0.4])
        np.testing.assert_allclose(median["pred_post"], [0.2, 0.2])

    def test_joint_track_exposes_recipe_faithful_summary_and_current_columns(self):
        dataset = _synthetic_dataset(groups=("E1", "E2", "E3"), records=10)
        base = b.build_feature_table(dataset, 12, 32, {"protocol": {
            "relative_eps": {"floor": 1e-8, "relative": 1e-6}}})
        meta, features = b.make_joint_event_track(dataset.meta, base, list(base.columns))
        current = b.select_feature_columns(list(features.columns),
                                           _recipe(representation="current")["features"],
                                           GROUP_SETS, "s2")
        summary = b.select_feature_columns(list(features.columns),
                                           _recipe(representation="summary")["features"],
                                           GROUP_SETS, "s2")
        self.assertTrue(all(c.startswith("m1__") for c in current[:3]))
        self.assertTrue(any("dmean__" in c for c in summary))
        self.assertFalse(any("dmean__" in c for c in current))
        self.assertGreater(len(summary), len(current))
        self.assertEqual(len(summary) % 6, 0)
        self.assertEqual(len(current) % 6, 0)

class ResumeStageTests(unittest.TestCase):
    """Screen/tune boundaries and stage phases survive a restart."""

    def _cfg(self, families, **stages):
        cfg = _small_config(tuple(families))
        cfg["legacy_quick_control"]["recipes"] = []
        cfg["stages"].update({"screen_trials": 2, "batch_size": 1, "tune_extra_trials": 2,
                              "tune_top_k": 2, "tune_min_total_trials": 3,
                              "tune_patience_trials": 5})
        cfg["stages"].update(stages)
        return cfg

    def _search(self, tmp, cfg, runner=None):
        meta, table = _synthetic_meta()
        train = meta.loc[meta["equipment"] != "A",
                         ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        return b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                             tables=lambda w, c: table, runner=runner or _runner(tmp),
                             config=cfg, candidates=b.CandidateStore(Path(tmp) / "candidates"),
                             study_dir=Path(tmp) / "study", identity={}, variance_key="s2")

    def test_finished_search_resume_adds_no_proposals_or_fits(self):
        cfg = self._cfg(("ridge",), tuning_enabled=False) if False else self._cfg(("ridge",))
        cfg["tuning_enabled"] = False
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            reference = self._search(first, cfg)
            expected = reference.run()
            self.assertGreater(reference.states["ridge"].proposed, 0)
            resumed_runner = _runner(first)
            again = self._search(first, cfg, resumed_runner)
            actual = again.run()
            self.assertEqual(actual["chosen"], expected["chosen"])
            self.assertEqual(resumed_runner.stats.fits_started, 0)
            self.assertEqual(again.states["ridge"].proposed,
                             reference.states["ridge"].proposed)

    def test_crash_between_screen_and_tune_resumes_to_the_same_decisions(self):
        cfg = self._cfg(("ridge", "extra_trees"))
        with tempfile.TemporaryDirectory() as baseline, tempfile.TemporaryDirectory() as crashed:
            expected = self._search(baseline, cfg).run()
            search = self._search(crashed, cfg)
            original = search._run_family

            def fail_on_tune(family, max_total, phase, **kwargs):
                if phase == "tune":
                    raise RuntimeError("crash between screen and tune")
                return original(family, max_total, phase, **kwargs)

            search._run_family = fail_on_tune
            with self.assertRaisesRegex(RuntimeError, "between screen and tune"):
                search.run()
            resumed_runner = _runner(crashed)
            resumed = self._search(crashed, cfg, resumed_runner)
            actual = resumed.run()
            self.assertEqual(actual["chosen"]["candidate_id"],
                             expected["chosen"]["candidate_id"])
            self.assertEqual([row["score_pp"] for row in actual["trials"]],
                             [row["score_pp"] for row in expected["trials"]])
            phases = {row["phase"] for row in actual["trials"]}
            self.assertIn("tune", phases)
            self.assertIn("screen", phases)

    def test_crash_during_tuning_preserves_phase_and_stays_within_the_cap(self):
        cfg = self._cfg(("ridge", "extra_trees"))
        cap = (cfg["stages"]["screen_trials"] + cfg["stages"]["tune_extra_trials"]
               + cfg["stages"]["ablation_retune_trials"])
        with tempfile.TemporaryDirectory() as tmp:
            search = self._search(tmp, cfg)
            original = search._evaluate
            state = {"calls": 0, "crashed": False}

            def crash_in_tune(items, seed_offset=0):
                if any(phase == "tune" for _n, phase, _r in items) and not state["crashed"]:
                    state["crashed"] = True
                    raise RuntimeError("crash inside tuning")
                return original(items, seed_offset=seed_offset)

            search._evaluate = crash_in_tune
            with self.assertRaisesRegex(RuntimeError, "inside tuning"):
                search.run()
            resumed = self._search(tmp, cfg, _runner(tmp))
            result = resumed.run()
            for family, state_info in result["family_notes"].items():
                self.assertLessEqual(state_info["proposed"], cap, family)
            keys = [row["candidate_key"] for row in result["trials"] if row["candidate_key"]]
            self.assertEqual(len(keys), len(set(keys)))

    def test_frozen_primary_selection_still_completes_the_event_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = b.Run.__new__(b.Run)
            run.dataset_name = "gearbox"
            run.pilot = False
            run.identity = {}
            run.dataset = mock.Mock(meta=pd.DataFrame({"equipment": ["E1", "E2"]}),
                                    outer_order=("E3",))
            run.outer_dir = lambda group: Path(tmp) / "outer" / group
            run.write_manifest = lambda: None
            run._write_stage_costs = lambda stage: None
            run._restore_selection_state = lambda group, selection: mock.Mock(states={})
            calls = []
            run._search_event_track = lambda group, search, selection: calls.append(group)
            run.outer_dir("E3").mkdir(parents=True)
            b.write_json_atomic(run.outer_dir("E3") / "selection.json",
                                {"frozen": True, "outer_group": "E3",
                                 "dataset": "gearbox",
                                 "search_train_groups": ["E1", "E2"]})
            run.search()
        self.assertEqual(calls, ["E3"])

class EventOuterSeedDiagnosticsTests(unittest.TestCase):
    """Per-seed diagnostics must use aligned endpoint frames, not per-expert arrays."""

    def _meta_table(self):
        rows, values = [], []
        for falha in range(1, 19):
            event = b.gearbox_event_of(falha)
            for day in range(4):
                rows.append({"endpoint_id": f"F{falha:02d}-{day}", "Falha": falha,
                             "equipment": f"E{event}",
                             "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                             "phase": "saudavel", "y": 0.9 - 0.1 * day})
                values.append({"cur__s0": float(falha / 20 + day / 10),
                               "cur__s1": float(day + 1)})
        meta = pd.DataFrame(rows)
        table = pd.DataFrame(values, index=pd.Index(meta.endpoint_id, name="endpoint_id"))
        return meta, table

    def test_three_seed_component_expert_outer_refits_align_and_average(self):
        meta, table = self._meta_table()
        cfg = _small_config(("extra_trees",))
        cfg["feature_group_sets"] = b.load_config()["feature_group_sets"]
        recipe = b.normalize_recipe("extra_trees", {
            "n_estimators": 4, "max_depth": 3, "min_samples_leaf": 2,
            "max_features": "sqrt", "n_jobs": 1},
            {"strategy": "component_experts"}, cfg["default_recipe"])
        selection = {"family_best": [{"candidate_id": b.recipe_id(recipe),
                                      "recipe": recipe, "seed_offsets": [0, 1, 2]}],
                     "ensembles": [], "chosen": {"candidate_id": b.recipe_id(recipe),
                                                 "kind": "family"}}
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp, max_fits=2000)
            run = b.Run.__new__(b.Run)
            run.dataset_name = "gearbox"
            run.config = cfg
            run.identity = {}
            run.runner = runner
            run.variance_key = "s4"
            frames, statuses, diagnostics = run._outer_frames(
                meta, lambda _recipe: table, "E3", selection)
            self.assertEqual(statuses[b.recipe_id(recipe)], "ok")
            self.assertEqual(diagnostics[b.recipe_id(recipe)]["offsets"], [0, 1, 2])
            self.assertGreaterEqual(
                diagnostics[b.recipe_id(recipe)]["prediction_std_pp_mean"], 0.0)
            frame = frames[b.recipe_id(recipe)]
            expected = meta.loc[meta.equipment == "E3", "endpoint_id"]
            self.assertEqual(list(frame["endpoint_id"]), list(expected))
            self.assertEqual(len(frame), len(expected))

class CorrectionRound1GapTests(unittest.TestCase):
    """Missing-track verification, cgroup availability and pre-fork budget reservation."""

    def test_missing_event_selection_or_evaluation_is_a_failed_check_in_a_full_run(self):
        run = b.Run.__new__(b.Run)
        run.dataset_name = "gearbox"
        run.pilot = False
        with tempfile.TemporaryDirectory() as tmp:
            run.event_dir = lambda group: Path(tmp) / "outer" / group / "event"
            checks = []
            run._check_event_track("E3", lambda name, ok, detail="": checks.append((name, ok)))
            self.assertIn(("E3/event: selection present", False), checks)
        self.assertFalse(any(ok for _name, ok in checks))

    def test_pilot_event_track_is_reported_as_absent_without_claiming_success(self):
        run = b.Run.__new__(b.Run)
        run.dataset_name = "gearbox"
        run.pilot = True
        with tempfile.TemporaryDirectory() as tmp:
            run.event_dir = lambda group: Path(tmp) / "outer" / group / "event"
            checks = []
            run._check_event_track("E3", lambda name, ok, detail="": checks.append((name, ok)))
        self.assertIn(("E3/event: selection present", True), checks)

    def test_effective_available_uses_the_smaller_of_host_and_cgroup_headroom(self):
        resources = dict(b.load_config()["resources"])
        policy = b.ResourcePolicy(resources, available_bytes=lambda: 4 * b.GIB,
                                  cgroup_limit=lambda: 2 * b.GIB,
                                  cgroup_current=lambda: int(1.5 * b.GIB))
        self.assertEqual(policy.available(), int(0.5 * b.GIB))

    def test_budget_is_reserved_on_disk_before_the_worker_is_forked(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            context = runner._context
            with mock.patch.object(context, "Process",
                                   side_effect=RuntimeError("fork refused")):
                with self.assertRaisesRegex(RuntimeError, "fork refused"):
                    runner.run([_task(key_salt="prefork")])
            ledger = b.read_json(Path(tmp) / "ledger.json")
            self.assertEqual(ledger["charged"], 1)

class InvalidProposalResumeTests(unittest.TestCase):
    """Terminal FAILED proposals must never be mistaken for corruption on resume."""

    def _search(self, tmp):
        meta, table = _synthetic_meta()
        train = meta.loc[meta["equipment"] != "A",
                         ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("ridge",))
        cfg["tuning_enabled"] = False
        cfg["legacy_quick_control"]["recipes"] = []
        return b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                             tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                             candidates=b.CandidateStore(Path(tmp) / "candidates"),
                             study_dir=Path(tmp) / "study", identity={}, variance_key="s2")

    def test_crash_after_telling_an_invalid_recipe_is_not_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            search = self._search(tmp)
            state = search._state("ridge")
            study = search._study("ridge", state)
            trial = study.ask()
            search.journal.save("ridge", [{"number": trial.number, "recipe": None,
                                           "reason": "no such column", "phase": "screen",
                                           "order": 0, "seeds": {}}], study.sampler)
            trial.set_user_attr("candidate_invalid", True)
            trial.set_user_attr("failure_reason", "invalid recipe: no such column")
            study.tell(trial, state=b.optuna.trial.TrialState.FAIL)
            # journal intentionally not reconciled: the crash happens right after
            fresh = self._search(tmp)
            fresh_state = fresh._state("ridge")
            recovered = fresh._restore_family("ridge", fresh_state)
            self.assertEqual(recovered, [])
            self.assertEqual(fresh_state.invalid, 1)
            self.assertEqual(fresh_state.proposed, 1)
            self.assertEqual(fresh_state.failed, 0)
            self.assertEqual(fresh.journal.load("ridge")[0], [])
            self.assertEqual(fresh.runner.stats.fits_started, 0)

    def test_invalid_items_are_told_and_dropped_from_the_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            search = self._search(tmp)
            state = search._state("ridge")
            study = search._study("ridge", state)
            trial = study.ask()
            item = {"number": trial.number, "trial": trial, "recipe": None,
                    "reason": "bad combination", "phase": "screen"}
            search.journal.save("ridge", [{"number": trial.number, "recipe": None,
                                           "reason": "bad combination", "phase": "screen",
                                           "order": 0, "seeds": {}}], study.sampler)
            search._apply_items("ridge", state, [item])
            self.assertEqual(search.journal.load("ridge")[0], [])
            self.assertEqual(state.invalid, 1)
            self.assertEqual(study.get_trials(deepcopy=False)[0].state,
                             b.optuna.trial.TrialState.FAIL)
            attrs = study.get_trials(deepcopy=False)[0].user_attrs
            self.assertTrue(attrs["candidate_invalid"])
            self.assertNotIn("candidate_failed", attrs)

    def test_mixed_batch_resume_keeps_counts_and_charges_no_extra_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            search = self._search(tmp)
            state = search._state("ridge")
            study = search._study("ridge", state)
            invalid_trial = study.ask()
            valid_trial = study.ask()
            recipes = [None, b.normalize_recipe(
                "ridge", {"alpha": 1.0},
                {"features.feature_groups": "sensors"}, search.config["default_recipe"])]
            numbers = [invalid_trial.number, valid_trial.number]
            search.journal.save("ridge", [
                {"number": number, "recipe": recipe,
                 "reason": "bad" if recipe is None else None, "phase": "screen",
                 "order": order, "seeds": {}}
                for order, (number, recipe) in enumerate(zip(numbers, recipes))],
                study.sampler)
            items = [{"number": numbers[0], "trial": invalid_trial, "recipe": None,
                      "reason": "bad", "phase": "screen"},
                     {"number": numbers[1], "trial": valid_trial, "recipe": recipes[1],
                      "reason": None, "phase": "screen"}]
            search._apply_items("ridge", state, items)
            charged = search.runner.budget.charged
            self.assertEqual(state.invalid, 1)
            self.assertIsNotNone(state.best)
            self.assertEqual(search.journal.load("ridge")[0], [])

            resumed = self._search(tmp)
            resumed_state = resumed._state("ridge")
            resumed._restore_family("ridge", resumed_state)
            self.assertEqual(resumed_state.invalid, 1)
            self.assertEqual(resumed_state.failed, 0)
            self.assertEqual(resumed_state.proposed, 2)
            self.assertEqual(resumed.runner.stats.fits_started, 0)
            self.assertEqual(search.runner.budget.charged, charged)

def _synthetic_gearbox_dataset(tmp, records=6):
    """Three events x six measurements, aligned timestamps and targets, tiny features."""
    rng = np.random.default_rng(3)
    sensors = b.cer.GEARBOX_SENSOR_COLUMNS
    clean_rows, endpoint_rows, meta_rows = [], [], []
    for falha in range(1, 19):
        event = (falha - 1) // 6 + 1
        baseline = rng.uniform(0.2, 0.8, size=len(sensors))
        for index in range(records):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=index)
            values = baseline + 0.01 * index + 0.001 * rng.standard_normal(len(sensors))
            clean_rows.append({"Date": date, "Falha": falha,
                               **dict(zip(sensors, values))})
            if index >= 3:
                endpoint_id = f"F{falha:02d}-r{index}"
                endpoint_rows.append({"endpoint_id": endpoint_id, "Falha": falha,
                                      "Date": date, "record_index": index})
                meta_rows.append({"endpoint_id": endpoint_id, "Falha": falha,
                                  "equipment": f"E{event}", "Date": date,
                                  "phase": "saudavel" if index < 4 else "pos_FPT",
                                  "y": float(np.clip(1 - index / records, 0, 1))})
    csv_path = Path(tmp) / "synthetic-gearbox.csv"
    csv_path.write_text("Date,Falha,RUL\n", encoding="utf-8")
    profile = {"measurement_by_id": {fid: b.cer.GEARBOX_MEASUREMENT_NAMES[(fid - 1) % 6]
                                     for fid in range(1, 19)},
               "sensor_columns": sensors}
    return b.Dataset(name="gearbox", profile=profile, clean=pd.DataFrame(clean_rows),
                     endpoints=pd.DataFrame(endpoint_rows), meta=pd.DataFrame(meta_rows),
                     csv_path=csv_path, csv_sha256=b.sha256_file(csv_path),
                     endpoints_sha256="synthetic", outer_order=("E1", "E2", "E3"),
                     sensor_columns=tuple(sensors))


class GearboxEndToEndRunTests(unittest.TestCase):
    """Production Run constructor, only the dataset provider is patched."""

    def test_search_evaluate_verify_cover_both_tracks_without_real_gearbox_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _synthetic_gearbox_dataset(tmp)
            config = b.load_config()
            config["datasets"]["gearbox"]["outer_order"] = ["E1", "E2", "E3"]
            config["gearbox_strategies"]["joint_event"]["expected_endpoints"] = 9
            config["families"] = {name: config["families"][name]
                                  for name in ("mean", "ridge")}
            config["legacy_quick_control"]["recipes"] = []
            # Specialist strategies are covered by the dedicated synthetic fit test;
            # this test proves the Run-level flow across both availability tracks.
            config["gearbox_strategies"]["screening"] = ["pooled_anonymous"]
            config["tuning_enabled"] = False
            config["pilot"]["fixed_candidates"] = {
                name: config["pilot"]["fixed_candidates"][name]
                for name in ("mean", "ridge")}
            # effective_config enables tuning for non-pilot runs; zero it to keep the
            # end-to-end flow small (the stages themselves are covered elsewhere).
            config["stages"].update({"screen_trials": 1, "batch_size": 1,
                                     "tune_top_k": 0, "tune_extra_trials": 0,
                                     "ablation_max": 0, "ablation_retune_trials": 0,
                                     "gearbox_strategy_recipes_per_family": 1,
                                     "gearbox_strategy_extra_per_family": 1,
                                     "gearbox_strategy_extra_top": 0,
                                     "ensemble_sizes": [2], "ensemble_how": ["mean"]})
            config_path = Path(tmp) / "config.json"
            b.write_json_atomic(config_path, config)
            with mock.patch.object(b, "load_dataset", return_value=dataset):
                run = b.Run("gearbox", pilot=False, config_path=config_path,
                            output_root=Path(tmp) / "out")
                run.search()
                for group in dataset.outer_order:
                    self.assertTrue((run.outer_dir(group) / "selection.json").exists())
                    self.assertTrue((run.event_dir(group) / "selection.json").exists())
                run.evaluate()
                result = run.verify()
            self.assertEqual(result["status"], "evaluated_complete",
                             [c for c in result["checks"] if not c[1]][:3])
            event_eval = b.read_json(run.event_dir("E1") / "evaluation.json")
            self.assertEqual(event_eval["track"], "event")
            self.assertIn("aggregation", event_eval["candidates"])

class DeclaredPresetCompletenessTests(unittest.TestCase):
    """Declared presets with only meaningful knobs must build; omissions fail loudly."""

    def test_real_legacy_presets_get_definitional_const_parameters(self):
        dataset = _synthetic_dataset(groups=("A", "B", "C"), records=12)
        table = b.build_feature_table(dataset, 12, 32, {"protocol": {
            "relative_eps": {"floor": 1e-8, "relative": 1e-6}}})
        train = dataset.meta.loc[dataset.meta["equipment"] != "C",
                                 ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = b.effective_config(b.load_config(), False)
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="C", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            recipes, named, _unavailable = search._legacy_recipes()
        svr = [r for r in recipes if r["family"] == "svr"]
        self.assertEqual(len(svr), 2)
        for recipe in svr:
            for name in ("max_iter", "cache_size", "gamma"):
                self.assertIn(name, recipe["params"])
            b.build_estimator("svr", recipe["params"], 0)
        for recipe in recipes:
            b.build_estimator(recipe["family"], recipe["params"], 0)

    def test_missing_declared_parameter_is_an_invalid_recipe_not_a_crash(self):
        with self.assertRaises(b.InvalidRecipe):
            b.build_estimator("svr", {"C": 1.0, "epsilon": 0.1}, 0)

class LegacyControlIsolationTests(unittest.TestCase):
    """Legacy quick presets are descriptive: they must never touch selection state."""

    def test_legacy_controls_leave_every_family_state_untouched(self):
        dataset = _synthetic_dataset(groups=("A", "B", "C"), records=12)
        table = b.build_feature_table(dataset, 12, 32, {"protocol": {
            "relative_eps": {"floor": 1e-8, "relative": 1e-6}}})
        train = dataset.meta.loc[dataset.meta["equipment"] != "C",
                                 ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = b.effective_config(b.load_config(), False)
        with tempfile.TemporaryDirectory() as tmp:
            search = b.OuterSearch(dataset_name="gerador", outer_group="C", train_meta=train,
                                   tables=lambda w, c: table, runner=_runner(tmp), config=cfg,
                                   candidates=b.CandidateStore(Path(tmp) / "candidates"),
                                   identity={}, variance_key="s2")
            states = search.states
            before = {family: (state.best, state.anchor_score, state.last_improve_at,
                               state.proposed, state.invalid, state.failed)
                      for family, state in states.items()}

            def fake_evaluate(items, seed_offset=0):
                return [b.CandidateResult(
                    key=f"legacy-{number}", family=recipe["family"], recipe=recipe,
                    trial=number, phase="legacy_control", status="ok", reasons=[],
                    score_pp=0.01, rmse_pp=0.02, complexity=1, oof=None, fit_keys=[],
                    candidate_id=f"legacy-{number}")
                    for number, _phase, recipe in items]

            search._evaluate = fake_evaluate
            search._legacy_controls()
            after = {family: (state.best, state.anchor_score, state.last_improve_at,
                              state.proposed, state.invalid, state.failed)
                     for family, state in states.items()}
        self.assertEqual(before, after)
        self.assertTrue(search._legacy_results)
        self.assertTrue(all(row["status"] == "ok" for row in search._legacy_results))
        self.assertTrue(all(row["candidate_id"].startswith("legacy-")
                            for row in search._legacy_results))
        self.assertTrue(all("rmse_pp" in row for row in search._legacy_results))
        self.assertFalse(any(row["score_pp"] == 1.0 for row in search._legacy_results))

    def test_legacy_controls_do_not_change_the_selected_candidate(self):
        meta, table = _synthetic_meta()
        train = meta.loc[meta["equipment"] != "A",
                         ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]]
        cfg = _small_config(("mean", "ridge"))
        cfg["tuning_enabled"] = False
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for with_controls in (False, True):
                local = b.effective_config(cfg, False)
                if not with_controls:
                    local["legacy_quick_control"]["recipes"] = []
                search = b.OuterSearch(dataset_name="gerador", outer_group="A", train_meta=train,
                                       tables=lambda w, c: table, runner=_runner(tmp),
                                       config=local,
                                       candidates=b.CandidateStore(Path(tmp) / f"c{with_controls}"),
                                       study_dir=Path(tmp) / f"s{with_controls}", identity={},
                                       variance_key="s2")
                results.append(search.run()["chosen"])
        self.assertEqual(results[0]["candidate_id"], results[1]["candidate_id"])

class CliSubprocessTests(unittest.TestCase):
    SCRIPT = ROOT / "baseline_experiments.py"

    def _run(self, *args):
        return subprocess.run([sys.executable, str(self.SCRIPT), *args], capture_output=True,
                              text=True, timeout=300)

    def test_help_lists_the_commands(self):
        result = self._run("--help")
        self.assertEqual(result.returncode, 0)
        for command in ("diagnose", "search", "evaluate", "report", "verify"):
            self.assertIn(command, result.stdout)

    def test_verify_on_an_empty_output_root_fails_and_names_the_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("verify", "--dataset", "gerador", "--pilot", "--output-root", tmp)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL manifest present", result.stdout)
        self.assertIn("RESULT: failed", result.stdout)

    def test_pilot_gearbox_is_refused_with_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run("search", "--dataset", "gearbox", "--pilot", "--output-root", tmp)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pilot is limited", result.stderr)


if __name__ == "__main__":
    unittest.main()
