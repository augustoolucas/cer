"""Small unittest suite for cer_rul (synthetic contract + one real-data check)."""

import math
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cer_rul as c  # noqa: E402


def synthetic_clean(n=40, fault_id=1):
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({
        "Date": dates,
        "Falha": [fault_id] * n,
        "RUL": np.linspace(80.0, 0.0, n),
        "raw_count": [1] * n,
    })
    for position, column in enumerate(c.SENSOR_COLUMNS):
        frame[column] = np.linspace(1.0 + position, 2.0 + position, n)
    return frame.loc[:, ["Date", "Falha", "RUL", "raw_count"] + list(c.SENSOR_COLUMNS)]


def same_features(first, second):
    return all(
        (math.isnan(first[key]) and math.isnan(second[key]))
        or abs(first[key] - second[key]) < 1e-12
        for key in first
    )


class ConsolidationTests(unittest.TestCase):
    def test_duplicate_target_mean_weighting(self):
        base = pd.Timestamp("2024-01-01 00:00:00")
        raw = pd.DataFrame({
            "Date": [base, base, base, base + pd.Timedelta(hours=1)],
            "Falha": [1, 1, 1, 1],
            "RUL": [1.0, 1.0, 0.0, 5.0],
        })
        for position, column in enumerate(c.SENSOR_COLUMNS):
            raw[column] = [1.0 + position] * 3 + [1.01 + position]
        raw = raw.loc[:, list(c.CSV_COLUMNS)]

        clean, audit, _ = c.consolidate_frame(raw, "synthetic",
                                              enforce_expectations=False)
        first_target = float(clean.loc[clean["Date"] == base, "RUL"].iloc[0])
        # Mean of ALL original rows is 2/3, not the mean of unique values (0.5).
        self.assertAlmostEqual(first_target, 2.0 / 3.0, places=12)
        self.assertGreater(abs(first_target - 0.5), 1e-6)
        self.assertEqual(audit["target_conflict_keys"], 1)
        self.assertEqual(audit["raw_max_multiplicity"], 3)
        self.assertEqual(int(clean.loc[clean["Date"] == base, "raw_count"].iloc[0]), 3)
        self.assertEqual(list(clean.columns),
                         ["Date", "Falha", "RUL", "raw_count"] + list(c.SENSOR_COLUMNS))

    def test_sensor_conflict_is_rejected(self):
        base = pd.Timestamp("2024-01-01 00:00:00")
        raw = pd.DataFrame({
            "Date": [base, base],
            "Falha": [1, 1],
            "RUL": [1.0, 1.0],
        })
        for position, column in enumerate(c.SENSOR_COLUMNS):
            raw[column] = [1.0 + position, 1.0 + position]
        raw.loc[1, c.SENSOR_COLUMNS[0]] = 999.0
        raw = raw.loc[:, list(c.CSV_COLUMNS)]
        with self.assertRaises(ValueError):
            c.consolidate_frame(raw, "synthetic", enforce_expectations=False)

    def test_fractional_fault_ids_are_rejected(self):
        self.assertEqual(c.coerce_fault_ids(["1", "7"]).tolist(), [1, 7])
        with self.assertRaises(ValueError):
            c.coerce_fault_ids(["1", "1.9"])


class SplitAndMaskTests(unittest.TestCase):
    def test_group_separation(self):
        splits = c.make_splits()
        self.assertEqual(len(splits), 4)
        self.assertEqual(set(splits["test_equipment"]), set(c.EQUIPMENT_ORDER))
        for row in splits.to_dict(orient="records"):
            test = set(row["test_ids"])
            val = set(row["val_ids"])
            inner = set(row["inner_train_ids"])
            outer = set(row["outer_train_ids"])
            self.assertTrue(test.isdisjoint(outer))
            self.assertTrue(val.isdisjoint(inner))
            self.assertEqual(val | inner, outer)
            self.assertEqual(test | outer, set(c.FAULT_IDS))
            self.assertIn(row["val_equipment"], c.EQUIPMENT_ORDER)
            self.assertNotEqual(row["val_equipment"], row["test_equipment"])
            self.assertEqual(val, set(c.EQUIPMENT_GROUPS[row["val_equipment"]]))

        target = pd.DataFrame({"Falha": list(c.FAULT_IDS)})
        masks = c.masks_for_fold(target, splits.iloc[0].to_dict())
        self.assertFalse((masks["test"] & masks["outer_train"]).any())
        self.assertFalse((masks["inner_train"] & masks["val"]).any())
        self.assertEqual(masks["test"].sum(), len(splits.iloc[0]["test_ids"]))
        self.assertGreater(masks["val"].sum(), 0)


class EndpointAndFeatureTests(unittest.TestCase):
    def test_final_endpoint_included(self):
        group = synthetic_clean(n=40)
        endpoints = c.build_endpoint_rows(group, warmup=5, stride=7, enforce=False)
        for fault_id, part in endpoints.groupby("Falha", sort=True):
            self.assertGreaterEqual(part["record_index"].min(), 5)
            self.assertEqual(part["record_index"].max(), 39)
            self.assertEqual(part["n_records_after"].min(), 0)
            self.assertEqual(part["endpoint_id"].iloc[-1], f"F{fault_id:02d}-r0039")

    def test_future_perturbation_does_not_change_history(self):
        group = synthetic_clean(n=80)
        endpoint = types.SimpleNamespace(record_index=60, Date=group.loc[60, "Date"])
        before = c.build_feature_row(endpoint, group, 12)

        future_row = group.iloc[[-1]].copy()
        future_row["Date"] = group["Date"].max() + pd.Timedelta(hours=1)
        for column in c.SENSOR_COLUMNS:
            future_row[column] = future_row[column] + 1000.0
        after = c.build_feature_row(
            endpoint, pd.concat([group, future_row], ignore_index=True), 12)
        self.assertTrue(same_features(before, after))

    def test_bin_left_boundary_and_last_value(self):
        endpoint_date = pd.Timestamp("2024-01-03 12:00:00")
        dates = pd.Series([
            endpoint_date - pd.Timedelta(hours=13),
            endpoint_date - pd.Timedelta(hours=12),
            endpoint_date - pd.Timedelta(hours=11),
            endpoint_date,
        ])
        values = np.vstack([np.ones(8) * value for value in (10, 20, 30, 40)])
        last_values, last_times, counts = c.bin_stats(dates, values, endpoint_date, 12)
        self.assertEqual(counts[30], 2)
        self.assertEqual(counts[31], 2)
        self.assertEqual(last_values[30, 0], 20)
        self.assertEqual(last_values[31, 0], 40)
        self.assertEqual(last_times.iloc[30], endpoint_date - pd.Timedelta(hours=12))

    def test_slope_is_per_actual_day(self):
        day = pd.Timestamp("2024-01-01")
        _, _, slope = c.daily_stats(pd.Series([day, day + pd.Timedelta(days=1)]),
                                    np.array([0.0, 2.0]))
        self.assertAlmostEqual(slope, 2.0, places=12)

    def test_nondefault_permitted_width_is_honored(self):
        group = synthetic_clean(n=80)
        endpoints = c.build_endpoint_rows(group, warmup=5, stride=20, enforce=False)
        width = 48
        permitted = (6, 12, 24, 48)
        frame = c.build_feature_frame(
            group, endpoints, width, bin_hours_candidates=permitted)
        expected = c.feature_columns(
            width, "summary", True, bin_hours_candidates=permitted)
        self.assertEqual(frame.columns.tolist(), expected)
        self.assertAlmostEqual(float(frame["timing__window_days"].iloc[0]), 64.0)


class MetricTests(unittest.TestCase):
    def test_known_metrics(self):
        metrics = c.regression_metrics([1.0, 2.0], [2.0, 4.0])
        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["mae"], 1.5, places=12)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.5), places=12)
        self.assertAlmostEqual(metrics["bias"], 1.5, places=12)

    def test_macro_keeps_families_separate(self):
        prediction_frame = pd.DataFrame([
            {"label": "dias-90", "family": "mean", "fold_id": "fold_A",
             "Falha": 1, "equipment": "A", "phase": "saudavel",
             "y_metric": 10.0, "pred_metric": 12.0},
            {"label": "dias-90", "family": "mean", "fold_id": "fold_B",
             "Falha": 2, "equipment": "B", "phase": "saudavel",
             "y_metric": 10.0, "pred_metric": 8.0},
            {"label": "dias-90", "family": "ridge", "fold_id": "fold_A",
             "Falha": 1, "equipment": "A", "phase": "saudavel",
             "y_metric": 10.0, "pred_metric": 14.0},
            {"label": "dias-90", "family": "ridge", "fold_id": "fold_B",
             "Falha": 2, "equipment": "B", "phase": "saudavel",
             "y_metric": 10.0, "pred_metric": 6.0},
        ])
        macro = c.aggregate_metric_rows(c.fold_metric_rows(prediction_frame))
        mean_macro = macro[(macro["family"] == "mean")
                           & (macro["scope"] == "overall")
                           & (macro["name"] == "all")]["mae"].iloc[0]
        ridge_macro = macro[(macro["family"] == "ridge")
                            & (macro["scope"] == "overall")
                            & (macro["name"] == "all")]["mae"].iloc[0]
        self.assertAlmostEqual(mean_macro, 2.0, places=12)
        self.assertAlmostEqual(ridge_macro, 4.0, places=12)
        self.assertEqual(
            set(macro["aggregation"]),
            {"media_simples_entre_folds_por_familia",
             "media_igual_entre_4_equipamentos_por_familia"},
        )


class RealDataTests(unittest.TestCase):
    CSV = ROOT / "data" / "Features_gerador_score_RUL-porcentagem-data.csv"

    @unittest.skipUnless(CSV.exists(), "real CSV not available")
    def test_real_csv_counts_and_audit(self):
        raw = c.load_raw_label(self.CSV, label="porcentagem-data",
                               max_allowed=1.0, print_summary=False)
        self.assertEqual(len(raw), 2710)

        clean, audit, terminal = c.consolidate_frame(
            raw, "porcentagem-data",
            label_meta=c.LABEL_META["porcentagem-data"],
            expected_clean_keys=1910,
            expected_target_conflicts=0,
            enforce_expectations=True,
        )
        self.assertEqual(len(clean), 1910)
        self.assertEqual(audit["raw_rows"], 2710)
        self.assertEqual(audit["clean_keys"], 1910)
        self.assertEqual(audit["target_conflict_keys"], 0)
        self.assertEqual(audit["feature_conflict_keys"], 0)
        self.assertEqual(audit["exact_duplicate_surplus_rows"], 800)
        self.assertEqual(len(terminal), len(c.FAULT_IDS))


def _profile_csv_columns(profile):
    return ("Date",) + tuple(profile["sensor_columns"]) + ("Falha", "RUL")


def _profile_id_to_equipment(profile):
    return {int(fault_id): name
            for name, fault_ids in profile["equipment_groups"].items()
            for fault_id in fault_ids}


def _real_csv(name):
    profile = c.get_dataset_profile(name)
    return ROOT / "data" / profile["labels"]["porcentagem-data"]["filename"]


class DatasetProfileTests(unittest.TestCase):
    EXPECTED_KEYS = {
        "dataset_id", "unit_kind", "sensor_columns", "equipment_groups",
        "measurement_by_id", "fpt_annotations", "fpt_note", "labels",
        "expected_raw_rows", "expected_clean_keys", "expected_target_conflicts",
        "expected_endpoints",
    }

    def test_profile_is_fresh_copy(self):
        first = c.get_dataset_profile("gerador")
        first["sensor_columns"] = ("tampered",)
        first["equipment_groups"]["A"] = (99,)
        first["labels"]["porcentagem-data"]["divisor"] = 123.0
        first["fpt_annotations"][1] = "1999-01-01"
        second = c.get_dataset_profile("gerador")
        self.assertIsNot(first, second)
        self.assertEqual(tuple(second["sensor_columns"]), c.SENSOR_COLUMNS)
        self.assertEqual(tuple(second["equipment_groups"]["A"]), (1, 2))
        self.assertEqual(second["labels"]["porcentagem-data"]["divisor"], 1.0)
        self.assertEqual(second["fpt_annotations"][1], "2024-07-25")
        self.assertEqual(c.LABEL_META["porcentagem-data"]["divisor"], 1.0)
        self.assertEqual(c.FPT_ANNOTATIONS[1], "2024-07-25")
        self.assertEqual(tuple(c.get_dataset_profile("gearbox")["equipment_groups"]["E1"]),
                         (1, 2, 3, 4, 5, 6))

    def test_unknown_profile_rejected(self):
        for name in ("dias", "Gerador", "", None, 7, ["gerador"]):
            with self.subTest(name=name), self.assertRaises(ValueError):
                c.get_dataset_profile(name)

    def test_profile_keys_and_labels_exact(self):
        for name in ("gerador", "gearbox"):
            profile = c.get_dataset_profile(name)
            self.assertEqual(set(profile), self.EXPECTED_KEYS)
            self.assertEqual(list(profile["labels"]), ["porcentagem-data"])
            self.assertEqual(profile["dataset_id"], name)

    def test_generator_profile_matches_module_constants(self):
        profile = c.get_dataset_profile("gerador")
        self.assertEqual(tuple(profile["sensor_columns"]), c.SENSOR_COLUMNS)
        self.assertEqual({k: tuple(v) for k, v in profile["equipment_groups"].items()},
                         {"A": (1, 2), "B": (3,), "C": (4, 5), "D": (6, 7)})
        self.assertEqual(profile["unit_kind"], "equipment")
        self.assertEqual(sorted(profile["measurement_by_id"]), list(range(1, 8)))
        self.assertEqual(set(profile["measurement_by_id"].values()), {"gerador"})
        self.assertEqual(profile["fpt_annotations"], dict(c.FPT_ANNOTATIONS))
        self.assertEqual(
            profile["fpt_note"],
            "Datas herdadas da célula 19 de RUL_CMS_TCN.ipynb, usadas apenas em "
            "relatório de fase e pesos opcionais de treino. Não foram verificadas "
            "fisicamente de forma independente.")
        label = profile["labels"]["porcentagem-data"]
        self.assertEqual(label["filename"], "Features_gerador_score_RUL-porcentagem-data.csv")
        self.assertEqual(label["kind"], "fraction")
        self.assertEqual(label["divisor"], 1.0)
        self.assertEqual(label["metric_factor"], 100.0)
        self.assertEqual(label["unit"], "pontos percentuais (pp)")
        self.assertEqual(label["max_allowed"], 1.0)
        self.assertEqual(profile["expected_raw_rows"], 2710)
        self.assertEqual(profile["expected_clean_keys"], 1910)
        self.assertEqual(profile["expected_target_conflicts"], {"porcentagem-data": 0})
        self.assertEqual(profile["expected_endpoints"], 574)

    def test_gearbox_profile_schema_mapping_and_fpt(self):
        profile = c.get_dataset_profile("gearbox")
        self.assertEqual(tuple(profile["sensor_columns"]), (
            "g_1_Meanfreq", "v_1_Centerfreq", "g_0_Peak_peak", "g_0_RMS",
            "g_0_Variance", "v_1_Rmsfreq",
            "e_HS-S-RS (GBX)_BSF_2X|NCF 2232 ECN1JB/L4BC3",
            "e_HS-S-GS (GBX) (Pista dupla)_BPFI_2X|BT2-0246 C/L4BPEX"))
        self.assertEqual(profile["unit_kind"], "event")
        self.assertEqual({k: tuple(v) for k, v in profile["equipment_groups"].items()}, {
            "E1": (1, 2, 3, 4, 5, 6), "E2": (7, 8, 9, 10, 11, 12),
            "E3": (13, 14, 15, 16, 17, 18)})
        names = ("HSS-Radial", "HSS-Axial", "IMS", "LSS", "Planetária 0°", "Planetária 90°")
        self.assertEqual(sorted(profile["measurement_by_id"]), list(range(1, 19)))
        for fault_id in range(1, 19):
            self.assertEqual(profile["measurement_by_id"][fault_id],
                             names[(fault_id - 1) % 6])
        self.assertEqual(
            {fault_id: date for fault_id, date in profile["fpt_annotations"].items()},
            {**{i: "2023-06-14" for i in range(1, 7)},
             **{i: "2022-11-01" for i in range(7, 13)},
             **{i: "2023-08-13" for i in range(13, 19)}})
        self.assertIn("não foram verificadas", profile["fpt_note"])
        self.assertIn("features de inferência", profile["fpt_note"])
        label = profile["labels"]["porcentagem-data"]
        self.assertEqual(label["filename"],
                         "Features_gearbox_score_todos_RUL-porcentagem-data.csv")
        self.assertEqual(label["unit"], "pontos percentuais (pp)")
        self.assertEqual(label["max_allowed"], 1.0)
        self.assertEqual(profile["expected_raw_rows"], 2946)
        self.assertEqual(profile["expected_clean_keys"], 2946)
        self.assertEqual(profile["expected_target_conflicts"], {"porcentagem-data": 0})
        self.assertEqual(profile["expected_endpoints"], 822)


class DatasetRealDataTests(unittest.TestCase):
    """Single-file checks: each CSV is loaded alone, never merged."""

    def _check_dataset(self, name):
        profile = c.get_dataset_profile(name)
        label = "porcentagem-data"
        meta = profile["labels"][label]
        csv_columns = _profile_csv_columns(profile)
        sensors = tuple(profile["sensor_columns"])
        ids = tuple(sorted(int(fault_id) for fault_id in profile["measurement_by_id"]))
        id_to_equipment = _profile_id_to_equipment(profile)
        equipment_order = tuple(profile["equipment_groups"])

        raw = c.load_raw_label(
            _real_csv(name), label=label,
            expected_raw_rows=profile["expected_raw_rows"],
            csv_columns=csv_columns, sensor_columns=sensors, fault_ids=ids,
            max_allowed=meta["max_allowed"], print_summary=False)
        self.assertEqual(len(raw), profile["expected_raw_rows"])
        self.assertEqual(list(raw.columns), list(csv_columns))
        self.assertEqual(set(raw["Falha"].unique()), set(ids))

        clean, audit, terminal = c.consolidate_frame(
            raw, label, label_meta=meta, sensor_columns=sensors,
            csv_columns=csv_columns, fault_ids=ids,
            id_to_equipment=id_to_equipment, equipment_order=equipment_order,
            expected_clean_keys=profile["expected_clean_keys"],
            expected_target_conflicts=profile["expected_target_conflicts"][label],
            enforce_expectations=True)
        self.assertEqual(len(clean), profile["expected_clean_keys"])
        self.assertEqual(audit["clean_keys"], profile["expected_clean_keys"])
        self.assertEqual(audit["target_conflict_keys"], 0)
        self.assertEqual(audit["feature_conflict_keys"], 0)
        self.assertEqual(set(clean["Falha"].unique()), set(ids))
        self.assertEqual(len(terminal), len(ids))
        self.assertEqual(audit["divisor"], meta["divisor"])
        self.assertEqual(audit["metric_factor"], meta["metric_factor"])
        self.assertEqual(audit["unit"], meta["unit"])
        return profile, clean, audit, id_to_equipment, equipment_order

    def test_generator_csv_counts_folds_masks_endpoints(self):
        if not _real_csv("gerador").exists():
            self.skipTest("real generator CSV not available")
        profile, clean, audit, id_to_equipment, order = self._check_dataset("gerador")
        self.assertEqual(audit["raw_rows"], 2710)
        self.assertEqual(audit["exact_duplicate_surplus_rows"], 800)
        self.assertEqual(audit["clean_rows_per_equipment"],
                         {"A": 1110, "B": 130, "C": 466, "D": 204})
        self._check_folds_masks_endpoints(profile, clean, id_to_equipment, order,
                                          expected_folds=4, expected_endpoints=574)

    def test_gearbox_csv_counts_folds_masks_endpoints(self):
        if not _real_csv("gearbox").exists():
            self.skipTest("real gearbox CSV not available")
        profile, clean, audit, id_to_equipment, order = self._check_dataset("gearbox")
        self.assertEqual(audit["raw_rows"], 2946)
        self.assertEqual(audit["exact_duplicate_surplus_rows"], 0)
        self._check_folds_masks_endpoints(profile, clean, id_to_equipment, order,
                                          expected_folds=3, expected_endpoints=822)

    def test_gearbox_loader_rejects_default_generator_row_count(self):
        if not _real_csv("gearbox").exists():
            self.skipTest("real gearbox CSV not available")
        profile = c.get_dataset_profile("gearbox")
        with self.assertRaises(ValueError):
            c.load_raw_label(
                _real_csv("gearbox"), label="gearbox",
                csv_columns=_profile_csv_columns(profile),
                sensor_columns=tuple(profile["sensor_columns"]),
                fault_ids=tuple(range(1, 19)), max_allowed=1.0,
                print_summary=False)

    def _check_folds_masks_endpoints(self, profile, clean, id_to_equipment,
                                     order, expected_folds, expected_endpoints):
        folds = c.make_splits(equipment_order=order,
                              equipment_groups=profile["equipment_groups"])
        self.assertEqual(len(folds), expected_folds)
        self.assertEqual(set(folds["test_equipment"]), set(order))
        for fold in folds.to_dict(orient="records"):
            masks = c.masks_for_fold(clean, fold, id_to_equipment=id_to_equipment,
                                     equipment_order=order)
            total = int(masks["outer_train"].sum() + masks["test"].sum())
            self.assertEqual(total, len(clean))
            self.assertEqual(int(masks["inner_train"].sum() + masks["val"].sum()),
                             int(masks["outer_train"].sum()))
            self.assertGreater(int(masks["val"].sum()), 0)

        endpoints = c.build_endpoint_rows(
            clean, warmup=c.DEFAULT_WARMUP_RECORDS, stride=c.DEFAULT_ENDPOINT_STRIDE,
            id_to_equipment=id_to_equipment,
            fpt_annotations=profile["fpt_annotations"],
            expected_endpoint_count=profile["expected_endpoints"], enforce=True)
        self.assertEqual(len(endpoints), expected_endpoints)
        self.assertEqual(set(endpoints["Falha"].unique()), set(profile["measurement_by_id"]))
        self.assertTrue((endpoints["n_records_after"].groupby(endpoints["Falha"])
                         .min() == 0).all())
        self.assertTrue(set(endpoints["phase"]) <= {"saudavel", "pos_FPT"})


class TabularPreprocessorTests(unittest.TestCase):
    @staticmethod
    def _frames():
        rng = np.random.default_rng(7)
        base = pd.DataFrame(rng.normal(size=(30, 3)) * [1.0, 10.0, 0.5],
                            columns=["a", "b", "c"])
        base.loc[::5, "a"] = np.nan
        base["c"] = np.nan  # entirely non-finite in train -> zero fallback
        train = base.iloc[:20].copy()
        test = base.iloc[20:].copy()
        test.loc[test.index[0], "a"] = np.nan
        test.loc[test.index[1], "b"] = 1e12
        test.loc[test.index[2], "b"] = -1e9
        test.loc[test.index[3], "a"] = np.inf
        test["c"] = 5.0
        return train, test

    def test_all_modes_shapes_names_and_finiteness(self):
        train, test = self._frames()
        for method in c.PREPROCESSING_OPTIONS:
            with self.subTest(method=method):
                p = c.TabularPreprocessor(method).fit(train)
                out = p.transform(test)
                self.assertEqual(out.shape, (len(test), 6))
                self.assertTrue(np.isfinite(out).all())
                self.assertEqual(p.n_features_out_, 6)
                self.assertEqual(list(p.get_feature_names_out()),
                                 ["a", "b", "c", "na__a", "na__b", "na__c"])
                self.assertEqual(p.constant_fallback_columns_, ["c"])
                self.assertEqual(p.fill_values_["c"], 0.0)
                self.assertEqual(list(p.fit_transform(train).shape), [len(train), 6])

    def test_indicators_off_emit_only_base_columns(self):
        train, test = self._frames()
        for method in c.PREPROCESSING_OPTIONS:
            with self.subTest(method=method):
                on = c.TabularPreprocessor(method).fit(train).transform(test)
                off = c.TabularPreprocessor(method, add_missing_indicators=False).fit(train)
                out = off.transform(test)
                self.assertEqual(out.shape, (len(test), 3))
                self.assertEqual(off.n_features_out_, 3)
                self.assertEqual(list(off.get_feature_names_out()), ["a", "b", "c"])
                self.assertEqual(off.indicator_columns_, [])
                np.testing.assert_array_equal(out, on[:, :3])

    def test_median_and_zero_fallback_from_train_only(self):
        train, test = self._frames()
        p = c.TabularPreprocessor("none").fit(train)
        self.assertEqual(p.fill_values_["a"], float(np.median(train["a"].dropna())))
        self.assertEqual(p.fill_values_["c"], 0.0)
        out = p.transform(test)
        self.assertEqual(out[0, 0], p.fill_values_["a"])
        self.assertEqual(out[0, 3], 1.0)  # missing indicator
        self.assertEqual(out[1, 3], 0.0)

    def test_signed_log_applied_before_standard_scaler(self):
        from sklearn.preprocessing import StandardScaler
        train, test = self._frames()
        p = c.TabularPreprocessor("signed_log1p_standard", add_missing_indicators=False)
        p.fit(train)
        imputed_train = c.signed_log1p(p._base_array(train))
        reference = StandardScaler().fit(imputed_train)
        np.testing.assert_allclose(p.scaler_.mean_, reference.mean_)
        np.testing.assert_allclose(p.scaler_.scale_, reference.scale_)
        np.testing.assert_allclose(
            p.transform(test),
            reference.transform(c.signed_log1p(p._base_array(test))))

    def test_heldout_extremes_do_not_mutate_fitted_parameters(self):
        train, test = self._frames()
        for method in c.PREPROCESSING_OPTIONS:
            with self.subTest(method=method):
                p = c.TabularPreprocessor(method).fit(train)
                fills = dict(p.fill_values_)
                fallback = list(p.constant_fallback_columns_)
                scaler_state = {}
                if p.scaler_ is not None:
                    scaler_state = {k: np.copy(v) for k, v in vars(p.scaler_).items()
                                    if isinstance(v, np.ndarray)}
                p.transform(test)
                self.assertEqual(p.fill_values_, fills)
                self.assertEqual(p.constant_fallback_columns_, fallback)
                for key, value in scaler_state.items():
                    np.testing.assert_array_equal(getattr(p.scaler_, key), value)

    def test_signed_log_handles_negative_inputs(self):
        values = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        out = c.signed_log1p(values)
        np.testing.assert_allclose(out, [-np.log1p(3.0), -np.log1p(0.5), 0.0,
                                         np.log1p(0.5), np.log1p(3.0)])
        np.testing.assert_allclose(c.signed_log1p(-values), -out)
        frame = pd.DataFrame({"x": [-5.0, -1.0, 2.0, -3.0]})
        result = c.TabularPreprocessor("signed_log1p_standard").fit_transform(frame)
        self.assertTrue(np.isfinite(result).all())

    def test_unknown_config_and_bad_frames_rejected(self):
        train, test = self._frames()
        with self.assertRaises(ValueError):
            c.TabularPreprocessor("log_only")
        with self.assertRaises(ValueError):
            c.TabularPreprocessor(None)
        p = c.TabularPreprocessor("standard").fit(train)
        with self.assertRaises(ValueError):
            p.transform(test[["b", "a", "c"]])
        with self.assertRaises(ValueError):
            c.TabularPreprocessor("none").fit(np.zeros((3, 2)))
        duplicated = train.copy()
        duplicated.columns = ["a", "a", "c"]
        with self.assertRaises(ValueError):
            c.TabularPreprocessor("none").fit(duplicated)

    def test_module_import_does_not_load_sklearn(self):
        code = ("import sys, cer_rul; "
                "print('sklearn' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
