"""Small unittest suite for cer_rul (synthetic contract + one real-data check)."""

import math
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


if __name__ == "__main__":
    unittest.main()
