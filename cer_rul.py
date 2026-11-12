"""cer_rul — reusable RUL/CPFL data pipeline helpers.

Single-file, manually uploadable module extracted from
``RUL_CPFL_baselines_colab.ipynb`` / ``RUL_CPFL_timestral_colab.ipynb``.

It covers only the reusable, model-free part of those notebooks:

* reading and validating one CSV, consolidating duplicate
  ``(Falha, Date)`` keys, and producing the raw/clean audit;
* explicit leave-one-equipment-out splits and per-fold masks;
* causal endpoint selection and causal tabular features;
* regression metrics and macro aggregation across folds.

Everything else (input discovery/upload orchestration, all-label loops,
models, search, caches, manifests, exports, plots) stays in the notebooks.

Public API
----------
Reading / cleaning (group A)
    ``coerce_fault_ids``, ``load_raw_label``, ``consolidate_frame``

Splits / masks / phase (group B)
    ``make_splits``, ``masks_for_fold``, ``phase_for_date``

Endpoints and features (group C)
    ``build_endpoint_rows``, ``feature_columns``, ``daily_stats``,
    ``bin_stats``, ``build_feature_row``, ``build_feature_frame``

Metrics (group D)
    ``compute_weights``, ``regression_metrics``, ``metric_breakdown``,
    ``scaled_prediction_to_units``, ``fold_metric_rows``,
    ``aggregate_metric_rows``

Dataset profiles (declarative, fresh copy per call)
    ``get_dataset_profile`` for ``'gerador'`` and ``'gearbox'``

Preprocessing (learned in ``fit`` on training rows only)
    ``TabularPreprocessor``, ``signed_log1p``, ``PREPROCESSING_OPTIONS``

Protocol constants are module defaults only. Every function that depends on
protocol values also accepts them as explicit keyword arguments, so a
notebook can pass its own values without mutating any module state.

Colab usage
-----------
Upload this file and import it::

    from google.colab import files
    files.upload()          # select cer_rul.py
    import cer_rul

    frame = cer_rul.load_raw_label(
        "Features_gerador_score_RUL-porcentagem-data.csv",
        label="porcentagem-data",
        max_allowed=1.0,
    )
    clean, audit, terminal = cer_rul.consolidate_frame(
        frame, "porcentagem-data",
        label_meta=cer_rul.LABEL_META["porcentagem-data"],
        expected_clean_keys=1910,
        expected_target_conflicts=0,
        enforce_expectations=True,
    )
    splits = cer_rul.make_splits()
    endpoints = cer_rul.build_endpoint_rows(clean, enforce=True,
                                            expected_endpoint_count=574)
    features = cer_rul.build_feature_frame(clean, endpoints, 12)

The upload cell may need to be re-run after a Colab runtime restart.
"""

import copy
from types import MappingProxyType

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Protocol defaults (immutable; override per call with explicit arguments).
# ---------------------------------------------------------------------------

DEFAULT_WARMUP_RECORDS = 30
DEFAULT_ENDPOINT_STRIDE = 3
DEFAULT_CONTEXT_BINS = 32
DEFAULT_BIN_HOURS = 12
BIN_HOURS_CANDIDATES = (6, 12, 24)
EXPECTED_RAW_ROWS = 2710
EXPECTED_CLEAN_KEYS = 1910
EXPECTED_ENDPOINT_COUNT = 574

EQUIPMENT_GROUPS = MappingProxyType({"A": (1, 2), "B": (3,), "C": (4, 5), "D": (6, 7)})
EQUIPMENT_ORDER = tuple(EQUIPMENT_GROUPS)
ID_TO_EQUIPMENT = MappingProxyType({
    int(equipment_id): equipment_name
    for equipment_name, equipment_ids in EQUIPMENT_GROUPS.items()
    for equipment_id in equipment_ids
})
FAULT_IDS = tuple(sorted(ID_TO_EQUIPMENT))

SENSOR_COLUMNS = (
    "g_0_RMS",
    "g_1_Meanfreq",
    "g_0_Variance",
    "g_0_Peak_peak",
    "e_Gerador Dianteiro _BPFO_2X|6332-M",
    "e_Gerador Dianteiro _BPFI_2X_1SB|6332-M",
    "e_Gerador Dianteiro _BPFI_1X_1SB|6332-M",
    "e_Gerador Dianteiro _BPFI_2X|6332-M",
)
SENSOR_KEYS = tuple(f"s{index}" for index in range(len(SENSOR_COLUMNS)))
SENSOR_BY_KEY = MappingProxyType(dict(zip(SENSOR_KEYS, SENSOR_COLUMNS)))
CSV_COLUMNS = ("Date",) + SENSOR_COLUMNS + ("Falha", "RUL")

FPT_ANNOTATIONS = MappingProxyType({
    1: "2024-07-25", 2: "2024-07-25", 3: "2024-03-15",
    4: "2023-12-10", 5: "2023-12-10", 6: "2024-07-22", 7: "2024-07-22",
})

REPRESENTATIONS = ("current", "summary")
WEIGHT_SCHEMES = ("none", "component", "component_phase")
FORBIDDEN_INFERENCE_TOKENS = (
    "Falha", "Date", "RUL", "phase", "FPT", "failure", "falha", "equipment",
)

LABEL_META = MappingProxyType({
    "dias-90": {"kind": "days", "divisor": 100.0, "metric_factor": 1.0,
                "unit": "dias", "max_allowed": 100.0},
    "dias-ajustado": {"kind": "days", "divisor": 100.0, "metric_factor": 1.0,
                      "unit": "dias", "max_allowed": 100.0},
    "dias-FPT": {"kind": "days", "divisor": 100.0, "metric_factor": 1.0,
                 "unit": "dias", "max_allowed": 100.0},
    "porcentagem-data": {"kind": "fraction", "divisor": 1.0, "metric_factor": 100.0,
                         "unit": "pontos percentuais (pp)", "max_allowed": 1.0},
    "porcentagem-registro": {"kind": "fraction", "divisor": 1.0, "metric_factor": 100.0,
                             "unit": "pontos percentuais (pp)", "max_allowed": 1.0},
})

_DEFAULT_LABEL_META = {
    "kind": "synthetic", "divisor": 1.0, "metric_factor": 1.0,
    "unit": "unidade", "max_allowed": np.inf,
}

# Gearbox dataset (independent CSV; never merged with the generator CSV).
# Its FPT dates were inferred from linear labels and are not physically verified.
GEARBOX_SENSOR_COLUMNS = (
    "g_1_Meanfreq",
    "v_1_Centerfreq",
    "g_0_Peak_peak",
    "g_0_RMS",
    "g_0_Variance",
    "v_1_Rmsfreq",
    "e_HS-S-RS (GBX)_BSF_2X|NCF 2232 ECN1JB/L4BC3",
    "e_HS-S-GS (GBX) (Pista dupla)_BPFI_2X|BT2-0246 C/L4BPEX",
)
GEARBOX_EQUIPMENT_GROUPS = MappingProxyType({
    "E1": (1, 2, 3, 4, 5, 6),
    "E2": (7, 8, 9, 10, 11, 12),
    "E3": (13, 14, 15, 16, 17, 18),
})
GEARBOX_MEASUREMENT_NAMES = (
    "HSS-Radial", "HSS-Axial", "IMS", "LSS", "Planetária 0°", "Planetária 90°",
)
GEARBOX_FPT_ANNOTATIONS = MappingProxyType({
    **{fault_id: "2023-06-14" for fault_id in range(1, 7)},
    **{fault_id: "2022-11-01" for fault_id in range(7, 13)},
    **{fault_id: "2023-08-13" for fault_id in range(13, 19)},
})

_GERADOR_FPT_NOTE = (
    "Datas herdadas da célula 19 de RUL_CMS_TCN.ipynb, usadas apenas em "
    "relatório de fase e pesos opcionais de treino. Não foram verificadas "
    "fisicamente de forma independente."
)
_GEARBOX_FPT_NOTE = (
    "Datas de FPT do gearbox inferidas da rotulagem linear (RUL) dos eventos "
    "E1=2023-06-14, E2=2022-11-01 e E3=2023-08-13; não foram verificadas "
    "fisicamente de forma independente. Usadas apenas em relatório de fase e "
    "pesos opcionais de treino, nunca como features de inferência."
)


def _porcentagem_label(filename):
    return {"porcentagem-data": {"filename": filename,
                                 **dict(LABEL_META["porcentagem-data"])}}


_DATASET_PROFILES = {
    "gerador": {
        "dataset_id": "gerador",
        "unit_kind": "equipment",
        "sensor_columns": tuple(SENSOR_COLUMNS),
        "equipment_groups": {name: tuple(ids) for name, ids in EQUIPMENT_GROUPS.items()},
        "measurement_by_id": {fault_id: "gerador" for fault_id in FAULT_IDS},
        "fpt_annotations": {int(fault_id): date for fault_id, date
                            in FPT_ANNOTATIONS.items()},
        "fpt_note": _GERADOR_FPT_NOTE,
        "labels": _porcentagem_label("Features_gerador_score_RUL-porcentagem-data.csv"),
        "expected_raw_rows": EXPECTED_RAW_ROWS,
        "expected_clean_keys": EXPECTED_CLEAN_KEYS,
        "expected_target_conflicts": {"porcentagem-data": 0},
        "expected_endpoints": EXPECTED_ENDPOINT_COUNT,  # warmup 30, stride 3
    },
    "gearbox": {
        "dataset_id": "gearbox",
        "unit_kind": "event",
        "sensor_columns": GEARBOX_SENSOR_COLUMNS,
        "equipment_groups": {name: tuple(ids)
                             for name, ids in GEARBOX_EQUIPMENT_GROUPS.items()},
        "measurement_by_id": {
            fault_id: GEARBOX_MEASUREMENT_NAMES[(fault_id - 1) % len(GEARBOX_MEASUREMENT_NAMES)]
            for fault_id in range(1, 19)
        },
        "fpt_annotations": {int(fault_id): date for fault_id, date
                            in GEARBOX_FPT_ANNOTATIONS.items()},
        "fpt_note": _GEARBOX_FPT_NOTE,
        "labels": _porcentagem_label(
            "Features_gearbox_score_todos_RUL-porcentagem-data.csv"),
        "expected_raw_rows": 2946,
        "expected_clean_keys": 2946,
        "expected_target_conflicts": {"porcentagem-data": 0},
        "expected_endpoints": 822,  # warmup 30, stride 3
    },
}


def get_dataset_profile(name):
    """Return a fresh deep copy of the declarative profile for ``name``.

    Only ``'gerador'`` and ``'gearbox'`` are accepted. Mutating the returned
    dict never affects module state or later calls.
    """
    if not isinstance(name, str) or name not in _DATASET_PROFILES:
        raise ValueError(
            f"Unknown dataset profile: {name!r}. "
            f"Expected one of {sorted(_DATASET_PROFILES)}."
        )
    return copy.deepcopy(_DATASET_PROFILES[name])


__all__ = [
    "coerce_fault_ids", "load_raw_label", "consolidate_frame",
    "make_splits", "masks_for_fold", "phase_for_date",
    "build_endpoint_rows", "feature_columns", "daily_stats", "bin_stats",
    "build_feature_row", "build_feature_frame",
    "compute_weights", "regression_metrics", "metric_breakdown",
    "scaled_prediction_to_units", "fold_metric_rows", "aggregate_metric_rows",
    "get_dataset_profile",
    "PREPROCESSING_OPTIONS", "signed_log1p", "TabularPreprocessor",
]


# ---------------------------------------------------------------------------
# Group A — reading, validation, consolidation, audit.
# ---------------------------------------------------------------------------

def coerce_fault_ids(values):
    """Validate finite, exactly integral fault IDs; never truncate 1.9 to 1."""
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Falha contains non-finite values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Falha must contain exact integers; fractional values "
                         "cannot be truncated.")
    return numeric.astype(int)


def load_raw_label(
    path,
    label=None,
    *,
    expected_raw_rows=EXPECTED_RAW_ROWS,
    csv_columns=CSV_COLUMNS,
    sensor_columns=SENSOR_COLUMNS,
    fault_ids=FAULT_IDS,
    max_allowed=None,
    check_positive_sensors=True,
    print_summary=True,
):
    """Read and validate a single RUL CSV, preserving raw row order.

    ``label`` is only a display name. Pass ``expected_raw_rows=None`` to skip
    the row-count check and ``max_allowed=None`` to skip the upper RUL bound.
    """
    csv_columns = tuple(csv_columns)
    sensor_columns = tuple(sensor_columns)
    name = label or str(path)

    raw_text = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(raw_text.columns) != list(csv_columns):
        raise ValueError(
            f"{name}: unexpected column order/names.\n"
            f"expected={list(csv_columns)}\nreceived={list(raw_text.columns)}"
        )
    if expected_raw_rows is not None and len(raw_text) != int(expected_raw_rows):
        raise ValueError(f"{name}: {len(raw_text)} rows, expected {expected_raw_rows}.")

    frame = pd.DataFrame(index=raw_text.index)
    frame["Date"] = pd.to_datetime(raw_text["Date"], errors="raise")
    frame["Falha"] = coerce_fault_ids(raw_text["Falha"])
    frame["RUL"] = pd.to_numeric(raw_text["RUL"], errors="raise").astype(float)
    for column in sensor_columns:
        frame[column] = pd.to_numeric(raw_text[column], errors="raise").astype(float)
    frame = frame.loc[:, list(csv_columns)]

    if frame["Date"].isna().any():
        raise ValueError(f"{name}: Date contains NaT.")
    if set(frame["Falha"].unique()) != set(fault_ids):
        raise ValueError(
            f"{name}: unexpected Falha IDs: {sorted(frame['Falha'].unique())}"
        )
    if not np.isfinite(frame["RUL"].to_numpy(float)).all():
        raise ValueError(f"{name}: RUL is not finite.")
    if (frame["RUL"] < 0).any():
        raise ValueError(f"{name}: RUL below 0.")
    if max_allowed is not None and (frame["RUL"] > float(max_allowed) + 1e-12).any():
        raise ValueError(f"{name}: RUL outside [0, {max_allowed}].")
    sensor_matrix = frame.loc[:, list(sensor_columns)].to_numpy(float)
    if not np.isfinite(sensor_matrix).all():
        raise ValueError(f"{name}: sensors are not finite.")
    if check_positive_sensors and (sensor_matrix <= 0).any():
        raise ValueError(f"{name}: sensors must be strictly positive.")

    if print_summary:
        dmin, dmax = frame["Date"].min(), frame["Date"].max()
        print(f"{name}: {len(frame)} rows | {dmin} .. {dmax} | "
              f"RUL [{frame['RUL'].min():.6g}, {frame['RUL'].max():.6g}]")
    return frame


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "min": float(np.min(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
        "mean": float(np.mean(values)) if values.size else None,
    }


def consolidate_frame(
    frame,
    label="raw",
    *,
    sensor_columns=SENSOR_COLUMNS,
    csv_columns=CSV_COLUMNS,
    fault_ids=FAULT_IDS,
    id_to_equipment=ID_TO_EQUIPMENT,
    equipment_order=EQUIPMENT_ORDER,
    label_meta=None,
    expected_clean_keys=None,
    expected_target_conflicts=0,
    enforce_expectations=False,
):
    """Average ALL original RUL rows per ``(Falha, Date)``; never deduplicate targets.

    Sensor values must be identical inside a ``(Falha, Date)`` key; otherwise a
    ``ValueError`` is raised. Returns ``(clean, audit, terminal)`` where
    ``clean`` carries ``Date, Falha, RUL, raw_count`` followed by the sensors.
    """
    sensor_columns = tuple(sensor_columns)
    csv_columns = tuple(csv_columns)
    if list(frame.columns) != list(csv_columns):
        raise ValueError(f"{label}: unexpected columns for consolidation.")
    meta = dict(_DEFAULT_LABEL_META)
    if label_meta:
        meta.update(label_meta)

    duplicate_groups = frame.groupby(list(csv_columns), dropna=False).size()
    exact_duplicate_groups = int((duplicate_groups > 1).sum())
    exact_duplicate_surplus = int(np.maximum(duplicate_groups - 1, 0).sum())

    key_sizes = frame.groupby(["Falha", "Date"], sort=False).size()
    target_conflicts = 0
    feature_conflicts = 0
    conflict_examples = []
    rows = []

    for (fault_id, endpoint_date), group in frame.groupby(["Falha", "Date"], sort=False):
        matrix = group.loc[:, list(sensor_columns)].to_numpy(float)
        if matrix.shape[0] == 0:
            raise ValueError(f"{label}: empty group at {(fault_id, endpoint_date)}.")
        if not np.all(matrix == matrix[0]):
            feature_conflicts += 1
            conflict_examples.append(
                {"Falha": int(fault_id), "Date": pd.Timestamp(endpoint_date).isoformat(),
                 "kind": "feature_conflict"}
            )
            continue
        original_targets = group["RUL"].to_numpy(float)
        if np.unique(original_targets).size > 1:
            target_conflicts += 1
            if len(conflict_examples) < 20:
                conflict_examples.append(
                    {"Falha": int(fault_id),
                     "Date": pd.Timestamp(endpoint_date).isoformat(),
                     "kind": "target_conflict",
                     "values": [float(value) for value in original_targets]}
                )
        rows.append({
            "Date": pd.Timestamp(endpoint_date),
            "Falha": int(fault_id),
            "RUL": float(np.mean(original_targets)),  # mean of ALL original rows
            "raw_count": int(len(group)),
            **{key: float(matrix[0, index])
               for index, key in enumerate(sensor_columns)},
        })

    if feature_conflicts:
        raise ValueError(
            f"{label}: {feature_conflicts} key(s) with divergent sensors. "
            "The protocol requires identical sensors inside (Falha, Date)."
        )

    clean = pd.DataFrame(rows)
    clean = clean.sort_values(["Falha", "Date"], kind="mergesort").reset_index(drop=True)
    clean = clean.loc[:, ["Date", "Falha", "RUL", "raw_count"] + list(sensor_columns)]

    if enforce_expectations:
        if expected_clean_keys is not None and len(clean) != int(expected_clean_keys):
            raise ValueError(f"{label}: {len(clean)} clean keys, expected "
                             f"{expected_clean_keys}.")
        if (expected_target_conflicts is not None
                and target_conflicts != int(expected_target_conflicts)):
            raise ValueError(f"{label}: {target_conflicts} target conflicts, "
                             f"expected {expected_target_conflicts}.")
        if set(clean["Falha"].unique()) != set(fault_ids):
            raise ValueError(f"{label}: fault IDs lost during consolidation.")

    terminal = clean.groupby("Falha", sort=True).tail(1).copy()
    terminal["target_nonzero"] = terminal["RUL"] > 0
    equipment_of = lambda fault_id: id_to_equipment[int(fault_id)]
    audit = {
        "label": label,
        "raw_rows": int(len(frame)),
        "raw_unique_keys": int(len(key_sizes)),
        "raw_duplicate_keys": int((key_sizes > 1).sum()),
        "raw_max_multiplicity": int(key_sizes.max()),
        "raw_multiplicity_histogram": {
            str(int(size)): int(count)
            for size, count in key_sizes.value_counts().sort_index().items()
        },
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_surplus_rows": exact_duplicate_surplus,
        "target_conflict_keys": int(target_conflicts),
        "feature_conflict_keys": int(feature_conflicts),
        "clean_keys": int(len(clean)),
        "clean_rows_per_fault": {
            str(int(fid)): int(count)
            for fid, count in clean.groupby("Falha").size().items()
        },
        "clean_rows_per_equipment": {
            eq: int((clean["Falha"].map(equipment_of) == eq).sum())
            for eq in equipment_order
        },
        "raw_target_stats": _stats(frame["RUL"]),
        "clean_target_stats": _stats(clean["RUL"]),
        "date_min": clean["Date"].min().isoformat(),
        "date_max": clean["Date"].max().isoformat(),
        "terminal_nonzero_count": int(terminal["target_nonzero"].sum()),
        "terminal_targets": {
            str(int(row.Falha)): float(row.RUL)
            for row in terminal.itertuples(index=False)
        },
        "conflict_examples": conflict_examples,
        "divisor": float(meta["divisor"]),
        "metric_factor": float(meta["metric_factor"]),
        "unit": meta["unit"],
    }
    return clean, audit, terminal


# ---------------------------------------------------------------------------
# Group B — explicit splits, masks and phase.
# ---------------------------------------------------------------------------

def make_splits(*, equipment_order=EQUIPMENT_ORDER, equipment_groups=EQUIPMENT_GROUPS):
    """Four leave-one-equipment-out folds with ONE inner validation equipment."""
    folds = []
    equipment_names = list(equipment_order)
    for fold_index, test_equipment in enumerate(equipment_names):
        other_equipment = [name for name in equipment_names if name != test_equipment]
        val_equipment = other_equipment[fold_index % len(other_equipment)]
        inner_train_equipment = [name for name in other_equipment
                                 if name != val_equipment]
        outer_train_ids = tuple(
            sorted(equipment_id
                   for name in other_equipment
                   for equipment_id in equipment_groups[name])
        )
        test_ids = tuple(equipment_groups[test_equipment])
        val_ids = tuple(equipment_groups[val_equipment])
        inner_train_ids = tuple(
            sorted(equipment_id
                   for name in inner_train_equipment
                   for equipment_id in equipment_groups[name])
        )
        folds.append({
            "fold_id": f"fold_{test_equipment}",
            "fold_index": fold_index,
            "test_equipment": test_equipment,
            "val_equipment": val_equipment,
            "inner_train_equipment": inner_train_equipment,
            "test_ids": test_ids,
            "val_ids": val_ids,
            "inner_train_ids": inner_train_ids,
            "outer_train_ids": outer_train_ids,
        })
    result = pd.DataFrame(folds)
    for row in result.to_dict(orient="records"):
        test = set(row["test_ids"])
        val = set(row["val_ids"])
        inner = set(row["inner_train_ids"])
        outer = set(row["outer_train_ids"])
        if not (test.isdisjoint(val | inner) and val.isdisjoint(inner)
                and val | inner == outer and outer.isdisjoint(test)):
            raise ValueError(f"Invalid split isolation in {row['fold_id']}.")
    return result


def masks_for_fold(target, fold, *, id_to_equipment=ID_TO_EQUIPMENT,
                   equipment_order=EQUIPMENT_ORDER):
    """Boolean row masks for inner_train/val/outer_train/test of one fold."""
    equipment = target["Falha"].map(id_to_equipment)
    masks = {
        "inner_train": equipment.isin(fold["inner_train_equipment"]).to_numpy(),
        "val": equipment.isin([fold["val_equipment"]]).to_numpy(),
        "outer_train": equipment.isin(
            [name for name in equipment_order if name != fold["test_equipment"]]
        ).to_numpy(),
        "test": equipment.isin([fold["test_equipment"]]).to_numpy(),
    }
    if (masks["inner_train"] & masks["val"]).any():
        raise ValueError("inner_train and val overlap.")
    if masks["outer_train"].sum() != masks["inner_train"].sum() + masks["val"].sum():
        raise ValueError("outer_train is not inner_train + val.")
    if (masks["test"] & masks["outer_train"]).any():
        raise ValueError("test and outer_train overlap.")
    if masks["test"].sum() == 0 or masks["val"].sum() == 0:
        raise ValueError("Empty test or val mask.")
    return masks


def phase_for_date(date_value, fault_id, *, fpt_annotations=FPT_ANNOTATIONS):
    """Return ``saudavel`` before the FPT annotation and ``pos_FPT`` from it on."""
    fpt = pd.Timestamp(fpt_annotations[int(fault_id)])
    return "saudavel" if pd.Timestamp(date_value) < fpt else "pos_FPT"


# ---------------------------------------------------------------------------
# Group C — endpoints and causal tabular features.
# ---------------------------------------------------------------------------

def build_endpoint_rows(
    clean_reference,
    *,
    warmup=DEFAULT_WARMUP_RECORDS,
    stride=DEFAULT_ENDPOINT_STRIDE,
    id_to_equipment=ID_TO_EQUIPMENT,
    fpt_annotations=FPT_ANNOTATIONS,
    expected_endpoint_count=None,
    enforce=False,
):
    """Build post-warmup endpoint rows, always including the final record."""
    warmup = int(warmup)
    stride = int(stride)
    rows = []
    for fault_id, group in clean_reference.groupby("Falha", sort=True):
        group = group.sort_values("Date", kind="mergesort").reset_index(drop=True)
        n = len(group)
        if n <= warmup:
            raise ValueError(f"Falha {fault_id}: not enough records for warmup.")
        indices = list(range(warmup, n, stride))
        if not indices or indices[-1] != n - 1:
            indices.append(n - 1)
        for ordinal, record_index in enumerate(indices):
            date_value = pd.Timestamp(group.loc[record_index, "Date"])
            rows.append({
                "endpoint_id": f"F{fault_id:02d}-r{record_index:04d}",
                "endpoint_ordinal": int(ordinal),
                "Falha": int(fault_id),
                "equipment": id_to_equipment[int(fault_id)],
                "Date": date_value,
                "record_index": int(record_index),
                "n_records_to_t": int(record_index + 1),
                "n_records_after": int(n - record_index - 1),
                "phase": phase_for_date(date_value, fault_id,
                                        fpt_annotations=fpt_annotations),
                "post_warmup": True,
            })
    result = pd.DataFrame(rows).sort_values(
        ["Falha", "Date", "record_index"], kind="mergesort"
    ).reset_index(drop=True)
    if result["endpoint_id"].duplicated().any():
        raise ValueError("duplicated endpoint_id.")
    if result.duplicated(["Falha", "Date"]).any():
        raise ValueError("duplicated endpoint in (Falha, Date).")
    for fault_id, group in result.groupby("Falha", sort=True):
        if group["record_index"].min() < warmup:
            raise ValueError("endpoint inside warmup.")
        if group["n_records_after"].min() != 0:
            raise ValueError(f"Falha {fault_id}: missing final endpoint.")
    if enforce and expected_endpoint_count is not None:
        if len(result) != int(expected_endpoint_count):
            raise ValueError(f"{len(result)} endpoints, expected "
                             f"{expected_endpoint_count}.")
    return result


def feature_columns(
    width_hours,
    representation="summary",
    timing=True,
    *,
    sensor_keys=SENSOR_KEYS,
    representations=REPRESENTATIONS,
    bin_hours_candidates=BIN_HOURS_CANDIDATES,
    forbidden_tokens=FORBIDDEN_INFERENCE_TOKENS,
):
    """Ordered feature-column names for one representation/width."""
    if representation not in representations:
        raise ValueError(f"Unknown representation: {representation}")
    if int(width_hours) not in bin_hours_candidates:
        raise ValueError(f"Unapproved grid width: {width_hours}")
    columns = []
    for sensor_key in sensor_keys:
        columns.extend([
            f"cur__{sensor_key}",
            f"nobs__{sensor_key}",
            f"missbins__{sensor_key}",
        ])
    columns.extend([
        "window__records",
        "window__observed_bins",
        "window__empty_bins",
    ])
    if representation == "summary":
        for sensor_key in sensor_keys:
            columns.extend([
                f"dmean__{sensor_key}",
                f"dstd__{sensor_key}",
                f"dslope__{sensor_key}",
            ])
    if timing:
        columns.extend([
            "timing__window_days",
            "timing__observed_span_days",
            "timing__prior_gap_h",
            "timing__median_gap_h",
        ])
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicated feature columns.")
    for column in columns:
        if any(token in column for token in forbidden_tokens):
            raise ValueError(f"Forbidden feature: {column}")
    return columns


def daily_stats(times, values):
    """Mean/std of observations and slope per real elapsed day."""
    times = pd.to_datetime(pd.Series(times))
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Statistics received invalid values.")
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=0))
    days = (times - times.iloc[0]).dt.total_seconds().to_numpy(float) / 86400.0
    centered_days = days - days.mean()
    denominator = float(np.dot(centered_days, centered_days))
    if len(values) < 2 or denominator <= 0:
        slope = float("nan")
    else:
        slope = float(np.dot(centered_days, values - values.mean()) / denominator)
    return mean_value, std_value, slope


def bin_stats(context_dates, context_values, endpoint_date, width_hours, *,
              context_bins=DEFAULT_CONTEXT_BINS):
    """Regular ``(left, right]`` bins anchored at ``t``; last value/time per bin."""
    n_bins = int(context_bins)
    width = pd.Timedelta(hours=float(width_hours))
    last_values = np.full((n_bins, context_values.shape[1]), np.nan, dtype=float)
    last_times = pd.Series(pd.NaT, index=range(n_bins), dtype="datetime64[ns]")
    counts = np.zeros(n_bins, dtype=int)
    for bin_index in range(n_bins):
        right = pd.Timestamp(endpoint_date) - width * (n_bins - 1 - bin_index)
        left = right - width
        mask = (context_dates > left) & (context_dates <= right)
        mask_array = mask.to_numpy()
        count = int(mask_array.sum())
        counts[bin_index] = count
        if count:
            last_row = int(np.flatnonzero(mask_array)[-1])
            last_values[bin_index, :] = context_values[last_row, :]
            last_times.iloc[bin_index] = pd.Timestamp(context_dates.iloc[last_row])
    return last_values, last_times, counts


def build_feature_row(endpoint_row, clean_group, width_hours, *,
                      sensor_columns=SENSOR_COLUMNS,
                      sensor_keys=SENSOR_KEYS,
                      context_bins=DEFAULT_CONTEXT_BINS):
    """Build one causal feature row using only records up to the endpoint."""
    sensor_columns = tuple(sensor_columns)
    sensor_keys = tuple(sensor_keys)
    group = clean_group.sort_values("Date", kind="mergesort").reset_index(drop=True)
    record_index = int(endpoint_row.record_index)
    context = group.iloc[:record_index + 1]
    if len(context) != record_index + 1:
        raise ValueError("context truncated unexpectedly.")
    endpoint_date = pd.Timestamp(endpoint_row.Date)
    if pd.Timestamp(context["Date"].iloc[-1]) != endpoint_date:
        raise ValueError("endpoint is not the last record of the context.")
    if (context["Date"] > endpoint_date).any():
        raise ValueError("context contains a future record.")

    dates = context["Date"]
    values = context.loc[:, list(sensor_columns)].to_numpy(float)
    last_values, last_times, counts = bin_stats(
        dates, values, endpoint_date, width_hours, context_bins=context_bins)
    observed_bins = counts > 0
    observed_count = int(observed_bins.sum())
    window_hours = float(width_hours) * int(context_bins)
    window_start = endpoint_date - pd.Timedelta(hours=window_hours)
    window_mask = (dates > window_start) & (dates <= endpoint_date)
    window_dates = dates[window_mask]
    window_gaps = window_dates.diff().dt.total_seconds().dropna() / 3600.0

    observed_times = pd.Series(last_times[observed_bins]).reset_index(drop=True)
    if len(observed_times) >= 2:
        observed_span_days = float(
            (observed_times.iloc[-1] - observed_times.iloc[0]).total_seconds()
            / 86400.0
        )
    else:
        observed_span_days = 0.0

    # The prior gap may use the record immediately before t even if it lies
    # outside the grid; it is cadence information only, never a value/statistic.
    prior_gap_h = float("nan")
    if len(dates) >= 2:
        prior_gap_h = float(
            (dates.iloc[-1] - dates.iloc[-2]).total_seconds() / 3600.0
        )
    median_gap_h = float(window_gaps.median()) if len(window_gaps) else float("nan")

    row = {}
    for sensor_position, sensor_key in enumerate(sensor_keys):
        sensor_bins = last_values[observed_bins, sensor_position]
        finite = np.isfinite(sensor_bins)
        if finite.any():
            row[f"cur__{sensor_key}"] = float(sensor_bins[finite][-1])
        else:
            row[f"cur__{sensor_key}"] = float("nan")
        row[f"nobs__{sensor_key}"] = int(counts.sum())
        row[f"missbins__{sensor_key}"] = float((~observed_bins).mean())
    row["window__records"] = int(counts.sum())
    row["window__observed_bins"] = int(observed_count)
    row["window__empty_bins"] = int(int(context_bins) - observed_count)

    # The caller always builds both views from the same row.
    for sensor_position, sensor_key in enumerate(sensor_keys):
        sensor_bins = last_values[observed_bins, sensor_position]
        sensor_times = pd.Series(last_times[observed_bins]).reset_index(drop=True)
        finite = np.isfinite(sensor_bins)
        if finite.any():
            mean_value, std_value, slope = daily_stats(
                sensor_times[finite], sensor_bins[finite])
        else:
            mean_value, std_value, slope = float("nan"), float("nan"), float("nan")
        row[f"dmean__{sensor_key}"] = mean_value
        row[f"dstd__{sensor_key}"] = std_value
        row[f"dslope__{sensor_key}"] = slope
    row["timing__window_days"] = float(window_hours / 24.0)
    row["timing__observed_span_days"] = observed_span_days
    row["timing__prior_gap_h"] = prior_gap_h
    row["timing__median_gap_h"] = median_gap_h
    return row


def build_feature_frame(clean_reference, endpoints, width_hours, *,
                        sensor_columns=SENSOR_COLUMNS,
                        sensor_keys=SENSOR_KEYS,
                        context_bins=DEFAULT_CONTEXT_BINS,
                        bin_hours_candidates=BIN_HOURS_CANDIDATES,
                        representations=REPRESENTATIONS,
                        forbidden_tokens=FORBIDDEN_INFERENCE_TOKENS):
    """Build the summary+timing causal feature matrix indexed by endpoint_id."""
    groups = {
        int(fault_id): group.sort_values("Date", kind="mergesort").reset_index(drop=True)
        for fault_id, group in clean_reference.groupby("Falha", sort=True)
    }
    rows = []
    endpoint_ids = []
    for endpoint_row in endpoints.itertuples(index=False):
        group = groups[int(endpoint_row.Falha)]
        rows.append(build_feature_row(
            endpoint_row, group, width_hours,
            sensor_columns=sensor_columns, sensor_keys=sensor_keys,
            context_bins=context_bins,
        ))
        endpoint_ids.append(endpoint_row.endpoint_id)
    result = pd.DataFrame(rows, index=pd.Index(endpoint_ids, name="endpoint_id"))
    expected_columns = feature_columns(
        width_hours, "summary", True, sensor_keys=sensor_keys,
        bin_hours_candidates=bin_hours_candidates,
        representations=representations,
        forbidden_tokens=forbidden_tokens)
    if result.columns.tolist() != expected_columns:
        raise ValueError("feature order/shape changed.")
    current_columns = [c for c in result.columns if c.startswith("cur__")]
    missing_current = int(result.loc[:, current_columns].isna().sum().sum())
    if missing_current:
        print(f"  missing current values in window: {missing_current} "
              "(kept as NaN; the training imputer handles them).")
    if any(column.startswith("bin") for column in result.columns):
        raise ValueError("raw bin values cannot enter the tabular matrix.")
    return result


# ---------------------------------------------------------------------------
# Group D — weights, metrics and macro aggregation.
# ---------------------------------------------------------------------------

def compute_weights(ids, phases, scheme, *, weight_schemes=WEIGHT_SCHEMES):
    """Training weights normalized to mean one for the requested scheme."""
    ids = np.asarray(ids, dtype=int)
    phases = np.asarray(phases, dtype=str)
    n = len(ids)
    if n == 0 or len(phases) != n:
        raise ValueError("weights: empty or misaligned vectors.")
    if scheme not in weight_schemes:
        raise ValueError(f"unknown weight scheme: {scheme}")
    if scheme == "none":
        weights = np.ones(n, dtype=float)
    elif scheme == "component":
        counts = {int(key): int(value) for key, value in
                  zip(*np.unique(ids, return_counts=True))}
        weights = np.asarray([1.0 / counts[int(fault_id)] for fault_id in ids])
        weights = weights / weights.mean()
    else:
        weights = np.zeros(n, dtype=float)
        components = sorted(set(int(value) for value in ids))
        share_component = 1.0 / len(components)
        for component in components:
            mask_component = ids == component
            component_phases = sorted(set(phases[mask_component].tolist()))
            share_phase = share_component / len(component_phases)
            for phase in component_phases:
                mask_cell = mask_component & (phases == phase)
                weights[mask_cell] = share_phase / int(mask_cell.sum())
        weights = weights / weights.mean()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("invalid weights.")
    return weights


def regression_metrics(y_true, y_pred):
    """n, MAE, RMSE and bias (prediction - truth). No clamp, no MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("metrics: empty or misaligned vectors.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("metrics: non-finite values.")
    residual = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),  # pred - truth
    }


def metric_breakdown(y_true, y_pred, ids, groups, phases, unit):
    """Overall plus per-component, per-equipment and per-phase metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ids = np.asarray(ids, dtype=int)
    groups = np.asarray(groups, dtype=str)
    phases = np.asarray(phases, dtype=str)
    result = {
        "unit": unit,
        "overall": regression_metrics(y_true, y_pred),
        "by_component": {},
        "by_equipment": {},
        "by_phase": {},
        "clamp": False,
        "mape": False,
    }
    for fault_id in sorted(set(int(value) for value in ids)):
        mask = ids == fault_id
        result["by_component"][str(fault_id)] = regression_metrics(
            y_true[mask], y_pred[mask])
    for equipment in sorted(set(groups.tolist())):
        mask = groups == equipment
        result["by_equipment"][equipment] = regression_metrics(
            y_true[mask], y_pred[mask])
    for phase in sorted(set(phases.tolist())):
        mask = phases == phase
        result["by_phase"][phase] = regression_metrics(
            y_true[mask], y_pred[mask])
    return result


def scaled_prediction_to_units(prediction_scaled, label, *, label_meta=LABEL_META):
    """Convert a scaled prediction to native units and to metric units."""
    meta = label_meta[label]
    native = np.asarray(prediction_scaled, dtype=float) * float(meta["divisor"])
    metric = native * float(meta["metric_factor"])
    return native, metric


def fold_metric_rows(prediction_frame, *, label_meta=LABEL_META):
    """One metric row per label/family/fold/scope/name from predictions."""
    rows = []
    for (label, family), label_frame in prediction_frame.groupby(
            ["label", "family"], sort=False):
        meta = label_meta[label]
        for fold_id, fold_frame in label_frame.groupby("fold_id", sort=False):
            block = metric_breakdown(
                fold_frame["y_metric"].to_numpy(float),
                fold_frame["pred_metric"].to_numpy(float),
                fold_frame["Falha"].to_numpy(int),
                fold_frame["equipment"].to_numpy(str),
                fold_frame["phase"].to_numpy(str),
                meta["unit"],
            )
            for scope, values in (("overall", {"all": block["overall"]}),
                                  ("component", block["by_component"]),
                                  ("equipment", block["by_equipment"]),
                                  ("phase", block["by_phase"])):
                for name, metrics in values.items():
                    rows.append({
                        "label": label,
                        "family": family,
                        "unit": meta["unit"],
                        "fold_id": fold_id,
                        "scope": scope,
                        "name": name,
                        **metrics,
                        "aggregation": "fold_metric",
                    })
    return rows


def aggregate_metric_rows(rows):
    """Macro average across folds per family; never recombines fold samples."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    aggregates = []
    for keys, group in frame.groupby(["label", "family", "unit", "scope", "name"],
                                     sort=False):
        label, family, unit, scope, name = keys
        aggregates.append({
            "label": label,
            "family": family,
            "unit": unit,
            "scope": scope,
            "name": name,
            "fold_count": int(group["fold_id"].nunique()),
            "mae": float(group["mae"].mean()),
            "rmse": float(group["rmse"].mean()),
            "bias": float(group["bias"].mean()),
            "n_observations_total": int(group["n"].sum()),
            "aggregation": "media_simples_entre_folds_por_familia",
        })
    result = pd.DataFrame(aggregates)
    # The "all" row under scope=overall is the equal-weight macro across the
    # 4 equipment folds, separately per family.
    overall = result[(result["scope"] == "overall") & (result["name"] == "all")].copy()
    overall["scope"] = "equipment_macro"
    overall["aggregation"] = "media_igual_entre_4_equipamentos_por_familia"
    return pd.concat([result, overall], ignore_index=True)


# ---------------------------------------------------------------------------
# Group E — train-only tabular preprocessing (imputation, indicators, scaling).
# ---------------------------------------------------------------------------

PREPROCESSING_OPTIONS = ("none", "standard", "robust", "minmax",
                         "signed_log1p_standard")
_SCALED_PREPROCESSING = ("standard", "robust", "minmax", "signed_log1p_standard")


def signed_log1p(values):
    """Signed log1p: ``sign(x) * log1p(|x|)``; odd and defined for negatives."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


class TabularPreprocessor:
    """Median imputation, optional missing indicators and optional scaling.

    Median fill values, constant-fallback columns and the scaler are learned
    only in ``fit``; ``transform`` never refits. Columns that are entirely
    non-finite in training fall back to 0.0. For ``signed_log1p_standard`` the
    signed log is applied BEFORE the scaler. With ``add_missing_indicators``
    False, no ``na__`` columns are emitted.
    """

    def __init__(self, method, add_missing_indicators=True):
        if method not in PREPROCESSING_OPTIONS:
            raise ValueError(f"Pré-processamento inválido: {method}")
        self.method = method
        self.add_missing_indicators = bool(add_missing_indicators)

    def fit(self, frame):
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
            raise ValueError("Pré-processador exige DataFrame com colunas.")
        if frame.columns.duplicated().any():
            raise ValueError("Colunas duplicadas no pré-processador.")
        self.columns_ = list(frame.columns)
        self.indicator_columns_ = (
            [f"na__{column}" for column in self.columns_]
            if self.add_missing_indicators else []
        )
        self.fill_values_ = {}
        self.constant_fallback_columns_ = []
        for column in self.columns_:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if finite.size:
                fill = float(np.median(finite))
            else:
                fill = 0.0
                self.constant_fallback_columns_.append(column)
            self.fill_values_[column] = fill
        base = self._base_array(frame)
        base = self._transform_values(base)
        if self.method in _SCALED_PREPROCESSING:
            from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

            if self.method in ("standard", "signed_log1p_standard"):
                self.scaler_ = StandardScaler()
            elif self.method == "robust":
                self.scaler_ = RobustScaler()
            else:
                self.scaler_ = MinMaxScaler()
            self.scaler_.fit(base)
        else:
            self.scaler_ = None
        self.n_features_in_ = len(self.columns_)
        self.n_features_out_ = len(self.columns_) + len(self.indicator_columns_)
        return self

    def _base_array(self, frame):
        if list(frame.columns) != self.columns_:
            raise ValueError("Colunas diferentes do ajuste do pré-processador.")
        columns = []
        for column in self.columns_:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            if column in self.constant_fallback_columns_:
                values = np.full(values.shape, self.fill_values_[column], dtype=float)
            else:
                values = np.where(np.isfinite(values), values,
                                  self.fill_values_[column])
            columns.append(values)
        return np.column_stack(columns).astype(float)

    def _transform_values(self, base):
        if self.method == "signed_log1p_standard":
            return signed_log1p(base)
        return base

    def transform(self, frame):
        base = self._base_array(frame)
        base = self._transform_values(base)
        if self.scaler_ is not None:
            base = self.scaler_.transform(base)
        blocks = [base]
        if self.add_missing_indicators:
            blocks.append(np.column_stack([
                (~np.isfinite(pd.to_numeric(frame[column], errors="coerce")
                              .to_numpy(float))).astype(float)
                for column in self.columns_
            ]))
        result = np.column_stack(blocks)
        if not np.isfinite(result).all():
            raise ValueError("Pré-processador produziu valores não finitos.")
        return result

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

    def get_feature_names_out(self):
        return np.asarray(self.columns_ + self.indicator_columns_, dtype=object)
