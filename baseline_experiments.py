"""Local CPU baseline exploration for the RUL datasets (gerador, gearbox).

Standalone runner that reuses ``cer_rul`` for profiles, loading, consolidation,
endpoints, the existing causal features and metrics. It never reads or executes
notebooks. Behaviour is declared in ``configs/baseline_experiments.json``.

Protocol (per outer group, computed independently):
* outer leave-one-group-out; the search only receives outer-training rows;
* inner leave-one-group-out over all remaining groups;
* score = per inner group, mean MAE over components; mean over groups with equal
  weight; reported in percentage points (pp);
* tie: candidates within ``tie_tolerance_pp`` of the best prefer lower complexity,
  then lower inner RMSE, then a stable candidate ID;
* every choice is made inside the inner validation only and frozen to
  ``selection.json`` before ``evaluate`` reads the outer targets.

Implemented: config validation, dataset loading, feature representations,
train-only preprocessing, model factory, postprocessing, inner scoring, seeds,
fit cache, RAM-aware parallel fit runner with timeouts, Optuna-driven family
search with stop rules and uniform ensembles, declared legacy controls,
one-axis ablations with conditional retuning, fixed-offset seed confirmation,
bounded gearbox strategy screening, the joint event track with aggregation,
outer evaluation, verification, Markdown report and CLI.

Availability tracks are kept separate: the per-series track (one row per
endpoint) and the gearbox event track (one target per aligned event timestamp).
They are never ranked against each other, and no outer metric drives selection.
"""

import os

# Thread limits must be set before numpy/sklearn import so each fit is single
# threaded and parallelism is controlled only by the fit runner.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections import OrderedDict, deque  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,  # noqa: E402
                              RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import cer_rul as cer  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "baseline_experiments.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "baselines"
DATA_DIR = ROOT / "data"
DATASET_NAMES = ("gerador", "gearbox")
PILOT_DATASETS = ("gerador",)  # this task runs the pilot on the generator only
PP = 100.0  # targets are fractions; metrics are reported in percentage points
GIB = 1024 ** 3
MIB = 1024 ** 2
FAMILY_KINDS = {
    "mean": "constant", "median": "constant", "ridge": "model", "elasticnet": "model",
    "huber": "model", "svr": "model", "knn": "model", "extra_trees": "model",
    "random_forest": "model", "hist_gb": "model", "catboost": "model",
}
CONSTANT_FAMILIES = frozenset(f for f, kind in FAMILY_KINDS.items() if kind == "constant")
WEIGHT_CAPABLE = frozenset({"ridge", "elasticnet", "huber", "svr", "extra_trees",
                            "random_forest", "hist_gb", "catboost"})
REPRESENTATION_PREFIXES = {
    "current": ("cur__",),
    "summary": ("cur__", "dmean__", "dstd__", "dslope__"),
    "robust": ("cur__", "rob_"),
}
ACQUISITION_PREFIXES = ("nobs__", "missbins__", "window__", "timing__")
ELAPSED_COLUMN = "time__elapsed_days"
MEASUREMENT_PREFIX = "meas__"
REL_TEMPLATE = "rel{k}__"
JOINT_PREFIXES = tuple(f"m{index}__" for index in range(1, 7))


def variance_sensor_key(sensor_columns):
    """Sensor key of ``g_0_Variance`` in this dataset's ordered sensor columns.

    The generator stores it as ``s2`` and the gearbox as ``s4``; deriving it from the
    selected profile keeps ``drop_variance`` meaningful for both datasets.
    """
    columns = list(sensor_columns)
    if "g_0_Variance" not in columns:
        raise ConfigError("dataset sensor columns do not contain g_0_Variance")
    return f"s{columns.index('g_0_Variance')}"
ALLOWED_EMA_ALPHAS = (1.0, 0.5, 0.25)


class ConfigError(Exception):
    """Invalid configuration; always fatal and never caught as a model failure."""


class InvalidRecipe(ValueError):
    """A recipe combination that the protocol does not allow."""


class CorruptCacheError(RuntimeError):
    """A cached fit payload is incompatible or damaged; never silently reused."""


class BudgetExceeded(RuntimeError):
    """The per-dataset fit budget would be exceeded."""


class ResourceError(RuntimeError):
    """Memory headroom did not recover within the configured pause limit."""


class WorkerFatalError(RuntimeError):
    """A fit worker hit a non-model error (programming, config or data-integrity bug)."""


try:
    from catboost import CatBoostError as _CatBoostError
except ImportError:  # catboost is optional at import time; its absence is reported elsewhere
    class _CatBoostError(Exception):
        pass

# Errors that mean "this estimator/recipe failed on this data" (recorded as a failed
# candidate with a reason). Anything else inside a worker is fatal.
MODEL_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, _CatBoostError)


# ---------------------------------------------------------------------------
# Utilities: canonical JSON, hashing, deterministic seeds, atomic writes.
# ---------------------------------------------------------------------------

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def canonical_json(obj):
    """Deterministic JSON text used for every hash and key in this module."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_json_default, allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(namespace, *parts):
    """Deterministic 31-bit seed from a namespace and identifying parts."""
    text = "|".join([str(namespace)] + [str(part) for part in parts])
    return int(sha256_text(text)[:8], 16) % (2 ** 31 - 1)


def write_json_atomic(path, obj):
    """Write JSON to a temporary file and ``os.replace`` it (all or nothing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default,
                  allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def code_sha256():
    """Hash of the runner and the shared module; part of every fingerprint."""
    return sha256_text(sha256_file(Path(__file__).resolve())
                       + sha256_file(ROOT / "cer_rul.py"))


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

_SPACE_TYPES = ("float", "int", "categorical", "const")


def load_config(path=None):
    """Load and validate the declarative JSON configuration."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def _validate_space(family, space):
    for name, spec in space.items():
        kind = spec.get("type")
        if kind not in _SPACE_TYPES:
            raise ConfigError(f"{family}.{name}: unknown type {kind!r}")
        if kind in ("float", "int"):
            if not spec["low"] < spec["high"]:
                raise ConfigError(f"{family}.{name}: low must be below high")
            if spec.get("log") and spec["low"] <= 0:
                raise ConfigError(f"{family}.{name}: log scale needs low > 0")
        if kind == "categorical" and not spec.get("choices"):
            raise ConfigError(f"{family}.{name}: empty choices")
        if kind == "const" and "value" not in spec:
            raise ConfigError(f"{family}.{name}: const without value")


def validate_config(config):
    """Fail loudly on any structural problem in the configuration."""
    required = ("protocol", "datasets", "feature_group_sets", "default_recipe",
                "recipe_space", "families", "stages", "budgets", "resources",
                "seed_namespace", "pilot")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"missing configuration sections: {missing}")
    for family, entry in config["families"].items():
        if family not in FAMILY_KINDS:
            raise ConfigError(f"unknown family in config: {family}")
        if entry.get("kind") != FAMILY_KINDS[family]:
            raise ConfigError(f"family {family}: kind must be {FAMILY_KINDS[family]}")
        _validate_space(family, entry.get("space", {}))
    for key, spec in config["recipe_space"].items():
        if spec.get("type") != "categorical" or not spec.get("choices"):
            raise ConfigError(f"recipe_space.{key} must be a non-empty categorical")
        if "datasets" in spec and "default" not in spec:
            raise ConfigError(f"recipe_space.{key}: dataset-limited key needs a default")
    choices = config["recipe_space"].get("features.feature_groups", {}).get("choices", [])
    for choice in choices:
        if choice not in config["feature_group_sets"]:
            raise ConfigError(f"feature group set {choice!r} is not defined")
    for name in config["datasets"]:
        if name not in DATASET_NAMES:
            raise ConfigError(f"unknown dataset in config: {name}")
    stages = config["stages"]
    for name in ("screen_trials", "tune_top_k", "tune_extra_trials", "tune_min_total_trials",
                 "tune_patience_trials", "batch_size"):
        if int(stages[name]) < 0:
            raise ConfigError(f"stages.{name} must be non-negative")
    if int(stages["batch_size"]) < 1:
        raise ConfigError("stages.batch_size must be at least 1")
    budgets = config["budgets"]
    if budgets["fit_timeout_s"] <= 0 or budgets["max_fits_per_dataset"] <= 0:
        raise ConfigError("budgets must be positive")
    resources = config["resources"]
    if not 0 < resources["start_workers"] <= resources["max_workers"]:
        raise ConfigError("resources: need 0 < start_workers <= max_workers")
    for family, fixed in config["pilot"]["fixed_candidates"].items():
        if family not in config["families"]:
            raise ConfigError(f"pilot candidate for unknown family {family}")
        extra = set(fixed.get("params", {})) - set(config["families"][family].get("space", {}))
        if extra:
            raise ConfigError(f"pilot {family} sets unknown params {sorted(extra)}")


def effective_config(config, pilot):
    """The configuration actually executed: pilot limits families and the fit budget."""
    eff = copy.deepcopy(config)
    eff["pilot_mode"] = bool(pilot)
    eff["tuning_enabled"] = not pilot
    if pilot:
        eff["budgets"]["max_fits_per_dataset"] = int(config["pilot"]["max_fits_per_dataset"])
        eff["families"] = {name: config["families"][name]
                           for name in config["pilot"]["fixed_candidates"]}
    return eff


def catboost_status():
    """Return ``(available, reason)``; an unavailable family is logged, not replaced."""
    try:
        import catboost
    except ImportError as exc:
        return False, f"ImportError: {exc}"
    return True, f"catboost {catboost.__version__}"


def versions_info():
    import sklearn
    available, reason = catboost_status()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "optuna": optuna.__version__,
        "psutil": psutil.__version__,
        "catboost": reason if available else f"unavailable ({reason})",
    }


# ---------------------------------------------------------------------------
# Model factory.
# ---------------------------------------------------------------------------

class ConstantRegressor:
    """Training mean or median. Reference only: no weights and no search."""

    def __init__(self, statistic):
        if statistic not in ("mean", "median"):
            raise ConfigError(f"unknown constant statistic: {statistic}")
        self.statistic = statistic

    def fit(self, features, target, sample_weight=None):
        if sample_weight is not None:
            raise InvalidRecipe("constant references do not accept sample weights")
        values = np.asarray(target, dtype=float)
        self.value_ = float(np.mean(values) if self.statistic == "mean" else np.median(values))
        return self

    def predict(self, features):
        return np.full(len(features), self.value_, dtype=float)


def normalize_params(family, params):
    """Canonical parameter dict used in recipes, keys and hashes."""
    if family not in FAMILY_KINDS:
        raise InvalidRecipe(f"unknown family: {family}")
    canonical = dict(params)
    if family == "svr":
        mode = canonical.pop("gamma_mode", "scale")
        value = canonical.pop("gamma", None)
        canonical["gamma"] = "scale" if mode == "scale" else float(value)
    return dict(sorted(canonical.items()))


def build_estimator(family, params, seed):
    """Unfitted estimator from canonical parameters; unknown family is an error."""
    p = params
    if family in CONSTANT_FAMILIES:
        return ConstantRegressor(family)
    if family == "ridge":
        return Ridge(alpha=p["alpha"])
    if family == "elasticnet":
        return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"],
                          max_iter=p["max_iter"], random_state=seed)
    if family == "huber":
        return HuberRegressor(epsilon=p["epsilon"], alpha=p["alpha"], max_iter=p["max_iter"])
    if family == "svr":
        return SVR(kernel="rbf", C=p["C"], epsilon=p["epsilon"], gamma=p["gamma"],
                   max_iter=p["max_iter"], cache_size=p["cache_size"])
    if family == "knn":
        return KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])
    if family in ("extra_trees", "random_forest"):
        forest = ExtraTreesRegressor if family == "extra_trees" else RandomForestRegressor
        return forest(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                      min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                      n_jobs=1, random_state=seed)
    if family == "hist_gb":
        return HistGradientBoostingRegressor(
            max_iter=p["max_iter"], min_samples_leaf=p["min_samples_leaf"],
            max_leaf_nodes=p["max_leaf_nodes"], learning_rate=p["learning_rate"],
            l2_regularization=p["l2_regularization"], loss=p["loss"],
            early_stopping=False, random_state=seed)
    if family == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"],
            l2_leaf_reg=p["l2_leaf_reg"], loss_function=p["loss_function"], thread_count=1,
            allow_writing_files=False, random_seed=seed, verbose=False)
    raise InvalidRecipe(f"unknown family: {family}")


# ---------------------------------------------------------------------------
# Dataset loading (cer_rul only; notebooks are never read).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    name: str
    profile: dict
    clean: pd.DataFrame        # consolidated (Falha, Date) rows with sensors
    endpoints: pd.DataFrame    # cer endpoint rows (positions only; causal)
    meta: pd.DataFrame         # endpoints + y (fraction RUL); one row per endpoint
    csv_path: Path
    csv_sha256: str
    endpoints_sha256: str
    outer_order: tuple
    sensor_columns: tuple

    def identity(self, code_sha):
        return {"dataset": self.name, "csv_sha256": self.csv_sha256,
                "endpoints_sha256": self.endpoints_sha256, "code_sha256": code_sha}


def load_dataset(name, config, data_dir=DATA_DIR):
    """Load one CSV alone, consolidate it and build the endpoint table from its profile."""
    if name not in DATASET_NAMES:
        raise ConfigError(f"unknown dataset: {name}")
    profile = cer.get_dataset_profile(name)
    label = "porcentagem-data"
    label_meta = profile["labels"][label]
    csv_path = Path(data_dir) / label_meta["filename"]
    sensors = tuple(profile["sensor_columns"])
    csv_columns = ("Date",) + sensors + ("Falha", "RUL")
    fault_ids = tuple(sorted(int(fid) for fid in profile["measurement_by_id"]))
    equipment_of = {int(fid): equipment for equipment, ids in profile["equipment_groups"].items()
                    for fid in ids}
    outer_order = tuple(profile["equipment_groups"])
    if outer_order != tuple(config["datasets"][name]["outer_order"]):
        raise ConfigError(f"{name}: outer_order in config differs from the profile")
    raw = cer.load_raw_label(
        csv_path, label=label, expected_raw_rows=profile["expected_raw_rows"],
        csv_columns=csv_columns, sensor_columns=sensors, fault_ids=fault_ids,
        max_allowed=label_meta["max_allowed"], print_summary=False)
    clean, _audit, _terminal = cer.consolidate_frame(
        raw, label, sensor_columns=sensors, csv_columns=csv_columns, fault_ids=fault_ids,
        id_to_equipment=equipment_of, equipment_order=outer_order, label_meta=label_meta,
        expected_clean_keys=profile["expected_clean_keys"],
        expected_target_conflicts=profile["expected_target_conflicts"][label],
        enforce_expectations=True)
    protocol = config["protocol"]
    endpoints = cer.build_endpoint_rows(
        clean, warmup=protocol["warmup_records"], stride=protocol["endpoint_stride"],
        id_to_equipment=equipment_of, fpt_annotations=profile["fpt_annotations"],
        expected_endpoint_count=profile["expected_endpoints"], enforce=True)
    targets = clean[["Falha", "Date", "RUL"]].rename(columns={"RUL": "y"})
    meta = endpoints.merge(targets, on=["Falha", "Date"], how="left", validate="one_to_one")
    if meta["y"].isna().any():
        raise ValueError(f"{name}: endpoint without target")
    meta = meta.assign(endpoint_id=meta["endpoint_id"].astype(str)).reset_index(drop=True)
    digest = sha256_text(canonical_json(
        [[row.endpoint_id, row.Date.isoformat(), float(row.y)] for row in meta.itertuples()]))
    return Dataset(name=name, profile=profile, clean=clean, endpoints=endpoints, meta=meta,
                   csv_path=csv_path, csv_sha256=sha256_file(csv_path),
                   endpoints_sha256=digest, outer_order=outer_order, sensor_columns=sensors)


# ---------------------------------------------------------------------------
# Feature representations (cached lazily per dataset, width and bins).
# ---------------------------------------------------------------------------

def _robust_sensor_stats(dates, values, endpoint_date, width_hours, context_bins, sensor_keys):
    """Per-sensor robust summary over observed bins up to t (no whole-series reference)."""
    last_values, last_times, counts = cer.bin_stats(
        dates, values, endpoint_date, width_hours, context_bins=context_bins)
    observed = counts > 0
    times_observed = pd.Series(last_times[observed]).reset_index(drop=True)
    stats = {}
    for position, key in enumerate(sensor_keys):
        bin_values = last_values[observed, position]
        finite = np.isfinite(bin_values)
        bin_values = bin_values[finite]
        bin_times = times_observed[finite]
        if bin_values.size == 0:
            for stat in ("median", "iqr", "q10", "q90", "delta", "slope"):
                stats[f"rob_{stat}__{key}"] = float("nan")
            continue
        q10, q25, median, q75, q90 = np.quantile(bin_values, [0.1, 0.25, 0.5, 0.75, 0.9])
        _mean, _std, slope = cer.daily_stats(bin_times, bin_values)
        stats[f"rob_median__{key}"] = float(median)
        stats[f"rob_iqr__{key}"] = float(q75 - q25)
        stats[f"rob_q10__{key}"] = float(q10)
        stats[f"rob_q90__{key}"] = float(q90)
        stats[f"rob_delta__{key}"] = float(bin_values[-1] - bin_values[0])
        stats[f"rob_slope__{key}"] = float(slope)
    return stats


def _extra_feature_row(endpoint, group, sensor_columns, sensor_keys, eps_cfg,
                       width_hours, context_bins):
    """Robust, relative and elapsed features for one endpoint; past records only."""
    index = int(endpoint.record_index)
    context = group.iloc[:index + 1]
    values = context.loc[:, list(sensor_columns)].to_numpy(float)
    endpoint_date = pd.Timestamp(endpoint.Date)
    row = _robust_sensor_stats(context["Date"], values, endpoint_date, width_hours,
                               context_bins, sensor_keys)
    current = values[-1]
    for records in (10, 30):
        causal_prefix = context.iloc[:records]  # prefix of the observed rows, never full series
        reference = causal_prefix.loc[:, list(sensor_columns)].to_numpy(float).mean(axis=0)
        eps = np.maximum(eps_cfg["floor"], eps_cfg["relative"] * np.abs(reference))
        relative = (current - reference) / (np.abs(reference) + eps)
        for position, key in enumerate(sensor_keys):
            row[f"{REL_TEMPLATE.format(k=records)}{key}"] = float(relative[position])
    row[ELAPSED_COLUMN] = float((endpoint_date - pd.Timestamp(group["Date"].iloc[0]))
                                .total_seconds() / 86400.0)
    return row


def build_feature_table(dataset, width_hours, context_bins, config):
    """Every candidate feature column for one grid, indexed by ``endpoint_id``."""
    sensor_keys = cer.SENSOR_KEYS
    summary = cer.build_feature_frame(
        dataset.clean, dataset.endpoints, width_hours,
        sensor_columns=dataset.sensor_columns, context_bins=context_bins)
    summary.index = summary.index.astype(str)
    groups = {int(fid): group.sort_values("Date", kind="mergesort").reset_index(drop=True)
              for fid, group in dataset.clean.groupby("Falha", sort=True)}
    rows = [_extra_feature_row(endpoint, groups[int(endpoint.Falha)], dataset.sensor_columns,
                               sensor_keys, config["protocol"]["relative_eps"],
                               width_hours, context_bins)
            for endpoint in dataset.endpoints.itertuples(index=False)]
    endpoint_ids = pd.Index(dataset.endpoints["endpoint_id"].astype(str).to_numpy(),
                            name="endpoint_id")
    table = summary.join(pd.DataFrame(rows, index=endpoint_ids), how="left",
                         validate="one_to_one")
    measurement_names = sorted({str(value) for value in
                                dataset.profile["measurement_by_id"].values()})
    if len(measurement_names) > 1:  # known measurement metadata only, never event identity
        names_by_falha = {int(fid): str(value) for fid, value
                          in dataset.profile["measurement_by_id"].items()}
        onehot = {f"{MEASUREMENT_PREFIX}{name}": [
            float(names_by_falha[int(fid)] == name) for fid in dataset.endpoints["Falha"]]
            for name in measurement_names}
        table = table.join(pd.DataFrame(onehot, index=endpoint_ids))
    if table.index.has_duplicates:
        raise ValueError("feature table has duplicated endpoints")
    return table


class FeatureCache:
    """Feature tables computed once per grid and stored on disk.

    Each grid is keyed by the dataset/code identity and its width and bins; a stored
    table is reused only when its metadata and file hash match.
    """

    def __init__(self, dataset, config, identity, directory):
        self._dataset = dataset
        self._config = config
        self._identity = dict(identity)
        self._directory = Path(directory)
        self._memory = {}
        self.builds = 0
        self.disk_hits = 0

    def _meta(self, width_hours, context_bins):
        return {**self._identity, "width_hours": int(width_hours),
                "context_bins": int(context_bins)}

    def table(self, width_hours, context_bins):
        key = (int(width_hours), int(context_bins))
        if key in self._memory:
            return self._memory[key]
        base = self._directory / f"w{key[0]}_b{key[1]}"
        table_path, meta_path = base.with_suffix(".pkl"), base.with_suffix(".json")
        if meta_path.exists() and table_path.exists():
            table = self._load(table_path, meta_path, self._meta(*key))
            self.disk_hits += 1
        else:
            table = build_feature_table(self._dataset, key[0], key[1], self._config)
            self.builds += 1
            self._save(table, table_path, meta_path, self._meta(*key))
        self._memory[key] = table
        return table

    @staticmethod
    def _load(table_path, meta_path, expected_meta):
        try:
            stored = read_json(meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptCacheError(f"unreadable feature cache metadata {meta_path.name}") from exc
        if {k: v for k, v in stored.items() if k != "table_sha256"} != expected_meta:
            raise CorruptCacheError(f"feature cache {meta_path.name} belongs to another "
                                    "dataset, code version or grid")
        if sha256_file(table_path) != stored["table_sha256"]:
            raise CorruptCacheError(f"feature cache {table_path.name} failed its hash check")
        return pd.read_pickle(table_path)

    @staticmethod
    def _save(table, table_path, meta_path, meta):
        table_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = table_path.with_name(f".{table_path.name}.tmp-{os.getpid()}")
        table.to_pickle(tmp)
        os.replace(tmp, table_path)
        write_json_atomic(meta_path, {**meta, "table_sha256": sha256_file(table_path)})


def _legacy_control_columns(columns, features):
    """Exact historical column list: ``cer.feature_columns(w, representation, timing=False)``."""
    expected = cer.feature_columns(int(features["width_hours"]),
                                   features["representation"], False)
    available = set(columns)
    missing = [column for column in expected if column not in available]
    if missing:
        raise InvalidRecipe(f"legacy control columns unavailable: {missing[:3]}")
    return expected


def _base_feature_columns(columns, features, group_sets, variance_key):
    if features.get("legacy_control_columns"):
        return _legacy_control_columns(columns, features)
    if features["feature_groups"] not in group_sets:
        raise InvalidRecipe(f"unknown feature group set: {features['feature_groups']}")
    chosen_groups = set(group_sets[features["feature_groups"]])
    representation = features["representation"]
    if representation not in REPRESENTATION_PREFIXES:
        raise InvalidRecipe(f"unknown representation: {representation}")
    relative = features["relative_reference"]
    if (relative is not None or features["drop_variance"]) and "sensors" not in chosen_groups:
        raise InvalidRecipe("relative and variance options require the sensors group")
    sensor_cols = [c for c in columns if c.startswith(REPRESENTATION_PREFIXES[representation])]
    if representation == "current" and features["redundancy"] == "legacy":
        sensor_cols += [c for c in columns if c.startswith(("nobs__", "missbins__"))]
    if relative is not None:
        prefix = REL_TEMPLATE.format(k=int(relative))
        sensor_cols += [c for c in columns if c.startswith(prefix)]
    if features["drop_variance"]:
        sensor_cols = [c for c in sensor_cols if c.rsplit("__", 1)[-1] != variance_key]
    chosen = set()
    if "sensors" in chosen_groups:
        chosen.update(sensor_cols)
    if "cadence" in chosen_groups:
        chosen.update(c for c in columns if c.startswith(ACQUISITION_PREFIXES))
    if "elapsed" in chosen_groups:
        chosen.update(c for c in columns if c == ELAPSED_COLUMN)
    if features.get("measurement_onehot"):
        onehot = [c for c in columns if c.startswith(MEASUREMENT_PREFIX)]
        if not onehot:
            raise InvalidRecipe("measurement one-hot requested but not available")
        chosen.update(onehot)
    selected = [c for c in columns if c in chosen]
    if not selected:
        raise InvalidRecipe("recipe selects no feature columns")
    return selected


def select_feature_columns(columns, features, group_sets, variance_key):
    """Ordered feature columns for a recipe (static choice; no training data used).

    ``variance_key`` is the dataset-specific key of ``g_0_Variance`` so the same
    ``drop_variance`` option removes that sensor for generator and gearbox alike.
    Joint event tables (``m1__``..``m6__`` prefixes) are selected block by block with
    exactly the same rules, so no joint column is added blindly.
    """
    columns = list(columns)
    joint = [c for c in columns if c.startswith(JOINT_PREFIXES[0])]
    if joint:
        if features.get("measurement_onehot"):
            raise InvalidRecipe("measurement one-hot is not available in the joint event track")
        base = [c.split("__", 1)[1] for c in joint]
        selected = _base_feature_columns(base, features, group_sets, variance_key)
        return [f"{prefix}{column}" for prefix in JOINT_PREFIXES for column in selected]
    return _base_feature_columns(columns, features, group_sets, variance_key)


# ---------------------------------------------------------------------------
# Recipes: normalisation, validation, sampling, complexity.
# ---------------------------------------------------------------------------

def _set_path(target, dotted, value):
    head, *rest = dotted.split(".")
    if not rest:
        target[head] = value
        return
    _set_path(target.setdefault(head, {}), ".".join(rest), value)


def normalize_recipe(family, params, overrides, default_recipe):
    """Canonical recipe: family, canonical params, and the feature/pre/post options."""
    recipe = copy.deepcopy(default_recipe)
    for dotted, value in overrides.items():
        _set_path(recipe, dotted, value)
    recipe["family"] = family
    recipe["params"] = normalize_params(family, params)
    if recipe.get("strategy") == "measurement_onehot":
        recipe["features"]["measurement_onehot"] = True
    return recipe


def validate_recipe(recipe, dataset_name, group_sets):
    """Reject combinations that the protocol does not allow (raises InvalidRecipe)."""
    family = recipe["family"]
    if family not in FAMILY_KINDS:
        raise InvalidRecipe(f"unknown family: {family}")
    if recipe["weights"] != "none" and family not in WEIGHT_CAPABLE:
        raise InvalidRecipe(f"{family} does not accept sample weights ({recipe['weights']})")
    features = recipe["features"]
    strategy = recipe.get("strategy", "pooled_anonymous")
    allowed_strategies = {"pooled_anonymous", "measurement_onehot",
                          "component_experts", "measurement_experts", "joint_event"}
    if strategy not in allowed_strategies:
        raise InvalidRecipe(f"unknown per-series strategy: {strategy}")
    if dataset_name != "gearbox" and strategy != "pooled_anonymous":
        raise InvalidRecipe("specialist strategies are gearbox-only")
    if strategy == "measurement_onehot" and not features.get("measurement_onehot"):
        raise InvalidRecipe("measurement-onehot strategy requires its fixed metadata features")
    if features.get("measurement_onehot") and dataset_name != "gearbox":
        raise InvalidRecipe("measurement one-hot is only allowed for the gearbox dataset")
    if features["feature_groups"] not in group_sets:
        raise InvalidRecipe(f"unknown feature group set: {features['feature_groups']}")
    if recipe["postprocess"]["ema_alpha"] not in ALLOWED_EMA_ALPHAS:
        raise InvalidRecipe(f"ema_alpha must be one of {ALLOWED_EMA_ALPHAS}")
    return recipe


def recipe_complexity(recipe, n_feature_columns):
    """Nominal complexity: candidate feature columns plus active optional stages."""
    features = recipe["features"]
    preprocess = recipe["preprocess"]
    post = recipe["postprocess"]
    stages = sum([
        recipe["weights"] != "none",
        bool(post["clip01"]),
        post["ema_alpha"] != 1.0,
        bool(post["running_min"]),
        bool(preprocess["clip_quantiles"]),
        bool(preprocess["missing_indicators"]),
        features["relative_reference"] is not None,
        bool(features["drop_variance"]),
        features["redundancy"] == "dedup",
        bool(features.get("measurement_onehot")),
    ])
    return int(n_feature_columns) + int(stages)


def recipe_id(recipe):
    return sha256_text(canonical_json(recipe))[:16]


def _suggest(trial, name, spec, n_train_min=None):
    kind = spec["type"]
    if kind == "const":
        return spec["value"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        high = int(spec["high"])
        if spec.get("cap") == "n_train_min":
            high = min(high, int(n_train_min))
            if high < int(spec["low"]):
                raise InvalidRecipe(f"{name}: training rows too few for the lower bound")
        return trial.suggest_int(name, int(spec["low"]), high)
    return trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                               log=bool(spec.get("log", False)))


def sample_recipe(trial, family, config, dataset_name, n_train_min):
    """Draw one recipe from the JSON spaces through an Optuna trial."""
    params = {name: _suggest(trial, f"{family}.{name}", spec, n_train_min)
              for name, spec in config["families"][family].get("space", {}).items()}
    overrides = {}
    for dotted, spec in config["recipe_space"].items():
        if dotted == "strategy" and dataset_name == "gearbox":
            overrides[dotted] = "pooled_anonymous"  # specialist budgets are isolated below
        elif "datasets" in spec and dataset_name not in spec["datasets"]:
            overrides[dotted] = spec["default"]
        else:
            overrides[dotted] = _suggest(trial, dotted, spec)
    return normalize_recipe(family, params, overrides, config["default_recipe"])


SPECIALIST_FAMILIES = ("ridge", "extra_trees", "hist_gb")
SERIES_STRATEGIES = ("pooled_anonymous", "measurement_onehot",
                     "component_experts", "measurement_experts")


def _lattice_values(spec, limit):
    """Deterministic candidate values for one parameter (at most ``limit`` values)."""
    kind = spec["type"]
    if kind == "const":
        return [spec["value"]]
    if kind == "categorical":
        return list(spec["choices"][:max(1, int(limit))])
    low, high = spec["low"], spec["high"]
    if kind == "int":
        values = sorted({int(low), int(round((int(low) + int(high)) / 2)),
                         int(high)})
        return values[:max(1, int(limit))]
    if spec.get("log"):
        values = sorted({float(low), float(np.sqrt(low * high)), float(high)})
    else:
        values = sorted({float(low), float((low + high) / 2), float(high)})
    return values[:max(1, int(limit))]


def bounded_family_recipes(family, space, limit):
    """Up to ``limit`` deterministic parameter settings, first ones of a small lattice."""
    names = sorted(space)
    value_lists = [_lattice_values(space[name], max(1, limit)) for name in names]
    recipes = []
    for combination in itertools.product(*value_lists):
        params = normalize_params(family, dict(zip(names, combination)))
        if params not in recipes:
            recipes.append(params)
        if len(recipes) >= int(limit):
            break
    return recipes


def strategy_fit_experts(strategy):
    """Number of independent fits one recipe needs per fold for a series strategy."""
    return len(gearbox_expert_members(strategy))


# ---------------------------------------------------------------------------
# Train-only preprocessing and weights.
# ---------------------------------------------------------------------------

def _dedup_columns(frame):
    """Keep the first of each set of identical (NaN-aware) training columns."""
    kept, arrays = [], []
    for column in frame.columns:
        values = frame[column].to_numpy(float)
        if any(np.array_equal(values, other, equal_nan=True) for other in arrays):
            continue
        kept.append(column)
        arrays.append(values)
    return kept


def prepare_matrices(train_x, eval_x, preprocess, redundancy):
    """Constant selector, optional dedup and quantile clip, then preprocessing.

    Every statistic is learned from ``train_x`` only; ``eval_x`` is transformed.
    """
    keep = [c for c in train_x.columns if train_x[c].nunique(dropna=True) > 1]
    if not keep:
        raise InvalidRecipe("no non-constant training columns")
    train_x, eval_x = train_x[keep], eval_x[keep]
    if redundancy == "dedup":
        kept = _dedup_columns(train_x)
        train_x, eval_x = train_x[kept], eval_x[kept]
    if preprocess["clip_quantiles"]:
        low, high = train_x.quantile(0.01), train_x.quantile(0.99)
        train_x = train_x.clip(lower=low, upper=high, axis=1)
        eval_x = eval_x.clip(lower=low, upper=high, axis=1)
    pre = cer.TabularPreprocessor(preprocess["method"],
                                  add_missing_indicators=preprocess["missing_indicators"])
    pre.fit(train_x)
    names = [str(name) for name in pre.get_feature_names_out()]
    return pre.transform(train_x), pre.transform(eval_x), names


def training_weights(scheme, train_meta):
    """Training weights (mean one). FPT phase is used only here, never as a feature."""
    if scheme == "group_equal":
        equipment = train_meta["equipment"].to_numpy(str)
        counts = pd.Series(equipment).value_counts()
        weights = np.asarray([1.0 / counts[name] for name in equipment], dtype=float)
        return weights / weights.mean()
    return cer.compute_weights(train_meta["Falha"].to_numpy(int),
                               train_meta["phase"].to_numpy(str), scheme)


# ---------------------------------------------------------------------------
# Fit tasks, the fit itself and the on-disk cache.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitTask:
    key: str
    family: str
    params: dict
    seed: int
    X_train: np.ndarray
    y_train: np.ndarray
    weights: object
    X_eval: np.ndarray
    eval_ids: tuple
    timeout_s: float
    train_ids: tuple = ()


@dataclass
class FitRecord:
    key: str
    status: str                  # "ok" or "failed"
    reason: object
    predictions: object          # np.ndarray or None
    elapsed_s: float = 0.0
    peak_uss_mb: float = 0.0     # private memory of the worker (USS), sampled
    inherited_rss_mb: float = 0.0  # parent RSS at fork: shared copy-on-write, not new memory
    source: str = "fit"          # "fit" or "cache"
    deterministic: bool = True   # timeouts and worker crashes are transient: not cached


def fit_key(identity, family, params, feature_columns, feature_spec, preprocess, weights,
            train_ids, eval_ids, seed):
    """Key covering everything that changes the fitted model and its predictions."""
    return sha256_text(canonical_json({
        "identity": identity, "family": family, "params": params,
        "feature_spec": feature_spec, "feature_columns": list(feature_columns),
        "preprocess": preprocess, "weights": weights,
        "train_ids": list(train_ids), "eval_ids": list(eval_ids), "seed": int(seed),
    }))


def build_fold_task(meta, features, train_mask, eval_mask, recipe, group_sets, seed,
                    identity, timeout_s, fit_context=None, *, variance_key):
    """Prepare one fit. ``features`` and ``meta`` share row order; masks are boolean."""
    columns = select_feature_columns(list(features.columns), recipe["features"], group_sets,
                                     variance_key)
    x_all = features[columns]
    train_meta = meta.loc[train_mask]
    x_train, x_eval, names = prepare_matrices(
        x_all.loc[train_mask], x_all.loc[eval_mask], recipe["preprocess"],
        recipe["features"]["redundancy"])
    weights = None if recipe["weights"] == "none" else training_weights(recipe["weights"],
                                                                        train_meta)
    key = fit_key({**identity, "fit_context": fit_context or {}}, recipe["family"],
                  recipe["params"], names, recipe["features"],
                  recipe["preprocess"], recipe["weights"],
                  train_meta["endpoint_id"].tolist(),
                  meta.loc[eval_mask, "endpoint_id"].tolist(), seed)
    return FitTask(key=key, family=recipe["family"], params=recipe["params"], seed=int(seed),
                   X_train=x_train, y_train=train_meta["y"].to_numpy(float), weights=weights,
                   X_eval=x_eval, eval_ids=tuple(meta.loc[eval_mask, "endpoint_id"]),
                   timeout_s=float(timeout_s),
                   train_ids=tuple(train_meta["endpoint_id"].astype(str)))


def gearbox_expert_members(strategy):
    """Component/measurement expert IDs; pooled strategies have one global expert."""
    if strategy in ("pooled_anonymous", "measurement_onehot", "joint_event"):
        return [("pooled", tuple(range(1, 19)))]
    if strategy == "component_experts":
        return [(name, tuple(fid for fid in range(1, 19)
                             if gearbox_component_of(fid) == name))
                for name in GEARBOX_COMPONENTS]
    if strategy == "measurement_experts":
        return [(f"measurement_{position}", tuple(range(position, 19, 6)))
                for position in range(1, 7)]
    raise InvalidRecipe(f"unsupported gearbox per-series strategy: {strategy}")


def build_strategy_fold_tasks(meta, features, train_mask, eval_mask, recipe, group_sets,
                              seed_for, identity, timeout_s, dataset_name, variance_key):
    """Build pooled or specialist fits; each specialist sees only training events."""
    strategy = recipe.get("strategy", "pooled_anonymous")
    if dataset_name != "gearbox" or strategy in ("pooled_anonymous", "measurement_onehot",
                                                   "joint_event"):
        context = {"strategy": strategy, "expert": "pooled", "schema": "series-v1"}
        task = build_fold_task(meta, features, train_mask, eval_mask, recipe, group_sets,
                               seed_for("pooled"), identity, timeout_s, context,
                               variance_key=variance_key)
        return [("pooled", task)]
    result = []
    for expert, member_ids in gearbox_expert_members(strategy):
        members = meta["Falha"].isin(member_ids).to_numpy()
        expert_train = np.asarray(train_mask, bool) & members
        expert_eval = np.asarray(eval_mask, bool) & members
        if not expert_train.any() or not expert_eval.any():
            raise InvalidRecipe(f"missing train or eval rows for expert {expert}")
        train_events = set(meta.loc[expert_train, "equipment"])
        eval_events = set(meta.loc[expert_eval, "equipment"])
        if not train_events.isdisjoint(eval_events):
            raise ValueError(f"expert {expert} leaks a held-out event into training")
        context = {"strategy": strategy, "expert": expert,
                   "members": member_ids, "schema": "series-v1"}
        task = build_fold_task(meta, features, expert_train, expert_eval, recipe, group_sets,
                               seed_for(expert), identity, timeout_s, context,
                               variance_key=variance_key)
        result.append((expert, task))
    return result


def fit_task_locally(task):
    """Fit one estimator (runs in a worker). Failures are returned as explicit records."""
    started = time.perf_counter()
    own = psutil.Process()
    inherited_mb = own.memory_info().rss / MIB  # shared with the parent (copy-on-write)
    status, reason, predictions = "ok", None, None
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = build_estimator(task.family, task.params, task.seed)
            if task.weights is None:
                model.fit(task.X_train, task.y_train)
            else:
                model.fit(task.X_train, task.y_train, sample_weight=task.weights)
            predictions = np.asarray(model.predict(task.X_eval), dtype=float).reshape(-1)
        if predictions.shape[0] != len(task.X_eval) or not np.isfinite(predictions).all():
            status, reason, predictions = "failed", "non_finite_or_misshaped_prediction", None
    except ConvergenceWarning as exc:
        status, reason, predictions = "failed", f"convergence: {exc}", None
    except MODEL_ERRORS as exc:  # estimator/numerical failure: explicit failed candidate
        status, reason, predictions = "failed", f"{type(exc).__name__}: {exc}", None
    # Private memory of this worker at completion: pages it copied or allocated itself.
    # Measured inside the worker because fits are shorter than any parent polling interval.
    return {"status": status, "reason": reason,
            "predictions": None if predictions is None else predictions.tolist(),
            "elapsed_s": time.perf_counter() - started,
            "uss_mb": own.memory_full_info().uss / MIB,
            "inherited_rss_mb": inherited_mb}


def _child_entry(connection, task):
    try:
        message = fit_task_locally(task)
    except BaseException:  # not a model failure: report to the parent, which aborts the run
        message = {"status": "fatal", "reason": traceback.format_exc(),
                   "predictions": None, "elapsed_s": 0.0}
    connection.send(message)
    connection.close()


def _reap(process, grace_s=2.0):
    """Terminate, then kill if needed, and always join so no child is left behind."""
    if process.is_alive():
        process.terminate()
        process.join(grace_s)
    if process.is_alive():
        process.kill()
    process.join()


def cgroup_memory_limit_bytes():
    """Effective cgroup memory limit, or None when unlimited/unknown."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text in ("", "max"):
            return None
        try:
            value = int(text)
        except ValueError:
            return None
        if value <= 0 or value >= (1 << 62):  # kernel sentinel for "unlimited"
            return None
        return value
    return None


def cgroup_memory_current_bytes():
    """Current cgroup memory use, or None when it cannot be read."""
    for path in ("/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(path).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return None


def final_refit_reserve(outer_count, config, pilot=False):
    """Fits reserved for final refits, from the worst case the configuration allows.

    Per-series winners may be specialist recipes needing one fit per expert, and a
    confirmed stochastic winner refits once per seed offset. The event track adds its
    own joint selection. Being explicit keeps search from consuming the whole budget.
    """
    budgets = config["budgets"]
    families = len(config["families"])
    if pilot:
        # Pilot runs one fixed recipe per family on the per-series track only.
        return int(outer_count) * families
    experts = int(budgets["experts_per_fit_max"])
    seeds = int(budgets["seed_refits_max"])
    event_families = len([f for f in SPECIALIST_FAMILIES if f in config["families"]]) + \
        len(config["stages"]["ensemble_sizes"])
    per_outer = families * experts * seeds + event_families
    return int(outer_count) * per_outer


class FitCache:
    """One JSON file per fit key; predictions and status are written in one atomic step."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def _path(self, key):
        return self.directory / f"{key}.json"

    def load(self, key, expected_length):
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptCacheError(f"unreadable fit cache {path.name}: {exc}") from exc
        stored_sha = payload.pop("payload_sha256", None)
        if payload.get("key") != key or stored_sha != sha256_text(canonical_json(payload)):
            raise CorruptCacheError(f"fit cache {path.name} failed key/hash validation")
        predictions = payload["predictions"]
        if payload["status"] == "ok" and (predictions is None
                                          or len(predictions) != expected_length):
            raise CorruptCacheError(f"fit cache {path.name} has the wrong prediction length")
        return FitRecord(key=key, status=payload["status"], reason=payload["reason"],
                         predictions=None if predictions is None else np.asarray(predictions),
                         elapsed_s=payload["elapsed_s"], peak_uss_mb=payload["peak_uss_mb"],
                         inherited_rss_mb=payload["inherited_rss_mb"],
                         source="cache", deterministic=True)

    def store(self, record):
        if not record.deterministic:
            return
        payload = {
            "key": record.key,
            "status": record.status,
            "reason": record.reason,
            "predictions": None if record.predictions is None else record.predictions.tolist(),
            "elapsed_s": record.elapsed_s,
            "peak_uss_mb": record.peak_uss_mb,
            "inherited_rss_mb": record.inherited_rss_mb,
        }
        payload["payload_sha256"] = sha256_text(canonical_json(payload))
        write_json_atomic(self._path(record.key), payload)


@dataclass
class FitBudget:
    """Per-dataset fit budget. Search cannot use the fits reserved for final refits."""
    max_fits: int
    charged: int = 0
    reserved_for_final: int = 0

    def limit(self, purpose):
        if purpose == "search":
            return self.max_fits - self.reserved_for_final
        if purpose == "final":
            return self.max_fits
        raise ValueError(f"unknown budget purpose: {purpose}")

    def remaining(self, purpose="search"):
        return self.limit(purpose) - self.charged

    def check(self, count, purpose="search"):
        if self.charged + count > self.limit(purpose):
            raise BudgetExceeded(f"{purpose}: {self.charged}+{count} fits exceeds "
                                 f"{self.limit(purpose)}")

    def charge(self, count, purpose="search"):
        self.check(count, purpose)
        self.charged += count

    def to_dict(self):
        return {"max_fits": self.max_fits, "charged": self.charged,
                "reserved_for_final": self.reserved_for_final}

    @classmethod
    def load(cls, path, max_fits, reserved_for_final):
        path = Path(path)
        if not path.exists():
            return cls(max_fits=max_fits, reserved_for_final=reserved_for_final)
        stored = read_json(path)
        if (int(stored["max_fits"]) != int(max_fits)
                or int(stored["reserved_for_final"]) != int(reserved_for_final)):
            raise ConfigError("ledger budget differs from the current configuration")
        return cls(max_fits=max_fits, charged=int(stored["charged"]),
                   reserved_for_final=int(reserved_for_final))


class ResourcePolicy:
    """Chooses how many fit workers may run from available RAM. OS limits are never changed.

    The per-worker estimate is ``rss_safety_factor`` times the largest worker USS
    observed so far, floored at ``worker_floor_mb`` before and after observations.
    """

    def __init__(self, resources, available_bytes=None, cgroup_limit=None,
                 cgroup_current=None):
        self._resources = resources
        self._available = available_bytes or (lambda: psutil.virtual_memory().available)
        self._cgroup_limit = cgroup_limit or cgroup_memory_limit_bytes
        self._cgroup_current = cgroup_current or cgroup_memory_current_bytes
        self.cgroup_limit_bytes = self._cgroup_limit()
        self.initial_available = self.available()
        self.observed_peak_mb = 0.0
        self.live_peak_mb = 0.0
        # One worker until a fit has actually been observed (calibration), then grow.
        self.target = 1
        self.decisions = []

    def estimate_bytes(self):
        """Per-worker estimate: safety factor times the observed peak, floored.

        A tiny Ridge must not convince the scheduler that eight heavy forests fit, so
        the estimate never drops below ``worker_floor_mb``. An observed peak is always
        inflated by the safety factor: no cap may ever sit below the real peak.
        """
        resources = self._resources
        observed = max(self.observed_peak_mb, self.live_peak_mb)
        floor_mb = float(resources.get("worker_floor_mb", 0))
        inflated = float(resources["rss_safety_factor"]) * observed
        return int(max(floor_mb, inflated) * MIB)

    def available(self):
        host = int(self._available())
        if self.cgroup_limit_bytes:
            used = self._cgroup_current()
            if used is not None:
                return max(0, min(host, int(self.cgroup_limit_bytes) - int(used)))
        return host

    def limit(self, available):
        r = self._resources
        by_share = math.floor(r["max_memory_share_of_initial_available"]
                              * self.initial_available / self.estimate_bytes())
        by_headroom = math.floor((available - r["min_headroom_gib"] * GIB)
                                 / self.estimate_bytes())
        return max(0, min(int(r["max_workers"]), by_share, by_headroom))

    def can_start(self, available, running):
        return running < min(self.target, self.limit(available))

    def observe_peak(self, peak_uss_mb):
        self.observed_peak_mb = max(self.observed_peak_mb, float(peak_uss_mb))

    def grow(self):
        """Increase concurrency only after a fit has produced a real memory observation."""
        if self.observed_peak_mb <= 0:
            return
        self.target = min(self.target + 1, int(self._resources["max_workers"]))

    def record(self, available, running, action):
        if len(self.decisions) < 500:
            self.decisions.append({"available_mib": int(available / MIB), "running": running,
                                   "target": self.target, "limit": self.limit(available),
                                   "action": action})


@dataclass
class RunnerStats:
    fits_started: int = 0
    cache_hits: int = 0
    ok: int = 0
    deterministic_failures: int = 0
    timeouts: int = 0
    worker_failures: int = 0
    max_concurrent: int = 0
    peak_uss_mb_max: float = 0.0
    inherited_rss_mb_max: float = 0.0
    fit_elapsed_s: float = 0.0
    wall_s: float = 0.0

    def to_dict(self):
        return dict(self.__dict__)


class FitRunner:
    """Runs fit tasks in forked workers with a hard timeout and RAM-aware scheduling.

    The parent is the only writer of the fit cache and of the budget ledger. Timeouts
    and worker crashes become transient failures: reported, never cached as scores.
    A non-model error inside a worker aborts the run (WorkerFatalError).
    """

    def __init__(self, cache, budget, policy, *, timeout_s, poll_s, ledger_path,
                 pause_poll_s, max_pause_s):
        self.cache = cache
        self.budget = budget
        self.policy = policy
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.ledger_path = Path(ledger_path)
        self.pause_poll_s = float(pause_poll_s)
        self.max_pause_s = float(max_pause_s)
        self.stats = RunnerStats()
        self._context = mp.get_context("fork")
        self._purpose = "search"

    def run(self, tasks, purpose="search"):
        """Return ``{key: FitRecord}``. The batch is refused before any worker starts
        if it does not fit the budget for ``purpose``."""
        started = time.perf_counter()
        unique = OrderedDict()
        for task in tasks:
            unique.setdefault(task.key, task)
        records, pending = {}, []
        for key, task in unique.items():
            cached = self.cache.load(key, len(task.eval_ids))
            if cached is None:
                pending.append(task)
            else:
                self.stats.cache_hits += 1
                records[key] = cached
        self.budget.check(len(pending), purpose)
        self._purpose = purpose
        try:
            records.update(self._execute(pending))
        finally:
            write_json_atomic(self.ledger_path, self.budget.to_dict())
            self.stats.wall_s += time.perf_counter() - started
        for key in unique:
            record = records[key]
            if record.status == "ok":
                self.stats.ok += 1
            elif record.deterministic:
                self.stats.deterministic_failures += 1
        return {key: records[key] for key in unique}

    def _execute(self, pending):
        records, queue, running = {}, deque(pending), {}
        pause_started = None
        try:
            while queue or running:
                available = self.policy.available()
                if queue and self.policy.can_start(available, len(running)):
                    pause_started = None
                    while queue and self.policy.can_start(self.policy.available(), len(running)):
                        self._start(queue.popleft(), running)
                        self.policy.record(available, len(running), "start")
                elif queue and not running:
                    if pause_started is None:
                        pause_started = time.perf_counter()
                        self.policy.record(available, 0, "pause_low_memory")
                        print(f"fit scheduling paused: {available / MIB:.0f} MiB available, "
                              f"{self.policy.estimate_bytes() / MIB:.0f} MiB per worker, "
                              f"{self.policy._resources['min_headroom_gib']:g} GiB reserve",
                              flush=True)
                    if time.perf_counter() - pause_started > self.max_pause_s:
                        raise ResourceError(
                            "memory headroom did not recover within max_pause_s "
                            f"(available {available / MIB:.0f} MiB, per-worker estimate "
                            f"{self.policy.estimate_bytes() / MIB:.0f} MiB, reserve "
                            f"{self.policy._resources['min_headroom_gib']:g} GiB)")
                    time.sleep(self.pause_poll_s)
                    continue
                for record in self._poll(running):
                    records[record.key] = record
                    self._account(record)
                time.sleep(self.poll_s)
        finally:
            for entry in running.values():  # abnormal exit: no worker may survive the run
                _reap(entry["process"])
                entry["conn"].close()
        return records

    def _start(self, task, running):
        # Reserve the fit and persist the ledger BEFORE forking: a SIGKILL after the
        # fork must not lose the cost of a fit that really ran.
        self.budget.charge(1, self._purpose)
        write_json_atomic(self.ledger_path, self.budget.to_dict())
        self.stats.fits_started += 1
        parent_end, child_end = self._context.Pipe(duplex=False)
        process = self._context.Process(target=_child_entry, args=(child_end, task))
        process.start()
        child_end.close()
        running[task.key] = {"task": task, "process": process, "conn": parent_end,
                             "started": time.perf_counter(),
                             "peak_uss_mb": 0.0,
                             "live_probe": psutil.Process(process.pid)}
        self.stats.max_concurrent = max(self.stats.max_concurrent, len(running))

    def _account(self, record):
        if record.peak_uss_mb > 0:
            self.policy.observe_peak(record.peak_uss_mb)
        self.stats.peak_uss_mb_max = max(self.stats.peak_uss_mb_max, record.peak_uss_mb)
        self.stats.inherited_rss_mb_max = max(self.stats.inherited_rss_mb_max,
                                              record.inherited_rss_mb)
        self.stats.fit_elapsed_s += record.elapsed_s
        if record.status == "ok":
            self.policy.grow()
        self.cache.store(record)

    def _poll(self, running):
        finished = []
        for key in list(running):
            entry = running[key]
            process, conn, task = entry["process"], entry["conn"], entry["task"]
            elapsed = time.perf_counter() - entry["started"]
            self._sample_worker(entry)
            if conn.poll():
                try:
                    message = conn.recv()
                except (EOFError, OSError):
                    message = None  # worker closed the pipe without a result
                del running[key]
                if message is not None and message["status"] == "fatal":
                    _reap(process)
                    conn.close()
                    raise WorkerFatalError(
                        f"fit worker failed outside the model ({task.family}):\n"
                        f"{message['reason']}")
                process.join()
                conn.close()
                if message is None:
                    self.stats.worker_failures += 1
                    finished.append(self._transient(key, f"worker_exit_{process.exitcode}", elapsed))
                    continue
                predictions = message["predictions"]
                peak_uss = max(float(message["uss_mb"]), float(entry["peak_uss_mb"]))
                finished.append(FitRecord(
                    key=key, status=message["status"], reason=message["reason"],
                    predictions=None if predictions is None else np.asarray(predictions, float),
                    elapsed_s=float(message["elapsed_s"]), peak_uss_mb=peak_uss,
                    inherited_rss_mb=float(message["inherited_rss_mb"]), source="fit",
                    deterministic=True))
            elif elapsed > self.timeout_s:
                _reap(process)
                conn.close()
                del running[key]
                self.stats.timeouts += 1
                finished.append(self._transient(key, f"timeout>{self.timeout_s:g}s", elapsed))
            elif not process.is_alive():
                process.join()
                conn.close()
                del running[key]
                self.stats.worker_failures += 1
                finished.append(self._transient(key, f"worker_exit_{process.exitcode}", elapsed))
        return finished

    def _sample_worker(self, entry):
        """Record the live private memory of a running worker through its lifetime."""
        try:
            uss_mb = entry["live_probe"].memory_full_info().uss / MIB
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return
        if uss_mb > entry["peak_uss_mb"]:
            entry["peak_uss_mb"] = float(uss_mb)
            self.policy.live_peak_mb = max(self.policy.live_peak_mb, float(uss_mb))

    @staticmethod
    def _transient(key, reason, elapsed):
        """Timeouts and crashes: no worker measurements exist, so none are recorded."""
        return FitRecord(key=key, status="failed", reason=reason, predictions=None,
                         elapsed_s=elapsed, source="fit", deterministic=False)


# ---------------------------------------------------------------------------
# Postprocessing and inner scoring.
# ---------------------------------------------------------------------------

def postprocess_predictions(frame, *, clip01, ema_alpha, running_min):
    """Clip, EMA and running minimum per Falha in time order; only past values are used."""
    if ema_alpha not in ALLOWED_EMA_ALPHAS:
        raise InvalidRecipe(f"ema_alpha must be one of {ALLOWED_EMA_ALPHAS}")
    order = pd.DataFrame({"Falha": frame["Falha"].to_numpy(int),
                          "Date": frame["Date"].to_numpy(),
                          "pos": np.arange(len(frame)),
                          "value": frame["pred_raw"].to_numpy(float)})
    output = np.empty(len(frame), dtype=float)
    for _falha, group in order.sort_values(["Falha", "Date", "pos"]).groupby("Falha",
                                                                              sort=False):
        values = group["value"].to_numpy(float)
        if clip01:
            values = np.clip(values, 0.0, 1.0)
        if ema_alpha != 1.0:
            smoothed = np.empty_like(values)
            smoothed[0] = values[0]
            for index in range(1, len(values)):
                smoothed[index] = ema_alpha * values[index] + (1 - ema_alpha) * smoothed[index - 1]
            values = smoothed
        if running_min:
            values = np.minimum.accumulate(values)
        output[group["pos"].to_numpy()] = values
    result = frame.copy()
    result["pred_post"] = output
    return result


def inner_score(oof):
    """Equal-weight inner-group score in pp: component MAE, then mean per group, then mean."""
    error = oof["pred_post"].to_numpy(float) - oof["y"].to_numpy(float)
    group_scores = {}
    for equipment, group in oof.groupby("equipment", sort=True):
        component_mae = [float(np.mean(np.abs(comp["pred_post"] - comp["y"])))
                         for _falha, comp in group.groupby("Falha", sort=True)]
        group_scores[str(equipment)] = float(np.mean(component_mae)) * PP
    return {
        "score_pp": float(np.mean(list(group_scores.values()))),
        "rmse_pp": float(np.sqrt(np.mean(error ** 2))) * PP,
        "group_scores_pp": group_scores,
    }


def select_best(candidates, tolerance_pp):
    """Best score; ties within tolerance -> lower complexity -> lower RMSE -> stable ID."""
    valid = [c for c in candidates if c.get("score_pp") is not None]
    if not valid:
        return None
    best = min(c["score_pp"] for c in valid)
    eligible = [c for c in valid if c["score_pp"] <= best + tolerance_pp + 1e-12]
    return min(eligible, key=lambda c: (c["complexity"], c["rmse_pp"], c["candidate_id"]))


def ensemble_predictions(member_frames, how):
    """Uniform mean or median of member postprocessed predictions for each endpoint."""
    if how not in ("mean", "median"):
        raise InvalidRecipe(f"unknown ensemble rule: {how}")
    base = member_frames[0][["endpoint_id", "Falha", "equipment", "Date", "y"]].copy()
    if "phase" in member_frames[0].columns:
        base["phase"] = member_frames[0]["phase"].to_numpy()
    stacked = np.column_stack([
        frame.set_index("endpoint_id").loc[base["endpoint_id"], "pred_post"].to_numpy(float)
        for frame in member_frames])
    base["pred_post"] = stacked.mean(axis=1) if how == "mean" else np.median(stacked, axis=1)
    return base


def mean_seed_predictions(predictions):
    """Uniformly average per-seed prediction vectors after checking their alignment."""
    if not predictions or len({len(values) for values in predictions}) != 1:
        raise ValueError("seed predictions are empty or misaligned")
    return np.mean(np.column_stack(predictions), axis=1)


# ---------------------------------------------------------------------------
# Gearbox strategy helpers (pure functions; not wired into search or evaluate yet).
# ---------------------------------------------------------------------------

GEARBOX_MEASUREMENTS_PER_EVENT = 6
GEARBOX_COMPONENTS = OrderedDict([("HSS", (1, 2)), ("IMS", (3,)), ("LSS", (4,)),
                                  ("PLANETARIA", (5, 6))])


def gearbox_event_of(fault_id):
    """Event E1..E3 for Falha 1..18 (six measurements per event)."""
    fid = int(fault_id)
    if not 1 <= fid <= 18:
        raise ValueError(f"unknown gearbox Falha: {fid}")
    return (fid - 1) // GEARBOX_MEASUREMENTS_PER_EVENT + 1


def gearbox_position_of(fault_id):
    """Position 1..6 inside its event (HSS-R, HSS-A, IMS, LSS, PLAN 0, PLAN 90)."""
    return (int(fault_id) - 1) % GEARBOX_MEASUREMENTS_PER_EVENT + 1


def gearbox_component_of(fault_id):
    position = gearbox_position_of(fault_id)
    for name, positions in GEARBOX_COMPONENTS.items():
        if position in positions:
            return name
    raise ValueError(f"no component for position {position}")


def build_joint_event_rows(meta, features, feature_columns):
    """Strict joint rows: one per (event, Date) where all six measurements are aligned.

    Dates missing from any measurement of an event are dropped (counted in
    ``attrs['dropped_unaligned']``), never filled. Aligned dates whose targets
    disagree raise. Columns are the six measurement blocks in position order.
    """
    rows = []
    dropped = 0
    for event in sorted({gearbox_event_of(f) for f in meta["Falha"]}):
        falhas = [f for f in range(1, 19) if gearbox_event_of(f) == event]
        missing = [f for f in falhas if f not in set(meta["Falha"])]
        if missing:
            raise ValueError(f"event {event} incomplete: missing Falha {missing}")
        blocks = {f: meta.loc[meta["Falha"] == f].set_index("Date") for f in falhas}
        common = set.intersection(*[set(block.index) for block in blocks.values()])
        union = set.union(*[set(block.index) for block in blocks.values()])
        dropped += len(union) - len(common)
        for date in sorted(common):
            targets = [float(blocks[f].loc[date, "y"]) for f in falhas]
            if max(targets) - min(targets) > 1e-12:
                raise ValueError(f"event {event} at {date}: targets differ across measurements")
            phases = [str(blocks[f].loc[date, "phase"]) for f in falhas
                      if "phase" in blocks[f].columns]
            if phases and len(set(phases)) != 1:
                raise ValueError(f"event {event} at {date}: phases differ across measurements")
            row = {"event": event, "endpoint_id": f"EV{event:02d}-{pd.Timestamp(date).value}",
                   "Falha": event, "equipment": f"E{event}", "Date": date,
                   "phase": phases[0] if phases else "unknown", "y": targets[0]}
            for falha in falhas:
                endpoint_id = blocks[falha].loc[date, "endpoint_id"]
                position = gearbox_position_of(falha)
                for column in feature_columns:
                    row[f"m{position}__{column}"] = float(features.loc[endpoint_id, column])
            rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["dropped_unaligned"] = int(dropped)
    return result


def make_joint_event_track(meta, features, feature_columns):
    """Adapt strict synchronized rows into the standard OuterSearch metadata/table contract.

    ``feature_columns`` are the base per-series columns; six prefixed blocks (``m1__``..
    ``m6__``) are produced so the joint recipe keeps the full declared feature space.
    Known measurement one-hot metadata is excluded: it is a per-series strategy option.
    """
    feature_columns = [c for c in feature_columns if not c.startswith(MEASUREMENT_PREFIX)]
    rows = build_joint_event_rows(meta, features, feature_columns)
    required = {"endpoint_id", "Falha", "equipment", "Date", "phase", "y", "event"}
    if not required.issubset(rows.columns):
        raise ValueError("joint event rows lack required metadata")
    track_meta = rows[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]].copy()
    columns = [c for c in rows.columns if c.startswith(tuple(f"m{i}__" for i in range(1, 7)))]
    if len(columns) != 6 * len(feature_columns):
        raise ValueError("joint event track must contain six complete feature blocks")
    track_features = rows.set_index("endpoint_id")[columns]
    track_meta.attrs.update(rows.attrs)
    return track_meta.reset_index(drop=True), track_features


def aggregate_series_frame_to_events(frame, how):
    """Aggregate one event's six per-series predictions per (event, Date).

    The six measurements of an event timestamp must be complete and agree on the
    target; nothing is filled or forward-propagated.
    """
    if how not in ("mean", "median"):
        raise ValueError(f"unknown aggregation: {how}")
    work = frame[["Falha", "Date", "phase", "y", "pred_raw", "pred_post"]].copy()
    work["event"] = [gearbox_event_of(value) for value in work["Falha"]]
    work["position"] = [gearbox_position_of(value) for value in work["Falha"]]
    grouped = work.groupby(["event", "Date"], sort=True)
    if (grouped.size() != GEARBOX_MEASUREMENTS_PER_EVENT).any():
        raise ValueError("each event timestamp needs exactly six measurement predictions")
    rows = []
    for (event, date), block in grouped:
        if sorted(block["position"].tolist()) != list(range(1, 7)):
            raise ValueError(f"event {event} at {date}: incomplete measurement positions")
        if float(block["y"].max() - block["y"].min()) > 1e-12:
            raise ValueError(f"event {event} at {date}: targets differ across measurements")
        rows.append({
            "endpoint_id": f"EV{int(event):02d}-{pd.Timestamp(date).value}",
            "Falha": int(event), "equipment": f"E{int(event)}", "Date": pd.Timestamp(date),
            "phase": block["phase"].iloc[0], "y": float(block["y"].iloc[0]),
            "pred_raw": float(block["pred_raw"].mean() if how == "mean"
                              else block["pred_raw"].median()),
            "pred_post": float(block["pred_post"].mean() if how == "mean"
                               else block["pred_post"].median()),
        })
    return pd.DataFrame(rows)


def aggregate_measurement_predictions(predictions, how):
    """Mean or median of the six measurement predictions for each (event, Date)."""
    if how not in ("mean", "median"):
        raise ValueError(f"unknown aggregation: {how}")
    grouped = predictions.groupby(["event", "Date"], sort=True)
    if (grouped.size() != GEARBOX_MEASUREMENTS_PER_EVENT).any():
        raise ValueError("each event timestamp needs six measurement predictions")
    return grouped["pred_post"].agg(how).rename("pred_post").reset_index()


CANDIDATE_SCHEMA = 1
OOF_COLUMNS = ["endpoint_id", "Falha", "equipment", "Date", "phase", "y", "pred_raw", "pred_post"]


def candidate_key(recipe, trial, phase, fit_keys, inner_groups):
    """Identity of one search candidate: recipe, trial number, phase, fits and groups."""
    return sha256_text(canonical_json({
        "recipe": recipe, "trial": int(trial), "phase": phase,
        "fit_keys": list(fit_keys), "inner_groups": list(inner_groups)}))


class CandidateStore:
    """Atomic candidate payloads: inner OOF predictions, score and integrity metadata.

    A candidate is complete only when its single file exists and validates. Per-fit
    cache entries are not candidate completions.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    def _path(self, key):
        return self.directory / f"{key}.json"

    def save(self, key, result, inner_groups):
        if result.status != "ok" or result.oof is None:
            raise ValueError("only successful candidates are stored")
        oof_rows = result.oof.assign(
            Date=result.oof["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")).to_dict("records")
        payload = {
            "schema": CANDIDATE_SCHEMA, "candidate_key": key,
            "candidate_id": result.candidate_id, "family": result.family,
            "recipe": result.recipe, "trial": result.trial, "phase": result.phase,
            "inner_groups": list(inner_groups), "fit_keys": list(result.fit_keys),
            "score_pp": result.score_pp, "rmse_pp": result.rmse_pp,
            "complexity": result.complexity, "oof": oof_rows,
            "seed_offsets": list(result.seed_offsets), "stage_metrics": result.stage_metrics,
        }
        payload["payload_sha256"] = sha256_text(canonical_json(payload))
        write_json_atomic(self._path(key), payload)

    def load(self, key, inner_groups):
        """Return the stored candidate, None if absent; raise CorruptCacheError if invalid."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptCacheError(f"unreadable candidate {path.name}: {exc}") from exc
        stored = payload.pop("payload_sha256", None)
        if stored != sha256_text(canonical_json(payload)):
            raise CorruptCacheError(f"candidate {path.name}: payload hash mismatch")
        if payload.get("schema") != CANDIDATE_SCHEMA or payload.get("candidate_key") != key:
            raise CorruptCacheError(f"candidate {path.name}: schema or key mismatch")
        if payload.get("inner_groups") != list(inner_groups):
            raise CorruptCacheError(f"candidate {path.name}: inner groups differ")
        recipe = payload["recipe"]
        if recipe_id(recipe) != payload["candidate_id"]:
            raise CorruptCacheError(f"candidate {path.name}: recipe does not match its id")
        if candidate_key(recipe, payload["trial"], payload["phase"], payload["fit_keys"],
                         inner_groups) != key:
            raise CorruptCacheError(f"candidate {path.name}: recipe/fit identity mismatch")
        oof = pd.DataFrame(payload["oof"])
        if not payload["oof"] or set(oof.columns) != set(OOF_COLUMNS):
            raise CorruptCacheError(f"candidate {path.name}: missing or malformed predictions")
        oof = oof[OOF_COLUMNS]  # JSON writing sorts keys; the schema order is restored here
        oof["Date"] = pd.to_datetime(oof["Date"])
        oof["Falha"] = oof["Falha"].astype(int)
        for column in ("y", "pred_raw", "pred_post"):
            if not np.isfinite(oof[column].to_numpy(float)).all():
                raise CorruptCacheError(f"candidate {path.name}: non-finite {column}")
        if set(oof["equipment"]) != set(inner_groups):
            raise CorruptCacheError(f"candidate {path.name}: predictions do not cover the groups")
        post = recipe["postprocess"]
        recomputed = postprocess_predictions(oof, clip01=post["clip01"],
                                             ema_alpha=post["ema_alpha"],
                                             running_min=post["running_min"])
        if not np.allclose(recomputed["pred_post"].to_numpy(float),
                           oof["pred_post"].to_numpy(float), atol=1e-12):
            raise CorruptCacheError(f"candidate {path.name}: postprocessed predictions mismatch")
        score = inner_score(oof)
        if (abs(score["score_pp"] - payload["score_pp"]) > 1e-9
                or abs(score["rmse_pp"] - payload["rmse_pp"]) > 1e-9):
            raise CorruptCacheError(f"candidate {path.name}: stored score does not match")
        offsets = payload.get("seed_offsets") or []
        if offsets:
            metrics = payload.get("stage_metrics") or {}
            seed_scores = metrics.get("seed_scores_pp") or []
            if len(seed_scores) != len(offsets):
                raise CorruptCacheError(f"candidate {path.name}: missing per-seed scores")
            if abs(metrics.get("averaged_oof_score_pp", np.inf) - score["score_pp"]) > 1e-9:
                raise CorruptCacheError(f"candidate {path.name}: averaged-OOF score mismatch")
        return CandidateResult(
            candidate_id=payload["candidate_id"], family=payload["family"], recipe=recipe,
            trial=payload["trial"], phase=payload["phase"], status="ok", reasons=[],
            score_pp=payload["score_pp"], rmse_pp=score["rmse_pp"],
            complexity=payload["complexity"], oof=oof, fit_keys=payload["fit_keys"],
            seed_offsets=tuple(payload.get("seed_offsets", ())),
            stage_metrics=payload.get("stage_metrics", {}))


# ---------------------------------------------------------------------------
# Inner search for one outer fold (receives outer-training rows only).
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    candidate_id: str
    family: str
    recipe: dict
    trial: int
    phase: str
    status: str
    reasons: list
    score_pp: object
    rmse_pp: object
    complexity: int
    oof: object
    fit_keys: list
    key: str = ""
    seed_offsets: tuple = ()
    stage_metrics: dict = field(default_factory=dict)


@dataclass
class FrozenSearch:
    """Minimal search view rebuilt from a frozen selection (resume of later stages)."""
    states: dict


@dataclass
class FamilyState:
    family: str
    study: object = None
    proposed: int = 0
    failed: int = 0
    invalid: int = 0
    best: object = None
    anchor_score: object = None   # score at the last significant gain (cumulative anchor)
    last_improve_at: int = 0
    notes: list = field(default_factory=list)
    restored: bool = False


class ProposalJournal:
    """Integrity-checked pending batch and trusted-local Optuna sampler snapshot."""

    def __init__(self, directory, identity):
        self.directory = Path(directory) if directory is not None else None
        self.identity = identity

    def _paths(self, family):
        base = self.directory / family
        return (base.with_suffix(".pending.json"), base.with_suffix(".sampler.pkl"),
                base.with_suffix(".sampler.json"))

    def save(self, family, batch, sampler):
        if self.directory is None:
            return
        pending, sampler_path, state_path = self._paths(family)
        sampler_path.parent.mkdir(parents=True, exist_ok=True)
        data = pickle.dumps(sampler, protocol=pickle.HIGHEST_PROTOCOL)
        tmp = sampler_path.with_name(f".{sampler_path.name}.tmp-{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, sampler_path)
        state = {"identity": self.identity, "sampler_sha256": sha256_text(data.hex())}
        state["payload_sha256"] = sha256_text(canonical_json(state))
        write_json_atomic(state_path, state)
        payload = {"identity": self.identity, "batch": batch,
                   "sampler_sha256": sha256_text(data.hex())}
        payload["payload_sha256"] = sha256_text(canonical_json(payload))
        write_json_atomic(pending, payload)

    def load(self, family):
        if self.directory is None:
            return None
        pending, sampler_path, state_path = self._paths(family)
        if not state_path.exists():
            if pending.exists():
                raise CorruptCacheError(f"pending proposals for {family} lack sampler state")
            return None
        try:
            state = read_json(state_path)
            digest = state.pop("payload_sha256")
            if digest != sha256_text(canonical_json(state)) or state.get("identity") != self.identity:
                raise CorruptCacheError("sampler state identity/hash mismatch")
            data = sampler_path.read_bytes()
            if sha256_text(data.hex()) != state["sampler_sha256"]:
                raise CorruptCacheError("Optuna sampler snapshot hash mismatch")
            sampler = pickle.loads(data)  # trusted, locally generated artifact; hash checked
            if not pending.exists():
                return [], sampler
            payload = read_json(pending)
            digest = payload.pop("payload_sha256")
            if (digest != sha256_text(canonical_json(payload))
                    or payload.get("identity") != self.identity
                    or payload.get("sampler_sha256") != state["sampler_sha256"]):
                raise CorruptCacheError("pending proposal journal hash/identity mismatch")
            return payload["batch"], sampler
        except (OSError, json.JSONDecodeError, KeyError, pickle.UnpicklingError) as exc:
            raise CorruptCacheError(f"invalid pending proposal journal for {family}: {exc}") from exc

    def clear(self, family):
        if self.directory is None:
            return
        pending, _sampler, _state = self._paths(family)
        pending.unlink(missing_ok=True)


class OuterSearch:
    """Search for one outer fold. Targets of the outer test group are never supplied."""

    def __init__(self, *, dataset_name, outer_group, train_meta, tables, runner, config,
                 identity, candidates, study_dir=None, pilot_candidates=None,
                 forced_strategy=None, variance_key=None):
        if "y" not in train_meta.columns:
            raise ValueError("search requires training targets")
        if outer_group in set(train_meta["equipment"]):
            raise ValueError("outer group leaked into the search training rows")
        self.dataset_name = dataset_name
        self.outer_group = outer_group
        self.train_meta = train_meta.reset_index(drop=True)
        self.tables = tables            # callable(width, bins) -> table indexed by endpoint_id
        self.runner = runner
        self.config = config
        self.identity = identity
        self.candidates = candidates
        self.study_dir = Path(study_dir) if study_dir is not None else None
        self.journal = ProposalJournal(self.study_dir, {
            "dataset": dataset_name, "outer": outer_group,
            "identity": identity, "groups": self.train_meta["equipment"].drop_duplicates().tolist(),
            "config_sha256": sha256_text(canonical_json(config)),
        })
        self.tuning_enabled = bool(config["tuning_enabled"])
        self.pilot_candidates = pilot_candidates
        self.inner_groups = sorted(set(self.train_meta["equipment"]))
        if len(self.inner_groups) < 2:
            raise ValueError("inner leave-one-group-out needs at least two groups")
        self.n_train_min = min(int((self.train_meta["equipment"] != g).sum())
                               for g in self.inner_groups)
        self.states = OrderedDict()
        self.trials = []
        self.seed_namespace = config["seed_namespace"]
        self.group_sets = config["feature_group_sets"]
        self.forced_strategy = forced_strategy
        if variance_key is None:
            raise ConfigError("OuterSearch requires the dataset variance sensor key")
        self.variance_key = variance_key
        self.strategy_results = []
        self._strategy_note = "not run"

    # -- candidate evaluation -------------------------------------------------

    def _features_for(self, recipe):
        table = self.tables(recipe["features"]["width_hours"], recipe["features"]["context_bins"])
        return table.loc[self.train_meta["endpoint_id"].to_numpy()].set_axis(self.train_meta.index)

    def _check_recipe(self, recipe):
        """Raise InvalidRecipe for any combination the protocol does not allow."""
        validate_recipe(recipe, self.dataset_name, self.group_sets)
        table = self.tables(recipe["features"]["width_hours"], recipe["features"]["context_bins"])
        select_feature_columns(list(table.columns), recipe["features"], self.group_sets,
                               self.variance_key)
        return recipe

    def _evaluate(self, items, seed_offset=0):
        """items: (trial, phase, recipe) tuples -> CandidateResult list in the same order.

        A completed candidate (valid payload on disk) is reused without any fit. Other
        candidates are fitted; only successful ones are committed to the candidate store.
        A recipe whose fold preparation is impossible is a failed candidate with a reason.
        """
        slots, pending_tasks = [], []
        for trial, phase, recipe in items:
            try:
                features = self._features_for(recipe)
                n_nominal = len(select_feature_columns(
                    list(features.columns), recipe["features"], self.group_sets,
                    self.variance_key))
                tasks = []
                for group in self.inner_groups:
                    fold_tasks = build_strategy_fold_tasks(
                        self.train_meta, features,
                        (self.train_meta["equipment"] != group).to_numpy(),
                        (self.train_meta["equipment"] == group).to_numpy(), recipe,
                        self.group_sets,
                        lambda expert, group=group: derive_seed(
                            self.seed_namespace, self.dataset_name, self.outer_group,
                            recipe["family"], group, expert, int(seed_offset)),
                        self.identity, self.config["budgets"]["fit_timeout_s"],
                        self.dataset_name, self.variance_key)
                    tasks.extend((group, expert, task) for expert, task in fold_tasks)
            except InvalidRecipe as exc:
                slots.append(CandidateResult(
                    recipe_id(recipe), recipe["family"], recipe, trial, phase, "failed",
                    [f"invalid recipe: {exc}"], None, None, recipe_complexity(recipe, 0),
                    None, []))
                continue
            key = candidate_key(recipe, trial, phase, [t.key for _, _, t in tasks], self.inner_groups)
            completed = self.candidates.load(key, self.inner_groups)
            if completed is not None:
                completed.key = key
                slots.append(completed)
                continue
            pending_tasks.extend(task for _, _, task in tasks)
            slots.append((key, trial, phase, recipe, tasks, n_nominal))
        records = self.runner.run(pending_tasks) if pending_tasks else {}
        results = []
        for slot in slots:
            if isinstance(slot, CandidateResult):
                results.append(slot)
                continue
            key, trial, phase, recipe, tasks, n_nominal = slot
            result = self._assemble(trial, phase, recipe, tasks, n_nominal, records)
            result.key = key
            if result.status == "ok":
                self.candidates.save(key, result, self.inner_groups)
            results.append(result)
        return results

    def _assemble(self, trial, phase, recipe, tasks, n_nominal, records):
        candidate_id = recipe_id(recipe)
        complexity = recipe_complexity(recipe, n_nominal)
        fit_keys = [task.key for _, _, task in tasks]
        failures = [f"{records[t.key].reason}" for _, _, t in tasks if records[t.key].status != "ok"]
        if failures:
            return CandidateResult(candidate_id, recipe["family"], recipe, trial, phase,
                                   "failed", failures, None, None, complexity, None, fit_keys)
        frames = []
        for _group, _expert, task in tasks:
            rows = self.train_meta.loc[self.train_meta["endpoint_id"].isin(task.eval_ids)]
            frame = rows[["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]].copy()
            frame["pred_raw"] = records[task.key].predictions
            frames.append(frame)
        post = recipe["postprocess"]
        oof = postprocess_predictions(pd.concat(frames, ignore_index=True),
                                      clip01=post["clip01"], ema_alpha=post["ema_alpha"],
                                      running_min=post["running_min"])
        score = inner_score(oof)
        return CandidateResult(candidate_id, recipe["family"], recipe, trial, phase, "ok", [],
                               score["score_pp"], score["rmse_pp"], complexity, oof, fit_keys)

    # -- proposals and stop rules ----------------------------------------------

    def _state(self, family):
        if family not in self.states:
            self.states[family] = FamilyState(family=family)
        return self.states[family]

    def _study(self, family, state):
        if state.study is not None:
            return state.study
        seed = derive_seed(self.seed_namespace, self.dataset_name, self.outer_group,
                           family, "tpe")
        name = f"{self.dataset_name}-{self.outer_group}-{family}"
        if self.study_dir is not None:
            self.study_dir.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{(self.study_dir / 'studies.sqlite3').resolve()}"
            state.study = optuna.create_study(
                study_name=name, storage=storage, load_if_exists=True, direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed))
        else:
            state.study = optuna.create_study(direction="minimize",
                                              sampler=optuna.samplers.TPESampler(seed=seed))
        return state.study

    def _restore_family(self, family, state):
        """Restore study counts, sampler state and pending proposals before any new ask.

        Returns the proposals that were already asked but never evaluated (RUNNING trials).
        Their persisted phase is preserved, so an interrupted tuning batch is processed
        as tuning instead of silently becoming screening.
        """
        if self.pilot_candidates is not None or family in CONSTANT_FAMILIES:
            if state.proposed > 0:
                return []
            recipe = (self.pilot_candidates[family] if self.pilot_candidates is not None
                      else normalize_recipe(family, {}, {}, self.config["default_recipe"]))
            return [{"number": 0, "trial": None, "recipe": recipe, "reason": None,
                     "phase": "screen"}]
        study = self._study(family, state)
        if not state.restored and self.study_dir is not None:
            for frozen in study.get_trials(deepcopy=False):
                if frozen.state == optuna.trial.TrialState.RUNNING:
                    continue
                state.proposed += 1
                key = frozen.user_attrs.get("candidate_key")
                if key is None:
                    self._attribute_unsuccessful(state, frozen)
                    continue  # terminal trial without a payload; never needs one
                result = self.candidates.load(key, self.inner_groups)
                if result is None:
                    raise CorruptCacheError(f"completed SQLite trial {frozen.number} "
                                             "has no candidate payload")
                self.trials.append(result)
                self._note_result(state, result)
            state.restored = True
        pending = self.journal.load(family)
        if pending is not None:
            batch, sampler = pending
            study.sampler = sampler
        else:
            batch = None
        if not batch:
            return []
        recovered, kept = [], []
        frozen = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
        for item in batch:
            number = item["number"]
            if number not in frozen:
                raise CorruptCacheError(f"pending trial {number} is absent from SQLite")
            if frozen[number].state != optuna.trial.TrialState.RUNNING:
                key = frozen[number].user_attrs.get("candidate_key")
                if key is None:
                    # A crash after telling FAIL is not corruption: the terminal trial was
                    # already attributed above from its recorded metadata, and no candidate
                    # payload is required. It is simply dropped from the pending journal.
                    continue
                if self.candidates.load(key, self.inner_groups) is None:
                    raise CorruptCacheError(f"completed pending trial {number} lacks payload")
                continue  # told before the crash; never tell it twice
            trial = optuna.trial.Trial(study, frozen[number]._trial_id)
            recovered.append({"number": number, "trial": trial, "recipe": item["recipe"],
                              "reason": item.get("reason"),
                              "phase": item.get("phase", "screen")})
            kept.append(item)
        if recovered:
            self.journal.save(family, kept, study.sampler)
        else:
            self.journal.clear(family)
        return recovered

    @staticmethod
    def _attribute_unsuccessful(state, frozen):
        """Count a terminal unsuccessful trial from its own recorded metadata."""
        attrs = frozen.user_attrs
        if attrs.get("candidate_invalid"):
            state.invalid += 1
            return
        if attrs.get("candidate_failed"):
            state.failed += 1
            return
        if str(attrs.get("failure_reason", "")).startswith("invalid recipe:"):
            state.invalid += 1
            return
        state.failed += 1

    def _propose_new(self, family, state, count, phase, fixed_recipe=None):
        study = self._study(family, state)
        items = []
        for _ in range(count):
            trial = study.ask()
            try:
                if fixed_recipe is None:
                    recipe = sample_recipe(trial, family, self.config, self.dataset_name,
                                           self.n_train_min)
                else:
                    params = {name: _suggest(trial, f"{family}.{name}", spec, self.n_train_min)
                              for name, spec in self.config["families"][family]
                              .get("space", {}).items()}
                    recipe = copy.deepcopy(fixed_recipe)
                    recipe["params"] = normalize_params(family, params)
                recipe = self._check_recipe(recipe)
                items.append({"number": trial.number, "trial": trial, "recipe": recipe,
                              "reason": None, "phase": phase})
            except InvalidRecipe as exc:
                items.append({"number": trial.number, "trial": trial, "recipe": None,
                              "reason": str(exc), "phase": phase})
        if self.study_dir is not None:
            self.journal.save(family, [
                {"number": item["number"], "recipe": item["recipe"],
                 "reason": item["reason"], "phase": item["phase"], "order": order,
                 "seeds": {group: derive_seed(self.seed_namespace, self.dataset_name,
                                              self.outer_group, family, item["number"], group)
                           for group in self.inner_groups}}
                for order, item in enumerate(items)], study.sampler)
        return items

    def _note_result(self, state, result):
        """Update best/anchor from one evaluated candidate (inner scores only)."""
        if result.status == "ok":
            if state.best is None:
                state.anchor_score = result.score_pp
                state.last_improve_at = state.proposed
            elif result.score_pp < state.anchor_score - float(
                    self.config["protocol"]["min_gain_pp"]):
                # gains are cumulative against the anchor: 0.3 + 0.3 counts as 0.6
                state.anchor_score = result.score_pp
                state.last_improve_at = state.proposed
            if state.best is None or result.score_pp < state.best.score_pp:
                state.best = result
        else:
            state.failed += 1

    def _apply_items(self, family, state, items):
        """Evaluate proposed items, telling each trial exactly once."""
        valid, evaluated_phase = [], {}
        for item in items:
            state.proposed += 1
            if item["recipe"] is None:
                state.invalid += 1
                state.notes.append(f"invalid combination not evaluated: {item['reason']}")
                trial = item["trial"]
                if trial is not None:
                    trial.set_user_attr("candidate_invalid", True)
                    trial.set_user_attr("failure_reason", f"invalid recipe: {item['reason']}")
                    state.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    self._journal_forget(family, state, item["number"])
                continue
            valid.append(item)
            evaluated_phase[item["number"]] = item["phase"]
        if not valid:
            return
        results = self._evaluate([(item["number"], item["phase"], item["recipe"])
                                  for item in valid])
        for item, result in zip(valid, results):
            self.trials.append(result)
            trial = item["trial"]
            if trial is not None:
                if result.key and result.status == "ok":
                    trial.set_user_attr("candidate_key", result.key)
                    state.study.tell(trial, result.score_pp)
                else:
                    trial.set_user_attr("candidate_failed", True)
                    trial.set_user_attr("failure_reason", "; ".join(result.reasons)[:400])
                    state.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                self._journal_forget(family, state, item["number"])
            self._note_result(state, result)

    def _journal_forget(self, family, state, number):
        """Drop one terminal proposal from the pending journal (never keep it for replay)."""
        checkpoint = self.journal.load(family)
        if checkpoint is None:
            return
        batch, _sampler = checkpoint
        remaining = [entry for entry in batch if entry["number"] != number]
        if remaining:
            self.journal.save(family, remaining, state.study.sampler)
        else:
            self.journal.clear(family)

    def _run_family(self, family, max_total, phase, min_total=None, patience=None,
                    fixed_recipe=None):
        state = self._state(family)
        batch_size = int(self.config["stages"]["batch_size"])
        min_gain = float(self.config["protocol"]["min_gain_pp"])
        n_folds = len(self.inner_groups) * (6 if self.dataset_name == "gearbox" else 1)
        recovered = self._restore_family(family, state)
        if recovered:
            self._apply_items(family, state, recovered)
        while state.proposed < max_total:
            if (min_total is not None and patience is not None
                    and state.proposed >= min_total
                    and state.proposed - state.last_improve_at >= patience):
                state.notes.append(f"{phase}: stopped at {state.proposed} trials; no "
                                   f">= {min_gain:g} pp gain in the last {patience} proposals")
                break
            count = min(batch_size, max_total - state.proposed)
            if count < 1:
                break
            if self.runner.budget.remaining() < count * n_folds:
                state.notes.append(f"{phase}: stopped, fit budget would be exceeded")
                break
            items = self._propose_new(family, state, count, phase, fixed_recipe=fixed_recipe)
            if not items:
                break
            before = state.proposed
            self._apply_items(family, state, items)
            if state.proposed == before:  # defensive: never spin without consuming a proposal
                raise RuntimeError(f"{family}: {phase} proposed nothing")
        return state

    def _expansion_blocked(self, state):
        total = state.proposed - state.invalid
        limit = float(self.config["budgets"]["max_failed_fraction_for_expansion"])
        return total >= 4 and state.failed / total > limit

    def _bounded_recipe(self, family, params, strategy):
        recipe = normalize_recipe(family, params, {"strategy": strategy},
                                  self.config["default_recipe"])
        return self._check_recipe(recipe)

    def _bounded_batch(self, strategy, items, phase, trial_base, used):
        """Fit a deterministic recipe batch, skipping already-evaluated recipes."""
        recipes = []
        for family, params in items:
            recipe = self._bounded_recipe(family, params, strategy)
            signature = (family, canonical_json(recipe["params"]))
            if signature in used:
                continue
            used.add(signature)
            recipes.append(recipe)
        if not recipes:
            return []
        multiplier = strategy_fit_experts(strategy) if self.dataset_name == "gearbox" else 1
        if self.runner.budget.remaining() < len(recipes) * len(self.inner_groups) * multiplier:
            raise BudgetExceeded(f"{phase}: fit budget cannot cover {len(recipes)} recipes")
        items_eval = [(trial_base - index, phase, recipe)
                      for index, recipe in enumerate(recipes)]
        return self._evaluate(items_eval, seed_offset=0)

    def _screen_bounded(self, *, strategy, families, per_family, extra_per_family=0,
                        extra_top=0, reference_score=None, phase="bounded_screen",
                        trial_base=-7000):
        """Bounded deterministic screening of ``families`` for one strategy.

        Recipes come from a small lattice of the declared spaces (never more than
        ``per_family`` initially). With ``extra_top`` and ``extra_per_family`` the
        leading ``extra_top`` families receive additional bounded proposals, and only
        when they beat ``reference_score`` by the configured inner-gain threshold.
        """
        cfg = self.config
        min_gain = float(cfg["protocol"]["min_gain_pp"])
        used, best, notes = set(), {}, []
        base_items = [(family, params) for family in families
                      for params in bounded_family_recipes(
                          family, cfg["families"][family]["space"], per_family)]
        try:
            results = self._bounded_batch(strategy, base_items, phase, trial_base, used)
        except BudgetExceeded as exc:
            notes.append(str(exc))
            results = []
        for result in results:
            self.trials.append(result)
            if result.status == "ok":
                state = self._state(result.family)
                if state.best is None or result.score_pp < state.best.score_pp:
                    state.best = result
                current = best.get(result.family)
                if current is None or result.score_pp < current.score_pp:
                    best[result.family] = result
        expanded, extra_trials = [], 0
        if extra_top and extra_per_family:
            ranked = sorted(best.values(), key=lambda r: (r.score_pp, r.family))
            eligible = [r for r in ranked
                        if reference_score is None or r.score_pp <= reference_score - min_gain]
            for result in eligible[:int(extra_top)]:
                params_seen = bounded_family_recipes(
                    result.family, cfg["families"][result.family]["space"], per_family)
                wanted = bounded_family_recipes(
                    result.family, cfg["families"][result.family]["space"],
                    per_family + int(extra_per_family))
                extras = [p for p in wanted if p not in params_seen]
                try:
                    more = self._bounded_batch(
                        strategy, [(result.family, p) for p in extras],
                        f"{phase}_expand", trial_base - 500 - extra_trials * 100, used)
                except BudgetExceeded as exc:
                    notes.append(str(exc))
                    break
                extra_trials += len(more)
                for candidate in more:
                    self.trials.append(candidate)
                    if candidate.status == "ok":
                        state = self._state(candidate.family)
                        if state.best is None or candidate.score_pp < state.best.score_pp:
                            state.best = candidate
                        current = best.get(candidate.family)
                        if current is None or candidate.score_pp < current.score_pp:
                            best[candidate.family] = candidate
                expanded.append(result.family)
        best_result = min(best.values(), key=lambda r: (r.score_pp, r.family)) if best else None
        return {"strategy": strategy, "best": best, "best_result": best_result,
                "expanded": expanded, "extra_trials": extra_trials, "notes": notes}

    def _screen_gearbox_strategies(self):
        """Bounded per-strategy screening for the gearbox series track (pooled reference)."""
        cfg = self.config
        min_gain = float(cfg["protocol"]["min_gain_pp"])
        paired = tuple(family for family in SPECIALIST_FAMILIES
                       if family in cfg["families"])
        per_family = int(cfg["stages"]["gearbox_strategy_recipes_per_family"])
        extra_per_family = int(cfg["stages"]["gearbox_strategy_extra_per_family"])
        top_strategies = int(cfg["stages"]["gearbox_strategy_extra_top"])
        pooled = [s.best for s in self.states.values() if s.best is not None]
        if not pooled:
            self._strategy_note = "not run: no pooled reference candidate"
            return
        reference = min(c.score_pp for c in pooled)
        results, summary = {}, []
        screening = self.config["gearbox_strategies"].get("screening", list(SERIES_STRATEGIES))
        unknown = [name for name in screening if name not in SERIES_STRATEGIES]
        if unknown:
            raise ConfigError(f"unknown gearbox screening strategy: {unknown}")
        if "pooled_anonymous" not in screening:
            raise ConfigError("pooled_anonymous is the per-series reference and must be screened")
        for index, strategy in enumerate(screening):
            screened = self._screen_bounded(
                strategy=strategy, families=paired, per_family=per_family,
                reference_score=None, phase="strategy_screen",
                trial_base=-8000 - index * 1000)
            results[strategy] = screened
            best = screened["best_result"]
            summary.append({
                "strategy": strategy,
                "best_score_pp": None if best is None else best.score_pp,
                "best_family": None if best is None else best.family,
                "best_recipe": None if best is None else best.recipe,
                "gain_vs_pooled_pp": None if best is None else reference - best.score_pp,
                "notes": screened["notes"],
            })
        eligible = [row for row in summary
                    if row["strategy"] != "pooled_anonymous" and row["best_score_pp"] is not None
                    and row["gain_vs_pooled_pp"] >= min_gain]
        eligible.sort(key=lambda row: (row["best_score_pp"], row["strategy"]))
        expanded = []
        for position, row in enumerate(eligible[:top_strategies]):
            more = self._screen_bounded(
                strategy=row["strategy"], families=paired, per_family=per_family,
                extra_per_family=extra_per_family, extra_top=1,
                reference_score=reference, phase="strategy_expand",
                trial_base=-9000 - position * 1000)
            row["expanded_families"] = more["expanded"]
            row["extra_trials"] = more["extra_trials"]
            expanded.append(row["strategy"])
        self.strategy_results = summary
        self._strategy_note = (
            f"pooled reference {reference:.4f} pp; eligible {[r['strategy'] for r in eligible]}; "
            f"expanded {expanded}; per-strategy recipes <= {per_family} (+{extra_per_family} "
            f"per family for top {top_strategies})")

    def run_bounded(self, families=SPECIALIST_FAMILIES, per_family=2, extra_per_family=8,
                    extra_top=2):
        """Bounded family search used by the gearbox event track (no generic TPE stage)."""
        strategy = self.forced_strategy or "joint_event"
        reference = None
        if self.dataset_name == "gearbox":
            try:
                mean_recipe = self._bounded_recipe("mean", {}, strategy)
                [reference_result] = self._evaluate([(-6000, "bounded_reference", mean_recipe)])
                self.trials.append(reference_result)
                reference = reference_result.score_pp if reference_result.status == "ok" else None
            except BudgetExceeded:
                reference = None
        families = tuple(family for family in families if family in self.config["families"])
        self._screen_bounded(strategy=strategy, families=families, per_family=per_family,
                             extra_per_family=extra_per_family, extra_top=extra_top,
                             reference_score=reference, phase="bounded_screen",
                             trial_base=-7000)
        self._bounded_note = (f"bounded screen of {list(families)} with <={per_family} recipes "
                              f"per family; reference {reference}")
        return self._select()

    def _ablate_best(self):
        """Evaluate a bounded, one-axis-at-a-time inner-CV neighborhood of the best recipe."""
        stages = self.config["stages"]
        best = select_best([_family_row(s.best) for s in self.states.values()
                            if s.best is not None],
                           self.config["protocol"]["tie_tolerance_pp"])
        if best is None:
            return
        baseline_score = best["score_pp"]
        recipe = best["recipe"]
        hypotheses = [
            ("features.redundancy", "legacy" if recipe["features"]["redundancy"] == "dedup" else "dedup"),
            ("features.feature_groups", "sensors" if recipe["features"]["feature_groups"] == "sensors+cadence" else "sensors+cadence"),
            ("features.relative_reference", 10 if recipe["features"]["relative_reference"] is None else None),
            ("features.drop_variance", not recipe["features"]["drop_variance"]),
            ("postprocess.clip01", not recipe["postprocess"]["clip01"]),
            ("postprocess.ema_alpha", 0.5 if recipe["postprocess"]["ema_alpha"] != 0.5 else 1.0),
            ("postprocess.running_min", not recipe["postprocess"]["running_min"]),
            ("features.width_hours", 6 if recipe["features"]["width_hours"] != 6 else 24),
            ("features.width_hours", 24 if recipe["features"]["width_hours"] != 24 else 6),
            ("features.context_bins", 16 if recipe["features"]["context_bins"] != 16 else 64),
            ("features.context_bins", 64 if recipe["features"]["context_bins"] != 64 else 16),
            ("preprocess.method", "robust" if recipe["preprocess"]["method"] != "robust" else "standard"),
        ]
        variants, seen = [], set()
        for axis, value in hypotheses[:int(stages["ablation_max"])]:
            candidate = copy.deepcopy(recipe)
            _set_path(candidate, axis, value)
            signature = canonical_json(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            try:
                self._check_recipe(candidate)
            except InvalidRecipe:
                continue
            variants.append(candidate)
        for offset in range(0, len(variants), int(stages["batch_size"])):
            batch = variants[offset:offset + int(stages["batch_size"])]
            multiplier = 6 if self.dataset_name == "gearbox" else 1
            if self.runner.budget.remaining() < len(batch) * len(self.inner_groups) * multiplier:
                break
            items = [(-1 - offset - i, "ablation", candidate)
                     for i, candidate in enumerate(batch)]
            for result in self._evaluate(items, seed_offset=0):
                self.trials.append(result)
                if result.status == "ok":
                    state = self._state(result.family)
                    if state.best is None or result.score_pp < state.best.score_pp:
                        state.best = result
        self._ablation_note = f"evaluated {min(len(variants), offset + len(batch)) if variants else 0} " \
            f"one-axis variants; configured cap {stages['ablation_max']}"
        winner = select_best([_family_row(s.best) for s in self.states.values()
                              if s.best is not None],
                             self.config["protocol"]["tie_tolerance_pp"])
        self._ablation_gain = (winner["family"], baseline_score - winner["score_pp"])

    def _retune_ablation_winner(self):
        """Retune only the winning family parameters when ablation gain reaches threshold."""
        gain = getattr(self, "_ablation_gain", None)
        if gain is None:
            return
        family, improvement = gain
        minimum = float(self.config["stages"]["ablation_min_gain_pp"])
        if family in CONSTANT_FAMILIES:
            self._retune_note = "not run: constant reference has no hyperparameters"
            return
        if improvement < minimum:
            self._retune_note = f"not run: ablation gain {improvement:.4f} pp < {minimum:.4f} pp"
            return
        state = self._state(family)
        limit = int(self.config["stages"]["ablation_retune_trials"])
        start = state.proposed
        self._run_family(family, start + limit, "ablation_retune",
                         fixed_recipe=state.best.recipe)
        self._retune_note = f"retuned {family} parameters only for up to {limit} proposals"

    def _legacy_controls(self):
        """Evaluate the declared historical presets under this run's inner LOGO folds.

        These are descriptive controls, evaluated with the same inner protocol as the
        candidates but scored outside the search selection. Nothing is parsed from
        notebooks. ``features.redundancy = legacy`` keeps the duplicated count/absence
        columns of the original presets.
        """
        declaration = self.config.get("legacy_quick_control", {})
        defaults = declaration.get("defaults", {})
        recipes, named, unavailable = [], [], []
        for index, entry in enumerate(declaration.get("recipes", [])):
            family = entry["family"]
            if family not in self.config["families"]:
                continue
            features = {
                "features.representation": entry["representation"],
                "features.redundancy": defaults["redundancy"],
                "features.width_hours": defaults["width_hours"],
                "features.context_bins": defaults["context_bins"],
                "features.legacy_control_columns": True,
                "preprocess.method": defaults["normalization"],
                "preprocess.missing_indicators": defaults["missing_indicators"],
                "postprocess.clip01": defaults["clip_output"],
                "postprocess.ema_alpha": 1.0,
                "postprocess.running_min": False,
                "weights": entry["weights"],
            }
            recipe = normalize_recipe(family, entry["params"], features,
                                      self.config["default_recipe"])
            try:
                self._check_recipe(recipe)
            except InvalidRecipe as exc:
                unavailable.append({"name": entry["name"], "family": family,
                                    "status": "unavailable", "reason": str(exc)})
                continue
            recipes.append(recipe)
            named.append({"name": entry["name"], "family": family,
                          "weights": entry["weights"],
                          "representation": entry["representation"]})
        evaluated = []
        if recipes:
            multiplier = 6 if self.dataset_name == "gearbox" else 1
            needed = len(recipes) * len(self.inner_groups) * multiplier
            if self.runner.budget.remaining() >= needed:
                results = self._evaluate(
                    [(-1000 - index, "legacy_control", recipe)
                     for index, recipe in enumerate(recipes)], seed_offset=0)
                for info, result in zip(named, results):
                    self.trials.append(result)
                    evaluated.append({**info, "status": result.status,
                                      "score_pp": result.score_pp,
                                      "rmse_pp": result.rmse_pp,
                                      "reasons": result.reasons,
                                      "candidate_id": result.candidate_id})
            else:
                self._legacy_note = "not run: fit budget insufficient for the controls"
        self._legacy_results = evaluated + unavailable
        if not hasattr(self, "_legacy_note"):
            self._legacy_note = (f"evaluated {len(evaluated)} declared legacy quick presets "
                                 "under the current inner LOGO (descriptive control, not the "
                                 "historical protocol)")

    def _confirm_seeds(self):
        """Confirm the two strongest stochastic learned families with fixed offsets 0,1,2."""
        stochastic = {"extra_trees", "random_forest", "catboost"}
        ranked = sorted((state for state in self.states.values()
                         if state.best is not None and state.family in stochastic),
                        key=lambda state: (state.best.score_pp, state.family))
        confirmed = []
        for index, state in enumerate(ranked[:2]):
            base = state.best
            trials = [-3000 - index * 3 - offset for offset in range(3)]
            results = [self._evaluate([(trial, "seed_confirmation", base.recipe)],
                                      seed_offset=offset)[0]
                       for offset, trial in enumerate(trials)]
            if any(result.status != "ok" for result in results):
                state.notes.append("three-seed confirmation failed; original recipe retained")
                continue
            aligned = [r.oof.set_index("endpoint_id").loc[base.oof["endpoint_id"]]
                       for r in results]
            mean_raw = mean_seed_predictions([r["pred_raw"].to_numpy(float) for r in aligned])
            oof = base.oof.copy()
            oof["pred_raw"] = mean_raw
            post = base.recipe["postprocess"]
            oof = postprocess_predictions(oof, clip01=post["clip01"],
                                          ema_alpha=post["ema_alpha"],
                                          running_min=post["running_min"])
            seed_scores = [r.score_pp for r in results]
            score = inner_score(oof)
            fit_keys = [key for result in results for key in result.fit_keys]
            trial = -3100 - index
            key = candidate_key(base.recipe, trial, "seed_confirmation", fit_keys,
                                self.inner_groups)
            combined = CandidateResult(
                base.candidate_id, base.family, base.recipe, trial, "seed_confirmation", "ok", [],
                score["score_pp"], score["rmse_pp"], base.complexity, oof, fit_keys,
                key=key, seed_offsets=(0, 1, 2),
                stage_metrics={"seed_scores_pp": seed_scores,
                               "seed_score_mean_pp": float(np.mean(seed_scores)),
                               "seed_score_std_pp": float(np.std(seed_scores, ddof=0)),
                               "averaged_oof_score_pp": score["score_pp"]})
            self.candidates.save(key, combined, self.inner_groups)
            self.trials.append(combined)
            state.best = combined
            confirmed.append(state.family)
        self._seed_confirmation_note = f"confirmed {confirmed} with fixed offsets [0,1,2]"

    def run(self):
        """Screening, tuning of the top families, uniform ensembles, then selection."""
        stages = self.config["stages"]
        for family in self.config["families"]:
            if self.pilot_candidates is not None or family in CONSTANT_FAMILIES:
                self._run_family(family, 1, "screen")
            else:
                self._run_family(family, int(stages["screen_trials"]), "screen")
        self._legacy_controls()
        if self.tuning_enabled:
            ranked = sorted((s for s in self.states.values()
                             if s.best is not None and s.family not in CONSTANT_FAMILIES),
                            key=lambda s: (s.best.score_pp, s.family))
            total = int(stages["screen_trials"]) + int(stages["tune_extra_trials"])
            for state in ranked[:int(stages["tune_top_k"])]:
                if self._expansion_blocked(state):
                    state.notes.append("expansion blocked: failure fraction above limit")
                    continue
                self._run_family(state.family, total, "tune",
                                 min_total=int(stages["tune_min_total_trials"]),
                                 patience=int(stages["tune_patience_trials"]))
            self._ablate_best()
            self._retune_ablation_winner()
        if self.dataset_name == "gearbox" and self.pilot_candidates is None:
            self._screen_gearbox_strategies()
        if self.tuning_enabled:
            # Confirmation runs last so the winning stochastic specialist is confirmed too.
            self._confirm_seeds()
        return self._select()

    def _select(self):
        stages = self.config["stages"]
        tolerance = float(self.config["protocol"]["tie_tolerance_pp"])
        family_best = [s.best for s in self.states.values() if s.best is not None]
        learned = [c for c in family_best if c.family not in CONSTANT_FAMILIES]
        ranked = sorted(learned, key=lambda c: (c.score_pp, c.family))
        ensembles = []
        for size in stages["ensemble_sizes"]:
            members = ranked[:int(size)]
            if len(members) < int(size):
                continue
            for how in stages["ensemble_how"]:
                if how == "median" and int(size) == 2:
                    continue  # the median of two members equals their mean: not a new candidate
                oof = ensemble_predictions([m.oof for m in members], how)
                score = inner_score(oof)
                ensembles.append({
                    "candidate_id": sha256_text(canonical_json(
                        {"members": [m.candidate_id for m in members], "how": how}))[:16],
                    "kind": "ensemble", "how": how, "members": [m.candidate_id for m in members],
                    "member_families": [m.family for m in members],
                    "score_pp": score["score_pp"], "rmse_pp": score["rmse_pp"],
                    "complexity": sum(m.complexity for m in members)})
        family_rows = [_family_row(c) for c in family_best]
        chosen = select_best(family_rows + ensembles, tolerance)
        if chosen is None:
            raise ValueError(f"no valid candidate for outer {self.outer_group}")
        return {
            "chosen": chosen,
            "family_best": family_rows,
            "ensembles": ensembles,
            "family_notes": {s.family: {"proposed": s.proposed, "failed": s.failed,
                                        "invalid": s.invalid, "notes": s.notes}
                             for s in self.states.values()},
            "ablation_note": getattr(self, "_ablation_note", "not run"),
            "legacy_control_note": getattr(self, "_legacy_note", "not run"),
            "seed_confirmation_note": getattr(self, "_seed_confirmation_note", "not run"),
            "retune_note": getattr(self, "_retune_note", "not run"),
            "strategy_note": getattr(self, "_strategy_note", "not run"),
            "strategies": getattr(self, "strategy_results", []),
            "bounded_note": getattr(self, "_bounded_note", "not run"),
            "legacy_controls": getattr(self, "_legacy_results", []),
            "trials": [_trial_row(t) for t in self.trials],
        }


def _family_row(candidate):
    return {"candidate_id": candidate.candidate_id, "kind": "family",
            "family": candidate.family, "recipe": candidate.recipe,
            "score_pp": candidate.score_pp, "rmse_pp": candidate.rmse_pp,
            "complexity": candidate.complexity, "trial": candidate.trial,
            "seed_offsets": list(candidate.seed_offsets),
            "stage_metrics": candidate.stage_metrics}


def _trial_row(candidate):
    return {"trial": candidate.trial, "family": candidate.family, "phase": candidate.phase,
            "candidate_id": candidate.candidate_id, "status": candidate.status,
            "reasons": candidate.reasons, "score_pp": candidate.score_pp,
            "rmse_pp": candidate.rmse_pp, "complexity": candidate.complexity,
            "candidate_key": candidate.key, "fit_keys": candidate.fit_keys,
            "seed_offsets": list(candidate.seed_offsets),
            "stage_metrics": candidate.stage_metrics}


# ---------------------------------------------------------------------------
# Outer evaluation, metrics, verification, report.
# ---------------------------------------------------------------------------

PREDICTION_COLUMNS = ["candidate_id", "kind", "family", "endpoint_id", "Falha", "equipment",
                      "Date", "phase", "y_true", "pred_raw", "pred_post"]


def outer_metrics(frame):
    """Pooled and equal-per-component metrics in pp from one candidate's test frame."""
    y_pp = frame["y"].to_numpy(float) * PP
    pred_pp = frame["pred_post"].to_numpy(float) * PP
    block = cer.metric_breakdown(y_pp, pred_pp, frame["Falha"].to_numpy(int),
                                 frame["equipment"].to_numpy(str),
                                 frame["phase"].to_numpy(str), "pp")
    component_mae = [values["mae"] for values in block["by_component"].values()]
    raw = frame["pred_raw"].to_numpy(float)
    return {
        "n": block["overall"]["n"],
        "mae_pp": block["overall"]["mae"],
        "rmse_pp": block["overall"]["rmse"],
        "bias_pp": block["overall"]["bias"],
        "component_equal_mae_pp": float(np.mean(component_mae)),
        "worst_component_mae_pp": float(np.max(component_mae)),
        "by_component_mae_pp": {k: v["mae"] for k, v in block["by_component"].items()},
        "by_phase_mae_pp": {k: v["mae"] for k, v in block["by_phase"].items()},
        "raw_outside_unit_interval": int(np.sum((raw < 0) | (raw > 1))),
    }


def _ensure_frozen(selection_path, group):
    if not Path(selection_path).exists():
        raise ConfigError(f"outer {group}: no frozen selection; run search first")
    selection = read_json(selection_path)
    if not selection.get("frozen") or selection.get("outer_group") != group:
        raise ConfigError(f"outer {group}: selection file is not a valid frozen selection")
    if group in selection["search_train_groups"]:
        raise ConfigError(f"outer {group}: test group present in search training groups")
    return selection


class Run:
    """One dataset under one effective configuration; artifacts live in a fingerprint dir."""

    def __init__(self, dataset_name, *, pilot, config_path=None,
                 output_root=DEFAULT_OUTPUT_ROOT, data_dir=DATA_DIR):
        if dataset_name not in DATASET_NAMES:
            raise ConfigError(f"unknown dataset: {dataset_name}")
        if pilot and dataset_name not in PILOT_DATASETS:
            raise ConfigError(f"pilot is limited to {PILOT_DATASETS} in this task")
        self.dataset_name = dataset_name
        self.pilot = bool(pilot)
        self.raw_config = load_config(config_path)
        self.config = effective_config(self.raw_config, self.pilot)
        self.code_sha = code_sha256()
        self.dataset = load_dataset(dataset_name, self.config, data_dir)
        self.identity = self.dataset.identity(self.code_sha)
        self.unavailable = {}
        available, reason = catboost_status()
        if "catboost" in self.config["families"] and not available:
            del self.config["families"]["catboost"]
            self.unavailable["catboost"] = reason
        self.variance_key = variance_sensor_key(self.dataset.sensor_columns)
        self.versions = versions_info()
        self.fingerprint = sha256_text(canonical_json(
            {"config": self.config, "identity": self.identity,
             "versions": self.versions}))[:16]
        self.run_dir = Path(output_root) / dataset_name / self.fingerprint
        budgets = self.config["budgets"]
        # Worst-case final refits (specialist experts x seed refits) for both tracks.
        reserved = final_refit_reserve(len(self.dataset.outer_order), self.config,
                                       pilot=self.pilot)
        if reserved >= int(budgets["max_fits_per_dataset"]):
            raise ConfigError("final-refit reservation leaves no fits for the search")
        self.budget = FitBudget.load(self.run_dir / "ledger.json",
                                     int(budgets["max_fits_per_dataset"]), reserved)
        self.features = FeatureCache(self.dataset, self.config, self.identity,
                                     self.run_dir / "features")
        self.policy = ResourcePolicy(self.config["resources"])
        self.cache = FitCache(self.run_dir / "fits")
        self.runner = FitRunner(
            self.cache, self.budget, self.policy, timeout_s=budgets["fit_timeout_s"],
            poll_s=self.config["resources"]["poll_interval_s"],
            ledger_path=self.run_dir / "ledger.json",
            pause_poll_s=self.config["resources"]["pause_poll_s"],
            max_pause_s=self.config["resources"]["max_pause_s"])
        self.pilot_candidates = self._pilot_recipes() if self.pilot else None

    def _pilot_recipes(self):
        recipes = {}
        for family, fixed in self.config["pilot"]["fixed_candidates"].items():
            if family in self.config["families"]:
                recipe = normalize_recipe(family, fixed["params"], {},
                                          self.config["default_recipe"])
                validate_recipe(recipe, self.dataset_name, self.config["feature_group_sets"])
                recipes[family] = recipe
        return recipes

    def outer_dir(self, group):
        return self.run_dir / "outer" / group

    def event_dir(self, group):
        return self.outer_dir(group) / "event"

    def event_track(self):
        """Joint event meta (one row per fully aligned event timestamp)."""
        if getattr(self, "_event_meta", None) is None:
            if self.dataset_name != "gearbox":
                self._event_meta = None
            else:
                defaults = self.config["default_recipe"]["features"]
                base = self.features.table(defaults["width_hours"], defaults["context_bins"])
                meta, _features = make_joint_event_track(
                    self.dataset.meta, base, list(base.columns))
                expected = int(self.config["gearbox_strategies"]["joint_event"]
                               ["expected_endpoints"])
                if meta.attrs.get("dropped_unaligned", 0):
                    raise ConfigError(
                        f"joint event track drops {meta.attrs['dropped_unaligned']} unaligned "
                        "measurement rows; the six measurements must share timestamps and targets")
                if len(meta) != expected:
                    raise ConfigError(
                        f"joint event track has {len(meta)} aligned event endpoints, expected "
                        f"{expected}; the six measurements must share timestamps and targets")
                self._event_meta = meta
        return self._event_meta

    def _event_table_for(self, recipe):
        """Recipe-to-grid adapter for ``_outer_frames`` (mirrors ``_series_table_for``)."""
        features = recipe["features"]
        return self.event_table(features["width_hours"], features["context_bins"])

    def event_table(self, width_hours, context_bins):
        """Joint table for one grid: every recipe feature of all six measurements.

        Built from the same base feature table the per-series track uses, so declared
        representation, cadence, relative, robust and variance options all apply.
        """
        key = (int(width_hours), int(context_bins))
        if getattr(self, "_event_tables", None) is None:
            self._event_tables = {}
        if key not in self._event_tables:
            base = self.features.table(*key)
            _meta, features = make_joint_event_track(
                self.dataset.meta, base, list(base.columns))
            self._event_tables[key] = features
        return self._event_tables[key]

    def _event_aggregation_selection(self, search, selection):
        """Choose mean vs median of the six per-series inner OOF predictions (event track)."""
        chosen = selection["chosen"]
        members = (chosen["members"] if chosen.get("kind") == "ensemble"
                   else [chosen["candidate_id"]])
        by_id = {candidate.candidate_id: candidate
                 for candidate in (state.best for state in search.states.values())
                 if candidate is not None}
        missing = [member for member in members if member not in by_id]
        if missing:
            raise ValueError(f"aggregation source candidates missing: {missing}")
        combine_how = chosen.get("how", "mean") if chosen.get("kind") == "ensemble" else "mean"
        frames = self._combine_series_frames(members, {m: by_id[m].oof for m in members},
                                             combine_how)
        scores, problems = {}, []
        for how in ("mean", "median"):
            try:
                scores[how] = inner_score(aggregate_series_frame_to_events(frames, how))["score_pp"]
            except ValueError as exc:
                problems.append(f"{how}: {exc}")
        valid = {how: value for how, value in scores.items() if value is not None}
        how = min(valid, key=lambda key: (valid[key], key)) if valid else None
        return {"source_candidate_ids": members, "combine_how": combine_how, "how": how,
                "score_pp": None if how is None else valid[how],
                "scores_pp": scores, "problems": problems,
                "note": "totals are the six primary predictions aggregated per event timestamp"}

    @staticmethod
    def _combine_series_frames(members, frames, how):
        """Combine the chosen per-series pipeline exactly as selected (mean or median)."""
        if how not in ("mean", "median"):
            raise ValueError(f"unknown ensemble combination: {how}")
        base = frames[members[0]][["endpoint_id", "Falha", "Date", "phase", "y",
                                   "pred_raw", "pred_post"]].copy()
        if len(members) > 1:
            reducer = np.mean if how == "mean" else np.median
            for column in ("pred_raw", "pred_post"):
                base[column] = reducer(np.column_stack([
                    frames[m].set_index("endpoint_id").loc[base["endpoint_id"], column]
                    .to_numpy(float) for m in members]), axis=1)
        return base

    def _restore_selection_state(self, group, selection):
        """Rebuild the search view from frozen artifacts so later stages can resume."""
        trials_path = self.outer_dir(group) / "trials.json"
        if not trials_path.exists():
            raise CorruptCacheError(f"outer {group}: frozen selection without a trial log")
        trials = read_json(trials_path)["trials"]
        keys = {row["candidate_id"]: row.get("candidate_key") for row in trials}
        store = CandidateStore(self.outer_dir(group) / "candidates")
        states = OrderedDict()
        for row in selection["family_best"]:
            key = keys.get(row["candidate_id"])
            if key is None:
                raise CorruptCacheError(f"outer {group}: {row['candidate_id']} has no trial key")
            candidate = store.load(key, selection["inner_groups"])
            if candidate is None:
                raise CorruptCacheError(f"outer {group}: {row['candidate_id']} payload missing")
            states[candidate.family] = FamilyState(family=candidate.family, best=candidate)
        return FrozenSearch(states=states)

    def _search_event_track(self, group, search, selection):
        """Bounded joint-event search for one outer fold; never mixes with the series track."""
        if self.dataset_name != "gearbox":
            return
        e_meta = self.event_track()
        directory = self.event_dir(group)
        if (directory / "selection.json").exists():
            print(f"outer {group}: frozen event selection exists, skipped")
            return
        columns = ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]
        event_train = e_meta.loc[e_meta["equipment"] != group, columns].reset_index(drop=True)
        stages = self.config["stages"]
        event_search = OuterSearch(
            dataset_name=self.dataset_name, outer_group=group, train_meta=event_train,
            tables=self.event_table, runner=self.runner, config=self.config,
            identity={**self.identity, "track": "event"},
            candidates=CandidateStore(directory / "candidates"), study_dir=directory,
            forced_strategy="joint_event", variance_key=self.variance_key)
        result = event_search.run_bounded(
            families=SPECIALIST_FAMILIES,
            per_family=int(stages["gearbox_strategy_recipes_per_family"]),
            extra_per_family=int(stages["gearbox_strategy_extra_per_family"]),
            extra_top=int(stages["gearbox_strategy_extra_top"]))
        trials = result["trials"]
        write_json_atomic(directory / "trials.json", {"outer_group": group, "trials": trials,
                                                      "track": "event"})
        aggregation = self._event_aggregation_selection(search, selection)
        write_json_atomic(directory / "selection.json", {
            "dataset": self.dataset_name, "pilot": self.pilot, "track": "event",
            "availability": "one target per event timestamp; all six measurements aligned",
            "outer_group": group,
            "search_train_groups": sorted(set(event_train["equipment"])),
            "inner_groups": event_search.inner_groups,
            "identity": {**self.identity, "track": "event"},
            "trials_sha256": sha256_text(canonical_json(trials)),
            "joint": {key: result[key] for key in
                      ("chosen", "family_best", "ensembles", "family_notes",
                       "strategy_note", "bounded_note")},
            "aggregation": aggregation,
            "frozen": True,
        })
        chosen = result["chosen"]
        print(f"outer {group} event track: joint {chosen['candidate_id']} inner "
              f"{chosen['score_pp']:.4f} pp; aggregation {aggregation['how']} "
              f"inner {aggregation['score_pp']}")

    def _evaluate_event_track(self, group, selection, series_frames):
        """Evaluate the frozen event choices: joint model and six-prediction aggregation."""
        if self.dataset_name != "gearbox":
            return
        directory = self.event_dir(group)
        selection_path = directory / "selection.json"
        if not selection_path.exists():
            return
        event_selection = _ensure_frozen(selection_path, group)
        if event_selection.get("track") != "event":
            raise ConfigError(f"outer {group}: event selection has the wrong track")
        e_meta = self.event_track()
        selection_sha = sha256_file(selection_path)
        joint = event_selection["joint"]
        frames, statuses, diagnostics = self._outer_frames(
            e_meta, self._event_table_for, group, joint)
        rows, candidates = self._evaluation_rows(joint, frames)
        aggregation = event_selection["aggregation"]
        members = aggregation["source_candidate_ids"]
        if aggregation["how"] is not None and all(m in series_frames for m in members):
            combined = self._combine_series_frames(
                members, series_frames, aggregation.get("combine_how", "mean"))
            event_frame = aggregate_series_frame_to_events(combined, aggregation["how"])
            event_frame["pred_raw"] = np.nan
            candidates["aggregation"] = {
                "kind": "aggregation", "how": aggregation["how"], "members": members,
                "status": "ok", "metrics": outer_metrics(event_frame)}
            rows.append(_prediction_rows("aggregation", "aggregation", aggregation["how"],
                                         event_frame))
        else:
            statuses["aggregation"] = "failed: source predictions or choice unavailable"
        self._write_predictions(directory / "predictions.csv", rows)
        write_json_atomic(directory / "evaluation.json", {
            "track": "event", "unit": "pp", "outer_group": group,
            "selection_sha256": selection_sha,
            "chosen_candidate_id": joint["chosen"]["candidate_id"],
            "candidate_statuses": statuses,
            "seed_diagnostics": diagnostics,
            "candidates": candidates,
        })
        print(f"outer {group} event track: evaluated {len(candidates)} candidates")

    def write_manifest(self):
        write_json_atomic(self.run_dir / "manifest.json", {
            "fingerprint": self.fingerprint,
            "dataset": self.dataset_name,
            "pilot": self.pilot,
            "identity": self.identity,
            "csv_file": self.dataset.csv_path.name,
            "versions": self.versions,
            "unavailable_families": self.unavailable,
            "outer_order": list(self.dataset.outer_order),
            "families": sorted(self.config["families"]),
            "host": {
                "logical_cpus": psutil.cpu_count(logical=True),
                "total_ram_mib": int(psutil.virtual_memory().total / MIB),
                "initial_available_mib": int(self.policy.initial_available / MIB),
            },
            "effective_config": self.config,
        })

    def _write_stage_costs(self, stage):
        path = self.run_dir / "stage_costs.json"
        costs = read_json(path) if path.exists() else {}
        costs[stage] = {
            "runner": self.runner.stats.to_dict(),
            "budget": self.budget.to_dict(),
            "worker_estimate_mb": round(self.policy.estimate_bytes() / MIB, 1),
            "max_workers_used": self.runner.stats.max_concurrent,
            "feature_tables_built": self.features.builds,
            "feature_tables_from_disk": self.features.disk_hits,
            "resource_decisions": self.policy.decisions[-200:],
        }
        write_json_atomic(path, costs)

    def search(self):
        """Search each outer fold on outer-training rows only and freeze the choice."""
        self.write_manifest()
        meta = self.dataset.meta
        columns = ["endpoint_id", "Falha", "equipment", "Date", "phase", "y"]
        for group in self.dataset.outer_order:
            selection_path = self.outer_dir(group) / "selection.json"
            if selection_path.exists():
                print(f"outer {group}: frozen selection exists, skipped")
                selection = _ensure_frozen(selection_path, group)
                search = self._restore_selection_state(group, selection)
                # A crash after the primary freeze must not leave the event track missing.
                self._search_event_track(group, search, selection)
                continue
            train_meta = meta.loc[meta["equipment"] != group, columns]
            search = OuterSearch(
                dataset_name=self.dataset_name, outer_group=group, train_meta=train_meta,
                tables=self.features.table, runner=self.runner, config=self.config,
                identity=self.identity, candidates=CandidateStore(
                    self.outer_dir(group) / "candidates"),
                study_dir=self.outer_dir(group), pilot_candidates=self.pilot_candidates,
                variance_key=self.variance_key)
            result = search.run()
            trials = result["trials"]
            write_json_atomic(self.outer_dir(group) / "trials.json",
                              {"outer_group": group, "trials": trials})
            selection = {
                "dataset": self.dataset_name,
                "pilot": self.pilot,
                "outer_group": group,
                "search_train_groups": sorted(set(train_meta["equipment"])),
                "inner_groups": search.inner_groups,
                "identity": self.identity,
                "trials_sha256": sha256_text(canonical_json(trials)),
                "chosen": result["chosen"],
                "family_best": result["family_best"],
                "ensembles": result["ensembles"],
                "family_notes": result["family_notes"],
                "tuning": {"ablation": result["ablation_note"],
                           "legacy_control": result["legacy_control_note"],
                           "seed_confirmation": result["seed_confirmation_note"],
                           "retune": result["retune_note"],
                           "strategy": result["strategy_note"]},
                "strategies": result["strategies"],
                "legacy_controls": result["legacy_controls"],
                "frozen": True,
            }
            write_json_atomic(selection_path, selection)
            self._search_event_track(group, search, selection)
            print(f"outer {group}: chosen {result['chosen']['candidate_id']} "
                  f"({result['chosen'].get('family', result['chosen'].get('kind'))}) "
                  f"inner {result['chosen']['score_pp']:.4f} pp")
        self._write_stage_costs("search")

    def _outer_frames(self, meta, table_for, group, selection):
        """Refit the frozen selection on outer-training rows and predict the held-out group.

        Used for both availability tracks (per-series and joint event); ``table_for``
        resolves the feature table for a recipe. Returns frames, statuses, diagnostics.
        """
        group_sets = self.config["feature_group_sets"]
        train_mask = (meta["equipment"] != group).to_numpy()
        test_mask = ~train_mask
        seed_ns = self.config["seed_namespace"]
        tasks, recipes, tasks_by_candidate, offsets_by_candidate = [], OrderedDict(), {}, {}
        for row in selection["family_best"]:
            recipes[row["candidate_id"]] = row["recipe"]
        for candidate_id, recipe in recipes.items():
            table = table_for(recipe)
            features = table.loc[meta["endpoint_id"].to_numpy()].set_axis(meta.index)
            offsets = next((r.get("seed_offsets", []) for r in selection["family_best"]
                            if r["candidate_id"] == candidate_id), []) or [0]
            offsets_by_candidate[candidate_id] = offsets
            current_tasks = []
            for offset in offsets:
                strategy_tasks = build_strategy_fold_tasks(
                    meta, features, train_mask, test_mask, recipe, group_sets,
                    lambda expert, offset=offset: derive_seed(
                        seed_ns, self.dataset_name, group, recipe["family"],
                        expert, int(offset)),
                    self.identity, self.config["budgets"]["fit_timeout_s"],
                    self.dataset_name, self.variance_key)
                current_tasks.extend(task for _, task in strategy_tasks)
            tasks.extend(current_tasks)
            tasks_by_candidate[candidate_id] = current_tasks
        records = self.runner.run(tasks, purpose="final")
        frames, statuses, diagnostics = OrderedDict(), {}, {}
        for candidate_id, recipe in recipes.items():
            current_tasks = tasks_by_candidate[candidate_id]
            failed = next((records[t.key] for t in current_tasks
                           if records[t.key].status != "ok"), None)
            if failed is not None:
                statuses[candidate_id] = f"failed: {failed.reason}"
                continue
            seed_frames = []
            for offset in offsets_by_candidate[candidate_id]:
                strategy_tasks = build_strategy_fold_tasks(
                    meta, table_for(recipe), train_mask, test_mask, recipe, group_sets,
                    lambda expert, offset=offset: derive_seed(
                        seed_ns, self.dataset_name, group, recipe["family"],
                        expert, int(offset)),
                    self.identity, self.config["budgets"]["fit_timeout_s"],
                    self.dataset_name, self.variance_key)
                offset_frames = []
                for _expert, task in strategy_tasks:
                    rows = meta.loc[meta["endpoint_id"].isin(task.eval_ids),
                                    ["endpoint_id", "Falha", "equipment", "Date", "phase",
                                     "y"]].copy()
                    rows["pred_raw"] = records[task.key].predictions
                    offset_frames.append(rows)
                offset_frame = pd.concat(offset_frames, ignore_index=True)
                if offset_frame["endpoint_id"].duplicated().any():
                    raise ValueError("strategy produced duplicate outer endpoint predictions")
                seed_frames.append(offset_frame.set_index("endpoint_id").loc[
                    meta.loc[test_mask, "endpoint_id"].to_numpy()].reset_index())
            frame = seed_frames[0].copy()
            if len(seed_frames) > 1:
                aligned = np.column_stack([f["pred_raw"].to_numpy(float) for f in seed_frames])
                frame["pred_raw"] = mean_seed_predictions(
                    [f["pred_raw"].to_numpy(float) for f in seed_frames])
                diagnostics[candidate_id] = {
                    "offsets": offsets_by_candidate[candidate_id],
                    "prediction_std_pp_mean": float(np.mean(np.std(aligned, axis=1)) * PP),
                }
            post = recipe["postprocess"]
            frames[candidate_id] = postprocess_predictions(
                frame, clip01=post["clip01"], ema_alpha=post["ema_alpha"],
                running_min=post["running_min"])
            statuses[candidate_id] = "ok"
        all_frames = OrderedDict(frames)
        for ensemble in selection["ensembles"]:
            members = ensemble["members"]
            if not all(member in frames for member in members):
                statuses[ensemble["candidate_id"]] = "failed: member failed"
                continue
            frame = ensemble_predictions([frames[m] for m in members], ensemble["how"])
            frame["pred_raw"] = np.nan
            all_frames[ensemble["candidate_id"]] = frame
            statuses[ensemble["candidate_id"]] = "ok"
        return all_frames, statuses, diagnostics

    def _evaluation_rows(self, selection, frames):
        rows, candidates = [], OrderedDict()
        recipes = {row["candidate_id"]: row["recipe"] for row in selection["family_best"]}
        for candidate_id, frame in frames.items():
            if candidate_id in recipes:
                info = {"kind": "family", "family": recipes[candidate_id]["family"]}
            else:
                ensemble = next(e for e in selection["ensembles"]
                                if e["candidate_id"] == candidate_id)
                info = {"kind": "ensemble", "how": ensemble["how"],
                        "members": ensemble["members"]}
            candidates[candidate_id] = {**info, "status": "ok",
                                        "metrics": outer_metrics(frame)}
            rows.append(_prediction_rows(candidate_id, info["kind"],
                                         info.get("family", info.get("how")), frame))
        return rows, candidates

    def evaluate(self):
        """Refit frozen choices on outer training rows and predict the held-out group."""
        self.write_manifest()
        meta = self.dataset.meta
        for group in self.dataset.outer_order:
            selection_path = self.outer_dir(group) / "selection.json"
            selection = _ensure_frozen(selection_path, group)
            selection_sha = sha256_file(selection_path)
            frames, statuses, diagnostics = self._outer_frames(
                meta, self._series_table_for, group, selection)
            rows, candidates = self._evaluation_rows(selection, frames)
            self._write_predictions(self.outer_dir(group) / "predictions.csv", rows)
            write_json_atomic(self.outer_dir(group) / "evaluation.json", {
                "track": "series", "unit": "pp",
                "outer_group": group,
                "selection_sha256": selection_sha,
                "chosen_candidate_id": selection["chosen"]["candidate_id"],
                "candidate_statuses": statuses,
                "seed_diagnostics": diagnostics,
                "candidates": candidates,
            })
            print(f"outer {group}: evaluated {len(candidates)} candidates")
            self._evaluate_event_track(group, selection, frames)
        self._write_stage_costs("evaluate")

    def _series_table_for(self, recipe):
        return self.features.table(recipe["features"]["width_hours"],
                                   recipe["features"]["context_bins"])

    @staticmethod
    def _write_predictions(path, rows):
        frame = pd.concat(rows, ignore_index=True)[PREDICTION_COLUMNS] if rows else \
            pd.DataFrame(columns=PREDICTION_COLUMNS)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def verify(self):
        """Integrity checks plus per-outer states (search and evaluation reported apart).

        Status is ``evaluated_complete`` only when every outer fold is evaluated and every
        check passes. Pilot runs never report a final status.
        """
        checks = []

        def check(name, ok, detail=""):
            checks.append((name, bool(ok), detail))

        manifest_path = self.run_dir / "manifest.json"
        check("manifest present", manifest_path.exists())
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            check("fingerprint matches code/config/data", manifest["fingerprint"] == self.fingerprint)
            check("CSV hash unchanged", sha256_file(self.dataset.csv_path)
                  == manifest["identity"]["csv_sha256"], self.dataset.csv_path.name)
        budget = FitBudget.load(self.run_dir / "ledger.json",
                                int(self.config["budgets"]["max_fits_per_dataset"]),
                                final_refit_reserve(len(self.dataset.outer_order),
                                                    self.config, pilot=self.pilot))
        check("fit budget not exceeded", budget.charged <= budget.max_fits,
              f"{budget.charged}/{budget.max_fits}")
        check("fit cache entries valid", self._cache_checks())
        meta = self.dataset.meta
        outers = OrderedDict()
        for group in self.dataset.outer_order:
            outer = self.outer_dir(group)
            selection_path = outer / "selection.json"
            if not selection_path.exists():
                outers[group] = "search_pending"
                continue
            selection = read_json(selection_path)
            expected_train = sorted(set(meta["equipment"]) - {group})
            check(f"{group}: search groups exclude test group",
                  selection["search_train_groups"] == expected_train)
            trials = read_json(outer / "trials.json")["trials"]
            check(f"{group}: trial log matches selection hash",
                  sha256_text(canonical_json(trials)) == selection["trials_sha256"])
            self._check_candidates(group, trials, check)
            chosen = selection["chosen"]
            family_rows = {r["candidate_id"]: r for r in selection["family_best"]}
            known = set(family_rows) | {e["candidate_id"] for e in selection["ensembles"]}
            check(f"{group}: chosen candidate exists", chosen["candidate_id"] in known)
            if chosen["kind"] == "family":
                check(f"{group}: chosen recipe id consistent",
                      recipe_id(family_rows[chosen["candidate_id"]]["recipe"])
                      == chosen["candidate_id"])
            outers[group] = "search_complete"
            evaluation_path = outer / "evaluation.json"
            if not evaluation_path.exists():
                continue
            evaluation = read_json(evaluation_path)
            check(f"{group}: evaluation used the frozen selection",
                  evaluation["selection_sha256"] == sha256_file(selection_path))
            self._check_predictions(group, selection, evaluation, check)
            outers[group] = "evaluated_complete"
            before = len(checks)
            self._check_event_track(group, check)
            if any(not ok for _name, ok, _detail in checks[before:]):
                outers[group] = "event_incomplete"
        failed = any(not ok for _name, ok, _detail in checks)
        states = set(outers.values())
        if failed:
            status = "failed"
        elif states == {"evaluated_complete"}:
            status = "pilot_evaluated_complete" if self.pilot else "evaluated_complete"
        elif states <= {"search_complete", "evaluated_complete"} and "search_pending" not in states:
            status = "pilot_search_complete" if self.pilot else "search_complete_not_evaluated"
        else:
            status = "incomplete"
        return {"checks": checks, "status": status, "outers": outers}

    def _check_candidates(self, group, trials, check):
        store = CandidateStore(self.outer_dir(group) / "candidates")
        bad = []
        for row in trials:
            if row["status"] != "ok":
                continue
            try:
                loaded = store.load(row["candidate_key"], self._inner_groups(group))
            except CorruptCacheError as exc:
                bad.append(str(exc))
                continue
            if loaded is None:
                bad.append(f"missing payload {row['candidate_key'][:12]}")
        check(f"{group}: every successful candidate has a valid payload", not bad,
              "; ".join(bad[:3]))

    def _inner_groups(self, group):
        return sorted(set(self.dataset.meta["equipment"]) - {group})

    def _check_event_track(self, group, check):
        """Event track checks: frozen choice, alignment, separate availability."""
        if self.dataset_name != "gearbox":
            return
        directory = self.event_dir(group)
        selection_path = directory / "selection.json"
        required = not self.pilot  # a full gearbox run always has both availability tracks
        if not selection_path.exists():
            check(f"{group}/event: selection present", not required,
                  "event track not searched")
            return
        event_selection = read_json(selection_path)
        check(f"{group}/event: frozen and marked as its own track",
              event_selection.get("frozen") is True
              and event_selection.get("track") == "event"
              and (event_selection.get("identity") or {}).get("track") == "event")
        e_meta = self.event_track()
        expected_train = sorted(set(e_meta["equipment"]) - {group})
        check(f"{group}/event: search groups exclude the held event",
              event_selection["search_train_groups"] == expected_train)
        joint = event_selection["joint"]
        check(f"{group}/event: aggregation choice recorded",
              event_selection["aggregation"]["how"] in ("mean", "median")
              or event_selection["aggregation"]["problems"])
        self._check_event_candidates(group, directory, event_selection, check)
        evaluation_path = directory / "evaluation.json"
        if not evaluation_path.exists():
            check(f"{group}/event: evaluation present", not required,
                  "event track not evaluated")
            return
        evaluation = read_json(evaluation_path)
        check(f"{group}/event: evaluation used the frozen event selection",
              evaluation["selection_sha256"] == sha256_file(selection_path))
        self._check_predictions(group, joint, evaluation, check, meta=e_meta,
                                directory=directory, prefix="event/")

    def _check_event_candidates(self, group, directory, event_selection, check):
        """Event trials must hash-match and every chosen candidate must have a payload."""
        trials_path = directory / "trials.json"
        if not trials_path.exists():
            check(f"{group}/event: trial log present", False, "missing trials.json")
            return
        trials = read_json(trials_path)["trials"]
        check(f"{group}/event: trial log matches selection hash",
              sha256_text(canonical_json(trials)) == event_selection["trials_sha256"])
        joint = event_selection["joint"]
        store = CandidateStore(directory / "candidates")
        keys = {row["candidate_id"]: row.get("candidate_key") for row in trials}
        chosen = joint["chosen"]["candidate_id"]
        known = {row["candidate_id"] for row in joint["family_best"]} | \
            {row["candidate_id"] for row in joint["ensembles"]}
        check(f"{group}/event: chosen joint candidate exists", chosen in known)
        checked = 0
        for row in joint["family_best"]:
            key = keys.get(row["candidate_id"])
            ok = bool(key) and store.load(key, event_selection["inner_groups"]) is not None
            check(f"{group}/event: candidate payload for {row['candidate_id']}", ok)
            checked += 1
        check(f"{group}/event: candidate payloads were checked", checked > 0)

    def _cache_checks(self):
        directory = self.run_dir / "fits"
        if not directory.exists():
            return True
        for path in directory.glob("*.json"):
            payload = read_json(path)
            stored = payload.pop("payload_sha256", None)
            if payload.get("key") != path.stem or stored != sha256_text(canonical_json(payload)):
                return False
        return True

    def _check_predictions(self, group, selection, evaluation, check, *, meta=None,
                           directory=None, prefix=""):
        meta = self.dataset.meta if meta is None else meta
        test = meta.loc[meta["equipment"] == group]
        directory = self.outer_dir(group) if directory is None else directory
        frame = pd.read_csv(Path(directory) / "predictions.csv")
        group = f"{prefix}{group}"
        candidates = evaluation["candidates"]
        chosen = evaluation.get("chosen_candidate_id")
        check(f"{group}: at least one evaluated candidate", bool(candidates))
        check(f"{group}: chosen candidate was evaluated and succeeded",
              chosen in candidates and candidates.get(chosen, {}).get("status") == "ok")

        for candidate_id, info in candidates.items():
            part = frame.loc[frame["candidate_id"] == candidate_id]
            check(f"{group}/{candidate_id}: prediction endpoint ids are unique",
                  not part["endpoint_id"].duplicated().any())
            check(f"{group}/{candidate_id}: covers all test endpoints",
                  set(part["endpoint_id"].astype(str)) == set(test["endpoint_id"].astype(str)))
            merged = part.merge(test[["endpoint_id", "y"]].assign(
                endpoint_id=test["endpoint_id"].astype(str)), on="endpoint_id")
            check(f"{group}/{candidate_id}: targets match the data",
                  np.allclose(merged["y_true"], merged["y"], atol=1e-12))
            if info["kind"] == "family":
                recipe = next(r["recipe"] for r in selection["family_best"]
                              if r["candidate_id"] == candidate_id)
                post = recipe["postprocess"]
                recomputed = postprocess_predictions(
                    part.assign(pred_raw=part["pred_raw"].astype(float)),
                    clip01=post["clip01"], ema_alpha=post["ema_alpha"],
                    running_min=post["running_min"])
                check(f"{group}/{candidate_id}: postprocessing reproducible",
                      np.allclose(recomputed["pred_post"].to_numpy(float),
                                  part["pred_post"].to_numpy(float), atol=1e-12))
            metrics = outer_metrics(part.assign(
                phase=part["phase"].astype(str), equipment=part["equipment"].astype(str),
                y=part["y_true"].astype(float), pred_post=part["pred_post"].astype(float),
                pred_raw=part["pred_raw"].astype(float), Falha=part["Falha"].astype(int)))
            stored = info.get("metrics") or {}
            check(f"{group}/{candidate_id}: metrics are finite",
                  all(isinstance(stored.get(key), (int, float))
                      and np.isfinite(stored.get(key))
                      for key in ("mae_pp", "rmse_pp", "bias_pp")))
            check(f"{group}/{candidate_id}: stored metrics equal recomputed metrics",
                  all(abs(metrics[key] - stored.get(key, np.inf)) < 1e-9
                      for key in ("mae_pp", "rmse_pp", "bias_pp")))
            check(f"{group}/{candidate_id}: covers exactly the test endpoints",
                  len(part) == len(test) and set(part["endpoint_id"].astype(str))
                  == set(test["endpoint_id"].astype(str)))

    def _event_report_lines(self, group):
        """Markdown block for the event availability track (never merged with series rows)."""
        directory = self.event_dir(group)
        selection = read_json(directory / "selection.json")
        joint = selection["joint"]
        aggregation = selection["aggregation"]
        lines = ["", f"### Event track — outer {group}", "",
                 f"- availability: {selection['availability']}",
                 f"- joint candidate: `{joint['chosen']['candidate_id']}` "
                 f"({joint['chosen'].get('family', joint['chosen'].get('kind'))}), inner "
                 f"{joint['chosen']['score_pp']:.4f} pp",
                 f"- aggregation of the six primary predictions: {aggregation['how']} "
                 f"(inner {aggregation['score_pp']})",
                 "", "| joint family | inner pp | inner RMSE pp |", "|---|---:|---:|"]
        for row in joint["family_best"]:
            lines.append(f"| {row['family']} | {row['score_pp']:.4f} | {row['rmse_pp']:.4f} |")
        evaluation_path = directory / "evaluation.json"
        if evaluation_path.exists():
            evaluation = read_json(evaluation_path)
            lines += ["", "Descriptive event-level outer metrics (pp, not used for selection):",
                      "", "| candidate | kind | MAE pp | RMSE pp | bias pp |",
                      "|---|---|---:|---:|---:|"]
            for candidate_id, info in evaluation["candidates"].items():
                if info["status"] != "ok":
                    lines.append(f"| `{candidate_id}` | {info['kind']} | {info['status']} |||")
                    continue
                metrics = info["metrics"]
                lines.append(f"| `{candidate_id}` | {info['kind']} | {metrics['mae_pp']:.4f} | "
                             f"{metrics['rmse_pp']:.4f} | {metrics['bias_pp']:.4f} |")
        lines.append("")
        return lines

    def report(self):
        """Markdown summary built only from artifacts (no outer-based selection)."""
        lines = [f"# Baseline exploration: {self.dataset_name}"
                 + (" (pilot)" if self.pilot else ""), "",
                 f"- fingerprint: `{self.fingerprint}`",
                 f"- CSV SHA-256: `{self.identity['csv_sha256']}`",
                 f"- code SHA-256: `{self.identity['code_sha256']}`",
                 f"- versions: {json.dumps(self.versions, sort_keys=True)}",
                 f"- unavailable families: {json.dumps(self.unavailable, sort_keys=True) or 'none'}",
                 "", "Selection uses inner validation only; outer metrics are descriptive.",
                 ""]
        for group in self.dataset.outer_order:
            selection_path = self.outer_dir(group) / "selection.json"
            if not selection_path.exists():
                lines += [f"## Outer {group}", "", "not searched yet", ""]
                continue
            selection = read_json(selection_path)
            chosen = selection["chosen"]
            lines += [f"## Outer {group}", "",
                      f"- chosen by inner score: `{chosen['candidate_id']}` "
                      f"({chosen.get('family', chosen.get('kind'))}), inner "
                      f"{chosen['score_pp']:.4f} pp, complexity {chosen['complexity']}",
                      "", "| candidate | family | inner pp | inner RMSE pp | complexity |",
                      "|---|---|---:|---:|---:|"]
            for row in selection["family_best"]:
                lines.append(f"| `{row['candidate_id']}` | {row['family']} | "
                             f"{row['score_pp']:.4f} | {row['rmse_pp']:.4f} | {row['complexity']} |")
            for row in selection["ensembles"]:
                lines.append(f"| `{row['candidate_id']}` | ensemble {row['how']} | "
                             f"{row['score_pp']:.4f} | {row['rmse_pp']:.4f} | {row['complexity']} |")
            evaluation_path = self.outer_dir(group) / "evaluation.json"
            if evaluation_path.exists():
                evaluation = read_json(evaluation_path)
                lines += ["", "Descriptive outer metrics (pp, not used for selection):", "",
                          "| candidate | kind | MAE pp | component-equal MAE pp | RMSE pp | bias pp |",
                          "|---|---|---:|---:|---:|---:|"]
                for candidate_id, info in evaluation["candidates"].items():
                    if info["status"] != "ok":
                        lines.append(f"| `{candidate_id}` | {info['kind']} | {info['status']} |||||")
                        continue
                    m = info["metrics"]
                    lines.append(f"| `{candidate_id}` | {info['kind']} | {m['mae_pp']:.4f} | "
                                 f"{m['component_equal_mae_pp']:.4f} | {m['rmse_pp']:.4f} | "
                                 f"{m['bias_pp']:.4f} |")
            lines.append("")
            event_selection = self.event_dir(group) / "selection.json"
            if event_selection.exists():
                lines.extend(self._event_report_lines(group))
        lines += ["## Scope notes", "",
                  "- Primary track: one row per series endpoint (the 574/822 endpoints).",
                  "- Event track (gearbox): one target per event timestamp with the six "
                  "measurements aligned; reported separately and never ranked against the "
                  "per-series track.",
                  "- This file lives in the ignored run directory; versioned reports belong "
                  "to the later tasks."]
        path = self.run_dir / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _prediction_rows(candidate_id, kind, family, frame):
    rows = pd.DataFrame({
        "candidate_id": candidate_id, "kind": kind, "family": family,
        "endpoint_id": frame["endpoint_id"].to_numpy(str),
        "Falha": frame["Falha"].to_numpy(int), "equipment": frame["equipment"].to_numpy(str),
        "Date": pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").to_numpy(),
        "phase": frame["phase"].to_numpy(str), "y_true": frame["y"].to_numpy(float),
        "pred_raw": frame["pred_raw"].to_numpy(float),
        "pred_post": frame["pred_post"].to_numpy(float)})
    return rows


# ---------------------------------------------------------------------------
# Command line.
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="baseline_experiments.py",
        description="Local CPU baseline exploration (generator/gearbox).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("diagnose", "load data and report audit/folds/endpoints"),
                       ("search", "inner search per outer fold; freezes selections"),
                       ("evaluate", "refit frozen choices and predict outer groups"),
                       ("report", "write report.md from artifacts"),
                       ("verify", "check artifacts; non-zero exit on any failure")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--dataset", required=True, choices=DATASET_NAMES)
        command.add_argument("--config", default=None, help="JSON config (default: repo file)")
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        command.add_argument("--pilot", action="store_true",
                             help="tiny fixed candidates and budget (generator only)")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    run = Run(args.dataset, pilot=args.pilot, config_path=args.config,
              output_root=args.output_root)
    if args.command == "diagnose":
        ds = run.dataset
        print(f"{ds.name}: {len(ds.meta)} endpoints, groups {list(ds.outer_order)}")
        print(f"fingerprint {run.fingerprint}; unavailable families: "
              f"{run.unavailable or 'none'}; versions {run.versions}")
        print(f"CSV sha256 {ds.csv_sha256}")
        return 0
    if args.command == "search":
        run.search()
        return 0
    if args.command == "evaluate":
        run.evaluate()
        return 0
    if args.command == "report":
        print(run.report())
        return 0
    result = run.verify()
    for name, ok, detail in result["checks"]:
        print(f"{'PASS' if ok else 'FAIL'} {name}{(' - ' + detail) if detail else ''}")
    for group, state in result["outers"].items():
        print(f"outer {group}: {state}")
    print(f"RESULT: {result['status']}")
    return 1 if result["status"] == "failed" or any(
        not ok for _name, ok, _detail in result["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
