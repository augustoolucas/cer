"""chronos_rul — shared causal data, Chronos-2 extraction, caching, and training core.

Supports adapted methodological replications of:
- El-Ghoussani et al. (arXiv:2606.11990v3)
- Abdouni et al. (PHME 4906)

Top-level imports are intentionally lightweight so ``smoke`` audits, data/split
validation, and CLI ``--help`` work without importing ``torch`` or ``chronos``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import cer_rul

# ---------------------------------------------------------------------------
# Protocol & model constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"
EXTRACTOR_VERSION = "1.0.0"
TRAINING_VERSION = "1.0.0"
CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CHRONOS_PACKAGE_VERSION = "2.3.2"
CHRONOS_PATCH_SIZE = 16
CHRONOS_D_MODEL = 768
CHRONOS_MAX_CONTEXT_STEPS = 8192
EMBED_BATCH_VARIATES = 64

GRID_STEP_HOURS = 12
MAX_INTERP_GAP_HOURS = 48

DEFAULT_WARMUP = cer_rul.DEFAULT_WARMUP_RECORDS
DEFAULT_STRIDE = cer_rul.DEFAULT_ENDPOINT_STRIDE
DEFAULT_LABEL = "porcentagem-data"
DRIVE_MOUNT_ROOT = Path("/content/drive")

SUPPORTED_ARTICLES = ("elghoussani", "abdouni")
SUPPORTED_MODES = ("smoke", "extract", "train", "full")


# ---------------------------------------------------------------------------
# Hashing, JSON, atomic I/O, and environment provenance
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` deterministically with sorted keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(target_path: Path | str, data: bytes) -> Path:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def atomic_write_json(target_path: Path | str, payload: Any) -> Path:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return atomic_write_bytes(target_path, text.encode("utf-8"))


def atomic_save_npz(target_path: Path | str, **arrays: np.ndarray) -> Path:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npz", dir=str(target.parent))
    os.close(fd)
    try:
        with open(tmp_name, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def collect_environment_info(*, include_torch: bool = False) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if include_torch:
        try:
            import torch  # noqa: PLC0415
            info["torch"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_version"] = getattr(torch.version, "cuda", None)
        except ImportError:
            info["torch"] = None
        for pkg in ("chronos", "transformers"):
            try:
                mod = __import__(pkg)
                info[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                info[pkg] = None
    return info


# ---------------------------------------------------------------------------
# Dataset loading & explicit nested outer/inner splits
# ---------------------------------------------------------------------------

def _hash_clean_sensors(clean_df: pd.DataFrame, sensor_columns: Sequence[str]) -> str:
    """Hash clean sensor observations and timestamps WITHOUT RUL."""
    cols = ["Falha", "Date"] + list(sensor_columns)
    ordered = clean_df.loc[:, cols].sort_values(["Falha", "Date"], kind="mergesort")
    h = hashlib.sha256()
    h.update(canonical_json(cols).encode("utf-8"))
    h.update(ordered["Falha"].to_numpy(np.int64).tobytes())
    h.update(ordered["Date"].astype("int64").to_numpy(np.int64).tobytes())
    h.update(ordered.loc[:, list(sensor_columns)].to_numpy(np.float64).tobytes())
    return h.hexdigest()


def _hash_endpoints(endpoints_df: pd.DataFrame) -> str:
    """Hash endpoint identifiers, fault IDs, and timestamps WITHOUT RUL."""
    cols = ["endpoint_id", "Falha", "Date", "record_index"]
    ordered = endpoints_df.loc[:, cols].sort_values(["Falha", "Date", "record_index"], kind="mergesort")
    records = [
        (str(r.endpoint_id), int(r.Falha), pd.Timestamp(r.Date).isoformat(), int(r.record_index))
        for r in ordered.itertuples(index=False)
    ]
    return sha256_text(canonical_json(records))


def load_dataset_bundle(
    dataset_name: str,
    csv_path: Path | str,
    *,
    warmup: int = DEFAULT_WARMUP,
    stride: int = DEFAULT_STRIDE,
    enforce_expectations: bool = True,
) -> dict[str, Any]:
    """Load one CPFL dataset via ``cer_rul`` and build post-warmup endpoints."""
    profile = cer_rul.get_dataset_profile(dataset_name)
    label_meta = profile["labels"][DEFAULT_LABEL]
    sensor_columns = tuple(profile["sensor_columns"])
    csv_columns = ("Date",) + sensor_columns + ("Falha", "RUL")
    equipment_groups = {k: tuple(int(x) for x in v) for k, v in profile["equipment_groups"].items()}
    equipment_order = tuple(equipment_groups)
    id_to_equipment = {
        int(fid): eq for eq, fids in equipment_groups.items() for fid in fids
    }
    fault_ids = tuple(sorted(id_to_equipment))

    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")

    raw_frame = cer_rul.load_raw_label(
        str(csv_file),
        label=DEFAULT_LABEL,
        expected_raw_rows=profile["expected_raw_rows"] if enforce_expectations else None,
        csv_columns=csv_columns,
        sensor_columns=sensor_columns,
        fault_ids=fault_ids,
        max_allowed=label_meta["max_allowed"],
        print_summary=False,
    )
    clean_frame, audit, terminal = cer_rul.consolidate_frame(
        raw_frame,
        DEFAULT_LABEL,
        sensor_columns=sensor_columns,
        csv_columns=csv_columns,
        fault_ids=fault_ids,
        id_to_equipment=id_to_equipment,
        equipment_order=equipment_order,
        label_meta=label_meta,
        expected_clean_keys=profile["expected_clean_keys"] if enforce_expectations else None,
        expected_target_conflicts=profile["expected_target_conflicts"][DEFAULT_LABEL] if enforce_expectations else None,
        enforce_expectations=enforce_expectations,
    )
    endpoints = cer_rul.build_endpoint_rows(
        clean_frame,
        warmup=warmup,
        stride=stride,
        id_to_equipment=id_to_equipment,
        fpt_annotations=profile["fpt_annotations"],
        expected_endpoint_count=profile["expected_endpoints"] if enforce_expectations else None,
        enforce=enforce_expectations,
    )

    rul_lookup = clean_frame.loc[:, ["Falha", "Date", "RUL"]]
    endpoints = endpoints.merge(rul_lookup, on=["Falha", "Date"], how="left", validate="one_to_one")
    if endpoints["RUL"].isna().any():
        raise ValueError("Missing RUL after merging clean targets onto endpoints.")
    if not np.isfinite(endpoints["RUL"].to_numpy(float)).all():
        raise ValueError("Non-finite RUL in endpoints.")
    endpoints = endpoints.sort_values(["Falha", "Date", "record_index"], kind="mergesort").reset_index(drop=True)

    return {
        "dataset": dataset_name,
        "profile": profile,
        "sensor_columns": sensor_columns,
        "equipment_groups": equipment_groups,
        "equipment_order": equipment_order,
        "id_to_equipment": id_to_equipment,
        "fault_ids": fault_ids,
        "clean": clean_frame,
        "audit": audit,
        "terminal": terminal,
        "endpoints": endpoints,
        "csv_path": str(csv_file),
        "csv_sha256": sha256_file(csv_file),
        "sensor_data_sha256": _hash_clean_sensors(clean_frame, sensor_columns),
        "endpoints_sha256": _hash_endpoints(endpoints),
    }


def build_nested_splits(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Build explicit outer folds and ALL inner validation folds per outer fold.

    Unlike ``cer_rul.make_splits()``, which assigns only one validation group
    per outer fold, this iterates over all remaining groups in ``outer_train``
    so inner selection evaluates every non-test group.
    """
    equipment_groups = {k: tuple(int(x) for x in v) for k, v in profile["equipment_groups"].items()}
    equipment_order = tuple(equipment_groups)
    outer_folds: list[dict[str, Any]] = []

    for outer_idx, test_group in enumerate(equipment_order):
        other_groups = [g for g in equipment_order if g != test_group]
        test_ids = tuple(sorted(equipment_groups[test_group]))
        outer_train_ids = tuple(sorted(fid for g in other_groups for fid in equipment_groups[g]))

        inner_folds: list[dict[str, Any]] = []
        for inner_idx, val_group in enumerate(other_groups):
            inner_train_groups = [g for g in other_groups if g != val_group]
            val_ids = tuple(sorted(equipment_groups[val_group]))
            inner_train_ids = tuple(sorted(fid for g in inner_train_groups for fid in equipment_groups[g]))

            # Strict isolation assertions
            assert set(test_ids).isdisjoint(outer_train_ids)
            assert set(val_ids).isdisjoint(inner_train_ids)
            assert set(val_ids) | set(inner_train_ids) == set(outer_train_ids)

            inner_folds.append({
                "inner_fold_id": f"{test_group}__val_{val_group}",
                "inner_index": inner_idx,
                "val_group": val_group,
                "val_ids": val_ids,
                "inner_train_groups": tuple(inner_train_groups),
                "inner_train_ids": inner_train_ids,
            })

        outer_folds.append({
            "outer_fold_id": f"fold_{test_group}",
            "outer_index": outer_idx,
            "test_group": test_group,
            "test_ids": test_ids,
            "outer_train_groups": tuple(other_groups),
            "outer_train_ids": outer_train_ids,
            "inner_folds": inner_folds,
        })

    return outer_folds


# ---------------------------------------------------------------------------
# Causal 12h grid construction & interpolation
# ---------------------------------------------------------------------------

def build_causal_12h_grid(
    clean_fault_df: pd.DataFrame,
    endpoint_date: pd.Timestamp,
    sensor_columns: Sequence[str],
    *,
    context_steps: int | None = None,
    step_hours: int = GRID_STEP_HOURS,
    max_gap_hours: int = MAX_INTERP_GAP_HOURS,
    max_context_steps: int = CHRONOS_MAX_CONTEXT_STEPS,
) -> np.ndarray:
    """Build causal ``(8, L)`` 12h grid anchored at ``endpoint_date``.

    Interpolates linearly ONLY between observed timestamps ``<= endpoint_date``
    whose gap is ``<= max_gap_hours``. Longer gaps and unavailable leading
    prefixes remain ``NaN`` (masked), never physical zero.
    """
    t_end = pd.Timestamp(endpoint_date)
    sub = clean_fault_df.loc[clean_fault_df["Date"] <= t_end].sort_values("Date", kind="mergesort")
    if sub.empty:
        raise ValueError(f"No observations <= {t_end} for fault.")
    if pd.Timestamp(sub["Date"].iloc[-1]) != t_end:
        raise ValueError(f"Final observation {sub['Date'].iloc[-1]} != endpoint {t_end}.")

    obs_hours = (sub["Date"] - t_end).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
    obs_vals = sub.loc[:, list(sensor_columns)].to_numpy(dtype=np.float64)  # (T_obs, C)
    n_sensors = len(sensor_columns)

    if context_steps is None:
        min_hour = float(obs_hours[0])
        n_steps = int(math.floor(-min_hour / float(step_hours))) + 1
        if n_steps > max_context_steps:
            raise ValueError(
                f"Full observed prefix length {n_steps} exceeds Chronos-2 limit "
                f"{max_context_steps}; refusing silent truncation."
            )
    else:
        n_steps = int(context_steps)
        if n_steps <= 0 or n_steps > max_context_steps:
            raise ValueError(
                f"context_steps={n_steps} must be in [1, {max_context_steps}]."
            )

    grid_hours = np.arange(-(n_steps - 1), 1, dtype=np.float64) * float(step_hours)
    grid = np.full((n_sensors, n_steps), np.nan, dtype=np.float64)

    n_obs = len(obs_hours)
    insert_idx = np.searchsorted(obs_hours, grid_hours, side="right")

    for j in range(n_steps):
        g = grid_hours[j]
        idx = int(insert_idx[j])
        if idx > 0 and abs(obs_hours[idx - 1] - g) <= 1e-9:
            grid[:, j] = obs_vals[idx - 1]
        elif 0 < idx < n_obs:
            t_left = obs_hours[idx - 1]
            t_right = obs_hours[idx]
            gap = t_right - t_left
            if gap <= float(max_gap_hours) + 1e-9:
                alpha = (g - t_left) / gap
                grid[:, j] = (1.0 - alpha) * obs_vals[idx - 1] + alpha * obs_vals[idx]

    if not np.isfinite(grid[:, -1]).all():
        raise AssertionError("Endpoint final step must always be observed and finite.")

    return grid.astype(np.float32)


def build_endpoint_grids(
    bundle: dict[str, Any],
    *,
    context_steps: int | None = None,
    step_hours: int = GRID_STEP_HOURS,
    max_gap_hours: int = MAX_INTERP_GAP_HOURS,
    max_context_steps: int = CHRONOS_MAX_CONTEXT_STEPS,
) -> list[np.ndarray]:
    """Build causal ``(8, L)`` grids for all endpoints in ``bundle['endpoints']``."""
    clean_by_fault = {
        int(fid): grp.sort_values("Date", kind="mergesort").reset_index(drop=True)
        for fid, grp in bundle["clean"].groupby("Falha", sort=True)
    }
    sensor_columns = bundle["sensor_columns"]
    grids: list[np.ndarray] = []
    for row in bundle["endpoints"].itertuples(index=False):
        grid = build_causal_12h_grid(
            clean_by_fault[int(row.Falha)],
            pd.Timestamp(row.Date),
            sensor_columns,
            context_steps=context_steps,
            step_hours=step_hours,
            max_gap_hours=max_gap_hours,
            max_context_steps=max_context_steps,
        )
        grids.append(grid)
    return grids


# ---------------------------------------------------------------------------
# Train-only causal preprocessing
# ---------------------------------------------------------------------------

class CausalPreprocessor:
    """Train-only sensor clipping (p1/p99) and/or z-score preserving NaN masks."""

    def __init__(self, mode: str) -> None:
        if mode not in ("clip_p1_p99_zscore", "zscore"):
            raise ValueError(f"Unsupported preprocessing mode: {mode!r}")
        self.mode = mode
        self.train_ids: tuple[int, ...] = ()
        self.sensor_columns: tuple[str, ...] = ()
        self.q01: np.ndarray | None = None
        self.q99: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(
        self,
        clean_df: pd.DataFrame,
        train_ids: Sequence[int],
        sensor_columns: Sequence[str],
    ) -> "CausalPreprocessor":
        train_id_tuple = tuple(sorted(int(x) for x in train_ids))
        sub = clean_df.loc[clean_df["Falha"].isin(train_id_tuple), list(sensor_columns)]
        if sub.empty:
            raise ValueError(f"No training rows found for train_ids={train_id_tuple}.")
        mat = sub.to_numpy(dtype=np.float64)
        if not np.isfinite(mat).all():
            raise ValueError("Non-finite sensor values in training data.")

        if self.mode == "clip_p1_p99_zscore":
            q01 = np.percentile(mat, 1.0, axis=0)
            q99 = np.percentile(mat, 99.0, axis=0)
            work = np.clip(mat, q01, q99)
        else:
            q01 = None
            q99 = None
            work = mat

        mean = np.mean(work, axis=0)
        std = np.std(work, axis=0)
        std = np.where(std < 1e-12, 1.0, std)

        self.train_ids = train_id_tuple
        self.sensor_columns = tuple(sensor_columns)
        self.q01 = q01
        self.q99 = q99
        self.mean = mean
        self.std = std
        return self

    def transform(self, grid_8xL: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("CausalPreprocessor must be fit before transform.")
        arr = np.asarray(grid_8xL, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != len(self.sensor_columns):
            raise ValueError(f"Expected shape ({len(self.sensor_columns)}, L), got {arr.shape}.")
        if self.mode == "clip_p1_p99_zscore":
            assert self.q01 is not None and self.q99 is not None
            arr = np.clip(arr, self.q01[:, None], self.q99[:, None])
        normed = (arr - self.mean[:, None]) / self.std[:, None]
        return normed.astype(np.float32)

    def transform_many(self, grids: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [self.transform(g) for g in grids]

    def transform_current_vector(self, grids: Sequence[np.ndarray]) -> np.ndarray:
        """Return normalized current (endpoint) sensor matrix of shape ``(N, 8)``."""
        return np.stack([self.transform(g)[:, -1] for g in grids], axis=0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            raise RuntimeError("CausalPreprocessor not fitted.")
        return {
            "mode": self.mode,
            "train_ids": list(self.train_ids),
            "sensor_columns": list(self.sensor_columns),
            "q01": [float(x) for x in self.q01] if self.q01 is not None else None,
            "q99": [float(x) for x in self.q99] if self.q99 is not None else None,
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalPreprocessor":
        obj = cls(data["mode"])
        obj.train_ids = tuple(int(x) for x in data["train_ids"])
        obj.sensor_columns = tuple(data["sensor_columns"])
        obj.q01 = np.asarray(data["q01"], dtype=np.float64) if data.get("q01") is not None else None
        obj.q99 = np.asarray(data["q99"], dtype=np.float64) if data.get("q99") is not None else None
        obj.mean = np.asarray(data["mean"], dtype=np.float64)
        obj.std = np.asarray(data["std"], dtype=np.float64)
        return obj


# ---------------------------------------------------------------------------
# Chronos-2 pipeline loading & embedding extraction (lazy Torch/Chronos)
# ---------------------------------------------------------------------------

def validate_chronos_pipeline_schema(
    pipeline: Any,
    *,
    expected_d_model: int = CHRONOS_D_MODEL,
    expected_patch_size: int = CHRONOS_PATCH_SIZE,
    expected_use_reg_token: bool = True,
) -> dict[str, Any]:
    """Validate Chronos-2 model config before extracting embeddings."""
    model = getattr(pipeline, "model", None)
    if model is None:
        raise ValueError("Pipeline has no .model attribute.")
    chronos_cfg = getattr(model, "chronos_config", None)
    core_cfg = getattr(model, "config", None)
    if chronos_cfg is None or core_cfg is None:
        raise ValueError("Chronos2Model missing chronos_config or config.")

    d_model = int(core_cfg.d_model)
    patch_size = int(chronos_cfg.input_patch_size)
    use_reg_token = bool(chronos_cfg.use_reg_token)
    context_length = int(chronos_cfg.context_length)
    max_output_patches = int(chronos_cfg.max_output_patches)

    if d_model != expected_d_model:
        raise ValueError(f"Expected d_model={expected_d_model}, got {d_model}.")
    if patch_size != expected_patch_size:
        raise ValueError(f"Expected input_patch_size={expected_patch_size}, got {patch_size}.")
    if use_reg_token is not expected_use_reg_token:
        raise ValueError(f"Expected use_reg_token={expected_use_reg_token}, got {use_reg_token}.")

    return {
        "d_model": d_model,
        "input_patch_size": patch_size,
        "use_reg_token": use_reg_token,
        "context_length": context_length,
        "max_output_patches": max_output_patches,
    }


def load_chronos_pipeline(
    *,
    model_id: str = CHRONOS_MODEL_ID,
    revision: str = CHRONOS_MODEL_REVISION,
    device: str = "cuda",
) -> Any:
    """Load frozen Chronos-2 pipeline in float32. Never falls back silently to CPU."""
    import torch  # noqa: PLC0415
    from chronos import Chronos2Pipeline  # noqa: PLC0415

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA nao disponivel para extracao no modo padrao ('cuda'). "
            "No Colab, ative Runtime -> Change runtime type -> T4/GPU. "
            "Para rodar em CPU explicitamente (mais lento), defina device='cpu'."
        )
    if device not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported device: {device!r}. Expected 'cuda' or 'cpu'.")

    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        device_map=device,
    )
    pipeline.model.to(device=device, dtype=torch.float32)
    pipeline.model.eval()
    for param in pipeline.model.parameters():
        param.requires_grad_(False)

    validate_chronos_pipeline_schema(pipeline)
    return pipeline


def extract_last_context_patch_embeddings(
    pipeline: Any,
    grids_8xL: Sequence[np.ndarray],
    *,
    batch_size_variates: int = EMBED_BATCH_VARIATES,
) -> np.ndarray:
    """Extract last observed context-patch embeddings ``(N, 8, 768)`` via ``embed()``.

    Each element in ``grids_8xL`` has shape ``(8, L_i)`` and becomes its own
    group inside ``Chronos2Pipeline.embed`` (8 variates share attention within
    one endpoint, never across endpoints).

    Token layout in Chronos-2 ``embed()`` when ``use_reg_token=True`` and
    ``prediction_length=0`` (1 masked output patch):
      ``[context_patch_0, ..., context_patch_{P-1}, REG, masked_output_patch]``
    where ``P = ceil(L_i / patch_size)``. We validate ``total_tokens == P + 2``
    and select index ``P - 1`` (the last observed context patch), NEVER ``-1``.
    """
    import torch  # noqa: PLC0415

    schema = validate_chronos_pipeline_schema(pipeline)
    patch_size = schema["input_patch_size"]
    d_model = schema["d_model"]
    max_ctx = schema["context_length"]

    if not grids_8xL:
        return np.empty((0, 8, d_model), dtype=np.float32)

    inputs: list[np.ndarray] = []
    expected_patches: list[int] = []
    for idx, grid in enumerate(grids_8xL):
        arr = np.asarray(grid, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != 8:
            raise ValueError(f"Item {idx}: expected shape (8, L), got {arr.shape}.")
        length = arr.shape[1]
        if length <= 0 or length > max_ctx:
            raise ValueError(f"Item {idx}: length {length} outside [1, {max_ctx}].")
        if not np.isfinite(arr[:, -1]).all():
            raise ValueError(f"Item {idx}: final endpoint step contains non-finite values.")
        inputs.append(arr)
        expected_patches.append(int(math.ceil(length / patch_size)))

    pipeline.model.eval()
    with torch.no_grad():
        embeds_list, _loc_scales = pipeline.embed(
            inputs,
            batch_size=int(batch_size_variates),
            context_length=max_ctx,
        )

    if len(embeds_list) != len(inputs):
        raise RuntimeError(f"Expected {len(inputs)} embeddings, got {len(embeds_list)}.")

    out = np.empty((len(inputs), 8, d_model), dtype=np.float32)
    for i, (emb, p_min_ctx) in enumerate(zip(embeds_list, expected_patches)):
        # Chronos2Dataset left-pads variable-length items within a batch to the
        # batch's max length, so total_tokens = P_batch + 2 (>= p_min_ctx + 2),
        # where the final 2 tokens are [REG] (-2) and masked output patch (-1).
        min_tokens = p_min_ctx + 2
        if emb.ndim != 3 or emb.shape[0] != 8 or emb.shape[1] < min_tokens or emb.shape[2] != d_model:
            raise RuntimeError(
                f"Item {i}: unexpected embedding shape {tuple(emb.shape)}; "
                f"expected (8, >= {min_tokens}, {d_model})."
            )
        last_ctx_idx = emb.shape[1] - 3
        last_ctx_patch = emb[:, last_ctx_idx, :].detach().cpu().to(torch.float32).numpy()
        if not np.isfinite(last_ctx_patch).all():
            raise RuntimeError(f"Item {i}: non-finite extracted embedding.")
        out[i] = last_ctx_patch

    return out


# ---------------------------------------------------------------------------
# Atomic embedding cache (npz without pickle + JSON metadata)
# ---------------------------------------------------------------------------

def get_chronos_package_version(*, require_installed: bool = False) -> str:
    """Return installed chronos-forecasting version or pinned protocol version."""
    try:
        ver = importlib.metadata.version("chronos-forecasting")
    except importlib.metadata.PackageNotFoundError:
        if require_installed:
            raise
        return CHRONOS_PACKAGE_VERSION
    if require_installed and ver != CHRONOS_PACKAGE_VERSION:
        raise RuntimeError(
            f"Unsupported chronos-forecasting version {ver!r}; "
            f"protocol requires {CHRONOS_PACKAGE_VERSION!r}."
        )
    return ver


def compute_extractor_code_sha256() -> str:
    """Hash extractor, grid, and preprocessing implementation + pinned constants."""
    payload = {
        "extractor_version": EXTRACTOR_VERSION,
        "chronos_package_version": CHRONOS_PACKAGE_VERSION,
        "model_id": CHRONOS_MODEL_ID,
        "model_revision": CHRONOS_MODEL_REVISION,
        "patch_size": CHRONOS_PATCH_SIZE,
        "d_model": CHRONOS_D_MODEL,
        "max_context_steps": CHRONOS_MAX_CONTEXT_STEPS,
        "grid_step_hours": GRID_STEP_HOURS,
        "max_interp_gap_hours": MAX_INTERP_GAP_HOURS,
        "sources": [
            inspect.getsource(build_causal_12h_grid),
            inspect.getsource(CausalPreprocessor),
            inspect.getsource(validate_chronos_pipeline_schema),
            inspect.getsource(extract_last_context_patch_embeddings),
        ],
    }
    return sha256_text(canonical_json(payload))


def _canonicalize_path(path: Path | str) -> Path:
    """Resolve user-home, relative segments against cwd, and symlinks cleanly."""
    return Path(os.path.abspath(os.path.expanduser(str(path)))).resolve(strict=False)


def is_drive_mounted(mount_root: Path | str | None = None) -> bool:
    """Return True only when Google Drive is genuinely mounted at ``mount_root``."""
    raw_root = Path(os.path.expanduser(str(mount_root))) if mount_root is not None else DRIVE_MOUNT_ROOT
    res_root = _canonicalize_path(raw_root)
    return (
        res_root.is_dir()
        and (res_root / "MyDrive").is_dir()
        and (os.path.ismount(str(raw_root)) or os.path.ismount(str(res_root)))
    )


def _is_under_drive_root(path: Path | str, mount_root: Path | str | None = None) -> bool:
    root = mount_root if mount_root is not None else DRIVE_MOUNT_ROOT
    cand = _canonicalize_path(path)
    root_res = _canonicalize_path(root)
    return cand == root_res or cand.is_relative_to(root_res)


def validate_storage_path(
    path: Path | str | None,
    *,
    label: str = "output",
    mount_root: Path | str | None = None,
) -> Path | None:
    """Reject /content/drive/... paths unless Drive is genuinely mounted."""
    if path is None:
        return None
    p = Path(path)
    if _is_under_drive_root(p, mount_root=mount_root) and not is_drive_mounted(mount_root=mount_root):
        raise RuntimeError(
            f"Caminho '{label}' aponta para o Google Drive ({str(p)!r}), mas "
            f"{str(mount_root or DRIVE_MOUNT_ROOT)!r} nao esta montado. "
            "No Colab, habilite CONFIG['mount_drive'] = True (ou monte o Drive via "
            "google.colab.drive.mount('/content/drive')). Localmente, use um caminho "
            "fora de /content/drive (ex.: './chronos_outputs')."
        )
    return p


def compute_view_identity(
    *,
    dataset_name: str,
    sensor_columns: Sequence[str],
    sensor_data_sha256: str,
    endpoints_sha256: str,
    preprocessor_dict: dict[str, Any],
    context_spec: dict[str, Any],
    model_id: str = CHRONOS_MODEL_ID,
    model_revision: str = CHRONOS_MODEL_REVISION,
    extractor_dtype: str = "float32",
    extractor_version: str | None = None,
    extractor_code_sha256: str | None = None,
    chronos_package_version: str | None = None,
) -> str:
    """Compute deterministic scientific view ID (excludes mode, paths, RUL)."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": extractor_version or EXTRACTOR_VERSION,
        "extractor_code_sha256": extractor_code_sha256 or compute_extractor_code_sha256(),
        "chronos_package_version": chronos_package_version or CHRONOS_PACKAGE_VERSION,
        "dataset": dataset_name,
        "sensor_columns": list(sensor_columns),
        "sensor_data_sha256": sensor_data_sha256,
        "endpoints_sha256": endpoints_sha256,
        "preprocessor": preprocessor_dict,
        "context_spec": context_spec,
        "model_id": model_id,
        "model_revision": model_revision,
        "patch_size": CHRONOS_PATCH_SIZE,
        "d_model": CHRONOS_D_MODEL,
        "use_reg_token": True,
        "extractor_dtype": extractor_dtype,
    }
    return sha256_text(canonical_json(payload))[:24]


def cache_paths(cache_dir: Path | str, view_id: str) -> tuple[Path, Path]:
    base = Path(cache_dir)
    return base / f"{view_id}.npz", base / f"{view_id}.json"


def save_embedding_cache(
    cache_dir: Path | str,
    view_id: str,
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    npz_path, json_path = cache_paths(cache_dir, view_id)
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (8, CHRONOS_D_MODEL):
        raise ValueError(f"Invalid embedding cache shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Refusing to cache non-finite embeddings.")

    meta = copy.deepcopy(metadata)
    meta["view_id"] = view_id
    meta["shape"] = list(arr.shape)
    meta["dtype"] = "float32"
    meta["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["embeddings_sha256"] = sha256_text(arr.tobytes().hex()[:64]) if arr.size == 0 else hashlib.sha256(arr.tobytes()).hexdigest()

    atomic_save_npz(npz_path, embeddings=arr)
    atomic_write_json(json_path, meta)
    return npz_path, json_path


def load_embedding_cache(
    cache_dir: Path | str,
    view_id: str,
    expected_metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and strictly validate a cached embedding view without importing Chronos."""
    npz_path, json_path = cache_paths(cache_dir, view_id)
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Embedding cache missing for view_id={view_id} in {cache_dir}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if meta.get("view_id") != view_id:
        raise ValueError(f"Cache view_id mismatch: {meta.get('view_id')} != {view_id}")

    required_keys = (
        "schema_version",
        "extractor_version",
        "extractor_code_sha256",
        "chronos_package_version",
        "dataset",
        "sensor_columns",
        "sensor_data_sha256",
        "endpoints_sha256",
        "train_ids",
        "preprocessor",
        "context_spec",
        "model_id",
        "model_revision",
        "patch_size",
        "d_model",
        "use_reg_token",
        "extractor_dtype",
    )
    for key in required_keys:
        if meta.get(key) != expected_metadata.get(key):
            raise ValueError(
                f"Cache provenance mismatch on {key!r}: "
                f"cached={meta.get(key)!r} vs expected={expected_metadata.get(key)!r}"
            )

    with np.load(npz_path, allow_pickle=False) as archive:
        if "embeddings" not in archive:
            raise ValueError(f"Corrupt cache {npz_path}: missing 'embeddings' array.")
        arr = np.asarray(archive["embeddings"], dtype=np.float32)

    expected_shape = tuple(meta.get("shape", ()))
    if arr.shape != expected_shape or arr.ndim != 3 or arr.shape[1:] != (8, CHRONOS_D_MODEL):
        raise ValueError(f"Cache shape mismatch: {arr.shape} != {expected_shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Corrupt cache {npz_path}: non-finite embeddings.")

    actual_hash = hashlib.sha256(arr.tobytes()).hexdigest()
    if meta.get("embeddings_sha256") != actual_hash:
        raise ValueError(f"Corrupt cache {npz_path}: embeddings checksum mismatch.")

    return arr, meta


# ---------------------------------------------------------------------------
# Generic CPU head training & state_dict checkpointing
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import torch  # noqa: PLC0415

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def train_head_epochs(
    head: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    wide_train: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    wide_val: np.ndarray | None = None,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a PyTorch head on CPU over pre-extracted embeddings (+ optional wide)."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    head = head.to(device="cpu", dtype=torch.float32)
    if optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    elif optimizer_name.lower() == "adamw":
        opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

    loss_fn = nn.MSELoss()

    x_tr_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    y_tr_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1)
    if wide_train is not None:
        w_tr_t = torch.from_numpy(np.asarray(wide_train, dtype=np.float32))
        ds = TensorDataset(x_tr_t, w_tr_t, y_tr_t)
    else:
        ds = TensorDataset(x_tr_t, y_tr_t)

    effective_batch = min(int(batch_size), len(ds))
    loader = DataLoader(ds, batch_size=effective_batch, shuffle=True, generator=gen)

    has_val = X_val is not None and y_val is not None
    if has_val:
        x_va_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
        y_va_np = np.asarray(y_val, dtype=np.float64).reshape(-1)
        w_va_t = torch.from_numpy(np.asarray(wide_val, dtype=np.float32)) if wide_val is not None else None

    train_losses: list[float] = []
    val_rmses: list[float] = []
    best_val_rmse = float("inf")
    best_epoch = int(max_epochs)
    best_state = copy.deepcopy(head.state_dict())
    no_improve = 0

    for epoch in range(1, int(max_epochs) + 1):
        head.train()
        epoch_loss = 0.0
        n_seen = 0
        for batch in loader:
            opt.zero_grad()
            if wide_train is not None:
                bx, bw, by = batch
                pred = head(bx, bw)
            else:
                bx, by = batch
                pred = head(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()
            bs = by.shape[0]
            epoch_loss += float(loss.item()) * bs
            n_seen += bs
        train_losses.append(epoch_loss / max(n_seen, 1))

        if has_val:
            head.eval()
            with torch.no_grad():
                val_pred = head(x_va_t, w_va_t) if wide_val is not None else head(x_va_t)
                val_pred_np = val_pred.squeeze(-1).cpu().numpy().astype(np.float64)
            val_rmse = float(np.sqrt(np.mean((val_pred_np - y_va_np) ** 2)))
            val_rmses.append(val_rmse)
            if val_rmse < best_val_rmse - 1e-12:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(head.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if patience is not None and no_improve >= int(patience):
                    break

    if has_val and patience is not None:
        head.load_state_dict(best_state)
    else:
        best_state = copy.deepcopy(head.state_dict())
        best_epoch = len(train_losses)

    return {
        "head": head,
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse) if has_val else None,
        "train_losses": train_losses,
        "val_rmses": val_rmses,
    }


def predict_head(head: Any, X: np.ndarray, *, wide: np.ndarray | None = None) -> np.ndarray:
    import torch  # noqa: PLC0415

    head.eval()
    x_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        if wide is not None:
            w_t = torch.from_numpy(np.asarray(wide, dtype=np.float32))
            preds = head(x_t, w_t)
        else:
            preds = head(x_t)
    return preds.squeeze(-1).cpu().numpy().astype(np.float64)


def save_head_checkpoint(path: Path | str, state_dict: dict[str, Any]) -> Path:
    import torch  # noqa: PLC0415

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".pt", dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(state_dict, tmp_name)
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def load_head_checkpoint(path: Path | str) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Hierarchical metrics in fraction 0..1 (cer_rul.regression_metrics)
# ---------------------------------------------------------------------------

def compute_hierarchical_metrics(
    pred_df: pd.DataFrame,
    *,
    y_col: str = "y_true",
    pred_col: str = "pred_raw",
    near_zero_threshold: float = 0.10,
) -> dict[str, Any]:
    """Compute hierarchical macro metrics in RUL 0..1:

    1. Per ``Falha`` inside each group (``equipment``);
    2. Equal mean across ``Falha``s inside each group;
    3. Equal mean across outer groups.

    Pooled metrics, phase metrics, and near-zero metrics are reported separately.
    """
    required = {"Falha", "equipment", "phase", y_col, pred_col}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction frame missing columns: {sorted(missing)}")

    y_all = pred_df[y_col].to_numpy(dtype=np.float64)
    p_all = pred_df[pred_col].to_numpy(dtype=np.float64)
    pooled = cer_rul.regression_metrics(y_all, p_all)

    by_group: dict[str, Any] = {}
    group_maes: list[float] = []
    group_rmses: list[float] = []
    group_biases: list[float] = []

    for eq_name, eq_df in pred_df.groupby("equipment", sort=True):
        by_fault: dict[str, Any] = {}
        for fid, f_df in eq_df.groupby("Falha", sort=True):
            by_fault[str(int(fid))] = cer_rul.regression_metrics(
                f_df[y_col].to_numpy(dtype=np.float64),
                f_df[pred_col].to_numpy(dtype=np.float64),
            )
        eq_mae = float(np.mean([m["mae"] for m in by_fault.values()]))
        eq_rmse = float(np.mean([m["rmse"] for m in by_fault.values()]))
        eq_bias = float(np.mean([m["bias"] for m in by_fault.values()]))
        by_group[str(eq_name)] = {
            "n": int(len(eq_df)),
            "mae": eq_mae,
            "rmse": eq_rmse,
            "bias": eq_bias,
            "by_fault": by_fault,
        }
        group_maes.append(eq_mae)
        group_rmses.append(eq_rmse)
        group_biases.append(eq_bias)

    by_phase: dict[str, Any] = {}
    for phase_name, ph_df in pred_df.groupby("phase", sort=True):
        by_phase[str(phase_name)] = cer_rul.regression_metrics(
            ph_df[y_col].to_numpy(dtype=np.float64),
            ph_df[pred_col].to_numpy(dtype=np.float64),
        )

    near_zero_mask = y_all <= float(near_zero_threshold)
    near_zero = (
        cer_rul.regression_metrics(y_all[near_zero_mask], p_all[near_zero_mask])
        if near_zero_mask.any()
        else None
    )

    return {
        "unit": "fraction_0_1",
        "macro": {
            "n": int(len(pred_df)),
            "n_groups": int(len(by_group)),
            "mae": float(np.mean(group_maes)),
            "rmse": float(np.mean(group_rmses)),
            "bias": float(np.mean(group_biases)),
        },
        "pooled": pooled,
        "by_group": by_group,
        "by_phase": by_phase,
        "near_zero": near_zero,
    }


def aggregate_seed_metrics(seed_metrics_list: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean +/- std across seeds for macro and pooled metrics."""
    if not seed_metrics_list:
        raise ValueError("seed_metrics_list cannot be empty.")

    def _agg(section: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for metric in ("mae", "rmse", "bias"):
            vals = np.asarray([m[section][metric] for m in seed_metrics_list], dtype=np.float64)
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            out[metric] = {"mean": float(np.mean(vals)), "std": std}
        return out

    return {
        "n_seeds": len(seed_metrics_list),
        "macro": _agg("macro"),
        "pooled": _agg("pooled"),
    }


# ---------------------------------------------------------------------------
# Portable export & smoke runner
# ---------------------------------------------------------------------------

def create_portable_zip(
    run_dir: Path | str,
    zip_path: Path | str | None = None,
    *,
    extra_files: dict[str, Path | str] | None = None,
    override_texts: dict[str, str] | None = None,
) -> Path:
    """Create a portable ZIP of ``run_dir``, optionally bundling extra files (e.g. cache)."""
    run_path = Path(run_dir)
    target = Path(zip_path) if zip_path is not None else run_path.with_name(run_path.name + ".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    overrides = dict(override_texts or {})
    written_arcnames: set[str] = set()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(run_path.rglob("*")):
            if file_path.is_file():
                arc = file_path.relative_to(run_path).as_posix()
                if arc in overrides:
                    zf.writestr(arc, overrides[arc])
                else:
                    zf.write(file_path, arcname=arc)
                written_arcnames.add(arc)
        if extra_files:
            for arcname, src in sorted(extra_files.items()):
                if arcname in overrides:
                    zf.writestr(arcname, overrides[arcname])
                elif arcname not in written_arcnames:
                    zf.write(Path(src), arcname=arcname)
                    written_arcnames.add(arcname)
    return target


def effective_article_config(article: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build normalized complete configuration with explicit protocol defaults."""
    dataset = str(config.get("dataset", "gerador"))
    base: dict[str, Any] = {
        "article": article,
        "dataset": dataset,
        "mode": str(config.get("mode", "smoke")),
        "input_csv": _resolve_input_csv(dataset, config),
        "output_dir": str(config.get("output_dir", "results/chronos")),
        "cache_dir": str(config["cache_dir"]) if config.get("cache_dir") else None,
        "run_dir": str(config["run_dir"]) if config.get("run_dir") else None,
        "run_id": str(config["run_id"]) if config.get("run_id") else None,
        "device": str(config.get("device", "cuda")),
        "batch_size_variates": int(config.get("batch_size_variates", EMBED_BATCH_VARIATES)),
        "warmup": int(config.get("warmup", DEFAULT_WARMUP)),
        "stride": int(config.get("stride", DEFAULT_STRIDE)),
        "model_id": CHRONOS_MODEL_ID,
        "model_revision": CHRONOS_MODEL_REVISION,
        "chronos_package_version": CHRONOS_PACKAGE_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "training_version": TRAINING_VERSION,
    }
    if article == "elghoussani":
        base.update({
            "context_candidates": [int(x) for x in config.get("context_candidates", [5, 32, 80])],
            "epochs": int(config.get("epochs", 50)),
            "batch_size": int(config.get("batch_size", 64)),
            "lr": float(config.get("lr", 1e-3)),
            "hidden_dim": int(config.get("hidden_dim", 128)),
            "dropout": float(config.get("dropout", 0.2)),
            "seeds": [int(s) for s in config.get("seeds", [42, 43, 44])],
            "controls": list(config.get("controls", ["linear_relu", "mlp4"])),
        })
    elif article == "abdouni":
        base.update({
            "d_comp": int(config.get("d_comp", 8)),
            "max_epochs": int(config.get("max_epochs", 150)),
            "patience": int(config.get("patience", 15)),
            "batch_size": int(config.get("batch_size", 1024)),
            "lr": float(config.get("lr", 1e-4)),
            "weight_decay": float(config.get("weight_decay", 1e-3)),
            "dropout": float(config.get("dropout", 0.2)),
            "seeds": [int(s) for s in config.get("seeds", list(range(42, 52)))],
            "variants": list(config.get("variants", ["fusion", "deep", "wide"])),
        })
    else:
        raise ValueError(f"Unsupported article: {article!r}")
    return base


def compute_training_code_sha256(article: str | None = None) -> str:
    """Hash head architecture, training, prediction, selection, and metric code.

    Deliberately decoupled from ``compute_extractor_code_sha256`` so changes to
    heads, optimization, selection, or predictions invalidate training identities
    and resumption checkpoints without invalidating compatible GPU embedding caches.
    Does not import ``torch``.
    """
    targets: list[tuple[str, Any]] = [
        ("seed_everything", seed_everything),
        ("train_head_epochs", train_head_epochs),
        ("predict_head", predict_head),
        ("compute_hierarchical_metrics", compute_hierarchical_metrics),
        ("aggregate_seed_metrics", aggregate_seed_metrics),
        ("effective_article_config", effective_article_config),
    ]
    if article in (None, "elghoussani"):
        targets.extend([
            ("build_elghoussani_head", build_elghoussani_head),
            ("run_elghoussani_pipeline", run_elghoussani_pipeline),
        ])
    if article in (None, "abdouni"):
        targets.extend([
            ("build_abdouni_head", build_abdouni_head),
            ("run_abdouni_pipeline", run_abdouni_pipeline),
        ])
    sources: dict[str, str] = {}
    for name, fn in targets:
        target_fn = fn
        if callable(getattr(target_fn, "_mock_side_effect", None)):
            target_fn = target_fn._mock_side_effect
        elif callable(getattr(target_fn, "_mock_wraps", None)):
            target_fn = target_fn._mock_wraps
        target_fn = inspect.unwrap(target_fn)
        try:
            sources[name] = inspect.getsource(target_fn)
        except (OSError, TypeError):
            sources[name] = repr(target_fn)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "training_version": TRAINING_VERSION,
        "article": article,
        "sources": sources,
    }
    return sha256_text(canonical_json(payload))


def compute_run_identity(
    article: str,
    bundle: dict[str, Any],
    effective_cfg: dict[str, Any],
) -> str:
    """Compute deterministic identity from data, scientific config, and training code.

    Local paths, ``mode``, ``device``, and Colab flags are excluded so runs with
    different scientific configs or data never collide, while GPU->CPU transfers
    preserve identity.
    """
    non_scientific_keys = {
        "mode",
        "input_csv",
        "output_dir",
        "cache_dir",
        "run_dir",
        "run_id",
        "device",
        "batch_size_variates",
        "install_dependencies",
        "mount_drive",
        "download_zip",
    }
    scientific_cfg = {
        k: v for k, v in effective_cfg.items() if k not in non_scientific_keys
    }
    training_code_sha = compute_training_code_sha256(article)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "training_version": TRAINING_VERSION,
        "training_code_sha256": training_code_sha,
        "article": article,
        "dataset": bundle["dataset"],
        "csv_sha256": bundle["csv_sha256"],
        "sensor_data_sha256": bundle["sensor_data_sha256"],
        "endpoints_sha256": bundle["endpoints_sha256"],
        "scientific_config": scientific_cfg,
    }
    return sha256_text(canonical_json(payload))[:16]


def _read_dir_run_identity(meta_path: Path) -> tuple[bool, str | None]:
    if not meta_path.is_file():
        return False, None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Nao foi possivel verificar a identidade existente em '{meta_path}': {exc}. "
            "Escolha outro diretorio ou 'run_id'."
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Metadados invalidos em '{meta_path}'; recusei para nao sobrescrever.")
    rid = data.get("run_identity")
    if not isinstance(rid, str) or not rid.strip():
        return True, None
    return True, rid.strip()


def _ensure_safe_run_dir(run_path: Path, expected_run_identity: str) -> None:
    """Validate before any mkdir/write that ``run_path`` does not belong to another run or legacy data."""
    if not run_path.exists():
        return
    if not run_path.is_dir():
        raise RuntimeError(
            f"Destino de execucao existe e nao e um diretorio: '{run_path}'."
        )
    if not any(run_path.iterdir()):
        return

    has_manifest, m_id = _read_dir_run_identity(run_path / "manifest.json")
    has_config, c_id = _read_dir_run_identity(run_path / "config.json")
    if (has_manifest and m_id is None) or (has_config and c_id is None) or (not has_manifest and not has_config):
        raise RuntimeError(
            f"Destino '{run_path}' nao esta vazio e nao possui 'run_identity' verificavel "
            "(diretorio legado ou alheio). Recusei sem alterar arquivos anteriores; escolha outro 'run_id' ou 'run_dir'."
        )
    if m_id is not None and c_id is not None and m_id != c_id:
        raise RuntimeError(
            f"Destino '{run_path}' possui identidades inconsistentes (manifest={m_id!r}, config={c_id!r}). "
            "Recusei para nao corromper artefatos anteriores."
        )
    existing_id = m_id if m_id is not None else c_id
    if existing_id != expected_run_identity:
        raise RuntimeError(
            f"Conflito de identidade em '{run_path}': destino pertence a run_identity={existing_id!r}, "
            f"mas a execucao atual tem run_identity={expected_run_identity!r}. "
            "Use outro 'run_id'/'run_dir' ou omita-os para isolamento automatico."
        )


def _validate_configured_storage_paths(config: dict[str, Any]) -> None:
    mount_root = config.get("drive_mount_root")
    validate_storage_path(config.get("output_dir"), label="output_dir", mount_root=mount_root)
    validate_storage_path(config.get("cache_dir"), label="cache_dir", mount_root=mount_root)
    validate_storage_path(config.get("run_dir"), label="run_dir", mount_root=mount_root)


def run_smoke_audit(
    article: str,
    config: dict[str, Any],
    *,
    run_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run offline data/splits/grid audit without downloading weights or training."""
    if article not in SUPPORTED_ARTICLES:
        raise ValueError(f"Unsupported article: {article!r}. Expected one of {SUPPORTED_ARTICLES}.")

    mount_root = config.get("drive_mount_root")
    _validate_configured_storage_paths(config)
    if run_dir is not None:
        validate_storage_path(run_dir, label="run_dir", mount_root=mount_root)
    eff_cfg = effective_article_config(article, config)
    eff_cfg["mode"] = "smoke"
    dataset = eff_cfg["dataset"]
    input_csv = eff_cfg["input_csv"]

    bundle = load_dataset_bundle(
        dataset, input_csv, warmup=eff_cfg["warmup"], stride=eff_cfg["stride"]
    )
    outer_folds = build_nested_splits(bundle["profile"])
    run_identity = compute_run_identity(article, bundle, eff_cfg)
    training_code_sha = compute_training_code_sha256(article)

    if run_dir is None:
        run_path = _resolve_run_dir(article, dataset, config, "smoke", run_identity=run_identity)
    else:
        run_path = Path(run_dir)
    validate_storage_path(run_path, label="run_dir", mount_root=mount_root)
    _ensure_safe_run_dir(run_path, run_identity)
    run_path.mkdir(parents=True, exist_ok=True)

    atomic_write_json(
        run_path / "config.json",
        {**eff_cfg, "run_identity": run_identity, "training_code_sha256": training_code_sha},
    )
    manifest = {
        "status": "smoke",
        "article": article,
        "dataset": dataset,
        "mode": "smoke",
        "run_identity": run_identity,
        "run_dir": str(run_path),
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "training_version": TRAINING_VERSION,
        "training_code_sha256": training_code_sha,
        "model_id": CHRONOS_MODEL_ID,
        "model_revision": CHRONOS_MODEL_REVISION,
        "csv_sha256": bundle["csv_sha256"],
        "sensor_data_sha256": bundle["sensor_data_sha256"],
        "endpoints_sha256": bundle["endpoints_sha256"],
        "clean_keys": int(len(bundle["clean"])),
        "endpoint_count": int(len(bundle["endpoints"])),
        "config": eff_cfg,
        "outer_folds": [
            {
                "outer_fold_id": f["outer_fold_id"],
                "test_group": f["test_group"],
                "test_ids": list(f["test_ids"]),
                "inner_fold_count": len(f["inner_folds"]),
            }
            for f in outer_folds
        ],
        "environment": collect_environment_info(include_torch=False),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = atomic_write_json(run_path / "manifest.json", manifest)
    zip_path = create_portable_zip(run_path)
    manifest["manifest_path"] = str(manifest_path)
    manifest["zip_path"] = str(zip_path)
    return manifest


def _resolve_input_csv(dataset: str, config: dict[str, Any]) -> str:
    input_csv = config.get("input_csv")
    if input_csv:
        return str(input_csv)
    profile = cer_rul.get_dataset_profile(dataset)
    return str(Path("data") / profile["labels"][DEFAULT_LABEL]["filename"])


def _resolve_run_dir(
    article: str,
    dataset: str,
    config: dict[str, Any],
    mode: str,
    *,
    run_identity: str | None = None,
) -> Path:
    if config.get("run_dir"):
        return Path(config["run_dir"])
    if config.get("run_id"):
        run_id = str(config["run_id"])
    else:
        stage = "train" if mode in ("train", "full") else mode
        suffix = f"_{run_identity}" if run_identity else ""
        run_id = f"{stage}_{article}_{dataset}{suffix}"
    output_root = Path(config.get("output_dir", "results/chronos"))
    return output_root / article / dataset / run_id


def _resolve_cache_dir(article: str, dataset: str, config: dict[str, Any], run_dir: Path) -> Path:
    if config.get("cache_dir"):
        return Path(config["cache_dir"])
    return run_dir.parent / "cache"


def _build_view_metadata(
    bundle: dict[str, Any],
    prep: CausalPreprocessor,
    context_spec: dict[str, Any],
    *,
    extractor_env: dict[str, Any] | None = None,
    extractor_version: str | None = None,
    extractor_code_sha256: str | None = None,
    chronos_package_version: str | None = None,
) -> dict[str, Any]:
    prep_dict = prep.to_dict()
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": extractor_version or EXTRACTOR_VERSION,
        "extractor_code_sha256": extractor_code_sha256 or compute_extractor_code_sha256(),
        "chronos_package_version": chronos_package_version or CHRONOS_PACKAGE_VERSION,
        "dataset": bundle["dataset"],
        "sensor_columns": list(bundle["sensor_columns"]),
        "sensor_data_sha256": bundle["sensor_data_sha256"],
        "endpoints_sha256": bundle["endpoints_sha256"],
        "train_ids": list(prep.train_ids),
        "preprocessor": prep_dict,
        "context_spec": context_spec,
        "model_id": CHRONOS_MODEL_ID,
        "model_revision": CHRONOS_MODEL_REVISION,
        "patch_size": CHRONOS_PATCH_SIZE,
        "d_model": CHRONOS_D_MODEL,
        "use_reg_token": True,
        "extractor_dtype": "float32",
    }
    if extractor_env is not None:
        meta["extractor_env"] = extractor_env
    return meta


def _ensure_view_embeddings(
    bundle: dict[str, Any],
    train_ids: Sequence[int],
    prep_mode: str,
    context_spec: dict[str, Any],
    grids: Sequence[np.ndarray],
    cache_dir: Path,
    *,
    pipeline: Any | None,
    allow_extract: bool,
    batch_size_variates: int = EMBED_BATCH_VARIATES,
) -> tuple[str, np.ndarray, CausalPreprocessor, dict[str, Any]]:
    prep = CausalPreprocessor(prep_mode).fit(bundle["clean"], train_ids, bundle["sensor_columns"])
    prep_dict = prep.to_dict()
    ext_code_sha = compute_extractor_code_sha256()
    pkg_ver = get_chronos_package_version(require_installed=allow_extract)
    view_id = compute_view_identity(
        dataset_name=bundle["dataset"],
        sensor_columns=bundle["sensor_columns"],
        sensor_data_sha256=bundle["sensor_data_sha256"],
        endpoints_sha256=bundle["endpoints_sha256"],
        preprocessor_dict=prep_dict,
        context_spec=context_spec,
        extractor_version=EXTRACTOR_VERSION,
        extractor_code_sha256=ext_code_sha,
        chronos_package_version=pkg_ver,
    )
    expected_meta = _build_view_metadata(
        bundle,
        prep,
        context_spec,
        extractor_version=EXTRACTOR_VERSION,
        extractor_code_sha256=ext_code_sha,
        chronos_package_version=pkg_ver,
    )
    npz_path, json_path = cache_paths(cache_dir, view_id)
    if npz_path.is_file() and json_path.is_file():
        arr, meta = load_embedding_cache(cache_dir, view_id, expected_meta)
        return view_id, arr, prep, meta

    if not allow_extract:
        raise FileNotFoundError(
            f"Cached embeddings not found for view_id={view_id} (train_ids={list(prep.train_ids)}, "
            f"context={context_spec}) in {cache_dir}. Run mode='extract' or 'full' first."
        )
    if pipeline is None:
        raise RuntimeError("Chronos pipeline is required for embedding extraction.")

    normed_grids = prep.transform_many(grids)
    arr = extract_last_context_patch_embeddings(
        pipeline, normed_grids, batch_size_variates=batch_size_variates
    )
    ext_env = collect_environment_info(include_torch=True)
    full_meta = _build_view_metadata(
        bundle,
        prep,
        context_spec,
        extractor_env=ext_env,
        extractor_version=EXTRACTOR_VERSION,
        extractor_code_sha256=ext_code_sha,
        chronos_package_version=pkg_ver,
    )
    save_embedding_cache(cache_dir, view_id, arr, full_meta)
    return view_id, arr, prep, full_meta


def _collect_cache_zip_entries(cache_dir: Path, views_registry: dict[str, Any]) -> dict[str, Path]:
    entries: dict[str, Path] = {}
    for vid in sorted(views_registry):
        npz_path, json_path = cache_paths(cache_dir, vid)
        if not npz_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"Expected extracted cache files missing for view_id={vid} in {cache_dir}")
        entries[f"cache/{vid}.npz"] = npz_path
        entries[f"cache/{vid}.json"] = json_path
    return entries


def _save_prediction_plots(pred_df: pd.DataFrame, out_dir: Path, title_prefix: str) -> list[str]:
    if pred_df.empty:
        return []
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for eq_name, eq_df in pred_df.groupby("equipment", sort=True):
        fault_ids = sorted(eq_df["Falha"].unique())
        fig, axes = plt.subplots(len(fault_ids), 1, figsize=(8, 3 * len(fault_ids)), squeeze=False)
        for idx, fid in enumerate(fault_ids):
            ax = axes[idx, 0]
            sub = eq_df.loc[eq_df["Falha"] == fid].sort_values("record_index")
            ax.plot(sub["record_index"], sub["y_true"], label="RUL real", color="black", linewidth=1.5)
            ax.plot(sub["record_index"], sub["pred_raw"], label="Pred raw", alpha=0.85)
            ax.set_title(f"{title_prefix} | {eq_name} | Falha {int(fid)}")
            ax.set_xlabel("record_index")
            ax.set_ylabel("RUL (0..1)")
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc="best")
        fig.tight_layout()
        out_path = plots_dir / f"curves_{eq_name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(str(out_path))
    return saved


# ---------------------------------------------------------------------------
# Method A — El-Ghoussani et al. (arXiv:2606.11990v3)
# ---------------------------------------------------------------------------

def count_parameters(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def build_elghoussani_head(
    arch: str = "mlp2",
    *,
    d_model: int = CHRONOS_D_MODEL,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> Any:
    """Build Method A regression head over channel-averaged Chronos-2 embeddings.

    Hypothesis note: El-Ghoussani et al. do not specify multivariate channel
    pooling, hidden width ``m``, or dropout ``p``. Averaging across the 8
    channels (``E.mean(dim=1)`` -> 768), ``m=128``, and ``p=0.2`` are explicit
    methodological choices of this replication, not claims of original defaults.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    if arch == "mlp2":
        layers: list[nn.Module] = [
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        ]
    elif arch == "linear_relu":
        layers = [
            nn.Linear(d_model, 1),
            nn.ReLU(),
        ]
    elif arch == "mlp4":
        layers = [
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        ]
    else:
        raise ValueError(f"Unknown El-Ghoussani head architecture: {arch!r}")

    net = nn.Sequential(*layers)

    class ElGhoussaniHead(nn.Module):
        def __init__(self, body: nn.Module, arch_name: str) -> None:
            super().__init__()
            self.arch = arch_name
            self.body = body

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 8, 768) -> channel mean (B, 768) -> (B, 1)
            pooled = x.mean(dim=1)
            return self.body(pooled)

    return ElGhoussaniHead(net, arch)


def run_elghoussani_pipeline(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Run Method A (El-Ghoussani et al.) for mode in {'extract', 'train', 'full'}."""
    _validate_configured_storage_paths(config)
    eff_cfg = effective_article_config("elghoussani", config)
    eff_cfg["mode"] = mode
    dataset = eff_cfg["dataset"]
    input_csv = eff_cfg["input_csv"]
    mount_root = config.get("drive_mount_root")

    context_candidates = eff_cfg["context_candidates"]
    epochs = eff_cfg["epochs"]
    batch_size = eff_cfg["batch_size"]
    lr = eff_cfg["lr"]
    seeds = eff_cfg["seeds"]
    hidden_dim = eff_cfg["hidden_dim"]
    dropout = eff_cfg["dropout"]
    controls = eff_cfg["controls"]
    device = eff_cfg["device"]
    batch_size_variates = eff_cfg["batch_size_variates"]

    bundle = load_dataset_bundle(
        dataset, input_csv, warmup=eff_cfg["warmup"], stride=eff_cfg["stride"]
    )
    run_identity = compute_run_identity("elghoussani", bundle, eff_cfg)
    training_code_sha = compute_training_code_sha256("elghoussani")
    run_dir = _resolve_run_dir("elghoussani", dataset, config, mode, run_identity=run_identity)
    cache_dir = _resolve_cache_dir("elghoussani", dataset, config, run_dir)
    validate_storage_path(run_dir, label="run_dir", mount_root=mount_root)
    validate_storage_path(cache_dir, label="cache_dir", mount_root=mount_root)
    _ensure_safe_run_dir(run_dir, run_identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "config.json",
        {**eff_cfg, "run_identity": run_identity, "training_code_sha256": training_code_sha},
    )

    outer_folds = build_nested_splits(bundle["profile"])
    endpoints = bundle["endpoints"]
    ep_faults = endpoints["Falha"].to_numpy(int)
    ep_rul = endpoints["RUL"].to_numpy(np.float32)

    grids_by_ctx: dict[int, list[np.ndarray]] = {
        ctx: build_endpoint_grids(bundle, context_steps=ctx) for ctx in context_candidates
    }

    allow_extract = mode in ("extract", "full")
    pipeline = load_chronos_pipeline(device=device) if allow_extract else None

    views_registry: dict[str, Any] = {}
    if allow_extract:
        for outer in outer_folds:
            for ctx in context_candidates:
                ctx_spec = {"kind": "fixed", "context_steps": ctx, "step_hours": GRID_STEP_HOURS, "max_gap_hours": MAX_INTERP_GAP_HOURS}
                for inner in outer["inner_folds"]:
                    vid, _, _, meta = _ensure_view_embeddings(
                        bundle, inner["inner_train_ids"], "clip_p1_p99_zscore", ctx_spec,
                        grids_by_ctx[ctx], cache_dir, pipeline=pipeline, allow_extract=True,
                        batch_size_variates=batch_size_variates,
                    )
                    views_registry[vid] = {"role": "inner", "outer_fold": outer["outer_fold_id"], "inner_fold": inner["inner_fold_id"], "context_steps": ctx}
                vid_out, _, _, meta_out = _ensure_view_embeddings(
                    bundle, outer["outer_train_ids"], "clip_p1_p99_zscore", ctx_spec,
                    grids_by_ctx[ctx], cache_dir, pipeline=pipeline, allow_extract=True,
                    batch_size_variates=batch_size_variates,
                )
                views_registry[vid_out] = {"role": "outer", "outer_fold": outer["outer_fold_id"], "context_steps": ctx}

    if mode == "extract":
        manifest = {
            "status": "extracted",
            "article": "elghoussani",
            "dataset": dataset,
            "mode": mode,
            "run_identity": run_identity,
            "run_dir": str(run_dir),
            "schema_version": SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "extractor_code_sha256": compute_extractor_code_sha256(),
            "training_version": TRAINING_VERSION,
            "training_code_sha256": training_code_sha,
            "chronos_package_version": get_chronos_package_version(require_installed=True),
            "model_id": CHRONOS_MODEL_ID,
            "model_revision": CHRONOS_MODEL_REVISION,
            "csv_sha256": bundle["csv_sha256"],
            "sensor_data_sha256": bundle["sensor_data_sha256"],
            "endpoints_sha256": bundle["endpoints_sha256"],
            "cache_dir": str(cache_dir),
            "portable_cache_relpath": "cache",
            "views_count": len(views_registry),
            "config": eff_cfg,
            "environment": collect_environment_info(include_torch=True),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(run_dir / "views.json", views_registry)
        atomic_write_json(run_dir / "manifest.json", manifest)
        extra_cache_files = _collect_cache_zip_entries(cache_dir, views_registry)
        portable_cfg = {
            **eff_cfg,
            "run_identity": run_identity,
            "training_code_sha256": training_code_sha,
            "cache_dir": "cache",
            "run_dir": None,
            "run_id": None,
        }
        zip_path = create_portable_zip(
            run_dir,
            extra_files=extra_cache_files,
            override_texts={"config.json": json.dumps(portable_cfg, indent=2, sort_keys=True, ensure_ascii=False) + "\n"},
        )
        manifest["zip_path"] = str(zip_path)
        return manifest

    # mode in ("train", "full"): run inner CV + outer refit on CPU
    head_sample = build_elghoussani_head("mlp2", hidden_dim=hidden_dim, dropout=dropout)
    param_counts = {"mlp2": count_parameters(head_sample)}
    for ctrl in controls:
        param_counts[ctrl] = count_parameters(build_elghoussani_head(ctrl, hidden_dim=hidden_dim, dropout=dropout))

    selections: dict[str, Any] = {}
    loss_histories: dict[str, Any] = {}
    all_pred_frames: dict[str, list[pd.DataFrame]] = {"mlp2": [], **{c: [] for c in controls}}
    ckpt_dir = run_dir / "checkpoints"
    resume_dir = run_dir / "resume"
    resumed_steps = 0
    trained_steps = 0

    for outer in outer_folds:
        outer_id = outer["outer_fold_id"]
        inner_scores_by_ctx: dict[int, dict[str, Any]] = {}

        for ctx in context_candidates:
            ctx_spec = {"kind": "fixed", "context_steps": ctx, "step_hours": GRID_STEP_HOURS, "max_gap_hours": MAX_INTERP_GAP_HOURS}
            seed_inner_maes: dict[int, float] = {}
            inner_views: list[tuple[dict[str, Any], str, np.ndarray]] = []
            for inner in outer["inner_folds"]:
                vid, emb_all, _, _ = _ensure_view_embeddings(
                    bundle, inner["inner_train_ids"], "clip_p1_p99_zscore", ctx_spec,
                    grids_by_ctx[ctx], cache_dir, pipeline=pipeline, allow_extract=False,
                )
                views_registry[vid] = {"role": "inner", "outer_fold": outer_id, "inner_fold": inner["inner_fold_id"], "context_steps": ctx}
                inner_views.append((inner, vid, emb_all))
            inner_view_ids = [vid for _, vid, _ in inner_views]

            for seed in seeds:
                inner_resume_path = resume_dir / f"inner_{outer_id}_ctx{ctx}_seed{seed}.json"
                reused_inner = False
                if inner_resume_path.is_file():
                    r_data = json.loads(inner_resume_path.read_text(encoding="utf-8"))
                    if (
                        r_data.get("run_identity") == run_identity
                        and r_data.get("training_version") == TRAINING_VERSION
                        and r_data.get("training_code_sha256") == training_code_sha
                        and r_data.get("view_ids") == inner_view_ids
                    ):
                        seed_inner_maes[seed] = float(r_data["macro_mae"])
                        resumed_steps += len(inner_views)
                        reused_inner = True
                if reused_inner:
                    continue

                val_frames: list[pd.DataFrame] = []
                for inner, _vid, emb_all in inner_views:
                    tr_mask = np.isin(ep_faults, inner["inner_train_ids"])
                    va_mask = np.isin(ep_faults, inner["val_ids"])
                    seed_everything(seed)
                    head = build_elghoussani_head("mlp2", hidden_dim=hidden_dim, dropout=dropout)
                    res = train_head_epochs(
                        head, emb_all[tr_mask], ep_rul[tr_mask],
                        optimizer_name="adam", lr=lr, batch_size=batch_size, max_epochs=epochs, seed=seed,
                    )
                    trained_steps += 1
                    va_pred_raw = predict_head(res["head"], emb_all[va_mask])
                    va_df = endpoints.loc[va_mask, ["endpoint_id", "Falha", "equipment", "Date", "record_index", "phase"]].copy()
                    va_df["y_true"] = ep_rul[va_mask].astype(np.float64)
                    va_df["pred_raw"] = va_pred_raw
                    val_frames.append(va_df)
                combined_va = pd.concat(val_frames, ignore_index=True)
                macro_mae = compute_hierarchical_metrics(combined_va, pred_col="pred_raw")["macro"]["mae"]
                seed_inner_maes[seed] = macro_mae
                atomic_write_json(inner_resume_path, {
                    "run_identity": run_identity,
                    "training_version": TRAINING_VERSION,
                    "training_code_sha256": training_code_sha,
                    "view_ids": inner_view_ids,
                    "macro_mae": macro_mae,
                })

            inner_scores_by_ctx[ctx] = {
                "mean_macro_mae": float(np.mean(list(seed_inner_maes.values()))),
                "by_seed_macro_mae": {str(k): v for k, v in seed_inner_maes.items()},
            }

        best_ctx = min(context_candidates, key=lambda c: (inner_scores_by_ctx[c]["mean_macro_mae"], c))
        selections[outer_id] = {
            "selected_context_steps": int(best_ctx),
            "inner_scores": {str(k): v for k, v in inner_scores_by_ctx.items()},
            "outer_train_ids": list(outer["outer_train_ids"]),
            "test_ids": list(outer["test_ids"]),
        }

        # Outer refit with frozen best_ctx
        best_ctx_spec = {"kind": "fixed", "context_steps": best_ctx, "step_hours": GRID_STEP_HOURS, "max_gap_hours": MAX_INTERP_GAP_HOURS}
        vid_out, emb_out, _, _ = _ensure_view_embeddings(
            bundle, outer["outer_train_ids"], "clip_p1_p99_zscore", best_ctx_spec,
            grids_by_ctx[best_ctx], cache_dir, pipeline=pipeline, allow_extract=False,
        )
        views_registry[vid_out] = {"role": "outer", "outer_fold": outer_id, "context_steps": best_ctx}

        tr_mask = np.isin(ep_faults, outer["outer_train_ids"])
        te_mask = np.isin(ep_faults, outer["test_ids"])

        for arch in ["mlp2"] + controls:
            for seed in seeds:
                ckpt_path = ckpt_dir / f"{arch}_{outer_id}_seed{seed}.pt"
                outer_resume_path = resume_dir / f"outer_{arch}_{outer_id}_seed{seed}.json"
                reused_outer = False
                if ckpt_path.is_file() and outer_resume_path.is_file():
                    o_data = json.loads(outer_resume_path.read_text(encoding="utf-8"))
                    if (
                        o_data.get("run_identity") == run_identity
                        and o_data.get("training_version") == TRAINING_VERSION
                        and o_data.get("training_code_sha256") == training_code_sha
                        and o_data.get("view_id") == vid_out
                        and o_data.get("context_steps") == best_ctx
                    ):
                        head = build_elghoussani_head(arch, hidden_dim=hidden_dim, dropout=dropout)
                        head.load_state_dict(load_head_checkpoint(ckpt_path))
                        loss_histories[f"{arch}_{outer_id}_seed{seed}"] = list(o_data["train_losses"])
                        resumed_steps += 1
                        reused_outer = True

                if not reused_outer:
                    seed_everything(seed)
                    head = build_elghoussani_head(arch, hidden_dim=hidden_dim, dropout=dropout)
                    res = train_head_epochs(
                        head, emb_out[tr_mask], ep_rul[tr_mask],
                        optimizer_name="adam", lr=lr, batch_size=batch_size, max_epochs=epochs, seed=seed,
                    )
                    trained_steps += 1
                    save_head_checkpoint(ckpt_path, res["state_dict"])
                    loss_histories[f"{arch}_{outer_id}_seed{seed}"] = res["train_losses"]
                    atomic_write_json(outer_resume_path, {
                        "run_identity": run_identity,
                        "training_version": TRAINING_VERSION,
                        "training_code_sha256": training_code_sha,
                        "view_id": vid_out,
                        "context_steps": best_ctx,
                        "train_losses": res["train_losses"],
                    })
                    head = res["head"]

                te_pred_raw = predict_head(head, emb_out[te_mask])
                te_df = endpoints.loc[te_mask, ["endpoint_id", "Falha", "equipment", "Date", "record_index", "phase"]].copy()
                te_df["outer_fold"] = outer_id
                te_df["arch"] = arch
                te_df["seed"] = seed
                te_df["context_steps"] = best_ctx
                te_df["y_true"] = ep_rul[te_mask].astype(np.float64)
                te_df["pred_raw"] = te_pred_raw
                te_df["pred_clipped_0_1"] = np.clip(te_pred_raw, 0.0, 1.0)
                all_pred_frames[arch].append(te_df)

    metrics_summary: dict[str, Any] = {}
    metric_rows_csv: list[dict[str, Any]] = []
    combined_preds_list: list[pd.DataFrame] = []

    for arch, frames in all_pred_frames.items():
        arch_df = pd.concat(frames, ignore_index=True)
        combined_preds_list.append(arch_df)
        raw_by_seed = []
        clipped_by_seed = []
        for seed in seeds:
            sdf = arch_df.loc[arch_df["seed"] == seed]
            raw_by_seed.append(compute_hierarchical_metrics(sdf, pred_col="pred_raw"))
            clipped_by_seed.append(compute_hierarchical_metrics(sdf, pred_col="pred_clipped_0_1"))
        agg_raw = aggregate_seed_metrics(raw_by_seed)
        agg_clip = aggregate_seed_metrics(clipped_by_seed)
        metrics_summary[arch] = {
            "param_count": param_counts[arch],
            "raw": agg_raw,
            "clipped_0_1": agg_clip,
            "by_seed_raw": {str(s): m for s, m in zip(seeds, raw_by_seed)},
        }
        for variant, agg in (("raw", agg_raw), ("clipped_0_1", agg_clip)):
            for scope in ("macro", "pooled"):
                metric_rows_csv.append({
                    "article": "elghoussani",
                    "dataset": dataset,
                    "arch": arch,
                    "variant": variant,
                    "scope": scope,
                    "mae_mean": agg[scope]["mae"]["mean"],
                    "mae_std": agg[scope]["mae"]["std"],
                    "rmse_mean": agg[scope]["rmse"]["mean"],
                    "rmse_std": agg[scope]["rmse"]["std"],
                    "bias_mean": agg[scope]["bias"]["mean"],
                    "bias_std": agg[scope]["bias"]["std"],
                })

    all_preds_df = pd.concat(combined_preds_list, ignore_index=True)
    all_preds_df.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows_csv).to_csv(run_dir / "metrics.csv", index=False)

    primary_mean_df = (
        all_preds_df.loc[all_preds_df["arch"] == "mlp2"]
        .groupby(["endpoint_id", "Falha", "equipment", "Date", "record_index", "phase", "y_true"], as_index=False)[["pred_raw", "pred_clipped_0_1"]]
        .mean()
    )
    plots = _save_prediction_plots(primary_mean_df, run_dir, f"El-Ghoussani ({dataset})")

    atomic_write_json(run_dir / "views.json", views_registry)
    atomic_write_json(run_dir / "selections.json", selections)
    atomic_write_json(run_dir / "loss_histories.json", loss_histories)
    atomic_write_json(run_dir / "metrics.json", metrics_summary)

    manifest = {
        "status": "trained",
        "article": "elghoussani",
        "dataset": dataset,
        "mode": mode,
        "run_identity": run_identity,
        "run_dir": str(run_dir),
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "training_version": TRAINING_VERSION,
        "training_code_sha256": training_code_sha,
        "model_id": CHRONOS_MODEL_ID,
        "model_revision": CHRONOS_MODEL_REVISION,
        "csv_sha256": bundle["csv_sha256"],
        "sensor_data_sha256": bundle["sensor_data_sha256"],
        "endpoints_sha256": bundle["endpoints_sha256"],
        "cache_dir": str(cache_dir),
        "config": eff_cfg,
        "resumed_steps": resumed_steps,
        "trained_steps": trained_steps,
        "param_counts": param_counts,
        "selections": selections,
        "metrics_summary": metrics_summary,
        "plots": plots,
        "consumer_environment": collect_environment_info(include_torch=True),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    zip_path = create_portable_zip(run_dir)
    manifest["zip_path"] = str(zip_path)
    return manifest


# ---------------------------------------------------------------------------
# Method B — Abdouni et al. (PHME 4906): Compress-and-Flatten + Wide & Deep
# ---------------------------------------------------------------------------

def build_abdouni_head(
    variant: str = "fusion",
    *,
    d_model: int = CHRONOS_D_MODEL,
    n_channels: int = 8,
    d_comp: int = 8,
    dropout: float = 0.2,
) -> Any:
    """Build Method B head (Compress-and-Flatten + Wide & Deep).

    ``variant`` in {'fusion', 'deep', 'wide'}:
    - 'fusion': shared Linear(768->8)+GELU per channel -> flatten(64) + wide(8) -> 72 -> MLP(256, 64, 1)
    - 'deep':   shared Linear(768->8)+GELU per channel -> flatten(64) -> 64 -> MLP(256, 64, 1)
    - 'wide':   wide(8) -> 8 -> MLP(256, 64, 1) (control ablation)

    Dropout(0.2) after each hidden layer (256 and 64) is an explicit
    interpretation of PHME 4906 §3. Output is linear (no implicit clamp).
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    if variant not in ("fusion", "deep", "wide"):
        raise ValueError(f"Unknown Abdouni variant: {variant!r}")

    if variant == "fusion":
        in_dim = n_channels * d_comp + n_channels
    elif variant == "deep":
        in_dim = n_channels * d_comp
    else:
        in_dim = n_channels

    class AbdouniHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.variant = variant
            self.compressor = (
                nn.Sequential(nn.Linear(d_model, d_comp), nn.GELU())
                if variant in ("fusion", "deep")
                else None
            )
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(64, 1),
            )

        def forward(self, emb: torch.Tensor, wide: torch.Tensor | None = None) -> torch.Tensor:
            if self.variant == "wide":
                if wide is None:
                    raise ValueError("wide tensor is required for 'wide' variant.")
                return self.mlp(wide)
            assert self.compressor is not None
            deep_feat = self.compressor(emb).reshape(emb.shape[0], -1)
            if self.variant == "deep":
                return self.mlp(deep_feat)
            if wide is None:
                raise ValueError("wide tensor is required for 'fusion' variant.")
            return self.mlp(torch.cat([deep_feat, wide], dim=-1))

    return AbdouniHead()


def run_abdouni_pipeline(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Run Method B (Abdouni et al.) for mode in {'extract', 'train', 'full'}."""
    _validate_configured_storage_paths(config)
    eff_cfg = effective_article_config("abdouni", config)
    eff_cfg["mode"] = mode
    dataset = eff_cfg["dataset"]
    input_csv = eff_cfg["input_csv"]
    mount_root = config.get("drive_mount_root")

    d_comp = eff_cfg["d_comp"]
    max_epochs = eff_cfg["max_epochs"]
    patience = eff_cfg["patience"]
    batch_size = eff_cfg["batch_size"]
    lr = eff_cfg["lr"]
    weight_decay = eff_cfg["weight_decay"]
    dropout = eff_cfg["dropout"]
    seeds = eff_cfg["seeds"]
    variants = eff_cfg["variants"]
    device = eff_cfg["device"]
    batch_size_variates = eff_cfg["batch_size_variates"]

    bundle = load_dataset_bundle(
        dataset, input_csv, warmup=eff_cfg["warmup"], stride=eff_cfg["stride"]
    )
    run_identity = compute_run_identity("abdouni", bundle, eff_cfg)
    training_code_sha = compute_training_code_sha256("abdouni")
    run_dir = _resolve_run_dir("abdouni", dataset, config, mode, run_identity=run_identity)
    cache_dir = _resolve_cache_dir("abdouni", dataset, config, run_dir)
    validate_storage_path(run_dir, label="run_dir", mount_root=mount_root)
    validate_storage_path(cache_dir, label="cache_dir", mount_root=mount_root)
    _ensure_safe_run_dir(run_dir, run_identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "config.json",
        {**eff_cfg, "run_identity": run_identity, "training_code_sha256": training_code_sha},
    )

    outer_folds = build_nested_splits(bundle["profile"])
    endpoints = bundle["endpoints"]
    ep_faults = endpoints["Falha"].to_numpy(int)
    ep_rul = endpoints["RUL"].to_numpy(np.float32)

    ctx_spec = {
        "kind": "full_prefix",
        "context_steps": None,
        "step_hours": GRID_STEP_HOURS,
        "max_gap_hours": MAX_INTERP_GAP_HOURS,
        "max_context_steps": CHRONOS_MAX_CONTEXT_STEPS,
    }
    grids = build_endpoint_grids(bundle, context_steps=None)

    allow_extract = mode in ("extract", "full")
    pipeline = load_chronos_pipeline(device=device) if allow_extract else None

    views_registry: dict[str, Any] = {}
    if allow_extract:
        for outer in outer_folds:
            for inner in outer["inner_folds"]:
                vid, _, _, _ = _ensure_view_embeddings(
                    bundle, inner["inner_train_ids"], "zscore", ctx_spec,
                    grids, cache_dir, pipeline=pipeline, allow_extract=True,
                    batch_size_variates=batch_size_variates,
                )
                views_registry[vid] = {"role": "inner", "outer_fold": outer["outer_fold_id"], "inner_fold": inner["inner_fold_id"]}
            vid_out, _, _, _ = _ensure_view_embeddings(
                bundle, outer["outer_train_ids"], "zscore", ctx_spec,
                grids, cache_dir, pipeline=pipeline, allow_extract=True,
                batch_size_variates=batch_size_variates,
            )
            views_registry[vid_out] = {"role": "outer", "outer_fold": outer["outer_fold_id"]}

    if mode == "extract":
        manifest = {
            "status": "extracted",
            "article": "abdouni",
            "dataset": dataset,
            "mode": mode,
            "run_identity": run_identity,
            "run_dir": str(run_dir),
            "schema_version": SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "extractor_code_sha256": compute_extractor_code_sha256(),
            "training_version": TRAINING_VERSION,
            "training_code_sha256": training_code_sha,
            "chronos_package_version": get_chronos_package_version(require_installed=True),
            "model_id": CHRONOS_MODEL_ID,
            "model_revision": CHRONOS_MODEL_REVISION,
            "csv_sha256": bundle["csv_sha256"],
            "sensor_data_sha256": bundle["sensor_data_sha256"],
            "endpoints_sha256": bundle["endpoints_sha256"],
            "cache_dir": str(cache_dir),
            "portable_cache_relpath": "cache",
            "views_count": len(views_registry),
            "config": eff_cfg,
            "environment": collect_environment_info(include_torch=True),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(run_dir / "views.json", views_registry)
        atomic_write_json(run_dir / "manifest.json", manifest)
        extra_cache_files = _collect_cache_zip_entries(cache_dir, views_registry)
        portable_cfg = {
            **eff_cfg,
            "run_identity": run_identity,
            "training_code_sha256": training_code_sha,
            "cache_dir": "cache",
            "run_dir": None,
            "run_id": None,
        }
        zip_path = create_portable_zip(
            run_dir,
            extra_files=extra_cache_files,
            override_texts={"config.json": json.dumps(portable_cfg, indent=2, sort_keys=True, ensure_ascii=False) + "\n"},
        )
        manifest["zip_path"] = str(zip_path)
        return manifest

    param_counts = {
        v: count_parameters(build_abdouni_head(v, d_comp=d_comp, dropout=dropout))
        for v in variants
    }
    selections: dict[str, Any] = {}
    loss_histories: dict[str, Any] = {}
    all_pred_frames: dict[str, list[pd.DataFrame]] = {v: [] for v in variants}
    ckpt_dir = run_dir / "checkpoints"
    resume_dir = run_dir / "resume"
    resumed_steps = 0
    trained_steps = 0

    for outer in outer_folds:
        outer_id = outer["outer_fold_id"]
        selections[outer_id] = {"outer_train_ids": list(outer["outer_train_ids"]), "test_ids": list(outer["test_ids"]), "variants": {}}

        # Preload inner views & wide features once per inner fold
        inner_data: list[tuple[dict[str, Any], str, np.ndarray, np.ndarray]] = []
        for inner in outer["inner_folds"]:
            vid, emb_all, prep, _ = _ensure_view_embeddings(
                bundle, inner["inner_train_ids"], "zscore", ctx_spec,
                grids, cache_dir, pipeline=pipeline, allow_extract=False,
            )
            views_registry[vid] = {"role": "inner", "outer_fold": outer_id, "inner_fold": inner["inner_fold_id"]}
            wide_all = prep.transform_current_vector(grids)
            inner_data.append((inner, vid, emb_all, wide_all))
        inner_view_ids = [vid for _, vid, _, _ in inner_data]

        vid_out, emb_out, prep_out, _ = _ensure_view_embeddings(
            bundle, outer["outer_train_ids"], "zscore", ctx_spec,
            grids, cache_dir, pipeline=pipeline, allow_extract=False,
        )
        views_registry[vid_out] = {"role": "outer", "outer_fold": outer_id}
        wide_out = prep_out.transform_current_vector(grids)

        tr_out_mask = np.isin(ep_faults, outer["outer_train_ids"])
        te_out_mask = np.isin(ep_faults, outer["test_ids"])

        for variant in variants:
            refit_epochs_by_seed: dict[str, int] = {}
            inner_best_by_seed: dict[str, list[int]] = {}
            for seed in seeds:
                inner_resume_path = resume_dir / f"inner_{variant}_{outer_id}_seed{seed}.json"
                reused_inner = False
                if inner_resume_path.is_file():
                    r_data = json.loads(inner_resume_path.read_text(encoding="utf-8"))
                    if (
                        r_data.get("run_identity") == run_identity
                        and r_data.get("training_version") == TRAINING_VERSION
                        and r_data.get("training_code_sha256") == training_code_sha
                        and r_data.get("view_ids") == inner_view_ids
                    ):
                        best_epochs = [int(e) for e in r_data["best_epochs"]]
                        refit_epochs = int(r_data["refit_epochs"])
                        resumed_steps += len(inner_data)
                        reused_inner = True

                if not reused_inner:
                    best_epochs = []
                    for inner, _vid, emb_in, wide_in in inner_data:
                        tr_mask = np.isin(ep_faults, inner["inner_train_ids"])
                        va_mask = np.isin(ep_faults, inner["val_ids"])
                        seed_everything(seed)
                        head = build_abdouni_head(variant, d_comp=d_comp, dropout=dropout)
                        res = train_head_epochs(
                            head,
                            emb_in[tr_mask], ep_rul[tr_mask],
                            wide_train=wide_in[tr_mask],
                            X_val=emb_in[va_mask], y_val=ep_rul[va_mask],
                            wide_val=wide_in[va_mask],
                            optimizer_name="adamw", lr=lr, weight_decay=weight_decay,
                            batch_size=batch_size, max_epochs=max_epochs, patience=patience,
                            seed=seed,
                        )
                        trained_steps += 1
                        best_epochs.append(int(res["best_epoch"]))

                    med_epoch = float(np.median(best_epochs))
                    refit_epochs = max(1, int(math.floor(med_epoch + 0.5)))
                    atomic_write_json(inner_resume_path, {
                        "run_identity": run_identity,
                        "training_version": TRAINING_VERSION,
                        "training_code_sha256": training_code_sha,
                        "view_ids": inner_view_ids,
                        "best_epochs": best_epochs,
                        "refit_epochs": refit_epochs,
                    })

                inner_best_by_seed[str(seed)] = best_epochs
                refit_epochs_by_seed[str(seed)] = refit_epochs

                ckpt_path = ckpt_dir / f"{variant}_{outer_id}_seed{seed}.pt"
                outer_resume_path = resume_dir / f"outer_{variant}_{outer_id}_seed{seed}.json"
                reused_outer = False
                if ckpt_path.is_file() and outer_resume_path.is_file():
                    o_data = json.loads(outer_resume_path.read_text(encoding="utf-8"))
                    if (
                        o_data.get("run_identity") == run_identity
                        and o_data.get("training_version") == TRAINING_VERSION
                        and o_data.get("training_code_sha256") == training_code_sha
                        and o_data.get("view_id") == vid_out
                        and o_data.get("refit_epochs") == refit_epochs
                    ):
                        head_out = build_abdouni_head(variant, d_comp=d_comp, dropout=dropout)
                        head_out.load_state_dict(load_head_checkpoint(ckpt_path))
                        loss_histories[f"{variant}_{outer_id}_seed{seed}"] = list(o_data["train_losses"])
                        resumed_steps += 1
                        reused_outer = True

                if not reused_outer:
                    seed_everything(seed)
                    head_out = build_abdouni_head(variant, d_comp=d_comp, dropout=dropout)
                    res_out = train_head_epochs(
                        head_out,
                        emb_out[tr_out_mask], ep_rul[tr_out_mask],
                        wide_train=wide_out[tr_out_mask],
                        optimizer_name="adamw", lr=lr, weight_decay=weight_decay,
                        batch_size=batch_size, max_epochs=refit_epochs, patience=None,
                        seed=seed,
                    )
                    trained_steps += 1
                    save_head_checkpoint(ckpt_path, res_out["state_dict"])
                    loss_histories[f"{variant}_{outer_id}_seed{seed}"] = res_out["train_losses"]
                    atomic_write_json(outer_resume_path, {
                        "run_identity": run_identity,
                        "training_version": TRAINING_VERSION,
                        "training_code_sha256": training_code_sha,
                        "view_id": vid_out,
                        "refit_epochs": refit_epochs,
                        "train_losses": res_out["train_losses"],
                    })
                    head_out = res_out["head"]

                te_pred_raw = predict_head(head_out, emb_out[te_out_mask], wide=wide_out[te_out_mask])
                te_df = endpoints.loc[te_out_mask, ["endpoint_id", "Falha", "equipment", "Date", "record_index", "phase"]].copy()
                te_df["outer_fold"] = outer_id
                te_df["variant"] = variant
                te_df["seed"] = seed
                te_df["refit_epochs"] = refit_epochs
                te_df["y_true"] = ep_rul[te_out_mask].astype(np.float64)
                te_df["pred_raw"] = te_pred_raw
                te_df["pred_clipped_0_1"] = np.clip(te_pred_raw, 0.0, 1.0)
                all_pred_frames[variant].append(te_df)

            selections[outer_id]["variants"][variant] = {
                "inner_best_epochs_by_seed": inner_best_by_seed,
                "refit_epochs_by_seed": refit_epochs_by_seed,
            }

    metrics_summary: dict[str, Any] = {}
    metric_rows_csv: list[dict[str, Any]] = []
    combined_preds_list: list[pd.DataFrame] = []

    for variant, frames in all_pred_frames.items():
        var_df = pd.concat(frames, ignore_index=True)
        combined_preds_list.append(var_df)
        raw_by_seed = []
        clipped_by_seed = []
        for seed in seeds:
            sdf = var_df.loc[var_df["seed"] == seed]
            raw_by_seed.append(compute_hierarchical_metrics(sdf, pred_col="pred_raw"))
            clipped_by_seed.append(compute_hierarchical_metrics(sdf, pred_col="pred_clipped_0_1"))
        agg_raw = aggregate_seed_metrics(raw_by_seed)
        agg_clip = aggregate_seed_metrics(clipped_by_seed)
        metrics_summary[variant] = {
            "param_count": param_counts[variant],
            "raw": agg_raw,
            "clipped_0_1": agg_clip,
            "by_seed_raw": {str(s): m for s, m in zip(seeds, raw_by_seed)},
        }
        for scale_var, agg in (("raw", agg_raw), ("clipped_0_1", agg_clip)):
            for scope in ("macro", "pooled"):
                metric_rows_csv.append({
                    "article": "abdouni",
                    "dataset": dataset,
                    "variant": variant,
                    "scale_variant": scale_var,
                    "scope": scope,
                    "mae_mean": agg[scope]["mae"]["mean"],
                    "mae_std": agg[scope]["mae"]["std"],
                    "rmse_mean": agg[scope]["rmse"]["mean"],
                    "rmse_std": agg[scope]["rmse"]["std"],
                    "bias_mean": agg[scope]["bias"]["mean"],
                    "bias_std": agg[scope]["bias"]["std"],
                })

    all_preds_df = pd.concat(combined_preds_list, ignore_index=True)
    all_preds_df.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows_csv).to_csv(run_dir / "metrics.csv", index=False)

    primary_variant = "fusion" if "fusion" in all_pred_frames else variants[0]
    primary_mean_df = (
        all_preds_df.loc[all_preds_df["variant"] == primary_variant]
        .groupby(["endpoint_id", "Falha", "equipment", "Date", "record_index", "phase", "y_true"], as_index=False)[["pred_raw", "pred_clipped_0_1"]]
        .mean()
    )
    plots = _save_prediction_plots(primary_mean_df, run_dir, f"Abdouni ({dataset}, {primary_variant})")

    atomic_write_json(run_dir / "views.json", views_registry)
    atomic_write_json(run_dir / "selections.json", selections)
    atomic_write_json(run_dir / "loss_histories.json", loss_histories)
    atomic_write_json(run_dir / "metrics.json", metrics_summary)

    manifest = {
        "status": "trained",
        "article": "abdouni",
        "dataset": dataset,
        "mode": mode,
        "run_identity": run_identity,
        "run_dir": str(run_dir),
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "training_version": TRAINING_VERSION,
        "training_code_sha256": training_code_sha,
        "model_id": CHRONOS_MODEL_ID,
        "model_revision": CHRONOS_MODEL_REVISION,
        "csv_sha256": bundle["csv_sha256"],
        "sensor_data_sha256": bundle["sensor_data_sha256"],
        "endpoints_sha256": bundle["endpoints_sha256"],
        "cache_dir": str(cache_dir),
        "config": eff_cfg,
        "resumed_steps": resumed_steps,
        "trained_steps": trained_steps,
        "param_counts": param_counts,
        "selections": selections,
        "metrics_summary": metrics_summary,
        "plots": plots,
        "consumer_environment": collect_environment_info(include_torch=True),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    zip_path = create_portable_zip(run_dir)
    manifest["zip_path"] = str(zip_path)
    return manifest


def run_article_pipeline(article: str, config: dict[str, Any], mode: str) -> dict[str, Any]:
    if article not in SUPPORTED_ARTICLES:
        raise ValueError(f"Unsupported article: {article!r}")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode!r}")
    if mode == "smoke":
        return run_smoke_audit(article, config)
    if article == "elghoussani":
        return run_elghoussani_pipeline(config, mode)
    if article == "abdouni":
        return run_abdouni_pipeline(config, mode)
    raise ValueError(f"Unsupported article: {article!r}")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronos-2 RUL CPFL methodological replication runner."
    )
    parser.add_argument("--article", required=True, choices=SUPPORTED_ARTICLES)
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--dataset", default=None, help="Optional dataset override (gerador|gearbox).")
    parser.add_argument("--input-csv", default=None, help="Optional CSV path override.")
    parser.add_argument("--output-dir", default=None, help="Optional output root override.")
    parser.add_argument("--cache-dir", default=None, help="Optional cache dir override.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.input_csv is not None:
        config["input_csv"] = args.input_csv
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.cache_dir is not None:
        config["cache_dir"] = args.cache_dir
    elif config.get("cache_dir") and not Path(config["cache_dir"]).is_absolute():
        rel_cache = config_path.parent / config["cache_dir"]
        if rel_cache.is_dir():
            config["cache_dir"] = str(rel_cache)

    result = run_article_pipeline(args.article, config, args.mode)
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
