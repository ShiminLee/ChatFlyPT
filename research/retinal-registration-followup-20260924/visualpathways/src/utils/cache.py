"""Cache helpers for preprocessing outputs.

`cached_*(name, compute_fn, force=False, **kwargs)` returns the cached artefact
if `data/cache/<name>.<ext>` exists and `force=False`, otherwise calls
`compute_fn(**kwargs)`, writes the result, and returns it. Module name is
`cache` (not `io`) to avoid the stdlib collision.

**Cache↔config guard.** Each artefact is written with a tiny
`<name>.<ext>.buildinfo.json` sidecar recording the build context
(`dataset`, `side`, `neuprint_dataset`, `built`). On a cache *hit* the sidecar
is checked and a loud warning is printed if the cache was built against a
different NeuPrint dataset than the one `.env` selects now
(`NEUPRINT_DATASET_NAME`). This addresses the silent-stale-cache footgun called
out in `docs/run_order.md`: the cache filename encodes only the *local*
`config.DATASET` tag and `SIDE_CHAR`, which are decoupled from the actual data
source — so bumping the NeuPrint dataset (e.g. `:v1.0` → `:v1.1`) without
rebuilding leaves the filename unchanged and the stale cache silently loads.
The guard only warns (never raises or deletes); an intentional reuse is one
`force=True` away, and pre-guard caches (no sidecar) are loaded silently.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import pickle

import pandas as pd
import scipy.sparse as sp

from utils import config
from utils.config import CACHE_PROC_DIR


_GROUP_NAME_RENAMES = {
    "CB " + "input": "VCN",
    "OL " + "internal": "VIN",
    "OL " + "output": "VPN",
}


def _path(name: str, ext: str) -> Path:
    CACHE_PROC_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_PROC_DIR / f"{name}.{ext}"


def _build_context() -> dict:
    """Current build context for provenance stamping (read fresh each call)."""
    return {
        "dataset": config.DATASET,
        "side": config.SIDE,
        "neuprint_dataset": os.environ.get("NEUPRINT_DATASET_NAME"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _sidecar_path(p: Path) -> Path:
    return p.with_name(p.name + ".buildinfo.json")


def _write_sidecar(p: Path) -> None:
    """Best-effort provenance stamp beside a freshly written cache artefact.

    Never fatal: a failed stamp must not fail the cache write it accompanies.
    """
    try:
        with open(_sidecar_path(p), "w") as f:
            json.dump(_build_context(), f, indent=2)
    except OSError:
        pass


def _check_sidecar(p: Path) -> None:
    """On a cache hit, warn loudly if built against a different NeuPrint dataset.

    Compares only `neuprint_dataset` — the one axis not already encoded in the
    filename (which carries `config.DATASET`/side). Missing or unreadable
    sidecars are treated as "can't verify" and pass silently, so pre-guard
    caches and both-sides/whole-dataset artefacts never raise false alarms.
    """
    sc = _sidecar_path(p)
    if not sc.exists():
        return
    try:
        with open(sc) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return
    built_ds = info.get("neuprint_dataset")
    current_ds = os.environ.get("NEUPRINT_DATASET_NAME")
    if built_ds and current_ds and built_ds != current_ds:
        print(
            f"[cache] WARNING: {p.name} was built against NeuPrint dataset "
            f"'{built_ds}', but NEUPRINT_DATASET_NAME is now '{current_ds}'. "
            "The cached data may be stale for the current dataset - pass "
            "force=True (or re-run setup_data) to rebuild if this is unexpected."
        )


def _canonicalize_group_names(df: pd.DataFrame) -> pd.DataFrame:
    if not any(col for col in df.columns if col.startswith("main_group")):
        return df
    out = df.copy()
    for col in out.columns:
        if col.startswith("main_group"):
            out[col] = out[col].replace(_GROUP_NAME_RENAMES)
    return out


def cached_parquet(name, compute_fn, *, force=False, verbose=False, **kwargs):
    p = _path(name, "parquet")
    if p.exists() and not force:
        if verbose:
            print(f"[cache] loading from cache: {p}")
        _check_sidecar(p)
        return _canonicalize_group_names(pd.read_parquet(p))
    if verbose:
        reason = "force re-compute" if p.exists() else "no cache found"
        print(f"[cache] computing ({reason}), will write: {p}")
    df = _canonicalize_group_names(compute_fn(**kwargs))
    df.to_parquet(p, index=False)
    _write_sidecar(p)
    return df


def cached_npz(name, compute_fn, *, force=False, **kwargs):
    p = _path(name, "npz")
    if p.exists() and not force:
        _check_sidecar(p)
        return sp.load_npz(p)
    m = compute_fn(**kwargs)
    sp.save_npz(p, m)
    _write_sidecar(p)
    return m


def cached_pickle(name, compute_fn, *, force=False, **kwargs):
    p = _path(name, "pkl")
    if p.exists() and not force:
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"[cache] corrupt cache file {p} ({e}); recomputing")
    obj = compute_fn(**kwargs)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(p)
    return obj
