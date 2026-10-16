"""Dataset-agnostic lateral-flow matrix builders.

Split out of the former `setup_data.py` (its §9-10). Consumed by
`setup_data/s3_effconn.py` (and `s2_layers.py`); works for any dataset (malecns
or flywire). The malecns meta/matrix assembly (§1-8) now lives in `malecns.py`
(same `setup_data/` dir).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import csc_matrix


# === 9. Flow CSV ===


R_FLOW_SEEDS = ["L1_R", "L2_R", "L3_R", "R7_R", "R8_R", "R7d_R", "R8d_R", "HBeyelet_R"]
L_FLOW_SEEDS = [s[:-2] + "_L" for s in R_FLOW_SEEDS]


def fetch_instance_flow(
    inprop,
    idx_to_instance: dict,
    seed_groups: list[str],
    data_dir: Path,
    save_prefix: str,
    steps: int = 20,
    thre: float = 0.1,
) -> pd.DataFrame:
    """Per-neuron hitting times. Writes
    `{save_prefix}{steps}step_{thre}thre_hit.csv` into `data_dir` and returns
    the resulting DataFrame. Takes less than 2 hours.

    The OL-subset sidecar (`{dataset}_OL_{side}_flow_*.csv`) is derived from
    this full-brain CSV on demand by `preprocessing._ensure_ol_sidecars`.
    """
    from connectome_interpreter import find_instance_flow

    find_instance_flow(
        inprop,
        idx_to_instance,
        flow_seed_groups=seed_groups,
        file_path=str(data_dir),
        save_prefix=save_prefix,
        flow_steps=steps,
        flow_thre=thre,
    )
    # Upstream writes two CSVs with `cell_group`; rename to the canonical
    # `instance` column in place so downstream readers see a single schema.
    out = Path(data_dir) / f"{save_prefix}{steps}step_{thre}thre_hit.csv"
    out_per_group = Path(data_dir) / f"{save_prefix}{steps}step_{thre}thre_hit_per_group.csv"
    df = pd.read_csv(out).rename(columns={"cell_group": "instance"})
    df.to_csv(out, index=False)
    if out_per_group.exists():
        pd.read_csv(out_per_group).rename(
            columns={"cell_group": "instance"}
        ).to_csv(out_per_group, index=False)
    return df


# === 10. Lateral flow ===


def trim_inprop_by_flow(
    inprop: csc_matrix,
    idx_to_instance: dict,
    flow_type_df: pd.DataFrame,
    flow_diff_min: float = 0.0,
    flow_diff_max: float = 20.0,
) -> csc_matrix:
    """Keep edges where `hitting_time_post − hitting_time_pre ∈ (min, max)`.

    Per-instance hitting times come from `flow_type_df` (`instance`,
    `hitting_time`). Edges whose pre or post idx has no matching hitting
    time are dropped. Vectorised over COO; equivalent to a DataFrame-merge
    but avoids the O(nnz) join.
    """
    hit_map = dict(zip(flow_type_df["instance"], flow_type_df["hitting_time"]))
    n = inprop.shape[0]
    hit_by_idx = np.full(n, np.nan, dtype=np.float64)
    for i, instance in idx_to_instance.items():
        h = hit_map.get(instance)
        if h is not None:
            hit_by_idx[i] = h

    coo = inprop.tocoo()
    diff = hit_by_idx[coo.col] - hit_by_idx[coo.row]
    mask = (diff > flow_diff_min) & (diff < flow_diff_max)
    return csc_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])),
        shape=inprop.shape,
    )


def save_lat_flow_0(
    trimmed_inprop: csc_matrix,
    data_dir: Path,
    save_prefix: str,
    output_threshold: float = 1e-6,
) -> None:
    """Write `{save_prefix}0.npz`. Values with `|x| <= output_threshold` are
    dropped so the stored matrix matches `compress_paths` step 0 exactly."""
    m = trimmed_inprop.tocsc().copy()
    m.data[np.abs(m.data) <= output_threshold] = 0
    m.eliminate_zeros()
    sp.sparse.save_npz(Path(data_dir) / f"{save_prefix}0.npz", m)


def compute_lat_flow_sum(
    trimmed_inprop: csc_matrix,
    step_number: int = 10,
    output_threshold: float = 1e-6,
    chunk_size: int = 5000,
) -> csc_matrix:
    """Sum of `A^1 + A^2 + … + A^step_number` for `A = trimmed_inprop`.

    Wraps `compress_paths(save_to_disk=False)` + `add_first_n_matrices`;
    no per-step files are written. GPU-accelerated if available.
    `chunk_size` is capped so `chunk_size * n_cols < 2**31` (the limit of
    `torch.nonzero`); on 8× A100 the full run takes ~15 min.
    """
    from connectome_interpreter import add_first_n_matrices, compress_paths

    steps = compress_paths(
        trimmed_inprop,
        step_number=step_number,
        output_threshold=output_threshold,
        chunkSize=chunk_size,
        save_to_disk=False,
        return_results=True,
    )
    return add_first_n_matrices(steps, len(steps))


def save_lat_flow_sum(
    summed: csc_matrix,
    data_dir: Path,
    save_prefix: str,
) -> None:
    """Write `{save_prefix}sum.npz`."""
    sp.sparse.save_npz(Path(data_dir) / f"{save_prefix}sum.npz", summed.tocsc())
