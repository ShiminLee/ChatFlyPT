"""Path-level filters for pathway example diagrams.

Consumers (``plotting.py::plot_pathway``) feed in a list of per-length edge
DataFrames (output of ``find_paths_of_length`` + ``group_paths``) and get
back the subset of edges that belong to a selected set of complete paths.

Two selection policies:

* :func:`filter_all_paths_to_top_set` — smallest top-weight prefix whose
  cumulative ``path_weight`` first exceeds ``thre_cumsum * w_all``.
* :func:`filter_all_paths_to_top_n` — the ``top_n`` strongest paths.

Both apply ``thre_step_min`` as a strict per-path filter on
``min_edge_weight`` before selection, and optionally route
``necessary_intermediate`` through the upstream ``filter_paths`` so its
semantics are preserved exactly.

``w_all`` is summed over every reconstructed path before any filtering, so
the denominator of ``w_filter / w_all`` is stable as the knobs are tuned.
The returned ``thre_step`` is the weakest surviving edge.
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from connectome_interpreter.path_finding import filter_paths


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _reconstruct_paths_with_trace(
    paths_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Chain-join per-layer edges into complete paths and keep the node trace.

    Analogous to ``connectome_interpreter.external_paths.effective_conn_per_path_from_paths``
    but returns a DataFrame with one row per complete path and columns:

    * ``n0, n1, ..., nL``  — node trace.
    * ``w1, w2, ..., wL``  — per-edge weights along the path.
    * ``path_weight``      — product of edge weights.
    * ``min_edge_weight``  — minimum edge weight along the path.

    Returns ``None`` if ``paths_df`` is empty or missing a required layer.

    The chain-join runs on int64 node codes (labels factorized once up front,
    decoded on the final frame only): hub-heavy instances can reconstruct to
    millions of rows, and object-dtype trace columns plus the full-frame copy
    that ``rename`` makes per layer were enough to exhaust memory.
    """
    if paths_df is None or paths_df.empty:
        return None
    max_layer = int(paths_df["layer"].max())
    layer = paths_df["layer"].astype(int).to_numpy()
    codes, uniques = pd.factorize(
        pd.concat([paths_df["pre"], paths_df["post"]], ignore_index=True)
    )
    uniques = np.asarray(uniques)
    pre_c, post_c = codes[: len(paths_df)], codes[len(paths_df):]
    weight = paths_df["weight"].to_numpy()
    layer_edges = {
        int(L): (pre_c[layer == L], post_c[layer == L], weight[layer == L])
        for L in np.unique(layer)
    }
    if 1 not in layer_edges:
        return None

    p1, q1, w1 = layer_edges[1]
    cur = pd.DataFrame({"n0": p1, "n1": q1, "w1": w1})
    cur["path_weight"] = cur["w1"]
    cur["min_edge_weight"] = cur["w1"]

    for L in range(2, max_layer + 1):
        if L not in layer_edges:
            return None
        pL, qL, wL = layer_edges[L]
        # joining on the shared node column needs no post-merge rename/drop
        nxt = pd.DataFrame({f"n{L-1}": pL, f"n{L}": qL, f"w{L}": wL})
        cur = cur.merge(nxt, on=f"n{L-1}", how="inner")
        wcol = cur[f"w{L}"].to_numpy()
        cur["path_weight"] = cur["path_weight"].to_numpy() * wcol
        cur["min_edge_weight"] = np.minimum(
            cur["min_edge_weight"].to_numpy(), wcol
        )

    if cur.empty:
        return None
    cur = cur.reset_index(drop=True)
    for i in range(max_layer + 1):
        cur[f"n{i}"] = uniques[cur[f"n{i}"].to_numpy()]
    return cur


def _trace_to_edges(trace: pd.DataFrame, max_layer: int) -> pd.DataFrame:
    """Flatten per-path traces into a deduplicated edge DataFrame."""
    parts = []
    for L in range(1, max_layer + 1):
        sub = trace[[f"n{L-1}", f"n{L}", f"w{L}"]].copy()
        sub.columns = ["pre", "post", "weight"]
        sub["layer"] = L
        parts.append(sub.drop_duplicates(subset=["layer", "pre", "post"]))
    if not parts:
        return pd.DataFrame(columns=["layer", "pre", "post", "weight"])
    return pd.concat(parts, ignore_index=True)[["layer", "pre", "post", "weight"]]


def _collect_paths(
    all_paths: List[pd.DataFrame],
    necessary_intermediate: Optional[Dict[int, object]],
):
    """Reconstruct per-length traces and flatten per-path weights for ranking.

    Returns ``(all_paths, traces, w_all, w_prod, w_min, origins_i, origins_row)``
    where ``all_paths`` is the (possibly necessary-intermediate-filtered)
    input list, ``traces[i]`` is the per-path reconstruction for length ``i``
    (or ``None``), and ``w_prod[k]`` / ``w_min[k]`` / ``origins_i[k]`` /
    ``origins_row[k]`` together describe the ``k``th reconstructed path.
    """
    if necessary_intermediate is not None:
        all_paths = [
            filter_paths(p, 0, necessary_intermediate)
            if (p is not None and not p.empty)
            else p
            for p in all_paths
        ]

    traces = [_reconstruct_paths_with_trace(p) for p in all_paths]

    w_all = 0.0
    w_prod: list = []
    w_min: list = []
    origins_i: list = []
    origins_row: list = []
    for i, tr in enumerate(traces):
        if tr is None or tr.empty:
            continue
        wp_i = tr["path_weight"].to_numpy()
        wm_i = tr["min_edge_weight"].to_numpy()
        w_all += float(wp_i.sum())
        w_prod.append(wp_i)
        w_min.append(wm_i)
        origins_i.append(np.full(len(wp_i), i, dtype=np.intp))
        origins_row.append(np.arange(len(wp_i), dtype=np.intp))

    if w_prod:
        w_prod_arr = np.concatenate(w_prod, axis=0)
        w_min_arr = np.concatenate(w_min, axis=0)
        origins_i_arr = np.concatenate(origins_i, axis=0)
        origins_row_arr = np.concatenate(origins_row, axis=0)
    else:
        w_prod_arr = np.array([], dtype=float)
        w_min_arr = np.array([], dtype=float)
        origins_i_arr = np.array([], dtype=np.intp)
        origins_row_arr = np.array([], dtype=np.intp)

    return all_paths, traces, w_all, w_prod_arr, w_min_arr, origins_i_arr, origins_row_arr


def _emit_top(
    all_paths: List[pd.DataFrame],
    traces: List[Optional[pd.DataFrame]],
    top_positions: np.ndarray,
    origins_i: np.ndarray,
    origins_row: np.ndarray,
):
    """Given chosen path positions, emit per-length edge DataFrames and sum
    ``w_filter`` / track the weakest surviving edge.
    """
    empty = [
        pd.DataFrame(columns=["layer", "pre", "post", "weight"]) for _ in all_paths
    ]

    top_by_trace: Dict[int, List[int]] = {i: [] for i in range(len(all_paths))}
    for pos in top_positions:
        top_by_trace[int(origins_i[pos])].append(int(origins_row[pos]))

    w_filter = 0.0
    weakest: List[float] = []
    for i, tr in enumerate(traces):
        rows = top_by_trace[i]
        if not rows or tr is None or tr.empty:
            all_paths[i] = empty[i]
            continue
        sub = tr.iloc[rows]
        w_filter += float(sub["path_weight"].sum())
        weakest.append(float(sub["min_edge_weight"].min()))
        max_layer = sum(
            1 for c in sub.columns if c.startswith("n") and c[1:].isdigit()
        ) - 1
        all_paths[i] = _trace_to_edges(sub, max_layer)

    thre_step = float(min(weakest)) if weakest else 0.0
    return all_paths, w_filter, thre_step


# ---------------------------------------------------------------------------
# Public filters
# ---------------------------------------------------------------------------


def filter_all_paths_to_top_set(
    all_paths: Union[pd.DataFrame, List[pd.DataFrame]],
    thre_cumsum: float = 0.5,
    thre_step_min: float = 0.0,
    necessary_intermediate: Optional[Dict[int, object]] = None,
):
    """Keep the smallest top-weight prefix of paths whose cumulative effective
    weight first exceeds ``thre_cumsum`` of the total effective weight; drop
    any such path whose weakest edge is ``<= thre_step_min``.

    Args:
        all_paths (pd.DataFrame | list[pd.DataFrame]): DataFrame or list of
            DataFrames like the output of ``find_paths_of_length`` — each row
            one edge at the given ``layer``.
        thre_cumsum (float): Cumulative effective-weight fraction the top set
            must cover. Between 0 and 1. Strict ``>``. Defaults to 0.5.
        thre_step_min (float, optional): Minimum edge weight a path's weakest
            edge must exceed (strict ``>``). Defaults to 0.0.
        necessary_intermediate (dict, optional): ``{layer: indices}`` —
            delegated to ``filter_paths`` with ``threshold=0``.

    Returns:
        paths, w_filter, w_all, thre_step — ``thre_step`` is the weakest
        edge among surviving paths (``0.0`` if nothing survives).
    """
    df_bool = False
    if isinstance(all_paths, pd.DataFrame):
        df_bool = True
        all_paths = [all_paths]

    all_paths, traces, w_all, w_prod, w_min, origins_i, origins_row = _collect_paths(
        all_paths, necessary_intermediate,
    )
    empty = [
        pd.DataFrame(columns=["layer", "pre", "post", "weight"]) for _ in all_paths
    ]
    if len(w_prod) == 0:
        return (empty[0] if df_bool else empty, 0.0, 0.0, 0.0)

    idx_sort = np.argsort(-w_prod)
    keep = w_min[idx_sort] > thre_step_min
    idx_sort = idx_sort[keep]
    if len(idx_sort) == 0:
        return (empty[0] if df_bool else empty, 0.0, w_all, 0.0)
    cum = np.cumsum(w_prod[idx_sort] / w_all)
    exceed = np.where(cum > thre_cumsum)[0]
    idx_thre = int(exceed[0]) if len(exceed) else len(idx_sort) - 1
    top = idx_sort[: idx_thre + 1]

    all_paths, w_filter, thre_step = _emit_top(
        all_paths, traces, top, origins_i, origins_row,
    )

    if df_bool:
        all_paths = all_paths[0]
    return all_paths, w_filter, w_all, thre_step


def filter_all_paths_to_top_n(
    all_paths: Union[pd.DataFrame, List[pd.DataFrame]],
    top_n: int = 100,
    thre_step_min: float = 0.0,
    necessary_intermediate: Optional[Dict[int, object]] = None,
):
    """Keep the ``top_n`` highest-weight paths among those whose weakest edge
    is ``> thre_step_min``.

    Mirrors :func:`filter_all_paths_to_top_set`'s input/return shape —
    selection is by path count instead of cumulative-weight fraction.

    Args:
        all_paths, thre_step_min, necessary_intermediate: see
            :func:`filter_all_paths_to_top_set`.
        top_n (int): Number of strongest paths to keep. If ``top_n`` exceeds
            the number of surviving paths, all survivors are kept. Defaults
            to 100.

    Returns:
        paths, w_filter, w_all, thre_step — same as
        :func:`filter_all_paths_to_top_set`.
    """
    df_bool = False
    if isinstance(all_paths, pd.DataFrame):
        df_bool = True
        all_paths = [all_paths]

    all_paths, traces, w_all, w_prod, w_min, origins_i, origins_row = _collect_paths(
        all_paths, necessary_intermediate,
    )
    empty = [
        pd.DataFrame(columns=["layer", "pre", "post", "weight"]) for _ in all_paths
    ]
    if len(w_prod) == 0 or top_n <= 0:
        return (empty[0] if df_bool else empty, 0.0, w_all, 0.0)

    idx_sort = np.argsort(-w_prod)
    keep = w_min[idx_sort] > thre_step_min
    idx_sort = idx_sort[keep]
    if len(idx_sort) == 0:
        return (empty[0] if df_bool else empty, 0.0, w_all, 0.0)
    top = idx_sort[:top_n]

    all_paths, w_filter, thre_step = _emit_top(
        all_paths, traces, top, origins_i, origins_row,
    )

    if df_bool:
        all_paths = all_paths[0]
    return all_paths, w_filter, w_all, thre_step
