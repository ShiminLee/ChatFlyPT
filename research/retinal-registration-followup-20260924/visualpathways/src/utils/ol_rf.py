"""Receptive-field utilities for OL neurons (input RFs, ellipse fits, alpha-hull
column extents). Imported as a `utils` submodule — expects `src` on `sys.path`
(handled by the entry-point scripts/notebooks)."""

from utils.config import (
    DATA_DIR, DATASET, PARAMS_DIR, SIDE_CHAR,
)

import pandas as pd
import numpy as np
import re
import warnings
import scipy.sparse as sp
from scipy.spatial import distance, ConvexHull

from utils import olc_client
c = olc_client.connect(verbose=False)

from neuprint import fetch_neurons, fetch_synapses, fetch_adjacencies
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_all_rois, fetch_roi_hierarchy

import navis
import navis.interfaces.neuprint as neu
# from fafbseg import flywire

# The cached RF functions below need cached_parquet plus several `get_*`
# functions from sibling utils modules.
from utils.cache import cached_parquet
from utils.core_data import _SIDE_WORD, get_flow, get_flow_per_group, get_meta, get_ol_flow_type, get_ol_meta
from utils.ol_data import get_ol_connectivity, get_ol_stepsn_sum, get_ol_type_directedness
from utils.vic import get_ol_cb_vic_type, get_ol_type_vic, get_vcbn_types
from utils.alpha_shape import alpha_shape_2d


def count_col_ahull(df, thr_mode="max", remove_frac=0.1, min_remove_frac=0.0,
                    alpha=0.5, min_alpha=0.0, alpha_reduction=0.1):
    """Alpha-hull area of the columns retained from a hex-grid weight distribution.

    Thresholds the weights (``thr_mode`` ``"max"`` = keep points above
    ``remove_frac`` of the max; ``"cumsum"`` = drop the smallest ``remove_frac``
    of total weight), morphs the kept square-grid coordinates into a regular hex
    grid, pads each point with its six hex neighbours, and fits an alpha shape
    (concave hull). If the kept points are insufficient for a hull (empty, <3
    points, or collinear / near-zero-area convex hull) ``remove_frac`` is lowered
    by 0.1 and retried down to ``min_remove_frac``.

    ``df`` is a single-column DataFrame or Series of weights indexed by ``'h,v'``
    coordinate strings (the effwt_vis* column convention). Returns
    ``(edge_points, edges, area, (com_x, com_y), xy_coord_kept, used_frac,
    used_alpha)``; ``area`` is a length-1 list in units of hex-cell area
    (``3*sqrt(3)/2``). On failure ``edge_points``/``edges`` are ``None`` and
    ``area`` reports only the kept-column count.
    """
    if not (0 <= float(remove_frac) <= 1):
        raise ValueError("remove_frac must be in [0, 1]")
    if not (0 <= float(min_remove_frac) <= float(remove_frac)):
        raise ValueError("min_remove_frac must be in [0, remove_frac]")

    # check df is a single column DataFrame or Series
    if not isinstance(df, (pd.DataFrame, pd.Series)) or (isinstance(df, pd.DataFrame) and df.shape[1] != 1):
        raise ValueError("df must be a pandas DataFrame with a single column or a pandas Series")

    # check df is not empty
    if df.values.sum() == 0:
        warnings.warn("weights sum to 0 Returning NaN results.")
        return [], set(), [np.nan], (np.nan, np.nan), None, float(remove_frac), alpha

    # convert df to series if it's a single column DataFrame
    if isinstance(df, pd.DataFrame):
        df = df.squeeze(axis=1)

    # get xy coord from index; entries without a finite 'h,v' coordinate (e.g.
    # the 'nan' group of column-less visual neurons) would NaN-poison ConvexHull
    # at every retry, so drop them up front
    xy_coord = df.index.to_series().str.split(',', expand=True).apply(
        pd.to_numeric, errors='coerce')
    xy_coord.columns = ['x', 'y']
    _valid = np.isfinite(xy_coord).all(axis=1)
    if not _valid.all():
        warnings.warn(f"Dropping {int((~_valid).sum())} entries without finite "
                      "coordinates before hull fitting.")
        df = df[_valid.values]
        xy_coord = xy_coord[_valid.values]

    # thresholding effective weight points for alpha shape, "max" or "cumsum".
    # remove_frac starts at the user-supplied value, but if the kept points are
    # insufficient for an alpha shape (empty, fewer than 3 points, or collinear /
    # near-zero-area convex hull), we lower remove_frac by 0.1 and retry. If
    # remove_frac reaches min_remove_frac and the checks still fail, we give up
    # and return.
    if thr_mode not in ("max", "cumsum"):
        raise ValueError("Invalid thr_mode. Choose 'max' or 'cumsum'.")

    cur_frac = float(remove_frac)
    while True:
        if thr_mode == "max":
            # remove points with effective weight smaller than x% of max
            thr_val = cur_frac * df.values.max()
        else:  # cumsum
            # remove smallest x% of total weight
            weights_sorted = df.sort_values()
            cutoff = cur_frac * weights_sorted.sum()
            thr_val = weights_sorted[weights_sorted.cumsum() >= cutoff].iloc[0]

        # thresholding
        mask_pos = df.values > thr_val
        xy_coord_kept = xy_coord[mask_pos]
        df_kept = df[mask_pos]

        # Check if we have enough non-collinear points
        check_failed = False
        if len(xy_coord_kept) == 0:
            check_failed = True
        elif len(xy_coord_kept) < 3:
            check_failed = True
        else:
            try:
                hull = ConvexHull(xy_coord_kept.values)
                if hull.volume < 1e-10:  # Nearly zero area
                    check_failed = True
            except Exception:
                check_failed = True

        if not check_failed:
            break  # enough points, proceed with alpha shape

        if cur_frac <= min_remove_frac:
            # give up: return the (failing) result at the current threshold
            if len(xy_coord_kept) == 0:
                return None, None, [0], (np.nan, np.nan), xy_coord_kept, cur_frac, alpha
            com_x = np.average(xy_coord_kept['x'], weights=df_kept.values)
            com_y = np.average(xy_coord_kept['y'], weights=df_kept.values)
            return None, None, [xy_coord_kept.shape[0]], (com_x, com_y), xy_coord_kept, cur_frac, alpha

        # lower the threshold fraction and retry, not going below min_remove_frac
        cur_frac = round(cur_frac - 0.1, 10)
        if cur_frac < min_remove_frac:
            cur_frac = float(min_remove_frac)

    df = df_kept

    # compute com weighted by values
    com_x = np.average(xy_coord_kept['x'], weights=df.values)
    com_y = np.average(xy_coord_kept['y'], weights=df.values)

    # df.index is a square grid, with range x ~ [-15 19], range y ~ [7 72]
    # nb points in square grid is [+1 +1], in hex grid of side = 1 is [1+1/2, 3/2/sqrt(3)]
    # to morph into a regular hex grid with side length = 1
    # multiply x by 3/2 and y by sqrt(3)/2
    xy_coord_kept.loc[:, 'x'] = xy_coord_kept['x'] * 3/2
    xy_coord_kept.loc[:, 'y'] = xy_coord_kept['y'] * np.sqrt(3)/2

    # correction for edge points
    # for each xy_coord_kept, add six points around it with distance 1 in the corrected hex grid
    directions = np.array([[1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
    aux_points = []
    for _, row in xy_coord_kept.iterrows():
        for d in directions:
            aux_points.append([row['x'] + d[0], row['y'] + d[1]])
    # combine edge points with xy_coord_kept for alpha shape
    aux_points = np.array(aux_points)
    aux_points = np.vstack([xy_coord_kept.values, aux_points])

    edge_points, edges, area, used_alpha = alpha_shape_2d(aux_points, alpha=alpha, min_alpha=min_alpha, alpha_reduction=alpha_reduction)

    if len(area) != 1:
        area = [np.nan]

    # in units of hex cell area, which is 3*sqrt(3)/2 * (side length)^2.
    # For side length = 1 a in regular hex grid:
    area = [a / (3 * np.sqrt(3) / 2) for a in area]

    # go back to the original coord
    xy_coord_kept.loc[:, 'x'] = xy_coord_kept['x'] / (3/2)
    xy_coord_kept.loc[:, 'y'] = xy_coord_kept['y'] / (np.sqrt(3)/2)
    # un-morph every hull polygon (not just the single-hull case) so callers can
    # overlay them on the original square-grid coordinates
    for _ep in edge_points:
        _ep[:, 0] = _ep[:, 0] / (3/2)
        _ep[:, 1] = _ep[:, 1] / (np.sqrt(3)/2)
    if len(edge_points) > 1:
        warnings.warn("Multiple ahulls.")

    return edge_points, edges, area, (com_x, com_y), xy_coord_kept, cur_frac, used_alpha


def hexw_columnar(
    bodyId:int,
    thr_qt:float=0.0,
    syntype:str='post',
    roi_str:str='ME(R)',
) -> pd.DataFrame:
    """
    get RF for columnar neurons using upstream connections in column rois

    Args
    ------
    bodyId : int
        bodyId of the neuron
    thr_qt : float
        quantile threshold for synapse count
    syntype : str
        default as post synapse, where to receive input
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    df : pd.DataFrame
        data frame with columns ['bodyId', 'type', 'instance']
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"
    
    # remove (R) from roi_str
    roi = roi_str.replace('(R)', '')

    # query synapse data
    _, roi_counts_df = fetch_neurons(NC(bodyId=bodyId))
    df = roi_counts_df[roi_counts_df['roi'].str.contains(f'{roi}_R_col_')] # keep col rois 
    # thresholding synapse count
    thr_post = df[syntype].quantile(thr_qt)
    df = df[(df[syntype] >= thr_post)]

    # return the input col numbers
    hex = np.array([int(s) for t in df['roi'] for s in re.findall(r'\d+', t)]).reshape(-1,2)
    # add weights
    hexw = pd.DataFrame(
        np.concatenate((hex, df[syntype].to_numpy().reshape(-1,1)), axis=1), 
        columns=['hex1', 'hex2', 'wt']
        )
    hexw.sort_values(by='wt', ascending=False, inplace=True)
    hexw.reset_index(drop=True, inplace=True)

    return hexw


# def pqw_columnar_flywire(
#     bodyId:np.int64,
#     thr_qt:float=0.0,
#     thr_dist:float=10000,
#     syntype:str='post',
#     layer_str:str='M5', #only work with M1/5/10 for now
# ) -> pd.DataFrame:
#     """
#     get RF for columnar neuorns using column rois

#     Args
#     ------
#     bodyId : int
#         bodyId of the neuron
#     thr_qt : float
#         quantile threshold for synapse count
#     thr_dist : float
#         distance threshold between syn and col
#     syntype : str
#         default as post synapse, where to receive input
#     roi_str : str
#         neuprint ROI, can only be ME(R), LO(R), LOP(R)

#     Returns
#     -------
#     df : pd.DataFrame
#         data frame with columns ['bodyId', 'type', 'instance']
#     """

#     assert layer_str in ['M1', 'M5', 'M10'],\
#             f"ROI must be one of 'M1, M5, M10', but is actually '{layer_str}'"
#     assert syntype in ['pre', 'post'],\
#             f"syntype must be one of 'pre, post', but is actually '{syntype}'"
    
#     # load eyemap
#     pqxyztpind = pd.read_excel(EYEMAP_DIR / '783_20240513' / f'pqxyztpid_layercol_{SIDE}.xlsx')
#     pqxyztpind.reset_index(drop=True, inplace=True)
#     pqxyztpind.astype({'rootid': 'Int64'})

#     pre_bool = syntype == 'pre'
#     post_bool = syntype == 'post'
#     # query synapse data
#     syn = flywire.get_synapses(bodyId, pre=pre_bool, post=post_bool)
#     synxyz = syn[[f'{syntype}_x', f'{syntype}_y', f'{syntype}_z']].values
#     # col xyz
#     colxyz = pqxyztpind[[f'x_{layer_str}',f'y_{layer_str}',f'z_{layer_str}']].values
    
#     # compute the distance matrix
#     dist = distance.cdist(colxyz, synxyz)
#     # find the minimum distance for each row
#     min_dist = np.min(dist, axis=1)
#     # index of columns that are smaller than the threshold
#     col_idx = np.where(min_dist < thr_dist)[0]
    
#     # assign weights to each col
#     # find which col_xyz is cloest to each synxyz
#     dist = dist[col_idx,:]
#     if len(dist) == 0:
#         return pd.DataFrame(columns=['p', 'q', 'wt'])
#     # find the minimum distance for each colum
#     min_dist = np.min(dist, axis=0)
#     # remove columns with min_dist > thr
#     dist = dist[:, min_dist < thr_dist]
#     # find the index of the minimum distance for each column
#     min_idx = np.argmin(dist, axis=0)
#     # value count min_idx, sort by index
#     wts = pd.Series(min_idx).value_counts().sort_index()
#     # make an array with wts, wts.index as row numbers, fill missing rows with 0
#     wt_arr = np.zeros(len(col_idx))
#     wt_arr[wts.index] = wts.values

#     # combine pqxyztpind[['p','q']] and wt_arr
#     pqw = np.concatenate((pqxyztpind[['p','q']].values[col_idx], wt_arr[:,None]), axis=1)
#     # convert to df and change type to int, rename columns
#     pqw = pd.DataFrame(pqw, columns=['p','q','wt'])
#     pqw = pqw.astype({'p': int, 'q': int, 'wt': int})
    
#     # thresholding synapse count
#     thr_post = pqw['wt'].quantile(thr_qt)
#     pqw = pqw[(pqw['wt'] >= thr_post)]

#     pqw.sort_values(by='wt', ascending=False, inplace=True)
#     pqw.reset_index(drop=True, inplace=True)
        
#     return pqw
# _VISUAL_INPUT_STEMS = ["L1", "L2", "L3", "R7", "R8"]


def compute_rf(meta_ol, stepsn, inidx, outidx, *, idx_to_coords=None, idx_to_root=None):
    """Single-target RF from the OL-subset `lat_flow_sum` (stepsn).

    Returns a long df (coords, x, y, bodyId, effective weight).
    `idx_to_coords` / `idx_to_root` can be passed in to avoid rebuilding the
    OL-wide dicts on every call.
    """
    from connectome_interpreter.compress_paths import result_summary

    if idx_to_coords is None:
        idx_to_coords = dict(zip(meta_ol.idx, meta_ol.coords))
    if idx_to_root is None:
        idx_to_root = dict(zip(meta_ol.idx, meta_ol.bodyId))
    df = result_summary(
        stepsn, inidx, outidx,
        inidx_map=idx_to_coords, outidx_map=idx_to_root,
        display_threshold=0, display_output=False,
    )
    df = df[(df.index != "nan") & (df.index != "None") & (~df.index.isnull())]
    if df.empty:
        return df
    df = (
        df.iloc[:, 0].to_frame().stack().reset_index()
        .rename(columns={"level_0": "coords", "level_1": "bodyId", 0: "effective weight"})
    )
    xy = df["coords"].str.split(",", expand=True).astype(int).values
    df["x"] = xy[:, 0]
    # `y` is scaled by 1/sqrt(3) so the hex grid is isotropic for the 2D
    # Gaussian fit. The raw `coords` string is preserved for hex plotting.
    df["y"] = xy[:, 1] / np.sqrt(3)
    return df


def _compute_rf_params(meta_ol, stepsn, in_instances, out_instances,
                       cumsum_thre=0.7, *, ahull=True,
                       thr_mode="cumsum", remove_frac=0.6, min_rmfrac=0.1):
    """Per target bodyId, fit a 2D Gaussian and (optionally) an alpha hull from
    the SAME propagated RF.

    `compute_rf` runs once per body and feeds both reductions, so the ellipse
    fit and the alpha-hull size share a single (expensive) propagation pass.

    Returns a long df with the Gaussian columns [n_col, r2, x0, y0, a, b, phi,
    amp] plus, when `ahull=True`, the alpha-hull columns [ahull_size,
    ahull_com_x, ahull_com_y, ahull_kept_points, ahull_rmfrac, ahull_alpha];
    always [bodyId, idx, instance, seed]. `idx` is the full-brain index
    (`idx_full` when `meta_ol` is the OL sidecar, else `idx`) so per-neuron
    joins downstream are unambiguous.

    A body is skipped only when its RF is empty or its total effective weight is
    non-positive. Rows whose Gaussian fit yields NaN or whose hull is degenerate
    are kept with only that block NaN-filled, so ellipse-valid and hull-valid
    bodies are unioned; callers drop NaN per metric. Dicts are built once and
    reused for every body.
    """
    from utils.external_rf import fit_rf_gaussian

    inidx = meta_ol[meta_ol.instance.isin(in_instances)].idx.values
    in_string = "_".join([s[:-2] for s in in_instances])
    idx_to_coords = dict(zip(meta_ol.idx, meta_ol.coords))
    idx_to_root = dict(zip(meta_ol.idx, meta_ol.bodyId))
    # full-brain idx for per-neuron joins (OL sidecar reindexes idx to 0..N-1)
    idx_col = "idx_full" if "idx_full" in meta_ol.columns else "idx"

    rows = []
    for target in out_instances:
        sub = meta_ol[meta_ol.instance == target]
        outidx_arr = sub.idx.values
        bids = sub.bodyId.values
        gidx = sub[idx_col].values
        for i, bid in enumerate(bids):
            df = compute_rf(
                meta_ol, stepsn, inidx, outidx_arr[i:i + 1],
                idx_to_coords=idx_to_coords, idx_to_root=idx_to_root,
            )
            if df.empty:
                continue
            tot_weight = df["effective weight"].sum()
            if tot_weight <= 0:
                continue
            amp = float(df["effective weight"].max())
            params_ij, rf_fitted = fit_rf_gaussian(df, cumsum_thre=cumsum_thre)
            row = {
                "n_col": int(rf_fitted.shape[0]), "r2": params_ij[0],
                "x0": params_ij[1], "y0": params_ij[2],
                "a": params_ij[3], "b": params_ij[4], "phi": params_ij[5],
                "amp": amp, "bodyId": int(bid), "idx": int(gidx[i]),
                "instance": target, "seed": in_string,
            }
            if ahull:
                row.update({
                    "ahull_size": np.nan, "ahull_com_x": np.nan,
                    "ahull_com_y": np.nan, "ahull_kept_points": np.nan,
                    "ahull_rmfrac": np.nan, "ahull_alpha": np.nan,
                })
                try:
                    dfa = df.set_index("coords")[["effective weight"]]
                    (edge_points, edges, area, com, kept_points,
                     used_frac, used_alpha) = count_col_ahull(
                        dfa, thr_mode=thr_mode, remove_frac=remove_frac,
                        min_remove_frac=min_rmfrac,
                    )
                    if edges is not None and len(edge_points) == 1:
                        # count_col_ahull returns `area` as a length-1 list even
                        # on success (a hull-area list); take the scalar.
                        row.update({
                            "ahull_size": area[0], "ahull_com_x": com[0],
                            "ahull_com_y": com[1],
                            "ahull_kept_points": len(kept_points),
                            "ahull_rmfrac": used_frac, "ahull_alpha": used_alpha,
                        })
                except ValueError:
                    pass
            rows.append(row)

    params = pd.DataFrame(rows)
    if params.empty:
        return params
    # Keep the union of ellipse-valid and hull-valid bodies. The previous
    # blanket dropna() is scoped to the ellipse block so a degenerate hull no
    # longer discards a good ellipse row (and vice versa).
    ell_cols = ["r2", "x0", "y0", "a", "b", "phi"]
    if ahull:
        ell_ok = params[ell_cols].notna().all(axis=1)
        hull_ok = params["ahull_size"].notna()
        params = params[ell_ok | hull_ok].reset_index(drop=True)
    else:
        params = params.dropna(subset=ell_cols).reset_index(drop=True)
    return params


def _clean_rf_params(params_df):
    """Apply the RF-param cleaning step:
    clip r2 to [-1, 1], clip a/b to [0.5, inf), derive size / ecc / cell_type.
    """
    df = params_df.copy()
    df["r2"] = df["r2"].clip(lower=-1, upper=1)
    df["a"] = df["a"].clip(lower=0.5)
    df["b"] = df["b"].clip(lower=0.5)
    df["size"] = df["a"] * df["b"] * np.pi * np.sqrt(3) / 2
    df["ecc"] = np.sqrt(1 - (df["b"] / df["a"]) ** 2)
    df["cell_type"] = [s[:-2] for s in df["instance"]]
    return df


_VISUAL_INPUT_STEMS = ["L1", "L2", "L3", "R7", "R8", "R7d", "R8d", "HBeyelet"]
_PHOTORECEPTOR_TYPES = {"R7", "R8", "R7d", "R8d", "HBeyelet"}


def _default_visual_input_instances(side_char: str) -> list[str]:
    suffix = "_R" if side_char == "r" else "_L"
    return [s + suffix for s in _VISUAL_INPUT_STEMS]


def get_rf_raw_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    in_instances: list[str] | None = None, force: bool = False,
) -> pd.DataFrame:
    """Per-body OL receptive-field fits (2D-Gaussian ellipse + alpha hull).

    Fits a 2D Gaussian AND an alpha hull on the effective-weight RF from visual
    inputs (L1/L2/L3/R7/R8 by default) onto every non-visual-input OL instance
    (VIN + VPN) on the selected side, via the OL-subset `lat_flow_sum`.
    Both reductions come from one propagation pass per body (see
    `_compute_rf_params`). Params are cleaned (`_clean_rf_params`) and merged
    with per-instance flow (hitting_time, main_groups, sign). Carries the ellipse
    columns (size, r2, x0, y0, a, b, phi, ...) plus the alpha-hull columns
    (ahull_size, ahull_com_x/y, ahull_kept_points, ahull_rmfrac, ahull_alpha),
    per-neuron `idx` (full-brain), and `region` ('<side> OL').
    """
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        flow_type = get_ol_flow_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )

        ins = in_instances if in_instances is not None else _default_visual_input_instances(side_char)
        out_inst = ol_meta[ol_meta.main_groups != "visual input"].instance.unique()

        params = _compute_rf_params(ol_meta, stepsn, ins, out_inst)
        if params.empty:
            return params
        params = _clean_rf_params(params)
        params = params.merge(
            flow_type[["cell_type", "main_groups", "sign", "hitting_time"]].drop_duplicates("cell_type"),
            on="cell_type", how="left",
        )
        params["region"] = f"{_SIDE_WORD[side_char]} OL"
        return params

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_raw", _compute, force=force,
    )


def get_rf_type_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-OL-instance median RF params + merged directedness + VIC + flow.

    Phi is aggregated via circular mean (period π); other fit params via
    median. `n_neurons` is the count of fitted bodies per (instance, seed).
    """
    from utils.external_rf import circular_mean

    def _compute():
        raw = get_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
        # Ellipse aggregation uses only Gaussian-valid bodies; the raw parquet
        # now also carries hull-only rows (NaN ellipse) that must not skew the
        # per-instance median / n_neurons.
        raw = raw[raw["size"].notna()]
        dir_type = get_ol_type_directedness(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        vic = get_ol_type_vic(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        flow_type = get_ol_flow_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )

        agg = (
            raw.groupby(["instance", "seed"])[["n_col", "r2", "a", "b", "phi", "amp"]]
            .median().reset_index()
        )
        agg["phi"] = (
            raw.groupby(["instance", "seed"])["phi"]
            .apply(lambda s: circular_mean(s.values)).values
        )
        agg["n_neurons"] = (
            raw.groupby(["instance", "seed"])["bodyId"].count().values
        )
        agg = _clean_rf_params(agg)
        agg = agg.merge(
            flow_type[["cell_type", "main_groups", "sign", "hitting_time"]].drop_duplicates("cell_type"),
            on="cell_type", how="left",
        )
        agg = agg.merge(
            dir_type[["instance_pre", "frac_ff", "frac_la", "frac_fb"]],
            left_on="instance", right_on="instance_pre", how="left",
        ).drop(columns=["instance_pre"])
        agg = agg.merge(vic, on="instance", how="left")
        return agg

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_type", _compute, force=force,
    )


def get_rf_connectivity_edges_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, include_cb_post: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Long-form OL edge table with per-type `size_diff` and `layer_diff`.

    Edges come from `get_ol_connectivity`. Post-side sizes are pulled from
    `get_rf_types_combined` (OL + CB-VCBN), so CB-post edges are retained
    when `include_cb_post=True`. Set it to False to match the OL-only subset
    from Step 6a.
    """
    def _compute():
        conn = get_ol_connectivity(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        if include_cb_post:
            rf_type = get_rf_types_combined(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
                vic_thre=vic_thre,
            )
        else:
            rf_type = get_rf_type_ol(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
            )
        sizes = rf_type[["instance", "size"]].drop_duplicates("instance")
        conn = conn.merge(
            sizes.rename(columns={"instance": "instance_pre", "size": "size_pre"}),
            on="instance_pre", how="left",
        )
        conn = conn.merge(
            sizes.rename(columns={"instance": "instance_post", "size": "size_post"}),
            on="instance_post", how="left",
        )
        conn = conn.dropna(subset=["hitting_time_pre", "hitting_time_post", "size_pre", "size_post"])
        conn["layer_diff"] = (
            conn["hitting_time_post"].astype(float) - conn["hitting_time_pre"].astype(float)
        )
        conn["size_diff"] = (
            (conn["size_post"].astype(float) - conn["size_pre"].astype(float))
            / conn["size_pre"].astype(float)
        )
        return conn.reset_index(drop=True)

    tag = "cb" if include_cb_post else "ol"
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_edges_{tag}", _compute, force=force,
    )


def _rf_type_sx_sy(df):
    """Rotate (a, b) back to horizontal / vertical semi-axes based on phi.
    Scale factor `sqrt(sqrt(3)/2)` converts hex-column units to column-radius
    units. Returns (sx, sy) as numpy arrays.
    """
    fac = np.sqrt(np.sqrt(3) / 2)
    horizontal = np.abs(df["phi"]) < np.pi / 4
    sx = np.where(horizontal, fac * df["a"], fac * df["b"])
    sy = np.where(horizontal, fac * df["b"], fac * df["a"])
    return sx, sy


def _synthetic_visual_input_prop(side_char: str) -> pd.DataFrame:
    """Synthetic propagated-RF rows for visual-input seeds (L1/L2/L3) using
    the convention `size=1` column, `sx=sy=1/sqrt(pi)` (unit-area
    circular RF), `phi=0`. Used by `get_rf_comparison_type` so RF-size
    comparisons can include visual inputs as a lower-bound reference."""
    suffix = "_R" if side_char == "r" else "_L"
    r = 1.0 / np.sqrt(np.pi)
    return pd.DataFrame({
        "instance": [f"L1{suffix}", f"L2{suffix}", f"L3{suffix}"],
        "cell_type": ["L1", "L2", "L3"],
        "main_groups": ["visual input"] * 3,
        "sign": [-1.0, 1.0, 1.0],
        "size": [1.0, 1.0, 1.0],
        "sx": [r, r, r],
        "sy": [r, r, r],
        "phi": [0.0, 0.0, 0.0],
    })


def _ol_instances(
    dataset: str, data_dir, side_char: str,
) -> set[str]:
    """OL-side instances, matching the filter in `_ensure_ol_sidecars`.

    Read directly from `get_meta` so callers (e.g. ROI-count / ROI-adj
    downloads) don't need the OL sidecar — which can't be built until the
    full-brain flow and lat_flow files also exist.
    """
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    side_word = _SIDE_WORD[side_char]
    sub = meta[
        (meta.region == f"{side_word} OL")
        & (meta.main_groups != "other")
        & (meta.instance != "R1-R6_R")
    ]
    return set(sub.instance.unique())


def _ol_non_visual_input_instances(
    dataset: str, data_dir, side_char: str,
) -> set[str]:
    """`_ol_instances` minus photoreceptors (R7/R8/R7d/R8d/HBeyelet).

    Excludes only true photoreceptors, not lamina VINs (L1–L5), so direct
    RF fits are computed for L1–L5 alongside VIN/VPN neurons.
    """
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    return _ol_instances(dataset, data_dir, side_char) - set(
        meta[meta.cell_type.isin(_PHOTORECEPTOR_TYPES)].instance.unique()
    )


def get_input_rf_raw_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    source_stem: str | None = None, force: bool = False,
) -> pd.DataFrame:
    """Per-body direct input RF fits, computed from raw neuprint synapse data.

    Reads `data/{source_stem}_input_syn_per_col.parquet` (default
    `{dataset}_OL_{side_char}`), restricts to OL instances excluding
    photoreceptors (R7/R8/R7d/R8d/HBeyelet) and R1-R6 — L1–L5 are included.
    Keeps only each instance's **majority input neuropil** (the ROI holding the
    most post-synapses) so a neuron spanning multiple neuropils yields a single
    ARF / size estimate. Fits a 2D
    Gaussian per (instance, roi, bodyId) on per-column `synapse_frac` as
    `effective weight`, with `amp = sum(effective weight)`. Cleaned via
    `_clean_rf_params`, cached as parquet.

    The raw parquet is produced by `setup_data/malecns.fetch_input_syn_per_col`
    (called from `setup_data/s1_malecns_data.ipynb` section 6). If you rerun the
    download, pass `force=True` to invalidate this cache.
    """
    from utils.external_rf import fit_rf_gaussian
    stem = source_stem or f"{dataset}_OL_{side_char}"

    def _compute():
        raw = pd.read_parquet(data_dir / f"{stem}_input_syn_per_col.parquet")
        instances = _ol_non_visual_input_instances(dataset, data_dir, side_char)
        raw = raw[raw["instance"].isin(instances)].copy()
        # Retain only each instance's majority input neuropil: the ROI holding
        # the most post-synapses. `_fetch_input_neuropil` keeps every ROI with
        # >= 40% of post-synapses, so a type can span two neuropils; collapse to
        # the dominant one so each neuron contributes a single ARF.
        syn_by_roi = (
            raw.groupby(["instance", "roi"])["synapse_count"].sum().reset_index()
        )
        top_roi = syn_by_roi.loc[
            syn_by_roi.groupby("instance")["synapse_count"].idxmax(),
            ["instance", "roi"],
        ]
        raw = raw.merge(top_roi, on=["instance", "roi"], how="inner")
        raw["effective weight"] = raw["synapse_frac"]

        rows = []
        for (inst, roi, bid), sub in raw.groupby(["instance", "roi", "bodyId"]):
            sub_xy = sub[["x", "y", "effective weight"]]
            amp = float(sub_xy["effective weight"].sum())
            params_ij, rf_fitted = fit_rf_gaussian(sub_xy)
            rows.append({
                "n_col": int(rf_fitted.shape[0]),
                "r2": params_ij[0], "x0": params_ij[1], "y0": params_ij[2],
                "a": params_ij[3], "b": params_ij[4], "phi": params_ij[5],
                "amp": amp, "bodyId": int(bid),
                "instance": inst, "roi": roi,
            })
        params = pd.DataFrame(rows)
        if params.empty:
            return params
        params = params.dropna().reset_index(drop=True)
        return _clean_rf_params(params)

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_input_raw", _compute, force=force,
    )


def get_input_rf_type_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    source_stem: str | None = None, force: bool = False,
) -> pd.DataFrame:
    """Per-(instance, roi) median direct input RF fits.

    Aggregates `get_input_rf_raw_ol` by `(instance, roi)`: median on
    `[n_col, r2, a, b, amp]`, `circular_mean` on `phi`, count on `bodyId` ->
    `n_neurons`. Cleaned via `_clean_rf_params`, cached as parquet.
    """
    from utils.external_rf import circular_mean

    def _compute():
        raw = get_input_rf_raw_ol(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            source_stem=source_stem,
        )
        if raw.empty:
            return raw
        numeric = (
            raw.groupby(["instance", "roi"])[["n_col", "r2", "a", "b", "amp"]]
            .median().reset_index()
        )
        phi = (
            raw.groupby(["instance", "roi"])["phi"]
            .apply(lambda x: circular_mean(x.values)).reset_index()
        )
        n_neurons = (
            raw.groupby(["instance", "roi"])["bodyId"].count().reset_index()
            .rename(columns={"bodyId": "n_neurons"})
        )
        out = numeric.merge(phi, on=["instance", "roi"])
        out = out.merge(n_neurons, on=["instance", "roi"])
        return _clean_rf_params(out)

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_input_type", _compute, force=force,
    )


def get_rf_comparison_body(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Merge propagated + direct per-body RFs on (bodyId, instance) with suffixes
    `_prop` / `_dir`."""
    def _compute():
        prop = get_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
        direct = get_input_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
        common = set(prop.columns) & set(direct.columns) - {"bodyId", "instance"}
        prop = prop.rename(columns={c: f"{c}_prop" for c in common})
        direct = direct.rename(columns={c: f"{c}_dir" for c in common})
        return prop.merge(direct, on=["bodyId", "instance"], how="inner")

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_comparison_body", _compute, force=force,
    )


def get_rf_comparison_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Merge propagated + direct per-instance RFs with sx/sy added on each side.

    Visual-input seeds (L1/L2/L3) are appended to the propagated side via
    `_synthetic_visual_input_prop` so they appear in downstream RF-size
    comparisons even though they aren't fitted as outputs in `get_rf_raw_ol`.
    An inner-join is used so only instances present in both the propagated and
    direct-fit tables are kept; L1/L2/L3 appear after their direct-fit cache
    is rebuilt with `get_input_rf_raw_ol(force=True)`.
    """
    def _compute():
        prop = get_rf_type_ol(dataset=dataset, data_dir=data_dir, side_char=side_char).copy()
        direct = get_input_rf_type_ol(dataset=dataset, data_dir=data_dir, side_char=side_char).copy()
        prop["sx"], prop["sy"] = _rf_type_sx_sy(prop)
        direct["sx"], direct["sy"] = _rf_type_sx_sy(direct)
        prop = pd.concat(
            [prop, _synthetic_visual_input_prop(side_char)], ignore_index=True,
        )
        common = set(prop.columns) & set(direct.columns) - {"instance"}
        prop = prop.rename(columns={c: f"{c}_prop" for c in common})
        direct = direct.rename(columns={c: f"{c}_dir" for c in common})
        return prop.merge(direct, on="instance", how="inner")

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_comparison_type", _compute, force=force,
    )


def rebuild_input_rf_caches(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = True,
) -> None:
    """Force-rebuild the four input-RF derived caches in dependency order.

    Run after re-fetching the input syn-per-column parquet (setup_data
    s1_malecns_data section 6) or changing the RF-fitting code. Normal figure
    rendering does not need this: each get_* below lazily builds and caches on
    first call. Order matters -- rebuilding a comparison cache on top of a
    stale raw/type cache would give wrong results.
    """
    get_input_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char, force=force)
    get_input_rf_type_ol(dataset=dataset, data_dir=data_dir, side_char=side_char, force=force)
    get_rf_comparison_body(dataset=dataset, data_dir=data_dir, side_char=side_char, force=force)
    get_rf_comparison_type(dataset=dataset, data_dir=data_dir, side_char=side_char, force=force)


def get_experimental_rf_sizes(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    source_file: str = "TableS2_experimental_data.xlsx",
    sheet: str = "RF size", force: bool = False,
) -> pd.DataFrame:
    """Experimental RF sizes from the experimental-data spreadsheet sheet.

    Resolves `*` wildcards (e.g. `T4*` → `T4a..T4d`), replicating the row's
    RF values across the matching cell types. Adds `instance = cell_type +
    '_R'`. Keeps the sheet's native columns (`size`, `sx`, `sy`, `FWHMx`,
    `FWHMy`, `Reference` for sheet 'RF size'; `mean_short_axis`,
    `mean_long_axis`, `size_ellipse`, ... for sheet 'RF size CC').
    """
    def _compute():
        from utils.polarity import _expand_wildcards, _side_cell_types
        df = pd.read_excel(PARAMS_DIR / source_file, sheet_name=sheet)
        df = df[~df["Neuron type"].isna()].copy()
        universe = _side_cell_types(dataset, data_dir, "r")
        df["cell_type"] = df["Neuron type"].apply(
            lambda c: _expand_wildcards([str(c).strip()], universe) or [str(c).strip()]
        )
        df = df.explode("cell_type").drop(columns=["Neuron type"])
        df["instance"] = df["cell_type"].astype(str) + "_R"
        return df.reset_index(drop=True)

    tag = sheet.replace(" ", "_")
    return cached_parquet(f"{dataset}_exp_{tag}", _compute, force=force)


def get_rf_raw_cb(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, in_instances: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Per-body CB receptive-field fits for VCBN instances (VIC > vic_thre):
    2D-Gaussian ellipse + alpha hull from one propagation pass per body.

    Uses the full `lat_flow_sum.npz` (not OL-only). Heavy first-run compute.
    Output instances come from `get_vcbn_types`. Carries the ellipse columns
    (size, r2, x0, y0, a, b, phi, ...) plus the alpha-hull columns (ahull_size,
    ahull_com_x/y, ahull_kept_points, ahull_rmfrac, ahull_alpha), per-neuron
    `idx` (full-brain), region, VIC, hitting_time, sign and main_groups.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        vcbn = get_vcbn_types(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        ins = in_instances if in_instances is not None else _default_visual_input_instances(side_char)
        out_inst = vcbn["instance"].values
        params = _compute_rf_params(meta, stepsn, ins, out_inst)
        if params.empty:
            return params
        params = _clean_rf_params(params)
        # merge region and flow
        flow_type = meta.groupby("instance")["region"].first().reset_index()
        params = params.merge(flow_type, on="instance", how="left")
        vic_type = get_ol_cb_vic_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        params = params.merge(
            vic_type[["instance", "VIC"]], on="instance", how="left",
        )
        ht = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        params = params.merge(ht, on="instance", how="left")
        sign = meta.groupby("instance")["sign"].first().reset_index()
        params = params.merge(sign, on="instance", how="left")
        mg = meta.groupby("instance")["main_groups"].first().reset_index()
        params = params.merge(mg, on="instance", how="left")
        # VCNs are excluded from the VCBN set (see `is_cb_neuron`), so `params`
        # carries no VCN rows here — no region relabelling is needed.
        return params.reset_index(drop=True)

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_cb_raw_{tag}", _compute, force=force,
    )


def get_rf_type_cb(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-instance median CB RF fits with circular-mean phi and merged
    region/VIC/layer/sign/main_groups."""
    from utils.external_rf import circular_mean

    def _compute():
        raw = get_rf_raw_cb(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        if raw.empty:
            return raw
        # ellipse aggregation over Gaussian-valid bodies only (hull-only rows
        # carry NaN ellipse params)
        raw = raw[raw["size"].notna()]
        agg = (
            raw.groupby(["instance", "seed"])[["n_col", "r2", "a", "b", "phi", "amp"]]
               .median().reset_index()
        )
        agg["phi"] = (
            raw.groupby(["instance", "seed"])["phi"]
               .apply(lambda s: circular_mean(s.values)).values
        )
        agg["n_neurons"] = raw.groupby(["instance", "seed"])["bodyId"].count().values
        agg = _clean_rf_params(agg)
        side_cols = ["region", "VIC", "hitting_time", "sign", "main_groups"]
        meta_type = (
            raw.groupby("instance")[side_cols].first().reset_index()
        )
        return agg.merge(meta_type, on="instance", how="left")

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_cb_type_{tag}", _compute, force=force,
    )


def get_rf_types_combined(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Concatenated OL + CB per-instance RF fits. Output has a `region`
    column distinguishing `right OL` vs `CB`. Used by CB-vs-OL plots."""
    def _compute():
        ol = get_rf_type_ol(dataset=dataset, data_dir=data_dir, side_char=side_char).copy()
        ol["region"] = f"{_SIDE_WORD[side_char]} OL"
        cb = get_rf_type_cb(
            dataset=dataset, data_dir=data_dir, side_char=side_char, vic_thre=vic_thre,
        ).copy()
        return pd.concat([ol, cb], axis=0, ignore_index=True)

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_types_combined_{tag}", _compute, force=force,
    )


def get_rf_ahull_neuron(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
) -> pd.DataFrame:
    """Per-neuron alpha-hull RF size for Main Fig. 6b.

    One row per neuron, pooling VIN + VPN neurons (`get_rf_raw_ol`,
    `main_groups` in {'VIN', 'VPN'}) with VCBNs
    (`get_rf_raw_cb`). VCN OL neurons are excluded (they are
    already absent from the VCBN set). `ahull_size` is the cumsum-60 alpha hull
    in column² (×25 → deg² inside `plot_rf_size_per_layer`). Per-neuron
    `hitting_time` is joined from the flow table (`get_flow`) by full-brain
    `idx`; neurons with no flow hitting time or no finite ahull are dropped.
    `region` is 'right OL'/'left OL' for the OL side, 'CB' for VCBN.

    Keeps per-neuron resolution (every neuron of a type contributes its own
    hull), unlike the per-instance `get_rf_types_combined`. Derives from the
    combined RF parquet, so it needs no separate alpha-hull cache.
    """
    from utils.core_data import get_flow

    cols = ["bodyId", "idx", "instance", "cell_type", "ahull_size", "region"]
    ol = get_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
    ol = ol[ol["main_groups"] != "VCN"][cols].copy()
    cb = get_rf_raw_cb(dataset=dataset, data_dir=data_dir, side_char=side_char)
    cb = cb[cols].copy()
    cb["region"] = "CB"

    nrn = pd.concat([ol, cb], ignore_index=True).dropna(subset=["ahull_size"])
    flow_ht = (
        get_flow(dataset=dataset, data_dir=data_dir, side_char=side_char)
        .set_index("idx")["hitting_time"]
    )
    nrn["hitting_time"] = nrn["idx"].map(flow_ht)
    return nrn.dropna(subset=["hitting_time"]).reset_index(drop=True)


def get_rf_ahull_type_cb(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4,
) -> pd.DataFrame:
    """Per-instance median alpha-hull RF size for VPN + VCBN.

    VPN from `get_rf_raw_ol` (`main_groups == 'VPN'`), VCBN from
    `get_rf_raw_cb`. One row per instance with median `ahull_size` (col²,
    ×25 → deg²), median `hitting_time`, `region` ('right OL' for VPN, 'CB' for
    VCBN) and `main_groups`. Replaces the 2D-Gaussian `size` used in the CB
    RF-size SI panels (SI Fig. 10b/c/d). Per-instance companion to the
    per-neuron `get_rf_ahull_neuron` (Main Fig. 6b); both derive from the
    combined RF parquet, so neither needs a separate alpha-hull cache.
    """
    vpn = get_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
    vpn = vpn[vpn["main_groups"] == "VPN"].copy()
    vcbn = get_rf_raw_cb(
        dataset=dataset, data_dir=data_dir, side_char=side_char, vic_thre=vic_thre,
    ).copy()
    vcbn["region"] = "CB"

    cols = ["instance", "ahull_size", "hitting_time", "region", "main_groups"]
    both = pd.concat([vpn[cols], vcbn[cols]], ignore_index=True)
    both = both.dropna(subset=["ahull_size"])
    return (
        both.groupby("instance")
        .agg(
            ahull_size=("ahull_size", "median"),
            hitting_time=("hitting_time", "median"),
            region=("region", "first"),
            main_groups=("main_groups", "first"),
        )
        .reset_index()
    )


