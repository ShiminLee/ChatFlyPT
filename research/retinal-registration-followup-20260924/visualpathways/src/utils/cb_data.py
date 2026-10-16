"""CB-side connectivity, clustering, and OL↔CB joint analyses.

Hosts:
- CB connectivity / directedness (`get_cb_connectivity`, `get_cb_directedness`,
  `get_cb_type_directedness`)
- CB features / clustering (`get_cb_features`, `get_cb_clusters`,
  `get_cb_clusters_intra`, `get_cb_cluster_tbars_per_roi`)
- CB left/right sweeps (`get_cb_lr_sweep`, `get_cb_lr_cluster_match`,
  `get_cb_layer_lr`, `get_cb_layer_lr_homologue`)
- CB pathways (`get_cb_paths_to_instance`)
- OL → CB joint analyses (`get_ol_in_cb_participation`,
  `get_ol_in_cb_type_participation`, `get_ol_cb_cluster_connectivity`,
  `get_ol_to_cb_weights`)
- ROI coverage (`get_ol_roi_coverage`, originally bound for ol_data but
  depends on vic so moved here alongside the other late-stage analyses)
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.cache import cached_parquet, cached_pickle
from utils.config import DATA_DIR, DATASET, SIDE_CHAR
from utils.core_data import (
    _SIDE_WORD,
    _grouped_paths,
    _swap_lr_suffix,
    get_flow,
    get_flow_per_group,
    get_full_prop,
    get_meta,
    get_ol_meta,
    get_sector_map,
    grouped_effweight,
    is_cb_neuron,
)
from utils.ol_data import (
    _compute_directedness,
    _compute_hierarchical_clustering,
    _compute_participation,
    get_ol_clusters,
)
from utils.ol_rf import get_rf_types_combined
from utils.vic import get_ol_cb_vic_type, get_roi_syn_vic, get_vcbn_types


def get_cb_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Long-format connectivity for pre-neurons in the CB analysis mask.

    Filters the full `prop.npz` to rows whose `instance_pre` passes
    `core_data.is_cb_neuron`. Adds per-instance median hitting times and
    `frac` = weight / full Σ(output weight) per pre-neuron.
    `vic_thre` is retained for API/cache-key compatibility; it does not limit
    the directedness pre-neuron set.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        cb_instances = set(meta.loc[is_cb_neuron(meta), "instance"].unique())

        prop = sp.load_npz(data_dir / f"{dataset}_prop.npz").tocoo()
        conn = pd.DataFrame({"idx_pre": prop.row, "idx_post": prop.col, "weight": prop.data})

        meta_small = meta[["idx", "bodyId", "instance", "region", "main_groups", "sign"]]
        conn = conn.merge(meta_small, left_on="idx_pre", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_pre", "instance": "instance_pre", "region": "region_pre",
                "main_groups": "main_groups_pre", "sign": "sign_pre",
            }
        ).drop(columns="idx")
        conn = conn[conn["instance_pre"].isin(cb_instances)]
        conn["total_weight_pre"] = conn.groupby("idx_pre")["weight"].transform("sum")

        conn = conn.merge(meta_small, left_on="idx_post", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_post", "instance": "instance_post", "region": "region_post",
                "main_groups": "main_groups_post", "sign": "sign_post",
            }
        ).drop(columns="idx")

        flow_type = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        conn = conn.merge(
            flow_type, left_on="instance_pre", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_pre"}).drop(columns="instance")
        conn = conn.merge(
            flow_type, left_on="instance_post", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_post"}).drop(columns="instance")

        conn = conn[~pd.isna(conn["hitting_time_pre"]) & ~pd.isna(conn["hitting_time_post"])]
        conn["hitting_time_diff"] = conn["hitting_time_post"] - conn["hitting_time_pre"]
        conn["frac"] = conn["weight"] / conn["total_weight_pre"]
        return conn.drop(columns="total_weight_pre").reset_index(drop=True)

    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_connectivity_cbmask_fullfrac_{vtag}", _compute, force=force,
    )


def get_cb_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron feedforward / feedback / lateral fractions for VCBN pre connections.

    `main_groups_pre` is overridden to `'nonOL'`.
    """
    def _compute():
        conn_cb = get_cb_connectivity(
            dataset=dataset, data_dir=data_dir, side_char=side_char, vic_thre=vic_thre,
        )
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        dir_df = _compute_directedness(conn_cb, hit_diff_thre=hit_diff_thre)
        dir_df = dir_df.merge(
            meta[["instance", "sign"]].drop_duplicates(),
            left_on="instance_pre", right_on="instance", how="left",
        ).rename(columns={"sign": "sign_pre"}).drop(columns="instance")
        dir_df["main_groups_pre"] = "nonOL"
        return dir_df

    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_directedness_cbmask_fullfrac_{vtag}_{tag}", _compute, force=force,
    )


def get_cb_type_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-CB-instance median directedness, renormalised to sum=1."""
    def _compute():
        d = get_cb_directedness(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre, hit_diff_thre=hit_diff_thre,
        )
        t = (
            d.groupby(["instance_pre", "cell_type_pre", "main_groups_pre", "sign_pre"])
            [["frac_ff", "frac_fb", "frac_la"]].median().reset_index()
        )
        t[["frac_ff", "frac_fb", "frac_la"]] = (
            t[["frac_ff", "frac_fb", "frac_la"]]
            / t[["frac_ff", "frac_fb", "frac_la"]].sum(1).values[:, np.newaxis]
        )
        return t.sort_values("main_groups_pre").reset_index(drop=True)

    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_type_directedness_cbmask_fullfrac_{vtag}_{tag}", _compute, force=force,
    )


def get_cb_features(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Visual-input → VCBN feature vectors on one side.

    Rows are VCBN instance names (from `get_vcbn_types(side_char)`); cols are
    visual-input instances on that side. Computed on the full
    `lat_flow_sum.npz` for the side. Heavy compute on first run.
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
        side_word = _SIDE_WORD[side_char]
        in_inst = meta[
            (meta.main_groups == "visual input") & (meta.side == side_word)
        ].instance.unique()
        out_inst = vcbn["instance"].values
        inidx = meta[meta.instance.isin(in_inst)].idx.values
        outidx = meta[meta.instance.isin(out_inst)].idx.values
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        feat = grouped_effweight(
            stepsn, inidx, outidx, idx_to_instance, idx_to_instance,
            combining_method="median",
        )
        # Drop VCBN rows where ALL visual input columns are NaN (no propagation at all)
        vis_cols = [c for c in feat.columns if c in set(in_inst)]
        if vis_cols:
            feat = feat[~feat[vis_cols].isna().all(axis=1)]
        feat.index.name = "instance"
        return feat.reset_index()

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    df = cached_parquet(
        f"{dataset}_cb_{side_char}_features_{tag}", _compute, force=force,
    )
    return df.set_index("instance")


def get_cb_clusters(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.19, n_clusters: int | None = None,
    method: str = "ward", metric: str = "euclidean",
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Flat clusters of VCBN instances (cached).

    If `n_clusters` is given, uses `criterion='maxclust'` to force exactly that
    many clusters (useful when a distance cutoff skips the desired count due
    to ties). Otherwise uses `frac_thre * max_distance` as a distance cutoff.
    """
    import scipy.cluster.hierarchy as sch

    def _compute():
        feat = get_cb_features(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        Z = _compute_hierarchical_clustering(feat, method=method, metric=metric)
        if n_clusters is not None:
            cluster_idx = sch.fcluster(Z, n_clusters, criterion="maxclust")
        else:
            thre = frac_thre * Z[:, 2].max()
            cluster_idx = sch.fcluster(Z, thre, criterion="distance")
        # Reverse cluster numbering so c1 corresponds to scipy's last leaf
        # (matches the convention used by get_cb_clusters_intra and the OL
        # cluster numbering). Without this, downstream plots would be mirror
        # images of the intra heatmap.
        n_clu = int(cluster_idx.max())
        cluster_idx = n_clu + 1 - cluster_idx
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        df = pd.DataFrame({"instance": feat.index.values, "cluster": cluster_idx})
        df = df.merge(
            meta.groupby("instance")["bodyId"].count().reset_index(name="count"),
            on="instance", how="left",
        )
        return df

    tag = f"n{n_clusters}" if n_clusters is not None else f"{frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_clusters_{tag}_{vtag}",
        _compute, force=force, verbose=True,
    )


def get_cb_lr_sweep(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    n_points: int = 48, frac_log2_min: float = -5.0, frac_log2_max: float = -0.1,
    method: str = "ward", metric: str = "euclidean",
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Sweep CB clustering threshold on both sides and compare L/R.

    Each row: frac, n_clu_R, n_clu_L, ari_LR (cell-type-matched via
    stripping `_R`/`_L` suffix), ari_fromR (L clusters vs. centroid
    assignment from R).
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score

    def _compute():
        feat_R = get_cb_features(
            dataset=dataset, data_dir=data_dir, side_char="r", vic_thre=vic_thre,
        )
        feat_L = get_cb_features(
            dataset=dataset, data_dir=data_dir, side_char="l", vic_thre=vic_thre,
        )

        Z_R = _compute_hierarchical_clustering(feat_R, method=method, metric=metric)
        Z_L = _compute_hierarchical_clustering(feat_L, method=method, metric=metric)

        frac_list = 2.0 ** np.linspace(frac_log2_min, frac_log2_max, n_points)
        rows = []
        for frac in frac_list:
            cl_R = sch.fcluster(Z_R, frac * Z_R[:, 2].max(), criterion="distance")
            cl_L = sch.fcluster(Z_L, frac * Z_L[:, 2].max(), criterion="distance")
            n_R = int(cl_R.max())
            n_L = int(cl_L.max())

            df_R = pd.DataFrame({"instance": feat_R.index.values, "cluster": cl_R})
            df_L = pd.DataFrame({"instance": feat_L.index.values, "cluster": cl_L})
            df_R["cell_type"] = [s[:-2] for s in df_R["instance"]]
            df_L["cell_type"] = [s[:-2] for s in df_L["instance"]]
            compare = df_R.merge(df_L, on="cell_type", how="inner", suffixes=("_R", "_L"))
            ari_lr = adjusted_rand_score(
                compare["cluster_R"].values, compare["cluster_L"].values,
            ) if len(compare) else float("nan")

            centroids_R = np.vstack([
                feat_R.values[cl_R == c].mean(axis=0) for c in range(1, n_R + 1)
            ])
            labels_L_from_R = cdist(
                feat_L.values, centroids_R, metric="euclidean",
            ).argmin(axis=1) + 1
            ari_from_r = adjusted_rand_score(cl_L, labels_L_from_R)

            rows.append({
                "frac": float(frac),
                "n_clu_R": n_R, "n_clu_L": n_L,
                "ari_LR": float(ari_lr), "ari_fromR": float(ari_from_r),
            })
        return pd.DataFrame(rows)

    tag = f"n{n_points}_{frac_log2_min}_{frac_log2_max}".replace("-", "m").replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_cb_lr_sweep_{tag}_{vtag}", _compute, force=force,
    )


def get_cb_lr_cluster_match(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    frac_thre: float = 0.19, n_clusters: int | None = None,
    vic_thre: float = 5e-4, force: bool = False,
):
    """L↔R CB-cluster confusion matrix + Hungarian assignment.

    If `n_clusters` is set, forces both sides to that many clusters via
    `criterion='maxclust'` (useful when the L-side distance cutoff skips the
    desired count due to ties). Otherwise uses `frac_thre` on both sides.

    Left instance names are flipped `_L`↔`_R` before merging on instance.
    Returns
    `{confusion, row_ind, col_ind, n_clu_L, n_clu_R, x_tick_labels,
    y_tick_labels, tick_labels}`. The confusion matrix is rectangular when
    the left and right sides have different cluster counts.
    """
    from utils.cache import cached_pickle
    from scipy.optimize import linear_sum_assignment

    def _compute():
        kw = dict(n_clusters=n_clusters) if n_clusters is not None else dict(frac_thre=frac_thre)
        cl_R = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char="r",
            vic_thre=vic_thre, **kw,
        )
        cl_L = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char="l",
            vic_thre=vic_thre, **kw,
        )
        cl_L = cl_L.copy()
        cl_L["instance"] = [_swap_lr_suffix(s) for s in cl_L["instance"].values]
        merged = cl_R.merge(cl_L, on="instance", how="inner", suffixes=("_R", "_L"))
        n_L = int(merged["cluster_L"].max())
        n_R = int(merged["cluster_R"].max())
        conf = np.zeros((n_L, n_R), dtype=int)
        np.add.at(
            conf,
            (merged["cluster_L"].values.astype(int) - 1,
             merged["cluster_R"].values.astype(int) - 1),
            1,
        )
        row_ind, col_ind = linear_sum_assignment(-conf)
        n = max(n_L, n_R)
        tick_labels = [f"cc{i}" for i in range(1, n + 1)]
        return {
            "confusion": conf,
            "row_ind": row_ind, "col_ind": col_ind,
            "n_clu_L": n_L, "n_clu_R": n_R,
            "x_tick_labels": [f"cc{i}" for i in range(1, n_R + 1)],
            "y_tick_labels": [f"cc'{i}" for i in range(1, n_L + 1)],
            "tick_labels": tick_labels,
        }

    tag = f"n{n_clusters}" if n_clusters is not None else f"{frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_cb_lr_cluster_match_{tag}_{vtag}_ccv2", _compute, force=force,
    )


def get_ol_in_cb_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron participation (VPNs + CB VCBN) in each CB cluster.

    Mirrors `_compute_participation` but restricts neurons to VPNs on
    the chosen side + CB VCBN instances, and the target clusters are CB
    clusters.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        side_word = _SIDE_WORD[side_char]
        ol_out_mask = (meta.region == f"{side_word} OL") & (meta.main_groups == "VPN")
        cb_mask = meta.instance.isin(cb_clu.instance.values)
        meta_sub = meta[ol_out_mask | cb_mask].copy()

        idx_sub = np.sort(meta_sub.index.values)
        meta_sub = meta_sub.set_index("idx").loc[idx_sub].reset_index(
            names="idx_full"
        ).reset_index(names="idx")
        stepsn_sub = stepsn[idx_sub][:, idx_sub]

        df = _compute_participation(meta_sub, stepsn_sub, cb_clu)
        return df

    ftag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_{side_char}_ol_in_cb_participation_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_ol_in_cb_type_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-instance median participation in CB clusters + max / idxmax /
    n_clusters summary + instance-level region / main_groups / sign /
    hitting_time."""
    def _compute():
        part = get_ol_in_cb_participation(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            cb_frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        cluster_names = list(part.columns[part.columns.str.contains(r"^c\d+$")])
        df = part.groupby("instance")[cluster_names].median().reset_index()
        vals = df[cluster_names].values
        df[cluster_names] = vals / vals.sum(1)[:, np.newaxis]
        df = df[~pd.isna(df[cluster_names[0]])]
        df["max"] = np.sort(df[cluster_names].values, axis=1)[:, -1:].sum(1)
        df["idxmax"] = df[cluster_names].idxmax(1)
        df["n_clusters"] = (df[cluster_names] >= 1 / len(cluster_names)).sum(1)
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ht = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        df = df.merge(ht, on="instance", how="left")
        df = df.merge(
            meta[["instance", "region", "main_groups", "sign"]].drop_duplicates("instance"),
            on="instance", how="left",
        )
        return df

    ftag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_{side_char}_ol_in_cb_type_participation_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_cb_cluster_tbars_per_roi(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-(CB-cluster, ROI) tbar count wide df (clusters × ROI) for the
    `cb_clusters_rois` heatmap. Merges `tbar_VIC_r.pkl` with CB
    cluster assignments via bodyId → instance → cluster."""
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        tbar = pd.read_pickle(data_dir / "tbar_VIC_r.pkl")
        tbar = tbar.merge(
            meta[["bodyId", "instance"]], on="bodyId", how="left",
        ).merge(
            cb_clu[["instance", "cluster"]], on="instance", how="left",
        )
        tbar = tbar[~tbar["cluster"].isna()].copy()
        tbar["cluster"] = tbar["cluster"].astype(int)
        counts = tbar.groupby(["cluster", "roi"])["bodyId"].count().unstack(fill_value=0)
        return counts

    from utils.cache import cached_pickle
    ftag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_{side_char}_cb_cluster_tbars_per_roi_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_cb_clusters_intra(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.19, vic_thre: float = 5e-4, force: bool = False,
):
    """CB-cluster × CB-cluster connectivity matrix (output-normalised).

    Mirrors `get_ol_clusters_intra` but uses full `prop.npz` and the CB
    clustering. Returns `{'matrix': ndarray, 'total_weight': float,
    'n_clu': int}`.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=frac_thre, vic_thre=vic_thre,
        )
        n = int(cb_clu["cluster"].max())
        outidx = meta[meta.instance.isin(cb_clu.instance.values)].idx.values
        instance_to_cluster = dict(zip(cb_clu["instance"], cb_clu["cluster"]))
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        idx_to_cluster = {
            i: instance_to_cluster[inst]
            for i, inst in idx_to_instance.items()
            if inst in instance_to_cluster
        }
        m = result_summary(
            prop, outidx, outidx, idx_to_cluster, idx_to_cluster,
            combining_method="sum", display_output=False,
        )
        m.index = [int(i) for i in m.index]
        m.columns = [int(i) for i in m.columns]
        m = m.reindex(index=range(1, n + 1), columns=range(1, n + 1)).fillna(0).values
        total = m.sum()
        m = m / m.sum(axis=1)[:, np.newaxis]
        return {"matrix": m, "total_weight": float(total), "n_clu": n}

    ftag = f"{frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_cb_{side_char}_clusters_intra_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_ol_cb_cluster_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    ol_frac_thre: float = 0.19, cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
):
    """OL-cluster × CB-cluster connectivity via full `prop.npz`.

    Aggregates per-body effective weights by cluster membership on both
    sides, returning a dense `(n_ol_clu, n_cb_clu)` DataFrame indexed/colored
    by cluster id.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)
        ol_clu = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=ol_frac_thre,
        )
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        n_ol = int(ol_clu["cluster"].max())
        n_cb = int(cb_clu["cluster"].max())

        inidx = meta[meta.instance.isin(ol_clu.instance.values)].idx.values
        outidx = meta[meta.instance.isin(cb_clu.instance.values)].idx.values

        idx_to_instance = dict(zip(meta.idx, meta.instance))
        instance_to_ol = dict(zip(ol_clu["instance"], ol_clu["cluster"]))
        instance_to_cb = dict(zip(cb_clu["instance"], cb_clu["cluster"]))
        idx_to_ol = {
            i: instance_to_ol[inst]
            for i, inst in idx_to_instance.items()
            if inst in instance_to_ol
        }
        idx_to_cb = {
            i: instance_to_cb[inst]
            for i, inst in idx_to_instance.items()
            if inst in instance_to_cb
        }
        conn = result_summary(
            prop, inidx, outidx, idx_to_ol, idx_to_cb,
            combining_method="sum", display_output=False,
        )
        conn.index = [int(i) for i in conn.index]
        conn.columns = [int(i) for i in conn.columns]
        conn = conn.reindex(index=range(1, n_ol + 1), columns=range(1, n_cb + 1)).fillna(0)
        return conn

    otag = f"{ol_frac_thre:.3f}".replace(".", "p")
    ctag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_{side_char}_ol_cb_cluster_conn_{otag}_{ctag}_{vtag}",
        _compute, force=force,
    )


def get_cb_layer_lr_homologue(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False,
) -> pd.DataFrame:
    """Cell-type-matched right + left layer (hitting_time) comparison.

    Per side: median hitting_time per instance, strip `_R`/`_L` suffix to
    cell_type, keep the variant with the **min** hitting_time (earliest
    layer). Inner-merge on cell_type gives one row per bilateral homologue.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        def _side(side_char):
            per_inst = get_flow_per_group(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
            ).copy()
            per_inst["cell_type"] = [
                s[:-2] if s[-2:] in ("_R", "_L") else s
                for s in per_inst["instance"]
            ]
            idx = per_inst.groupby("cell_type")["hitting_time"].idxmin()
            return per_inst.loc[idx].reset_index(drop=True)

        r = _side("r").rename(columns={"instance": "instance_r", "hitting_time": "hitting_time_r"})
        l = _side("l").rename(columns={"instance": "instance_l", "hitting_time": "hitting_time_l"})
        df = r.merge(l, on="cell_type", how="inner")
        df = df.merge(
            meta[["cell_type", "region", "main_groups"]].drop_duplicates("cell_type"),
            on="cell_type", how="left",
        )
        return df

    return cached_parquet(
        f"{dataset}_cb_layer_lr_homologue", _compute, force=force,
    )


def get_cb_layer_lr(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False,
) -> pd.DataFrame:
    """Per-instance right + left median hitting_time joined with region + main_groups."""
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ht_r = get_flow_per_group(
            dataset=dataset, data_dir=data_dir, side_char="r",
        ).rename(columns={"hitting_time": "hitting_time_r"})
        ht_l = get_flow_per_group(
            dataset=dataset, data_dir=data_dir, side_char="l",
        ).rename(columns={"hitting_time": "hitting_time_l"})
        df = ht_r.merge(ht_l, on="instance", how="inner")
        df = df.merge(
            meta[["instance", "region", "main_groups"]].drop_duplicates("instance"),
            on="instance", how="left",
        )
        return df

    return cached_parquet(
        f"{dataset}_cb_layer_lr", _compute, force=force,
    )


def get_cb_paths_to_instance(
    instance: str, *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, in_main_group: str = 'visual input',
    n_interm_max: int = 4, force: bool = False,
) -> dict:
    """Effective-weight paths from OL `in_main_group` → CB `instance` (full-brain
    `lat_flow_0`), for each path length 1..min(ceil(hit_diff), n_interm_max).
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(data_dir / f"{dataset}_{side_char}_lat_flow_0.npz")
        flow_type = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        side_word = _SIDE_WORD[side_char]
        in_instances = meta[
            (meta.main_groups == in_main_group) & (meta.region == f'{side_word} OL')
        ].instance.unique()
        inidx = meta[meta.instance.isin(in_instances)].idx.values
        outidx = meta[meta.instance == instance].idx.values
        hit_self = float(flow_type[flow_type.instance == instance].hitting_time.values[0])
        hit_min = float(flow_type[flow_type.instance.isin(in_instances)].hitting_time.min())
        hit_inst = hit_self - hit_min
        n_interm = min(int(np.ceil(hit_inst)), n_interm_max)
        all_paths = _grouped_paths(stepsn, inidx, outidx, n_interm, idx_to_instance)
        return {'paths': all_paths, 'hit_inst': hit_inst}

    tag = instance.replace('/', '_')
    return cached_pickle(
        f"{dataset}_{side_char}_cb_paths_to_{tag}", _compute, force=force,
    )


def get_ol_to_cb_weights(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, r2_thre: float = 0.05, force: bool = False,
) -> pd.DataFrame:
    """Normalised effective-weight matrix VPN × CB-type.

    Uses the full `lat_flow_sum.npz` with instance-level aggregation.
    Restricted to CB types with `r2 > r2_thre` (from `get_rf_type_cb`) and
    sorted by each side's median RF `size`. Column-normalised (each CB type
    column sums to 1 after taking absolute values).
    """
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        combined = get_rf_types_combined(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        side_word = _SIDE_WORD[side_char]
        ol_types = combined[
            (combined["region"] == f"{side_word} OL")
            & (combined["main_groups"] == "VPN")
            & (combined["r2"] > r2_thre)
        ].sort_values("size")["instance"].values
        cb_types = combined[
            (combined["region"] == "CB") & (combined["r2"] > r2_thre)
        ].sort_values("size")["instance"].values

        inidx = meta[
            (meta.region == f"{side_word} OL") & (meta.main_groups == "VPN")
        ].idx.values
        outidx = meta[meta.instance.isin(cb_types)].idx.values
        idx_to_instance = dict(zip(meta.idx, meta.instance))

        m = result_summary(
            stepsn, inidx, outidx,
            inidx_map=idx_to_instance, outidx_map=idx_to_instance,
            display_threshold=0, display_output=False,
        )
        m = m.div(np.abs(m).sum(axis=0), axis=1)
        # keep only rows/cols that actually appear in the matrix
        ol_rows = [i for i in ol_types if i in m.index]
        cb_cols = [c for c in cb_types if c in m.columns]
        return m.loc[ol_rows, cb_cols]

    from utils.cache import cached_pickle
    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_OL_{side_char}_ol_to_cb_weights_{tag}", _compute, force=force,
    )


def get_visual_system_sankey_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    ol_frac_thre: float = 0.19, cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4,
    in_instances=(
        "L1_R", "L2_R", "L3_R", "R7_R", "R8_R",
        "R7d_R", "R8d_R", "HBeyelet_R",
    ),
    weight_thre: float = 5.0, min_plot_neurons: int = 120,
    output_normalized: bool = True, force: bool = False,
) -> dict:
    """Full visual-system Sankey connectivity from visual inputs through OL/CB.

    Categories are mutually exclusive at the neuron level. VCBN cluster
    membership has priority over non-VCBN CB/VCN membership; VIN, VPN, VCBN, and
    non-VCBN CB nodes are split by integer hitting-time layer. Connectivity is
    computed only for neuron pairs where the true integer post layer is greater
    than the true integer pre layer. If `output_normalized` is false, edge
    weights are raw synapse counts. Plot-only node filtering and layer-6
    placement for left OL / VNC are recorded in `node_table` but do not affect
    `conn_diagram`.
    """
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir).copy()
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow = get_flow_per_group(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )[["instance", "hitting_time"]].copy()
        ol_clusters = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=ol_frac_thre,
        )
        cb_clusters = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)

        max_layer = int(np.ceil(flow["hitting_time"].max()))
        bins = np.arange(-0.5, max_layer + 1.5, 1)
        labels = np.arange(0, max_layer + 1)
        flow["layer"] = pd.cut(
            flow["hitting_time"], bins=bins, labels=labels,
        ).astype(float)

        ol_cluster_map = ol_clusters.set_index("instance")["cluster"].to_dict()
        cb_cluster_map = cb_clusters.set_index("instance")["cluster"].to_dict()
        vcbn_instances = set(cb_cluster_map)
        vpn_instances = set(ol_cluster_map)
        vin_instances = set(
            ol_meta.loc[ol_meta["main_groups"] == "VIN", "instance"]
        )
        input_instances = set(in_instances)

        meta_sub = meta[[
            "idx", "bodyId", "instance", "cell_type", "region", "main_groups",
        ]].copy()
        meta_sub = meta_sub.merge(flow, on="instance", how="left")
        meta_sub["category"] = pd.NA
        meta_sub["diagram"] = pd.NA
        meta_sub["cluster"] = np.nan

        assigned = pd.Series(False, index=meta_sub.index)

        def _assign(mask, category, diagram=None, cluster_map=None):
            idx = meta_sub.index[mask & ~assigned]
            if len(idx) == 0:
                return
            meta_sub.loc[idx, "category"] = category
            if cluster_map is not None:
                meta_sub.loc[idx, "cluster"] = meta_sub.loc[idx, "instance"].map(cluster_map)
            if diagram is not None:
                meta_sub.loc[idx, "diagram"] = diagram
            assigned.loc[idx] = True

        _assign(meta_sub["instance"].isin(input_instances), "visual input")
        input_idx = meta_sub.index[meta_sub["category"] == "visual input"]
        meta_sub.loc[input_idx, "diagram"] = meta_sub.loc[input_idx, "cell_type"]
        meta_sub.loc[input_idx, "layer"] = 0

        _assign(meta_sub["instance"].isin(vcbn_instances), "VCBN", cluster_map=cb_cluster_map)
        _assign(
            meta_sub["region"].eq("CB"),
            "CB",
        )
        _assign(meta_sub["instance"].isin(vpn_instances), "VPN", cluster_map=ol_cluster_map)
        _assign(meta_sub["instance"].isin(vin_instances), "VIN")
        _assign(meta_sub["region"].eq("VNC"), "VNC")
        _assign(meta_sub["region"].eq("left OL"), "left OL")

        meta_sub = meta_sub.dropna(subset=["category", "layer"]).copy()
        meta_sub["layer"] = meta_sub["layer"].astype(int)

        vcbn_idx = meta_sub["category"].eq("VCBN")
        meta_sub.loc[vcbn_idx, "diagram"] = [
            f"cc{int(c)}.{int(layer)}"
            for c, layer in zip(meta_sub.loc[vcbn_idx, "cluster"], meta_sub.loc[vcbn_idx, "layer"])
        ]

        vpn_idx = meta_sub["category"].eq("VPN")
        meta_sub.loc[vpn_idx, "diagram"] = [
            f"oc{int(c)}.{int(layer)}"
            for c, layer in zip(meta_sub.loc[vpn_idx, "cluster"], meta_sub.loc[vpn_idx, "layer"])
        ]

        vin_idx = meta_sub["category"].eq("VIN")
        meta_sub.loc[vin_idx, "diagram"] = [
            f"VIN.{int(layer)}" for layer in meta_sub.loc[vin_idx, "layer"]
        ]
        cb_idx = meta_sub["category"].eq("CB")
        meta_sub.loc[cb_idx, "diagram"] = [
            f"CB.{int(layer)}" for layer in meta_sub.loc[cb_idx, "layer"]
        ]
        meta_sub.loc[meta_sub["category"].eq("left OL"), "diagram"] = "left OL"
        meta_sub.loc[meta_sub["category"].eq("VNC"), "diagram"] = "VNC"
        meta_sub = meta_sub.dropna(subset=["diagram"]).copy()

        final_display_layer = 6.5
        meta_sub["display_layer"] = meta_sub["layer"].astype(float)
        final_mask = meta_sub["category"].isin(["left OL", "VNC"])
        meta_sub.loc[final_mask, "display_layer"] = final_display_layer

        # VCNs (visual centrifugal CB->OL feedback) are excluded from the
        # CB/VCBN categories. Any VCN that survives is contralateral and falls
        # into the "left OL" sink; none may be labelled CB or VCBN.
        vcn_rows = meta_sub["main_groups"] == "VCN"
        assert not meta_sub.loc[vcn_rows, "category"].isin(["CB", "VCBN"]).any()
        input_layers = meta_sub.loc[
            meta_sub["instance"].isin(input_instances), ["cell_type", "layer"]
        ].drop_duplicates()
        assert set(input_layers["cell_type"]) == {s[:-2] for s in in_instances}
        assert input_layers["layer"].eq(0).all()

        idx_to_diagram = dict(zip(meta_sub["idx"], meta_sub["diagram"]))
        diagram_names = list(meta_sub["diagram"].drop_duplicates())
        conn_diagram = pd.DataFrame(
            np.zeros((len(diagram_names), len(diagram_names))),
            index=diagram_names, columns=diagram_names,
        )

        if output_normalized:
            total_post = prop.sum(axis=1).A1.astype(float)
            inv = np.zeros_like(total_post)
            np.reciprocal(total_post, where=total_post != 0, out=inv)
            conn_matrix = prop.multiply(inv.reshape((-1, 1))).astype(np.float32)
        else:
            conn_matrix = prop.astype(np.float32)

        for pre_layer in sorted(meta_sub["layer"].unique()):
            inidx = meta_sub.loc[meta_sub["layer"].eq(pre_layer), "idx"].values
            outidx = meta_sub.loc[meta_sub["layer"].gt(pre_layer), "idx"].values
            if len(inidx) == 0 or len(outidx) == 0:
                continue
            conn_i = result_summary(
                conn_matrix, inidx, outidx, idx_to_diagram, idx_to_diagram,
                combining_method="sum", outprop=False,
                display_threshold=0, display_output=False,
            )
            conn_diagram.loc[conn_i.index, conn_i.columns] += conn_i

        for diagram in conn_diagram.index:
            if diagram in conn_diagram.columns:
                conn_diagram.loc[diagram, diagram] = 0.0

        node_table = (
            meta_sub.groupby("diagram", sort=False)
            .agg(
                category=("category", "first"),
                n_neurons=("bodyId", "nunique"),
                layer_min=("layer", "min"),
                layer_max=("layer", "max"),
                display_layer=("display_layer", "max"),
                cluster=("cluster", "first"),
            )
            .reset_index()
        )
        node_table["plot_node"] = node_table["n_neurons"] >= min_plot_neurons
        node_table["plot_node_reason"] = np.where(
            node_table["plot_node"], "min_neurons", "",
        )

        input_nodes = node_table["category"].eq("visual input")
        node_table.loc[input_nodes, "plot_node"] = True
        node_table.loc[input_nodes, "plot_node_reason"] = "visual_input"

        for category in ("VPN", "VCBN"):
            cluster_nodes = node_table[
                node_table["category"].eq(category) & node_table["cluster"].notna()
            ].copy()
            if cluster_nodes.empty:
                continue
            keep_idx = (
                cluster_nodes.sort_values(
                    ["cluster", "n_neurons", "layer_min"],
                    ascending=[True, False, True],
                    kind="stable",
                )
                .groupby("cluster", sort=False)
                .head(1)
                .index
            )
            node_table.loc[keep_idx, "plot_node"] = True
            empty_reason = node_table.loc[keep_idx, "plot_node_reason"].eq("")
            node_table.loc[
                node_table.loc[keep_idx].index[empty_reason],
                "plot_node_reason",
            ] = f"{category}_cluster_representative"

        return {
            "conn_diagram": conn_diagram,
            "meta_sub": meta_sub[[
                "idx", "bodyId", "instance", "cell_type", "region", "main_groups",
                "layer", "display_layer", "category", "cluster", "diagram",
            ]].copy(),
            "node_table": node_table,
            "n_ol_clu": int(ol_clusters["cluster"].max()),
            "n_cb_clu": int(cb_clusters["cluster"].max()),
            "weight_thre": weight_thre,
            "min_plot_neurons": min_plot_neurons,
            "output_normalized": output_normalized,
            "weight_units": (
                "output-normalized synapse fraction"
                if output_normalized else "synapse count"
            ),
            "final_display_layer": final_display_layer,
        }

    otag = f"{ol_frac_thre:.3f}".replace(".", "p")
    ctag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    norm_tag = "outnorm" if output_normalized else "rawsyn"
    return cached_pickle(
        f"{dataset}_{side_char}_visual_system_sankey_v7_{norm_tag}_ol{otag}_cb{ctag}_{vtag}_w{weight_thre:g}_n{min_plot_neurons}",
        _compute, force=force,
    )


def get_ol_roi_coverage(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    n_perm: int = 10000, cov0: float = 0.2, seed: int | None = None,
    force: bool = False,
):
    """Per-ROI retinotopic coverage + permutation p-value.

    For each ROI in `get_roi_syn_vic`, computes the synapse-weighted
    effective input per visual column from right-OL visual inputs, projects
    onto the 5 retinotopic sectors from `get_sector_map`, and tests
    uniformity (`cov0` per sector) via a permutation of per-column weights.

    Returns `{'summary': df, 'per_roi': {roi: coverage_df}}` (pickled).
    `summary` has columns `roi, n_neu, n_syn, sec1..sec5, pval`, sorted by
    synapse count descending. `per_roi[roi]` has columns `coords,
    effective weight` (normalised to sum=1).
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        sectors = get_sector_map(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        syn = get_roi_syn_vic(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )

        side_word = _SIDE_WORD[side_char]
        in_inst = ol_meta[
            (ol_meta.main_groups == "visual input") & (ol_meta.side == side_word)
        ].instance.unique()
        inidx = meta[meta.instance.isin(in_inst)].idx.values

        idx_to_coords = dict(zip(meta.idx, meta.coords))
        idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))

        sector_arr = sectors["sector"].values
        G = np.zeros((len(sectors), 5), dtype=int)
        for i in range(5):
            G[sector_arr == i + 1, i] = 1

        rng = np.random.default_rng(seed)
        rois = syn["roi"].unique()
        summary_cols = ["n_neu", "n_syn", "sec1", "sec2", "sec3", "sec4", "sec5", "pval"]
        summary = pd.DataFrame(0.0, index=rois, columns=summary_cols)
        per_roi = {}

        for roi in rois:
            syn_roi = syn[syn.roi == roi]
            outidx = meta[meta.bodyId.isin(syn_roi.bodyId.unique())].idx.values
            if len(outidx) == 0:
                continue
            summary.loc[roi, "n_neu"] = float(len(outidx))

            df = result_summary(
                stepsn, inidx, outidx,
                inidx_map=idx_to_coords, outidx_map=idx_to_bodyId,
                combining_method="sum", display_threshold=0, display_output=False,
            )
            df = df[(df.index != "nan") & (~df.index.isnull())]
            if df.empty:
                continue

            # weight by per-bodyId synapse count, sum across bodyIds
            syn_count = syn_roi.groupby("bodyId").size()
            syn_count.index = syn_count.index.astype(str)
            common = [c for c in df.columns if c in syn_count.index]
            if not common:
                continue
            weighted = df[common] * syn_count.loc[common].values
            weights = weighted.sum(axis=1).to_frame("effective weight")
            total = float(weights["effective weight"].sum())
            if total <= 0:
                continue
            weights["effective weight"] = weights["effective weight"] / total
            summary.loc[roi, "n_syn"] = float(syn_count.loc[common].sum())
            per_roi[roi] = weights.reset_index().rename(columns={"index": "coords"})

            w = (
                weights.reindex(sectors["coords"].values)
                .fillna(0)["effective weight"].values
            )
            per_sector = w @ G
            summary.loc[roi, ["sec1", "sec2", "sec3", "sec4", "sec5"]] = per_sector

            # permutation: argsort of random matrix = row-wise permutation
            rand = rng.random((n_perm, len(w)))
            perm_idx = rand.argsort(axis=1)
            T_perm = ((w[perm_idx] @ G - cov0) ** 2).sum(axis=1)
            T_obs = float(((per_sector - cov0) ** 2).sum())
            summary.loc[roi, "pval"] = (np.sum(T_perm >= T_obs) + 1) / (n_perm + 1)

        summary = (
            summary.sort_values("n_syn", ascending=False)
            .reset_index().rename(columns={"index": "roi"})
        )
        return {"summary": summary, "per_roi": per_roi}

    tag = f"n{n_perm}_cov{cov0:.2f}".replace(".", "p")
    if seed is not None:
        tag += f"_seed{seed}"
    return cached_pickle(
        f"{dataset}_OL_{side_char}_roi_coverage_{tag}", _compute, force=force,
    )


