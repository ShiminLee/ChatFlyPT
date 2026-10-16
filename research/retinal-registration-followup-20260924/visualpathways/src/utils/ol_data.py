"""OL-side connectivity, clustering, pathways, and related cached data.

Hosts:
- OL connectivity / directedness (`get_ol_connectivity`, `get_ol_directedness`,
  `get_ol_type_directedness`)
- OL sparse matrix accessors (`get_ol_prop`, `get_ol_stepsn_sum`,
  `get_ol_stepsn_0`)
- ROI / layer geometry (`get_ol_layers`, `get_ol_roi_adjacency`)
- Feature vectors and clustering (`get_ol_features`, `get_ol_clusters`,
  `get_ol_clusters_intra`, `get_ol_participation`,
  `get_ol_type_participation`)
- Connectivity pathways and Sankey diagrams (`get_ol_connectivity_pathways`,
  `get_ol_sankey_connectivity`, `get_paths_to_instance`,
  `get_participation_paths`)
- Left/right clustering sweeps (`get_ol_lr_sweep`,
  `get_ol_lr_cluster_match`)
- Private helpers co-located with their `get_*` wrapper:
  `_compute_directedness`, `_compute_hierarchical_clustering`,
  `_compute_participation`.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.cache import cached_parquet, cached_pickle
from utils.config import DATA_DIR, DATASET, HIT_THRE, N_FLOW, SIDE_CHAR
from utils.core_data import (
    _ensure_ol_sidecars,
    _grouped_paths,
    _swap_lr_suffix,
    grouped_effweight,
    get_flow,
    get_flow_per_group,
    get_full_prop,
    get_meta,
    get_ol_flow_type,
    get_ol_meta,
    is_cb_neuron,
)
def get_ol_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Long-format connectivity for pre-neurons in OL, with pre/post meta
    and per-instance median hitting times joined.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        ol_instances = set(ol_meta.instance.unique())

        prop = sp.load_npz(data_dir / f"{dataset}_prop.npz").tocoo()
        conn = pd.DataFrame({"idx_pre": prop.row, "idx_post": prop.col, "weight": prop.data})

        meta_small = meta[["idx", "bodyId", "instance", "region", "main_groups", "sign"]]
        conn = conn.merge(meta_small, left_on="idx_pre", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_pre", "instance": "instance_pre", "region": "region_pre",
                "main_groups": "main_groups_pre", "sign": "sign_pre",
            }
        ).drop(columns="idx")
        conn = conn[conn["instance_pre"].isin(ol_instances)]

        conn = conn.merge(meta_small, left_on="idx_post", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_post", "instance": "instance_post", "region": "region_post",
                "main_groups": "main_groups_post", "sign": "sign_post",
            }
        ).drop(columns="idx")

        flow_type = get_flow_per_group(dataset=dataset, data_dir=data_dir)
        conn = conn.merge(
            flow_type, left_on="instance_pre", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_pre"}).drop(columns="instance")
        conn = conn.merge(
            flow_type, left_on="instance_post", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_post"}).drop(columns="instance")

        conn = conn[~pd.isna(conn["hitting_time_pre"]) & ~pd.isna(conn["hitting_time_post"])]
        conn["hitting_time_diff"] = conn["hitting_time_post"] - conn["hitting_time_pre"]
        conn["frac"] = conn["weight"] / conn.groupby("idx_pre")["weight"].transform("sum")
        return conn.reset_index(drop=True)

    return cached_parquet(f"{dataset}_OL_{side_char}_connectivity", _compute, force=force)


def _compute_directedness(conn_df: pd.DataFrame, hit_diff_thre: float = 0.5) -> pd.DataFrame:
    """Per-pre-neuron fractions of feedforward / feedback / lateral weight.

    Requires `conn_df` to already have `hitting_time_diff` and `frac` columns.
    """
    dir_ff = (
        conn_df[conn_df["hitting_time_diff"] > hit_diff_thre]
        .groupby("idx_pre").agg({"frac": "sum", "instance_pre": "first"})
        .reset_index().rename(columns={"frac": "frac_ff"})
    )
    dir_fb = (
        conn_df[conn_df["hitting_time_diff"] < -hit_diff_thre]
        .groupby("idx_pre").agg({"frac": "sum", "instance_pre": "first"})
        .reset_index().rename(columns={"frac": "frac_fb"})
    )
    dir_la = (
        conn_df[np.abs(conn_df["hitting_time_diff"]) <= hit_diff_thre]
        .groupby("idx_pre").agg({"frac": "sum", "instance_pre": "first"})
        .reset_index().rename(columns={"frac": "frac_la"})
    )
    dir_df = (
        dir_ff.merge(dir_fb, on=["idx_pre", "instance_pre"], how="outer")
        .merge(dir_la, on=["idx_pre", "instance_pre"], how="outer")
        .fillna(0)
    )
    dir_df["cell_type_pre"] = [s[:-2] for s in dir_df["instance_pre"].values]
    return dir_df


def get_ol_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron feedforward / feedback / lateral fractions for OL-pre connections."""
    def _compute():
        conn_ol = get_ol_connectivity(dataset=dataset, data_dir=data_dir, side_char=side_char)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        dir_df = _compute_directedness(conn_ol, hit_diff_thre=hit_diff_thre)
        dir_df = dir_df.merge(
            ol_meta[["instance", "main_groups", "sign"]].drop_duplicates(),
            left_on="instance_pre", right_on="instance", how="left",
        ).rename(columns={"main_groups": "main_groups_pre", "sign": "sign_pre"}).drop(columns="instance")
        return dir_df

    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_directedness_{tag}", _compute, force=force
    )


def get_ol_prop(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
):
    """Sparse OL-subset connectivity matrix (sidecar in `data/`, built if missing)."""
    _ensure_ol_sidecars(dataset, data_dir, side_char)
    return sp.load_npz(data_dir / f"{dataset}_OL_{side_char}_prop.npz")


def get_ol_stepsn_sum(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
):
    """Sparse OL-subset summed lateral-flow matrix (sidecar in `data/`, built if missing)."""
    _ensure_ol_sidecars(dataset, data_dir, side_char)
    return sp.load_npz(data_dir / f"{dataset}_OL_{side_char}_lat_flow_sum.npz")


def get_ol_stepsn_0(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
):
    """Sparse OL-subset `lat_flow_0` matrix (sidecar in `data/`, built if missing)."""
    _ensure_ol_sidecars(dataset, data_dir, side_char)
    return sp.load_npz(data_dir / f"{dataset}_OL_{side_char}_lat_flow_0.npz")


def get_ol_type_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-instance (median) directedness, renormalised to sum=1."""
    def _compute():
        d = get_ol_directedness(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            hit_diff_thre=hit_diff_thre,
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

    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_type_directedness_{tag}", _compute, force=force
    )


def get_ol_layers(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-synapse (bodyId, roi, hitting_time) rows for OL anatomical layers.

    Keeps `ME_{S}_layer_*`, `LO_{S}_layer_*`, `LOP_{S}_layer_*`, and `AME({S})`
    (S = `side_char.upper()`). Each (bodyId, roi) row is expanded by the
    presynapse count so each synapse contributes one sample to violins.
    Hitting time is joined from the side-specific flow CSV.
    """
    def _compute():
        side_upper = side_char.upper()
        roi_counts = pd.read_pickle(
            data_dir / f"{dataset}_OL_{side_char}_roi_counts.pkl"
        )
        prefixes = (
            f"ME_{side_upper}_layer_",
            f"LO_{side_upper}_layer_",
            f"LOP_{side_upper}_layer_",
        )
        mask = (
            roi_counts["roi"].str.startswith(prefixes)
            | (roi_counts["roi"] == f"AME({side_upper})")
        )
        sub = roi_counts.loc[mask, ["bodyId", "roi", "pre"]]
        expanded = sub.loc[sub.index.repeat(sub["pre"])].drop(columns="pre")
        flow = get_flow(dataset=dataset, data_dir=data_dir, side_char=side_char)
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        flow = flow.merge(meta[["idx", "bodyId"]], on="idx", how="left")
        expanded = expanded.merge(
            flow[["bodyId", "hitting_time"]], on="bodyId", how="left"
        )
        return expanded.reset_index(drop=True)

    return cached_parquet(f"{dataset}_OL_{side_char}_layers", _compute, force=force)


def get_ol_roi_adjacency(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-(pre, post, roi) adjacency weights joined with pre/post hitting times.

    Loads `{dataset}_OL_{side_char}_roi_adj.pkl` (columns bodyId_pre,
    bodyId_post, roi, weight) and merges hitting_time from the full flow
    table onto both pre and post bodyIds. Adds `hit_diff = hitting_time_post
    - hitting_time_pre`. Rows with missing pre or post hitting time are
    dropped.
    """
    def _compute():
        adj = pd.read_pickle(data_dir / f"{dataset}_OL_{side_char}_roi_adj.pkl")
        flow = get_flow(dataset=dataset, data_dir=data_dir, side_char=side_char)
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        flow = flow.merge(meta[["idx", "bodyId"]], on="idx", how="left")
        hit = flow[["bodyId", "hitting_time"]]
        adj = adj.merge(
            hit.rename(columns={"bodyId": "bodyId_pre", "hitting_time": "hitting_time_pre"}),
            on="bodyId_pre", how="left",
        ).merge(
            hit.rename(columns={"bodyId": "bodyId_post", "hitting_time": "hitting_time_post"}),
            on="bodyId_post", how="left",
        )
        adj = adj.dropna(subset=["hitting_time_pre", "hitting_time_post"])
        adj["hit_diff"] = adj["hitting_time_post"] - adj["hitting_time_pre"]
        return adj.reset_index(drop=True)

    return cached_parquet(f"{dataset}_OL_{side_char}_roi_adjacency", _compute, force=force)


def get_ol_features(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Visual-input → VPN feature vectors (heavy compute via lat_flow_sum)."""
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        in_inst = ol_meta[ol_meta.main_groups == "visual input"].instance.unique()
        out_inst = ol_meta[ol_meta.main_groups == "VPN"].instance.unique()
        inidx = ol_meta[ol_meta.instance.isin(in_inst)].idx.values
        outidx = ol_meta[ol_meta.instance.isin(out_inst)].idx.values
        feat = grouped_effweight(
            stepsn, inidx, outidx, idx_to_instance, idx_to_instance,
            combining_method="median",
        )
        feat.index.name = "instance"
        return feat.reset_index()

    df = cached_parquet(f"{dataset}_OL_{side_char}_features", _compute, force=force)
    return df.set_index("instance")


def _compute_hierarchical_clustering(feature_df, method="ward", metric="euclidean"):
    import scipy.cluster.hierarchy as sch
    feature_vec = feature_df.values
    feature_vec = feature_vec / feature_vec.sum(1)[:, np.newaxis]
    return sch.linkage(feature_vec, method=method, metric=metric)


def get_ol_clusters(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, method: str = "ward", metric: str = "euclidean",
    force: bool = False,
) -> pd.DataFrame:
    """Flat clusters of VPN instances (cached). frac_thre is a fraction of `Z[:,2].max()`."""
    import scipy.cluster.hierarchy as sch

    def _compute():
        feat = get_ol_features(dataset=dataset, data_dir=data_dir, side_char=side_char)
        Z = _compute_hierarchical_clustering(feat, method=method, metric=metric)
        thre = frac_thre * Z[:, 2].max()
        cluster_idx = sch.fcluster(Z, thre, criterion="distance")
        # Reverse cluster numbering so c1 corresponds to scipy's last leaf
        # (matches the convention used by get_ol_clusters_intra). Without this,
        # the intra heatmap and other cluster plots would be mirror images.
        n_clu = int(cluster_idx.max())
        cluster_idx = n_clu + 1 - cluster_idx
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        df = pd.DataFrame({"instance": feat.index.values, "cluster": cluster_idx})
        df = df.merge(
            ol_meta.groupby("instance")["bodyId"].count().reset_index(name="count"),
            on="instance", how="left",
        )
        return df

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_clusters_{tag}", _compute, force=force,
    )


def _compute_participation(meta_ol, stepsn, cluster_df):
    n_clu = int(cluster_df["cluster"].max())
    in_out_mat = np.zeros((stepsn.shape[0], n_clu))
    for i in range(n_clu):
        out_inst_i = cluster_df[cluster_df.cluster == i + 1]["instance"].values
        outidx_i = meta_ol[meta_ol.instance.isin(out_inst_i)].index.values
        in_out_mat[:, i] = stepsn[:, outidx_i].toarray().sum(1) / max(len(outidx_i), 1)
    for i in range(n_clu):
        out_inst_i = cluster_df[cluster_df.cluster == i + 1]["instance"].values
        outidx_i = meta_ol[meta_ol.instance.isin(out_inst_i)].index.values
        in_out_mat[outidx_i, :] = 0
        in_out_mat[outidx_i, i] = 1
    in_out_mat = in_out_mat / in_out_mat.sum(1)[:, np.newaxis]
    cluster_names = [f"c{i + 1}" for i in range(n_clu)]
    df = pd.DataFrame(in_out_mat, columns=cluster_names, index=meta_ol.index)
    return pd.concat([df, meta_ol], axis=1)


def get_ol_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron participation in each VPN cluster (cached)."""
    def _compute():
        meta_ol = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(dataset=dataset, data_dir=data_dir, side_char=side_char)
        clu = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        return _compute_participation(meta_ol, stepsn, clu)

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_participation_{tag}", _compute, force=force,
    )


def get_ol_type_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, force: bool = False,
) -> pd.DataFrame:
    """Per-instance (median) participation, normalised, with idxmax / max / n_clusters."""
    def _compute():
        part = get_ol_participation(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        cluster_names = list(part.columns[part.columns.str.contains(r"^c\d+$")])
        df = part.groupby("instance")[cluster_names].median().reset_index()
        vals = df[cluster_names].values
        df[cluster_names] = vals / vals.sum(1)[:, np.newaxis]
        df = df[~pd.isna(df[cluster_names[0]])]
        df["max"] = np.sort(df[cluster_names].values, axis=1)[:, -1:].sum(1)
        df["idxmax"] = df[cluster_names].idxmax(1)
        df["n_clusters"] = (df[cluster_names] >= 1 / len(cluster_names)).sum(1)
        return df

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_type_participation_{tag}", _compute, force=force,
    )


def get_ol_connectivity_pathways(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, force: bool = False,
):
    """Pathway-level connectivity matrices (full + e/i breakdowns) and their diagonals.

    Returns a dict {'conn': list[5][N+2,N+2], 'diag': list[5][N+2]} (cached as pickle).
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)
        cluster_df = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        n_clu = int(cluster_df["cluster"].max())
        instance_list = np.zeros(n_clu + 2, dtype=object)
        instance_list[0] = meta[
            (meta.main_groups == "visual input") & (meta.region == "right OL")
        ].instance.values
        instance_list[-1] = meta[is_cb_neuron(meta)].instance.values
        for i in range(n_clu):
            instance_list[i + 1] = cluster_df[cluster_df.cluster == i + 1]["instance"].values

        pre_signs = [[-1, 0, 1], [1], [1], [-1], [-1]]
        post_signs = [[-1, 0, 1], [1], [-1], [1], [-1]]
        n_combi = len(pre_signs)
        conn_list = [np.zeros((n_clu + 2, n_clu + 2)) for _ in range(n_combi)]
        diag_list = [np.zeros(n_clu + 2) for _ in range(n_combi)]

        for i in range(n_combi):
            for k in range(n_clu + 2):
                pre_idx = meta[
                    meta.instance.isin(instance_list[k]) & meta.sign.isin(pre_signs[i])
                ].index.values
                for l in range(n_clu + 2):
                    post_idx = meta[
                        meta.instance.isin(instance_list[l]) & meta.sign.isin(post_signs[i])
                    ].index.values
                    conn_list[i][k, l] = prop[pre_idx][:, post_idx].sum()
                    if (k == l) and (k < n_clu + 1) and len(pre_idx) and len(post_idx):
                        self_conn = result_summary(
                            prop, pre_idx, post_idx,
                            inidx_map=idx_to_instance, outidx_map=idx_to_instance,
                            display_threshold=0, combining_method="sum",
                            display_output=False,
                        )
                        common = list(set(self_conn.index.values) & set(self_conn.columns.values))
                        diag_list[i][k] = self_conn.loc[common, common].values.diagonal().sum()
        return {"conn": conn_list, "diag": diag_list}

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_pickle(
        f"{dataset}_OL_{side_char}_connectivity_pathways_{tag}", _compute, force=force,
    )


def get_ol_clusters_intra(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.15, force: bool = False,
):
    """Cluster-cluster connectivity matrix (output-normalised) and total weight.

    Used by section 3's intra-cluster connectivity heatmap.
    Returns dict {'matrix': ndarray, 'total_weight': float, 'n_clu': int}.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary
    import scipy.cluster.hierarchy as sch

    def _compute():
        feat = get_ol_features(dataset=dataset, data_dir=data_dir, side_char=side_char)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        prop_ol = get_ol_prop(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))

        Z = _compute_hierarchical_clustering(feat)
        thre = frac_thre * Z[:, 2].max()
        cluster_idx = sch.fcluster(Z, thre, criterion="distance")
        clu = pd.DataFrame({"instance": feat.index.values, "cluster": cluster_idx})
        n_clu5 = int(clu["cluster"].max())
        clu["cluster"] = n_clu5 + 1 - clu["cluster"]

        outidx = ol_meta[ol_meta.instance.isin(clu["instance"].values)].idx.values
        instance_to_cluster = dict(zip(clu["instance"], clu["cluster"]))
        idx_to_cluster = {
            idx: instance_to_cluster[inst]
            for idx, inst in idx_to_instance.items() if inst in instance_to_cluster
        }
        m = result_summary(
            prop_ol, outidx, outidx, idx_to_cluster, idx_to_cluster,
            combining_method="sum", display_output=False,
        )
        m.index = [int(i) for i in m.index]
        m.columns = [int(i) for i in m.columns]
        m = m.loc[range(1, n_clu5 + 1), range(1, n_clu5 + 1)].fillna(0).values
        total = m.sum()
        m = m / m.sum(axis=1)[:, np.newaxis]
        return {"matrix": m, "total_weight": float(total), "n_clu": n_clu5}

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_pickle(
        f"{dataset}_OL_{side_char}_clusters_intra_{tag}", _compute, force=force,
    )


def get_ol_sankey_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.19,
    in_instances=(
        "L1_R", "L2_R", "L3_R", "R7_R", "R8_R",
        "R7d_R", "R8d_R", "HBeyelet_R",
    ),
    weight_thre: float = 100.0, min_plot_neurons: int = 20,
    output_normalized: bool = True,
    force: bool = False,
):
    """Pre-build the diagram-level connectivity matrix used by the Sankey plot.

    Bins OL hitting times into layers 1..3, builds diagram labels (input cell
    types, OLI.{i}, OLO{j}.{i}, plus 'CB' / 'left OL'), and aggregates connection
    weight via `result_summary`. If `output_normalized` is false, edge weights are
    raw synapse counts instead of output-normalized fractions. Returns a dict with
    `conn_diagram` (DataFrame), `meta_sub` (per-neuron diagram assignments), and
    `node_table` (per-diagram category / n_neurons / display_layer / plot_node,
    where `plot_node` marks nodes with at least `min_plot_neurons` neurons plus the
    forced visual inputs).
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        ol_flow_type = get_ol_flow_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        clusters = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)

        # bin OL hitting time into 1..3 (clamp >=3.5 to 3)
        flow_b = ol_flow_type[["instance", "hitting_time"]].copy()
        flow_b.loc[flow_b.hitting_time >= 3.5, "hitting_time"] = 3
        bins = np.arange(-0.5, 4 + 0.5, 1)
        labels = (bins[1:] + bins[:-1]) / 2
        flow_b["hit_bin"] = pd.cut(flow_b["hitting_time"], bins=bins, labels=labels).astype(float)

        ol_inst = ol_meta[
            ol_meta.main_groups.isin(["visual input", "VIN", "VPN"])
        ].instance.unique()
        cb_inst = meta[is_cb_neuron(meta) | (meta.region == "left OL")].instance.unique()
        meta_sub = meta[meta.instance.isin(ol_inst) | meta.instance.isin(cb_inst)].reset_index(drop=True).copy()
        meta_sub = meta_sub.merge(clusters[["instance", "cluster"]], how="left", on="instance")
        meta_sub = meta_sub.merge(flow_b[["instance", "hit_bin"]], how="left", on="instance")
        meta_sub = meta_sub[meta_sub.region != "other"].reset_index(drop=True).copy()

        # Capture terminal masks BEFORE main_groups gets overwritten below —
        # otherwise the diagram-label step can't recover the VCN rows.
        cb_mask = is_cb_neuron(meta_sub)
        loff_mask = (meta_sub.region == "left OL") & ~cb_mask
        terminal_mask = cb_mask | loff_mask
        meta_sub.loc[terminal_mask, "hit_bin"] = 4
        meta_sub.loc[terminal_mask, "main_groups"] = "other"

        n_clu = int(clusters["cluster"].max())
        meta_sub["diagram"] = "other"
        meta_sub["category"] = "other"
        in_idx = meta_sub[meta_sub.instance.isin(in_instances)].index
        meta_sub.loc[in_idx, "diagram"] = meta_sub.loc[in_idx, "cell_type"]
        meta_sub.loc[in_idx, "category"] = "visual input"
        meta_sub.loc[in_idx, "hit_bin"] = 0  # inputs are layer 0 (e.g. HBeyelet)
        for i in range(3):
            oli_idx = meta_sub[
                (meta_sub.main_groups == "VIN") & (meta_sub.hit_bin == i + 1)
            ].index
            meta_sub.loc[oli_idx, "diagram"] = f"OLI.{i + 1}"
            meta_sub.loc[oli_idx, "category"] = "VIN"
            for j in range(n_clu):
                olo_idx = meta_sub[
                    (meta_sub.cluster == j + 1) & (meta_sub.hit_bin == i + 1)
                ].index
                if len(olo_idx) > 0:
                    meta_sub.loc[olo_idx, "diagram"] = f"OLO{j + 1}.{i + 1}"
                    meta_sub.loc[olo_idx, "category"] = "VPN"

        meta_sub.loc[cb_mask, "diagram"] = "CB"
        meta_sub.loc[cb_mask, "category"] = "CB"
        meta_sub.loc[loff_mask, "diagram"] = "left OL"
        meta_sub.loc[loff_mask, "category"] = "left OL"
        meta_sub = meta_sub[meta_sub.diagram != "other"].copy()

        idx_to_diagram = dict(zip(meta_sub.idx, meta_sub.diagram))

        # edge weights: output-normalized fractions (inprop) or raw synapse counts
        if output_normalized:
            total_post = prop.sum(axis=1).A1.astype(float)
            inv = np.zeros_like(total_post)
            np.reciprocal(total_post, where=total_post != 0, out=inv)
            conn_matrix = prop.multiply(inv.reshape((-1, 1))).astype(np.float32)
        else:
            conn_matrix = prop.astype(np.float32)

        diagram_names = list(meta_sub["diagram"].unique())
        conn_diagram = pd.DataFrame(
            np.zeros((len(diagram_names), len(diagram_names))),
            index=diagram_names, columns=diagram_names,
        )

        # layer 0 → all later layers
        inidx = meta_sub[meta_sub.hit_bin == 0].idx.values
        outidx = meta_sub[meta_sub.hit_bin > 0].idx.values
        conn_i = result_summary(
            conn_matrix, inidx, outidx, idx_to_diagram, idx_to_diagram,
            combining_method="sum", outprop=False, display_output=False,
        )
        conn_diagram.loc[conn_i.index, conn_i.columns] += conn_i

        # layer i → later layers (i = 1..3)
        for i in range(3):
            inidx = meta_sub[meta_sub.hit_bin == i + 1].idx.values
            outidx = meta_sub[meta_sub.hit_bin > i + 1].idx.values
            conn_i = result_summary(
                conn_matrix, inidx, outidx, idx_to_diagram, idx_to_diagram,
                combining_method="sum", outprop=False, display_output=False,
            )
            conn_diagram.loc[conn_i.index, conn_i.columns] += conn_i

        # per-diagram node attributes for plotting (height ∝ flow, filtering)
        node_table = (
            meta_sub.groupby("diagram", sort=False)
            .agg(
                category=("category", "first"),
                n_neurons=("bodyId", "nunique"),
                display_layer=("hit_bin", "first"),
                cluster=("cluster", "first"),
            )
            .reset_index()
        )
        node_table["display_layer"] = node_table["display_layer"].astype(float)
        node_table["plot_node"] = node_table["n_neurons"] >= min_plot_neurons
        node_table.loc[node_table["category"].eq("visual input"), "plot_node"] = True

        return {
            "conn_diagram": conn_diagram,
            "meta_sub": meta_sub[["instance", "cell_type", "main_groups", "region",
                                  "cluster", "hit_bin", "diagram", "category"]].copy(),
            "node_table": node_table,
            "n_clu": n_clu,
            "weight_thre": weight_thre,
            "min_plot_neurons": min_plot_neurons,
            "output_normalized": output_normalized,
            "weight_units": (
                "output-normalized synapse fraction"
                if output_normalized else "synapse count"
            ),
        }

    tag = f"{frac_thre:.3f}".replace(".", "p")
    norm_tag = "outnorm" if output_normalized else "rawsyn"
    return cached_pickle(
        f"{dataset}_OL_{side_char}_sankey_{tag}_{norm_tag}_n{min_plot_neurons}_in{len(in_instances)}",
        _compute, force=force,
    )


def get_paths_to_instance(
    instance: str, *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, in_main_group: str = 'visual input',
    force: bool = False,
) -> dict:
    """Effective-weight paths from `in_main_group` → `instance` in OL, for each
    path length 1..ceil(hit_time[instance]).

    Returns `{'paths': list[pd.DataFrame], 'hit_inst': float}`, where `paths` is
    the grouped (per-instance) output of `find_paths_of_length` for each length.
    Filtering / `conn_paths_frac` is deferred to plot time.
    """
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_0(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow_type = get_ol_flow_type(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        in_instances = ol_meta[ol_meta.main_groups == in_main_group].instance.unique()
        inidx = ol_meta[ol_meta.instance.isin(in_instances)].idx.values
        outidx = ol_meta[ol_meta.instance == instance].idx.values
        hit_inst = float(flow_type[flow_type.instance == instance].hitting_time.values[0])
        n_interm = int(np.ceil(hit_inst))
        all_paths = _grouped_paths(stepsn, inidx, outidx, n_interm, idx_to_instance)
        return {'paths': all_paths, 'hit_inst': hit_inst}

    tag = instance.replace('/', '_')
    return cached_pickle(
        f"{dataset}_OL_{side_char}_paths_to_{tag}", _compute, force=force,
    )


def get_participation_paths(
    instance: str, *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, out_main_group: str = 'VPN',
    outprop: bool = False, force: bool = False,
) -> dict:
    """Effective-weight paths from `instance` → `out_main_group` in OL, for each
    path length 1..ceil(max(out_hit) - hit[instance]).

    `outprop=False` (default) — input-normalised edges.
    `outprop=True` — output-normalised edges.
    """
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_0(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow_type = get_ol_flow_type(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        out_instances = ol_meta[ol_meta.main_groups == out_main_group].instance.unique()
        inidx = ol_meta[ol_meta.instance == instance].idx.values
        outidx = ol_meta[ol_meta.instance.isin(out_instances)].idx.values
        hit_max = float(flow_type[flow_type.instance.isin(out_instances)].hitting_time.max())
        hit_self = float(flow_type[flow_type.instance == instance].hitting_time.values[0])
        hit_inst = hit_max - hit_self
        n_interm = int(np.ceil(hit_inst))
        all_paths = _grouped_paths(
            stepsn, inidx, outidx, n_interm, idx_to_instance, outprop=outprop,
        )
        return {'paths': all_paths, 'hit_inst': hit_inst}

    tag = instance.replace('/', '_')
    suffix = '_out' if outprop else ''
    return cached_pickle(
        f"{dataset}_OL_{side_char}_participation_paths_{tag}{suffix}",
        _compute, force=force,
    )

def get_ol_lr_sweep(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    n_points: int = 48, frac_log2_min: float = -5.0, frac_log2_max: float = -0.1,
    method: str = "ward", metric: str = "euclidean",
    force: bool = False,
) -> pd.DataFrame:
    """Sweep clustering threshold on both sides; per-frac compare L/R.

    Each row: frac, n_clu_R, n_clu_L, ari_LR (cell-type-matched),
    ari_fromR (L clusters vs. centroid-assignment from R), diag (mean
    diagonal of output-normalised R cluster-cluster connectivity),
    silhouette / ch / db on R features.
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import cdist
    from sklearn.metrics import (
        adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
        silhouette_score,
    )
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        feat_R = get_ol_features(dataset=dataset, data_dir=data_dir, side_char="r")
        feat_L = get_ol_features(dataset=dataset, data_dir=data_dir, side_char="l")
        meta_R = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char="r")
        prop_R = get_ol_prop(dataset=dataset, data_dir=data_dir, side_char="r")
        idx_to_instance_R = dict(zip(meta_R.idx, meta_R.instance))

        feat_R = feat_R[~feat_R.iloc[:, 0].isna()]
        feat_L = feat_L[~feat_L.iloc[:, 0].isna()]

        Z_R = _compute_hierarchical_clustering(feat_R, method=method, metric=metric)
        Z_L = _compute_hierarchical_clustering(feat_L, method=method, metric=metric)

        outidx_R = meta_R[meta_R.instance.isin(feat_R.index)].idx.values

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
            ari_lr = adjusted_rand_score(compare["cluster_R"].values, compare["cluster_L"].values)

            instance_to_cluster_R = dict(zip(df_R["instance"], df_R["cluster"]))
            idx_to_cluster_R = {
                i: instance_to_cluster_R[inst]
                for i, inst in idx_to_instance_R.items()
                if inst in instance_to_cluster_R
            }
            m = result_summary(
                prop_R, outidx_R, outidx_R, idx_to_cluster_R, idx_to_cluster_R,
                combining_method="sum", display_output=False,
            )
            m.index = [int(i) for i in m.index]
            m.columns = [int(i) for i in m.columns]
            m = m.loc[range(1, n_R + 1), range(1, n_R + 1)].fillna(0).values
            m = m / m.sum(axis=1, keepdims=True)
            diag_mean = float(np.diag(m).sum() / n_R)

            silh = silhouette_score(feat_R.values, cl_R)
            ch = calinski_harabasz_score(feat_R.values, cl_R)
            db = davies_bouldin_score(feat_R.values, cl_R)

            centroids_R = np.vstack([
                feat_R.values[cl_R == c].mean(axis=0) for c in range(1, n_R + 1)
            ])
            labels_L_from_R = cdist(feat_L.values, centroids_R, metric="euclidean").argmin(axis=1) + 1
            ari_from_r = adjusted_rand_score(cl_L, labels_L_from_R)

            rows.append({
                "frac": float(frac),
                "n_clu_R": n_R, "n_clu_L": n_L,
                "ari_LR": float(ari_lr), "ari_fromR": float(ari_from_r),
                "diag": diag_mean,
                "silhouette": float(silh),
                "ch": float(ch), "db": float(db),
            })
        return pd.DataFrame(rows)

    tag = f"n{n_points}_{frac_log2_min}_{frac_log2_max}".replace("-", "m").replace(".", "p")
    return cached_parquet(f"{dataset}_OL_lr_sweep_{tag}", _compute, force=force)


def get_ol_lr_cluster_match(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    frac_thre: float = 0.19, force: bool = False,
):
    """L↔R cluster confusion matrix + Hungarian assignment at fixed frac_thre.

    L instance names are flipped `_L`→`_R` before merging on `instance`.
    Returns a dict
    `{confusion, row_ind, col_ind, n_clu_L, n_clu_R, tick_labels}`.
    """
    from utils.cache import cached_pickle
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    def _compute():
        cl_R = get_ol_clusters(dataset=dataset, data_dir=data_dir,
                               side_char="r", frac_thre=frac_thre)
        cl_L = get_ol_clusters(dataset=dataset, data_dir=data_dir,
                               side_char="l", frac_thre=frac_thre)
        cl_L = cl_L.copy()
        cl_L["instance"] = [_swap_lr_suffix(s) for s in cl_L["instance"].values]
        merged = cl_R.merge(cl_L, on="instance", how="inner", suffixes=("_R", "_L"))
        conf = confusion_matrix(merged["cluster_L"].values, merged["cluster_R"].values)
        row_ind, col_ind = linear_sum_assignment(-conf)
        n_L = int(merged["cluster_L"].max())
        n_R = int(merged["cluster_R"].max())
        n = max(n_L, n_R)
        tick_labels = [f"c{i}" for i in range(1, n + 1)]
        return {
            "confusion": conf,
            "row_ind": row_ind, "col_ind": col_ind,
            "n_clu_L": n_L, "n_clu_R": n_R,
            "tick_labels": tick_labels,
        }

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_pickle(f"{dataset}_OL_lr_cluster_match_{tag}", _compute, force=force)
