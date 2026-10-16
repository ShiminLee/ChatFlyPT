"""Section 3 (Clusters) plot_* functions: dendrograms, correlation matrices,
cluster features/participation/connectivity, Sankeys, and cluster graphs.
"""


from pathlib import Path


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from utils.config import FIG_DATASET as DATASET, FIG_DIR, SIDE_CHAR


from utils.plotting_shared import _SIDE_WORD, _heatmatrix, _hist_lines, _save, _scatter, _stacked_bars


# %%
_CLUSTER_BLUES = ["#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
                  "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]

# VCBN cluster -> OL cluster colour key, so VCBN cluster colours stay consistent
# between the Main Fig. 6a Sankey (plot_visual_system_sankey) and the VCN-input
# pie (plot_vcn_input_pie). Each cc is keyed to the oc that supplies it the most
# input, i.e. argmax over OL rows of get_ol_cb_cluster_connectivity(0.19, 0.17)
# (input-normalised). Regenerate this map if the OL/CB clustering changes.
_VCBN_CC_TO_OC = {
    "cc1": "oc6", "cc2": "oc8", "cc3": "oc7", "cc4": "oc6", "cc5": "oc4",
    "cc6": "oc1", "cc7": "oc2", "cc8": "oc9", "cc9": "oc9", "cc10": "oc9",
}
_CLUSTER_TYPES = [
    "l-LNv", "MeTu2a", "aMe5", "MeTu1", "MeTu3c", "LC6", "MeTu4f", "LPLC2", "LC4",
]
_OUT_TYPES = [
    "aMe3", "LPLC2", "LC4", "LC6",
    "MeTu3a", "MeTu3b", "MeTu3c", "VS",
    "MeTu2a", "LPLC1", "LPLC2",
    "MeTu1", "LC10a", "aMe12", "MeVP11",
]
_PART_TYPES = [
    "L1", "L2", "L3", "R7", "R8", "R7d", "R8d", "HBeyelet",
    "T4a", "Dm4", "Dm9", "Tm31", "T5a",
    "Tm5", "Tm20", "Mi1", "Mi4", "Dm12",
    "Mi15", "TmY18", "Dm3a", "L4", "L5",
    "Y13", "Cm3", "Cm15",
    "aMe6b", "aMe6c", "Dm-DRA1", "Dm-DRA2", "Dm20", "Cm14",
    "Cm11d", "Mi9", "Tm1", "Tm2", "Tm3", "Tm9", "Tm5a",
    "TmY16", "Li25", "CT1", "Dm8", "TmY21", "TmY4", "Dm2", "Mi10",
    "Pm13", "Tm33", "Y11", "T1", "C2", "C3", "MeLo1",
    "Dm8a", "Dm8b", "Dm9", "Tlp1", "Am1", "LPi14", "Li38", "T2", "T2a", "T3",
]


# %%
def _cluster_color_df(n_clu: int) -> pd.DataFrame:
    names = [f"oc{i + 1}" for i in range(n_clu)]
    blues = _CLUSTER_BLUES[:n_clu] if n_clu <= len(_CLUSTER_BLUES) else (
        _CLUSTER_BLUES * ((n_clu // len(_CLUSTER_BLUES)) + 1)
    )[:n_clu]
    return pd.DataFrame(blues, index=names, columns=["color"])


# %%
def plot_clusters_dendrograms(
    feature_df: pd.DataFrame, frac_thre: float = 0.21, *,
    method: str = "ward", metric: str = "euclidean",
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
    compact_lastp: int = 10,
):
    """Three dendrogram PDFs (lastp=50, compact lastp, full vertical)."""
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as sch
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    save_dir.mkdir(parents=True, exist_ok=True)

    feature_vec = feature_df.values / feature_df.values.sum(1)[:, np.newaxis]
    Z = sch.linkage(feature_vec, method=method, metric=metric)
    thre = frac_thre * Z[:, 2].max()

    plt.figure(figsize=(7.5, 6))
    sch.dendrogram(Z, p=50, truncate_mode="lastp", above_threshold_color="black",
                   link_color_func=lambda k: "black")
    plt.xticks([])
    plt.savefig(save_dir / f"{dataset}{side_str}_dendrogram_clusters_50.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 6))
    sch.dendrogram(Z, p=compact_lastp, truncate_mode="lastp", above_threshold_color="black",
                   link_color_func=lambda k: "black")
    plt.xticks([])
    plt.savefig(save_dir / f"{dataset}{side_str}_dendrogram_clusters.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(3, 65))
    dendr = sch.dendrogram(
        Z, color_threshold=thre, show_leaf_counts=True,
        labels=[s[:-2] for s in feature_df.index.values], leaf_font_size=12,
        above_threshold_color="black", orientation="right",
    )
    plt.axvline(x=thre, color="gray", linestyle="--")
    plt.ylabel("clustered VPN types", fontsize=14)
    plt.xticks(
        ticks=[0, thre, 0.5 * Z[:, 2].max(), Z[:, 2].max()],
        labels=["0", f"{frac_thre:.2f}", "0.5", "1"],
    )
    plt.xlabel("linkage distance/max", fontsize=14)
    plt.savefig(save_dir / f"{dataset}{side_str}_dendrogram_full.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    return Z, thre, dendr


# %%
def plot_clusters_correlation_matrix(
    feature_df: pd.DataFrame, dendr: dict, cluster_num: pd.Series, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Heatmap of cosine-like correlation between feature vectors with cluster boundaries."""
    from scipy.stats import zscore
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    n_features = feature_df.shape[1]
    feature_vec = zscore(feature_df.values, axis=1)
    corr = (feature_vec @ feature_vec.T) / n_features
    corr[np.isnan(corr)] = 0
    order = dendr["leaves"][::-1]
    fig = _heatmatrix(corr[order][:, order], feature_df.index.values,
                      cmap="RdBu_r", height=600, width=600)
    n_clu = len(cluster_num)
    cumulative = np.cumsum(cluster_num.values)
    for i in range(n_clu):
        fig.add_hline(y=cumulative[i] - .5, line_width=.5, line_color="gray")
        fig.add_vline(x=cumulative[i] - .5, line_width=.5, line_color="gray")
    fig.update_layout(
        xaxis_title="clustered VPN types", xaxis_tickvals=[],
        yaxis_title="clustered VPN types", yaxis_tickvals=[],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(zmin=-1, zmax=1)
    _save(fig, save_dir, f"{dataset}{side_str}_clustering_matrix.pdf")
    return fig


# %%
def plot_clusters_example_features(
    feature_df: pd.DataFrame, in_instances, cluster_types,
    colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Stacked bar of normalised feature vector for example output types."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    sub = feature_df.loc[cluster_types, in_instances]
    plot_df = sub / sub.sum(1).values[:, np.newaxis]
    plot_df.index = [s[:-2] for s in plot_df.index]
    plot_df.columns = [s[:-2] for s in plot_df.columns]
    plot_df = plot_df.stack(0).reset_index(name="effective weight").rename(
        columns={"level_0": "out type", "level_1": "in type"}
    )
    fig = _stacked_bars(plot_df, "out type", "effective weight", "in type",
                        colored_seed_df["color"].to_dict(),
                        xrot=-45, height=400, width=550)
    _save(fig, save_dir, f"{dataset}{side_str}_in_out_examples.pdf", html=True)
    return fig


# %%
def plot_clusters_pathway_features(
    feature_df: pd.DataFrame, in_instances, cluster_df: pd.DataFrame,
    cluster_num: pd.Series, colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
    xtick_fontsize: int | None = None, width: int = 550, height: int = 400,
):
    """Stacked bar of cluster-mean normalised feature vector.

    `xtick_fontsize` overrides the x-tick label size (default inherits the
    base font, 18); pass a smaller value when many clusters crowd the axis.
    `width`/`height` set the figure canvas size.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    sub = feature_df.loc[:, in_instances] / feature_df.loc[:, in_instances].sum(1).values[:, np.newaxis]
    df = sub.reset_index(names="instance").merge(cluster_df, on="instance")
    df = df.groupby("cluster")[in_instances].mean()
    df.columns = [s[:-2] for s in df.columns]
    df = df.stack(0).reset_index(name="mean effective weight").rename(
        columns={"level_0": "cluster", "level_1": "in type"}
    )
    fig = _stacked_bars(df, "cluster", "mean effective weight", "in type",
                        colored_seed_df["color"].to_dict(), height=height, width=width)
    n_clu = len(cluster_num)
    fig.update_layout(
        xaxis_tickvals=np.arange(1, n_clu + 1),
        xaxis_ticktext=[f"oc{i}<br>({cluster_num[i]})" for i in range(1, n_clu + 1)],
    )
    if xtick_fontsize is not None:
        fig.update_xaxes(tickfont_size=xtick_fontsize)
    _save(fig, save_dir, f"{dataset}{side_str}_in_out_pathways.pdf", html=True)
    return fig


# %%
def plot_clusters_scatter(
    cluster_df: pd.DataFrame, flow_type_ol: pd.DataFrame, type_dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None, jitter: float = 0.2,
):
    """Scatter of layer / ff-fb-la fractions vs cluster, coloured by cluster and by sign."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    figs = []
    n_clu = int(cluster_df["cluster"].max())
    cluster_color = _cluster_color_df(n_clu)
    cluster_names = list(cluster_color.index)
    out_types_r = list(set([s + "_R" for s in _OUT_TYPES]) | set([s + "_R" for s in _CLUSTER_TYPES]))

    base = cluster_df.merge(flow_type_ol, how="left", on="instance").copy()
    base["cluster_name"] = ["oc" + str(i) for i in base["cluster"]]
    rng = np.random.default_rng(0)
    base["y"] = base["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(base))

    layer_df = base[~np.isnan(base.hitting_time)]
    fig = _scatter(layer_df, "hitting_time", "y", "cluster_name", "instance",
                   cluster_color, out_types_r, height=400, width=500)
    fig.update_layout(yaxis_title="cluster", xaxis_title="layer")
    fig.update_yaxes(autorange="reversed", tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names)
    figs.append((fig, f"{dataset}_examples_layer_vs_cluster.pdf"))

    fig = _scatter(layer_df, "hitting_time", "y", "sign", "instance",
                   colored_sign_df, out_types_r, height=400, width=500)
    fig.update_layout(yaxis_title="cluster", xaxis_title="layer")
    fig.update_yaxes(autorange="reversed", tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names)
    figs.append((fig, f"{dataset}_examples_layer_vs_cluster_sign.pdf"))

    dir_df = type_dir_ol.rename(
        columns={"instance_pre": "instance", "main_groups_pre": "main_groups", "sign_pre": "sign"}
    ).merge(cluster_df, how="right", on="instance")
    dir_df["cluster_name"] = ["oc" + str(i) for i in dir_df["cluster"]]
    dir_df["y"] = dir_df["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(dir_df))
    dir_df = dir_df[~np.isnan(dir_df.frac_ff)]

    for direct in ["ff", "fb", "la"]:
        for c_col, cmap in [("cluster_name", cluster_color), ("sign", colored_sign_df)]:
            fig = _scatter(dir_df, f"frac_{direct}", "y", c_col, "instance",
                           cmap, out_types_r, height=400, width=500)
            fig.update_layout(yaxis_title="cluster", xaxis_title=f"frac_{direct}")
            fig.update_yaxes(autorange="reversed", tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names)
            suffix = "" if c_col == "cluster_name" else "_sign"
            figs.append((fig, f"{dataset}_examples_{direct}_vs_cluster{suffix}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_clusters_participation(
    type_part_df: pd.DataFrame, ol_meta: pd.DataFrame, ol_flow_type: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None, jitter: float = 0.2,
):
    """Participation: n_clusters histogram, max-pi vs layer / vs idxmax, and per-cluster pi."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    figs = []
    cluster_names = list(type_part_df.columns[type_part_df.columns.str.contains(r"^c\d+$")].values)
    n_clu = len(cluster_names)

    type_part = (
        type_part_df
        .merge(ol_meta[["instance", "main_groups", "sign"]].drop_duplicates(),
               on="instance", how="left")
        .merge(ol_flow_type[["instance", "hitting_time"]].drop_duplicates(),
               on="instance", how="left")
    )

    bins = np.linspace(0, n_clu + 1, n_clu + 2) - 0.5
    color_val = {"main_groups": colored_main_groups_df, "sign": colored_sign_df}
    base = type_part[type_part["main_groups"].isin(["visual input", "VIN"])]
    for val in ["main_groups", "sign"]:
        plot_df = base if val != "sign" else base[base.sign != 0]
        fig = _hist_lines(plot_df, "n_clusters", val, bins, color_val[val])
        fig.update_layout(
            xaxis_title=f"no. clusters with pi>1/{n_clu}",
            xaxis_range=[-.1, n_clu + 0.1], xaxis_tickvals=np.arange(n_clu + 1),
            yaxis_title="frac. cell types", yaxis_range=[-.01, 0.66],
        )
        figs.append((fig, f"{dataset}_in_out_n_clu_participation_{val}.pdf"))

    part_types_r = [s + "_R" for s in _PART_TYPES]
    for val in ["main_groups", "sign"]:
        fig = _scatter(base, "hitting_time", "max", val, "instance",
                       color_val[val], part_types_r, height=400, width=500)
        fig.add_hline(y=1 / n_clu, line_dash="dot", line_color="gray")
        fig.update_layout(yaxis_title="max. pi", xaxis_title="layer", yaxis_range=[-0.05, 1.05])
        figs.append((fig, f"{dataset}_examples_layer_max_participation_{val}.pdf"))

    base2 = base.reset_index(drop=True).copy()
    cat_to_num = {cat[1:]: i for i, cat in enumerate(cluster_names)}
    rng = np.random.default_rng(0)
    base2["x"] = base2["idxmax"].map(cat_to_num).astype(float) + rng.uniform(-jitter, jitter, size=len(base2))
    for val in ["main_groups", "sign"]:
        fig = _scatter(base2, "x", "max", val, "instance",
                       color_val[val], part_types_r, height=400, width=500)
        fig.add_hline(y=1 / n_clu, line_dash="dot", line_color="gray")
        fig.update_layout(
            yaxis_title="max. pi", xaxis_title="max. ci", xaxis_tickangle=0,
            xaxis_tickvals=np.arange(n_clu),
            xaxis_ticktext=["o" + c for c in cluster_names],
            yaxis_range=[-0.05, 1.05],
        )
        figs.append((fig, f"{dataset}_examples_max_participation_{val}.pdf"))

    long_df = type_part.melt(
        id_vars=["instance", "main_groups", "sign", "hitting_time"],
        value_vars=cluster_names, var_name="cluster", value_name="participation",
    )
    long_df = long_df[long_df["participation"] >= 2 / n_clu]
    long_df = long_df[long_df["main_groups"].isin(["visual input", "VIN"])].reset_index(drop=True)
    x_num = np.array([int(c[1:]) for c in long_df["cluster"].values])
    long_df["x"] = x_num + rng.uniform(-jitter, jitter, size=len(long_df))
    for val in ["main_groups", "sign"]:
        fig = _scatter(long_df, "x", "participation", val, "instance",
                       color_val[val], part_types_r, height=400, width=500)
        fig.add_hline(y=2 / n_clu, line_dash="dot", line_color="gray")
        fig.update_layout(
            yaxis_title="pi", xaxis_title="ci", xaxis_tickangle=0,
            xaxis_tickvals=np.arange(n_clu) + 1,
            xaxis_ticktext=["o" + c for c in cluster_names],
            yaxis_range=[-0.05, 1.05],
        )
        figs.append((fig, f"{dataset}_examples_participation_{val}.pdf"))

    for fig, name in figs:
        # main_2_h panels (n_clu participation histograms) are static — no HTML
        _save(fig, save_dir, name, html="in_out_n_clu_participation" not in name)
    return figs


# %%
def plot_clusters_intra_connectivity(
    intra: dict, *, dataset: str = DATASET, save_dir: Path | None = None,
    filename_stem: str | None = None, font_size: int = 18,
):
    """Cluster-cluster connectivity heatmap (output-normalised).

    Filename defaults to `{dataset}_outrel_{n_clu}_conn_matrix_clusters.pdf`
    (n_clu inferred from `intra['n_clu']`); pass an explicit `filename_stem`
    to override. `font_size` sets the overall label/title font."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    n = intra["n_clu"]
    tick_labels = [f"c{i}" for i in range(1, n + 1)]
    fig = _heatmatrix(intra["matrix"], tick_labels, cmap="Greys", height=600, width=600,
                      font_size=font_size)
    fig.update_layout(
        title=f"all weights: {intra['total_weight']: .0f}",
        xaxis_title="post", yaxis_title="pre",
    )
    fig.update_layout(xaxis_tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                      zmin=0, zmax=1)
    if filename_stem is None:
        filename_stem = f"outrel_{n}_conn_matrix_clusters"
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf")
    return fig


# %%
def plot_clusters_pathway_connectivity(
    pathways: dict, n_clu: int, *,
    dataset: str = DATASET, save_dir: Path | None = None, font_size: int = 18,
):
    """Pathway-level connectivity heatmaps (cluster ↔ cluster, with/without self;
    plus the 5 e/i breakdowns with/without self). `font_size` sets the overall
    label/title font."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    figs = []
    cluster_names = [f"c{i}" for i in range(1, n_clu + 1)]
    title_list = ["all", "ee", "ei", "ie", "ii"]
    cmap_list = ["Greys", "Reds", "Reds", "Blues", "Blues"]

    conn0 = pathways["conn"][0]
    diag0 = pathways["diag"][0]

    # cluster-cluster (drop in/out endpoints)
    inner = conn0[1:-1, 1:-1]
    tot = inner.sum()
    plot_mat = inner / inner.sum(1)[:, np.newaxis]
    fig = _heatmatrix(plot_mat, cluster_names, cmap="Greys", height=600, width=600,
                      font_size=font_size)
    fig.update_layout(title=f"all weights: {tot: .0f}", xaxis_title="post", yaxis_title="pre")
    fig.update_layout(xaxis_tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                      zmin=0, zmax=1)
    figs.append((fig, f"{dataset}_outrel_conn_matrix_clusters.pdf"))

    inner_no_self = conn0[1:-1, 1:-1] - np.diag(diag0)[1:-1, 1:-1]
    tot = inner_no_self.sum()
    plot_mat = inner_no_self / inner_no_self.sum(1)[:, np.newaxis]
    fig = _heatmatrix(plot_mat, cluster_names, cmap="Greys", height=600, width=600,
                      font_size=font_size)
    fig.update_layout(title=f"without cell type-intrinsic weights: {tot: .0f}",
                      xaxis_title="post", yaxis_title="pre")
    fig.update_layout(xaxis_tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                      zmin=0, zmax=1)
    figs.append((fig, f"{dataset}_outrel_conn_matrix_clusters_wo_selfconn.pdf"))

    # pathway breakdowns with the in/out endpoints
    in_out_labels = ["in"] + cluster_names + ["out"]
    for i, (title, cmap) in enumerate(zip(title_list, cmap_list)):
        m = pathways["conn"][i]
        tot = m.sum()
        plot_mat = m / m.sum(1)[:, np.newaxis]
        fig = _heatmatrix(plot_mat, in_out_labels, cmap=cmap, height=600, width=600,
                          font_size=font_size)
        fig.update_layout(title=f"{title} weights: {tot: .0f}",
                          xaxis_title="post", yaxis_title="pre")
        fig.update_layout(xaxis_tickangle=0)
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                          zmin=0, zmax=1)
        figs.append((fig, f"{dataset}_outrel_conn_matrix_{title}.pdf"))

    for i, (title, cmap) in enumerate(zip(title_list, cmap_list)):
        m = pathways["conn"][i] - np.diag(pathways["diag"][i])
        tot = m.sum()
        plot_mat = m / m.sum(1)[:, np.newaxis]
        fig = _heatmatrix(plot_mat, in_out_labels, cmap=cmap, height=600, width=600,
                          font_size=font_size)
        fig.update_layout(title=f"{title} weights: {tot: .0f}",
                          xaxis_title="post", yaxis_title="pre")
        fig.update_layout(xaxis_tickangle=0)
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
                          zmin=0, zmax=1)
        figs.append((fig, f"{dataset}_outrel_conn_matrix_{title}_wo_selfconn.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def _prepare_flow_sankey_edges(
    conn: pd.DataFrame, node_table: pd.DataFrame, *,
    weight_thre: float, min_plot_neurons: int,
    clustered_categories=("VPN", "VCBN"),
):
    """Shared edge preparation for both flow Sankeys.

    Chooses visible nodes (>= `min_plot_neurons` neurons, forced visual inputs,
    plus one representative per cluster at its most-connected layer when the whole
    cluster is otherwise hidden), remaps hidden nodes onto the nearest visible
    same-category / same-cluster representative (tie-broken by total connections),
    keeps only feed-forward edges above `weight_thre`, then reinforces connectivity
    by layer position: inputs (first layer) keep an outgoing edge, terminal nodes
    (last layer) an incoming edge, and every middle node both -- re-adding the
    strongest sub-threshold edge in a missing direction. Returns
    `(edges, merged_n_neurons)`, where `merged_n_neurons` sums the neuron counts of
    every node folded into each visible node.
    """
    nt = node_table
    display_layer = nt["display_layer"].to_dict()
    total_conn = conn.sum(axis=0).add(conn.sum(axis=1), fill_value=0)

    visible = set(nt.index[
        (nt["n_neurons"] >= min_plot_neurons) | (nt["category"] == "visual input")
    ])
    for category in clustered_categories:
        cat_nodes = nt[nt["category"].eq(category)]
        for clu in sorted(cat_nodes["cluster"].dropna().unique()):
            members = cat_nodes.index[cat_nodes["cluster"].eq(clu)]
            if visible.isdisjoint(members):
                visible.add(total_conn.reindex(members).fillna(0).idxmax())

    visible_list = sorted(visible)

    def _remap(diagram):
        if diagram in visible:
            return diagram
        if diagram not in nt.index:
            return None
        row = nt.loc[diagram]
        category = row["category"]
        cands = nt.loc[visible_list]
        cands = cands[cands["category"].eq(category)]
        if category in clustered_categories:
            cands = cands[cands["cluster"].eq(row["cluster"])]
        if cands.empty:
            return None
        order = pd.DataFrame({
            "_ld": (cands["display_layer"].astype(float) - float(row["display_layer"])).abs(),
            "_dl": cands["display_layer"].astype(float),
            "_cr": -total_conn.reindex(cands.index).fillna(0).to_numpy(),
        }, index=cands.index).sort_values(["_ld", "_dl", "_cr"], kind="stable")
        return str(order.index[0])

    remap = {d: _remap(d) for d in nt.index}

    raw = conn.unstack().reset_index(name="weight").rename(
        columns={"level_0": "post", "level_1": "pre"},
    )
    raw = raw[raw["weight"] > 0]
    raw["pre"] = raw["pre"].map(remap)
    raw["post"] = raw["post"].map(remap)
    raw = raw.dropna(subset=["pre", "post"])
    raw = raw[raw["pre"] != raw["post"]]
    raw = raw.groupby(["post", "pre"], as_index=False, sort=False)["weight"].sum()
    raw = raw[raw["pre"].map(display_layer) < raw["post"].map(display_layer)]

    edges = raw[raw["weight"] > weight_thre].copy()

    layers = sorted({display_layer[d] for d in visible})
    first_layer, last_layer = layers[0], layers[-1]
    have_in = set(edges["post"])
    have_out = set(edges["pre"])
    reinforced = []
    for d in visible:
        dl = display_layer[d]
        if dl < last_layer and d not in have_out:
            out = raw[raw["pre"] == d]
            if not out.empty:
                reinforced.append(out.loc[out["weight"].idxmax()])
        if dl > first_layer and d not in have_in:
            inc = raw[raw["post"] == d]
            if not inc.empty:
                reinforced.append(inc.loc[inc["weight"].idxmax()])
    if reinforced:
        edges = pd.concat([edges, pd.DataFrame(reinforced)], ignore_index=True)
    edges = edges.groupby(["post", "pre"], as_index=False, sort=False)["weight"].sum()

    merged = nt.groupby([remap.get(d) for d in nt.index])["n_neurons"].sum()
    merged = merged.drop(index=[None], errors="ignore")
    return edges, merged


def _render_flow_sankey(
    nodes: pd.DataFrame, edges: pd.DataFrame, *,
    explicit_y: dict[str, float] | None = None,
    layer_bands: dict[float, tuple[float, float]] | None = None,
    terminal_band: tuple[float, float] = (0.84, 0.94),
    default_band: tuple[float, float] = (0.02, 0.98),
    width: int = 2080, height: int = 672,
    node_pad: float = 18.0, node_width: float = 0.012,
    show_labels: bool = True, label_size: int = 24,
    label_side: str = "right",
    ribbon_color: str = "rgba(123, 123, 123, 0.35)",
    weight_transform=None,
) -> go.Figure:
    """Render a flow-conserving Sankey with manual block / ribbon geometry.

    Block height is proportional to synaptic flow (the larger of a node's total
    incoming and outgoing edge weight), so with raw synapse-count edges the
    blocks reflect synaptic throughput. `nodes` is indexed by diagram label and
    must provide columns `display_layer`, `n_neurons`, `color`, `label`, and
    `min_height`; its row order sets the top-to-bottom stacking within each
    layer. `edges` has `pre`, `post`, `weight` already filtered to visible nodes.
    `weight_transform`, if given, is a monotonic concave map (e.g. `np.sqrt`)
    applied to every edge weight before sizing, to compress a wide dynamic range
    so small nodes are not slivers and large nodes are not oversized; blocks and
    ribbons stay mutually consistent because the same transform feeds both.
    `explicit_y` pins individual nodes to a y position; `layer_bands` maps a
    `display_layer` to the (y_min, y_max) band its nodes are stacked within
    (defaults to `default_band`, or `terminal_band` for layers with explicit
    nodes). Shared by `plot_clusters_sankey` and `plot_visual_system_sankey`.
    """
    explicit_y = explicit_y or {}
    layer_bands = layer_bands or {}
    nodes = nodes.copy()
    if weight_transform is not None:
        edges = edges.copy()
        edges["weight"] = weight_transform(edges["weight"].astype(float))

    max_display_layer = float(nodes["display_layer"].max())
    if max_display_layer <= 0:
        max_display_layer = 1.0
    x_lookup = {
        layer: 0.06 + float(layer) * 0.82 / max_display_layer
        for layer in sorted(nodes["display_layer"].unique())
    }

    plot_height = max(1, height - 80)
    pad_norm = node_pad / plot_height

    incoming = edges.groupby("post")["weight"].sum()
    outgoing = edges.groupby("pre")["weight"].sum()
    node_value = pd.Series(0.0, index=nodes.index)
    node_value = node_value.combine(incoming.reindex(nodes.index).fillna(0), max)
    node_value = node_value.combine(outgoing.reindex(nodes.index).fillna(0), max)

    scale_candidates = []
    for _, group in nodes.groupby("display_layer", sort=False):
        total = float(node_value.loc[group.index].sum())
        if total <= 0:
            continue
        available = max(0.01, 1.0 - pad_norm * (len(group) - 1))
        scale_candidates.append(available / total)
    value_to_y = min(scale_candidates) if scale_candidates else 0.0
    node_height = node_value * value_to_y
    min_height = nodes["min_height"].astype(float)
    visible_node_height = node_height.clip(lower=min_height)

    pos_y = pd.Series(index=nodes.index, dtype=float)

    def _stack(index, y_min, y_max):
        index = list(index)
        if not index:
            return
        if len(index) == 1:
            top = y_min + max(
                0, (y_max - y_min - float(visible_node_height.loc[index[0]])) / 2,
            )
            pos_y.loc[index[0]] = top
            return
        heights = visible_node_height.loc[index].astype(float)
        available = y_max - y_min
        min_gap = pad_norm * 1.05
        extra = available - float(heights.sum()) - min_gap * (len(index) - 1)
        gap = min_gap + max(0, extra) / (len(index) - 1)
        if extra < 0:
            gap = max(0, (available - float(heights.sum())) / (len(index) - 1))
        y = y_min
        for diagram in index:
            pos_y.loc[diagram] = y
            y += float(visible_node_height.loc[diagram]) + gap

    for layer, group in nodes.groupby("display_layer", sort=False):
        layer_explicit = [d for d in group.index if d in explicit_y]
        if layer_explicit:
            for diagram in layer_explicit:
                pos_y.loc[diagram] = explicit_y[diagram]
            leftovers = [d for d in group.index if d not in explicit_y]
            band = layer_bands.get(layer, terminal_band)
            _stack(leftovers, *band)
        else:
            band = layer_bands.get(layer, default_band)
            _stack(group.index, *band)

    pos_y = pos_y.fillna(0.5).clip(lower=0.02, upper=0.98)

    node_geom = pd.DataFrame({
        "x0": [x_lookup[nodes.loc[d, "display_layer"]] for d in nodes.index],
        "y0": [float(pos_y.loc[d]) for d in nodes.index],
        "height": [float(visible_node_height.loc[d]) for d in nodes.index],
        "color": [nodes.loc[d, "color"] for d in nodes.index],
    }, index=nodes.index)
    node_geom["x1"] = node_geom["x0"] + node_width
    node_geom["y1"] = node_geom["y0"] + node_geom["height"]

    edge_draw = edges.copy()
    edge_draw["source_width"] = edge_draw["weight"] * value_to_y
    edge_draw["target_width"] = edge_draw["weight"] * value_to_y
    edge_draw["_pre_y"] = edge_draw["pre"].map(node_geom["y0"])
    edge_draw["_post_y"] = edge_draw["post"].map(node_geom["y0"])

    source_slots = {}
    for node, group_i in edge_draw.sort_values(["_post_y", "post"]).groupby("pre", sort=False):
        y = float(node_geom.loc[node, "y0"])
        for idx, edge_i in group_i.iterrows():
            width_i = float(edge_i["source_width"])
            source_slots[idx] = (y, y + width_i)
            y += width_i

    target_slots = {}
    for node, group_i in edge_draw.sort_values(["_pre_y", "pre"]).groupby("post", sort=False):
        y = float(node_geom.loc[node, "y0"])
        for idx, edge_i in group_i.iterrows():
            width_i = float(edge_i["target_width"])
            target_slots[idx] = (y, y + width_i)
            y += width_i

    fig = go.Figure()
    for idx, edge_i in edge_draw.iterrows():
        sx = float(node_geom.loc[edge_i["pre"], "x1"])
        tx = float(node_geom.loc[edge_i["post"], "x0"])
        sy0, sy1 = source_slots[idx]
        ty0, ty1 = target_slots[idx]
        curve = max(0.02, abs(tx - sx) * 0.45)
        path = (
            f"M {sx},{sy0} "
            f"C {sx + curve},{sy0} {tx - curve},{ty0} {tx},{ty0} "
            f"L {tx},{ty1} "
            f"C {tx - curve},{ty1} {sx + curve},{sy1} {sx},{sy1} Z"
        )
        fig.add_shape(
            type="path", path=path,
            xref="x", yref="y",
            fillcolor=ribbon_color,
            line=dict(width=0, color="rgba(123, 123, 123, 0)"),
            layer="below",
        )

    for diagram in nodes.index:
        geom = node_geom.loc[diagram]
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=float(geom["x0"]), x1=float(geom["x1"]),
            y0=float(geom["y0"]), y1=float(geom["y1"]),
            fillcolor=str(geom["color"]),
            line=dict(color="black", width=0.5),
            layer="above",
        )
        fig.add_trace(go.Scatter(
            x=[(float(geom["x0"]) + float(geom["x1"])) / 2],
            y=[(float(geom["y0"]) + float(geom["y1"])) / 2],
            mode="markers",
            marker=dict(size=1, opacity=0),
            hovertemplate=(
                f"{diagram}<br>{int(nodes.loc[diagram, 'n_neurons'])} neurons"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        if show_labels:
            if label_side == "left":
                label_x, label_anchor = float(geom["x0"]) - 0.004, "right"
            else:
                label_x, label_anchor = float(geom["x1"]) + 0.004, "left"
            fig.add_annotation(
                xref="x", yref="y",
                x=label_x,
                y=(float(geom["y0"]) + float(geom["y1"])) / 2,
                text=str(nodes.loc[diagram, "label"]),
                showarrow=False,
                xanchor=label_anchor,
                yanchor="middle",
                font=dict(size=label_size, color="black", family="Arial"),
            )

    margin = (
        dict(l=120, r=30, t=40, b=40) if label_side == "left"
        else dict(l=40, r=180, t=40, b=40)
    )
    fig.update_layout(
        width=width, height=height,
        font=dict(size=label_size, color="black", family="Arial"),
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[1, 0], visible=False, fixedrange=True)
    return fig


def plot_clusters_sankey(
    sankey: dict, colored_main_groups_df: pd.DataFrame,
    colored_seed_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    filename: str | None = None, show_labels: bool = True,
    weight_transform=None,
):
    """Sankey diagram of inputs → VIN/output clusters by hit_bin → CB / left OL.

    Consumes the cached output of `pre.get_ol_sankey_connectivity()`. Uses the
    shared `_render_flow_sankey` renderer so block height is proportional to
    synaptic flow (raw synapse counts) like the full visual-system Sankey. Nodes
    with fewer than `min_plot_neurons` neurons (per the cached `node_table`) and
    edges at or below `weight_thre` synapses are dropped. Node labels are the VIN
    group, output cluster (`oc{i}`), input cell type, or terminal region; CB and
    left OL are stacked toward the vertical centre.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    conn = sankey["conn_diagram"]
    node_table = sankey["node_table"].copy().set_index("diagram")
    weight_thre = sankey["weight_thre"]
    n_clu = int(sankey["n_clu"])
    min_plot_neurons = int(sankey.get("min_plot_neurons", 20))

    edges, merged_counts = _prepare_flow_sankey_edges(
        conn, node_table, weight_thre=weight_thre, min_plot_neurons=min_plot_neurons,
    )
    if edges.empty:
        raise ValueError("No OL Sankey edges remain after plotting filters.")

    plotted = sorted(set(edges["pre"]) | set(edges["post"]))
    nodes = node_table.loc[plotted].copy()
    nodes["n_neurons"] = (
        merged_counts.reindex(plotted).fillna(nodes["n_neurons"]).astype(int)
    )

    cluster_colors = _cluster_color_df(n_clu)["color"].to_dict()  # oc1..ocN

    def _node_color(diagram, row):
        category = row["category"]
        if category == "visual input":
            return (
                colored_seed_df.loc[diagram, "color"]
                if diagram in colored_seed_df.index else "#cccccc"
            )
        if category == "VIN":
            return colored_main_groups_df.loc["VIN", "color"]
        if category == "VPN":
            return cluster_colors.get(f"oc{int(row['cluster'])}", "#64a0d1")
        if category in ("CB", "left OL"):
            return (
                colored_region_df.loc[category, "color"]
                if category in colored_region_df.index else "#cccccc"
            )
        return "#cccccc"

    def _node_label(diagram, row):
        category = row["category"]
        if category == "VIN":
            return "VIN"
        if category == "VPN":
            return f"oc{int(row['cluster'])}"
        if category in ("CB", "left OL"):
            return category
        return diagram  # input cell type (e.g. L1)

    nodes["color"] = [_node_color(d, nodes.loc[d]) for d in nodes.index]
    nodes["label"] = [_node_label(d, nodes.loc[d]) for d in nodes.index]
    nodes["min_height"] = 0.0

    # top-to-bottom stacking order within each layer
    seed_order = {
        "L1": 0, "L2": 1, "L3": 2, "R7": 3, "R8": 4,
        "R7d": 5, "R8d": 6, "HBeyelet": 7,
    }
    category_order = {
        "visual input": 0, "VIN": 1, "VPN": 2, "CB": 3, "left OL": 4,
    }

    def _sort_key(row):
        diagram = row.name
        category = row["category"]
        if category == "visual input":
            return (category_order[category], seed_order.get(diagram, 99), 0, diagram)
        cluster = int(row["cluster"]) if pd.notna(row["cluster"]) else 0
        return (category_order.get(category, 99), 0, cluster, diagram)

    sort_keys = nodes.apply(_sort_key, axis=1, result_type="expand")
    nodes[["_sort_group", "_sort_seed", "_sort_cluster", "_sort_name"]] = sort_keys
    nodes = nodes.sort_values(
        ["display_layer", "_sort_group", "_sort_seed", "_sort_cluster", "_sort_name"],
        kind="stable",
    )

    # stack the terminal CB / left OL nodes toward the vertical centre
    terminal_layer = float(nodes["display_layer"].max())

    fig = _render_flow_sankey(
        nodes[["display_layer", "n_neurons", "color", "label", "min_height"]],
        edges,
        layer_bands={terminal_layer: (0.30, 0.70)},
        width=1600, height=500,
        node_pad=14, node_width=0.016,
        show_labels=show_labels, label_size=27.5, label_side="left",
        weight_transform=weight_transform,
    )
    _save(fig, save_dir, filename or f"{dataset}_sankey_pathways_w_OLL.pdf")
    return fig


def plot_visual_system_sankey(
    sankey: dict, colored_main_groups_df: pd.DataFrame,
    colored_seed_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    filename: str | None = None, show_labels: bool = True,
    vcbn_cluster_colors: dict[str, str] | None = None,
):
    """Sankey diagram from visual inputs through OL/VPN and CB/VCBN groups."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    conn = sankey["conn_diagram"]
    node_table = sankey["node_table"].copy().set_index("diagram")
    weight_thre = sankey["weight_thre"]

    min_plot_neurons = int(sankey.get("min_plot_neurons", 120))
    edges, merged_counts = _prepare_flow_sankey_edges(
        conn, node_table, weight_thre=weight_thre, min_plot_neurons=min_plot_neurons,
    )
    if edges.empty:
        raise ValueError("No Sankey edges remain after plotting filters.")

    plotted = sorted(set(edges["pre"]) | set(edges["post"]))
    nodes = node_table.loc[plotted].copy()
    nodes["n_neurons"] = (
        merged_counts.reindex(plotted).fillna(nodes["n_neurons"]).astype(int)
    )
    final_layer_mask = nodes["category"].isin(["left OL", "VNC"])
    nodes.loc[final_layer_mask, "display_layer"] = 6.5
    seed_order = {
        "L1": 0, "L2": 1, "L3": 2, "R7": 3, "R8": 4,
        "R7d": 5, "R8d": 6, "HBeyelet": 7,
    }
    category_order = {
        "visual input": 0, "VIN": 1, "CB": 2, "VPN": 3,
        "VCBN": 4, "left OL": 5, "VNC": 6,
    }

    def _sort_key(row):
        diagram = row.name
        category = row["category"]
        if category == "visual input":
            return (category_order[category], seed_order.get(diagram, 99), diagram)
        cluster = row["cluster"]
        cluster = int(cluster) if pd.notna(cluster) else 0
        return (category_order.get(category, 99), cluster, diagram)

    sort_keys = nodes.apply(_sort_key, axis=1, result_type="expand")
    nodes[["_sort_group", "_sort_cluster", "_sort_name"]] = sort_keys
    nodes = nodes.sort_values(
        ["display_layer", "_sort_group", "_sort_cluster", "_sort_name"],
        kind="stable",
    )

    def _cluster_palette(prefix: str, n_clu: int, colors: list[str]) -> dict[str, str]:
        if n_clu <= len(colors):
            palette = colors[:n_clu]
        else:
            palette = (colors * ((n_clu // len(colors)) + 1))[:n_clu]
        return {f"{prefix}{i + 1}": color for i, color in enumerate(palette)}

    n_ol_clu = max(int(sankey.get("n_ol_clu", 1)), 1)
    n_cb_clu = max(int(sankey.get("n_cb_clu", 1)), 1)
    ol_colors = _cluster_palette("oc", n_ol_clu, _CLUSTER_BLUES)
    vcbn_cluster_colors = vcbn_cluster_colors or _VCBN_CC_TO_OC
    vcbn_colors = {}
    for cluster_name, color in vcbn_cluster_colors.items():
        vcbn_colors[cluster_name] = ol_colors.get(color, color)

    def _node_color(diagram, row):
        category = row["category"]
        if category == "visual input" and diagram in colored_seed_df.index:
            return colored_seed_df.loc[diagram, "color"]
        if category == "VIN":
            return colored_main_groups_df.loc["VIN", "color"]
        if category == "VPN":
            return ol_colors.get(f"oc{int(row['cluster'])}", "#64a0d1")
        if category == "VCBN":
            return vcbn_colors.get(f"cc{int(row['cluster'])}", "#cb181d")
        if category == "CB":
            return colored_region_df.loc["CB", "color"]
        if category == "left OL":
            return colored_region_df.loc["left OL", "color"]
        if category == "VNC":
            return colored_region_df.loc["VNC", "color"]
        return "#cccccc"

    nodes["color"] = [_node_color(d, nodes.loc[d]) for d in nodes.index]
    nodes["label"] = [d.split(".")[0] for d in nodes.index]
    nodes["min_height"] = np.where(nodes["category"].eq("VIN"), 0.025, 0.0)
    explicit_y = {
        k: v for k, v in {"CB.6": 0.02, "left OL": 0.42, "VNC": 0.72}.items()
        if k in nodes.index
    }

    fig = _render_flow_sankey(
        nodes[["display_layer", "n_neurons", "color", "label", "min_height"]],
        edges,
        explicit_y=explicit_y,
        terminal_band=(0.84, 0.94),
        default_band=(0.02, 0.98),
        width=2080, height=672,
        node_pad=18, node_width=0.012,
        show_labels=show_labels, label_size=24, label_side="left",
    )
    out_name = filename or f"{dataset}_visual_system_sankey.pdf"
    _save(fig, save_dir, out_name)
    return fig


def plot_vcn_input_pie(
    sankey: dict, meta: pd.DataFrame, prop,
    colored_region_df: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, side_char: str = SIDE_CHAR, min_pct: float = 1.0,
    dataset: str = DATASET, save_dir: Path | None = None,
    filename: str | None = None,
):
    """Pie of total synapses into <side> VCNs, split by presynaptic category.

    Uses the same neuron categorization that builds the Main Fig. 6a Sankey
    (`sankey` = output of `cb_data.get_visual_system_sankey_connectivity`): VIN,
    VPN clusters ``oc*``, VCBN clusters ``cc*``, and non-visual CB. Edge weights
    are raw synapse counts from the full `prop` matrix onto VCN cell types that
    project into the chosen optic lobe (``main_groups == "VCN"`` &
    ``region == "<side> OL"``). Colours match the 6a Sankey. Slices below
    ``min_pct`` percent of the total are dropped and the remainder renormalised.

    Returns ``(fig, breakdown_df)``; ``breakdown_df`` lists every category with
    its synapse count and full-total percentage (including dropped ones).
    """
    import scipy.sparse as sp

    meta_sub = sankey["meta_sub"]
    n_ol_clu, n_cb_clu = int(sankey["n_ol_clu"]), int(sankey["n_cb_clu"])

    def _category_column(category, cluster):
        if category == "VIN":
            return "VIN"
        if category == "VPN":
            return f"oc{int(cluster)}"
        if category == "VCBN":
            return f"cc{int(cluster)}"
        if category == "CB":
            return "non-visual CB"
        return None

    cols = [
        _category_column(c, k)
        for c, k in zip(meta_sub["category"], meta_sub["cluster"])
    ]
    ms = meta_sub.assign(_col=cols).dropna(subset=["_col"])
    col_order = (
        ["VIN"]
        + [f"oc{i}" for i in range(1, n_ol_clu + 1)]
        + [f"cc{i}" for i in range(1, n_cb_clu + 1)]
        + ["non-visual CB"]
    )
    col_to_row = {c: i for i, c in enumerate(col_order)}
    indicator = sp.coo_matrix(
        (np.ones(len(ms)), (ms["_col"].map(col_to_row).values, ms["idx"].values)),
        shape=(len(col_order), prop.shape[0]),
    ).tocsr()

    side_word = _SIDE_WORD[side_char]
    post_idx = meta[
        (meta["main_groups"] == "VCN") & (meta["region"] == f"{side_word} OL")
    ]["idx"].values
    synapses = np.asarray((indicator @ prop[:, post_idx]).sum(axis=1)).ravel()

    breakdown = pd.DataFrame({"category": col_order, "synapses": synapses})
    breakdown["pct"] = 100 * breakdown["synapses"] / breakdown["synapses"].sum()

    # Colours matched to plot_visual_system_sankey: VIN main-group pink, VPN/VCBN
    # clusters share the OL cluster blues (cc via the _VCBN_CC_TO_OC key), CB grey.
    oc_colors = _cluster_color_df(n_ol_clu)["color"].to_dict()
    cmap = {"VIN": colored_main_groups_df.loc["VIN", "color"]}
    cmap.update({f"oc{i}": oc_colors[f"oc{i}"] for i in range(1, n_ol_clu + 1)})
    cmap.update({
        cc: oc_colors.get(oc, "#cb181d")
        for cc, oc in _VCBN_CC_TO_OC.items() if cc in col_to_row
    })
    cmap["non-visual CB"] = colored_region_df.loc["CB", "color"]

    plot_df = breakdown[breakdown["pct"] >= min_pct].copy()
    plot_order = [c for c in col_order if c in set(plot_df["category"])]
    # Order the slices by explicitly sorting the rows rather than passing
    # px.pie's category_orders: plotly >=6 (narwhals backend) crashes in
    # process_dataframe_pie when category_orders is set (hardcoded 'b' column).
    # With sort=False below, this row order is preserved in slices and legend.
    _rank = {c: i for i, c in enumerate(plot_order)}
    plot_df = plot_df.sort_values("category", key=lambda s: s.map(_rank))

    fig = px.pie(
        plot_df, values="synapses", names="category", color="category",
        color_discrete_map=cmap,
        title=f"Synapses into {side_word} VCNs by presynaptic category (>={min_pct:g}%)",
    )
    fig.update_traces(sort=False, textposition="inside", texttemplate="%{percent:.0%}")
    fig.update_layout(
        width=720, height=520, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="arial", size=14),
    )
    out_name = filename or f"{dataset}_vcn_{side_char}_input_pie.pdf"
    _save(fig, save_dir, out_name, html=True)
    return fig, breakdown


def _save_graph_html(g, layout, vertex_colors, vertex_size, out_path: Path) -> None:
    """Save an igraph graph as an interactive Plotly HTML with node-name hover."""
    # Re-imported live (not a bare module-global) so this picks up whatever
    # utils.plotting_shared.set_html_output() last set, even though that
    # setter lives in a different module than this reader.
    from utils.plotting_shared import _WRITE_HTML
    if not _WRITE_HTML:
        return

    coords = layout.coords
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    edge_x, edge_y = [], []
    for src, tgt in g.get_edgelist():
        edge_x += [xs[src], xs[tgt], None]
        edge_y += [ys[src], ys[tgt], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="lightgray", width=0.8),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            color=vertex_colors,
            size=vertex_size,
            line=dict(color="black", width=0.5),
        ),
        text=g.vs["name"],
        hoverinfo="text",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=10, b=10),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


# %%
def plot_clusters_graph(
    type_part_df: pd.DataFrame, ol_meta: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    colored_seed_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    edge_color: str = "lightgray",
):
    """igraph network of cell-type / cluster participation (visual input + OL
    internal). Edges are undirected gray lines; `colored_sign_df` is accepted
    for API compatibility but no longer used for edge colouring."""
    import igraph as ig
    import matplotlib.pyplot as plt
    save_dir = save_dir if save_dir is not None else FIG_DIR / "pathways"
    save_dir.mkdir(parents=True, exist_ok=True)

    cluster_names_c = list(type_part_df.columns[type_part_df.columns.str.contains(r"^c\d+$")].values)
    cluster_names_no_c = [s[1:] for s in cluster_names_c]
    n_clu = len(cluster_names_c)

    base = type_part_df.merge(
        ol_meta[["instance", "main_groups", "sign"]].drop_duplicates(),
        on="instance", how="left",
    )
    base = base[base.main_groups.isin(["visual input", "VIN"])].reset_index(drop=True)
    long = base.melt(
        id_vars=["instance", "main_groups", "sign"],
        value_vars=cluster_names_c, var_name="post", value_name="weight",
    ).rename(columns={"instance": "pre"})
    long = long[long.weight >= 1 / n_clu].copy()
    long["pre"] = [s.split("_")[0] for s in long["pre"]]
    long["post"] = [s[1:] for s in long["post"]]

    nodes = list(set(long["pre"]).union(long["post"]))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[s], node_to_idx[t]) for s, t in zip(long["pre"], long["post"])]

    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.vs["name"] = nodes
    g.es["weight"] = long["weight"].tolist()

    cluster_color_map = _cluster_color_df(n_clu)["color"].to_dict()
    cluster_to_color = {f"{i + 1}": cluster_color_map[f"oc{i + 1}"] for i in range(n_clu)}

    pre_color = (
        long.assign(color=long["main_groups"].map(colored_main_groups_df["color"]))
            .set_index("pre")["color"].to_dict()
    )
    pre_color.update(colored_seed_df["color"].to_dict())
    pre_color.update(cluster_to_color)
    vertex_colors = [pre_color.get(v["name"], "gray") for v in g.vs]

    highlight = set(cluster_names_no_c) | {s for s in _PART_TYPES}
    vertex_label = [s if s in highlight else "" for s in g.vs["name"]]
    vertex_frame_color = ["black" if v["name"] in highlight else "white" for v in g.vs]
    vertex_size = [25 if v["name"][0].isdigit() else 15 for v in g.vs]

    layout = g.layout("kk")
    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        g, target=ax, layout=layout,
        vertex_label=vertex_label, vertex_label_size=14,
        vertex_color=vertex_colors, vertex_frame_color=vertex_frame_color,
        vertex_frame_width=1, vertex_size=vertex_size,
        edge_color=edge_color, edge_arrow_size=0.0,
        edge_width=g.es["weight"],
    )
    ax.axis("off")
    out = save_dir / f"{dataset}_graph_pathways_w_OLL_v1.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _save_graph_html(
        g, layout, vertex_colors, vertex_size,
        out.with_suffix(".html"),
    )
    return out
