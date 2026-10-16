"""Section 9 (LR CB clustering) plot_* functions: CB directedness,
cluster features/participation/connectivity/ROIs/graphs, left/right CB
clustering sweeps, and layer/VIC left-right homologue comparisons.
"""


import re
from pathlib import Path


import numpy as np
import pandas as pd
import plotly.graph_objects as go


from utils.config import FIG_DATASET as DATASET, FIG_DIR, SIDE_CHAR


from utils.plotting_shared import _SIDE_WORD, _AXIS, _add_gaussian_ellipses, _directedness_triangle, _heatmatrix, _hist_lines, _hist_lines_v, _point_line, _save, _scatter, _stacked_bars, _violin_median
from utils.plotting_clusters import _CLUSTER_TYPES, _save_graph_html


# %%
# Full-suffix cluster-type names with a deliberate L/R mix per side. Keep the
# order exactly as given so the example stacked bar keeps the same
# left-to-right layout.
_CB_CLUSTER_TYPES = {
    "r": [
        "pC1_2a_R", "LAL094_R", "IbSpsP_0", "LHPV5b3_R", "TuBu08_R",
        "SMP228_R", "ER4d_R", "DNp02_R", "pC1_5b_R", "DNg02_a_R",
    ],
    "l": [
        "pC1_2a_L", "LAL094_L", "IbSpsP_0", "LHPV5b3_L", "TuBu08_L",
        "SMP228_L", "ER4d_L", "DNp02_L", "pC1_5b_L", "DNg02_a_L",
    ],
}


# %%
def _cb_cluster_labels(n_clu: int, side_char: str = "r") -> list[str]:
    prefix = "cc'" if side_char.lower().startswith("l") else "cc"
    return [f"{prefix}{i}" for i in range(1, n_clu + 1)]


# %%
def _cb_cluster_side(cluster_df: pd.DataFrame, side_str: str = "") -> str:
    if side_str.startswith("_left"):
        return "l"
    if "instance" in cluster_df.columns:
        inst = cluster_df["instance"].astype(str)
        if len(inst) and inst.str.endswith("_L").all():
            return "l"
    return "r"


# %%
def plot_cb_directedness_histograms(
    type_dir_cb: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-direction (ff/fb/la) fraction-of-types histogram for VCBN pre-neurons.

    `type_dir_cb` comes from `pre.get_cb_type_directedness()`;
    `main_groups_pre` is collapsed to `'nonOL'` there, so all types share one
    colour. We build a one-row colour df that maps `'nonOL'` to the CB region
    colour.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    cmap = pd.DataFrame({"color": [colored_region_df.loc["CB", "color"]]}, index=["nonOL"])
    figs = []
    ff_bins = np.arange(-0.1, 1.2, 0.2)
    for direct in ["ff", "fb", "la"]:
        fig = _hist_lines(type_dir_cb, f"frac_{direct}", "main_groups_pre", ff_bins, cmap)
        fig.update_layout(
            xaxis_title=f"frac_{direct}", xaxis_range=[-.05, 1.05],
            yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
            xaxis_title_standoff=0, yaxis_title_standoff=2,
        )
        figs.append((fig, f"{dataset}_{direct}_frac_hist_all.pdf"))
    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_cb_directedness_triangle(
    type_dir_cb: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, cluster_types=None, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Ternary directedness triangle (la/fb/ff) for VCBN pre-neurons.

    Applies the eps=0.05 smoothing before plotting
    and highlights `cluster_types` (defaults to `_CB_CLUSTER_TYPES['r']`).
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    stars = list(cluster_types) if cluster_types is not None else _CB_CLUSTER_TYPES["r"]
    cmap = {"nonOL": colored_region_df.loc["CB", "color"]}
    d = type_dir_cb.copy()
    eps = 5e-2
    d[["frac_la", "frac_fb", "frac_ff"]] = d[["frac_la", "frac_fb", "frac_ff"]] * (1 - eps) + eps / 3
    fig = _directedness_triangle(d, "main_groups_pre", "instance_pre", cmap, stars)
    _save(fig, save_dir, f"{dataset}_examples_0.1thre_dir_triangle.pdf", html=True)
    return fig


# %%
def plot_cb_clusters_example_features(
    feature_df: pd.DataFrame, in_instances, cluster_types,
    colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Stacked bar of normalised CB feature vector for example output types.
    `cluster_types` are full instance names (with
    side suffix) from `_CB_CLUSTER_TYPES` — L/R mix allowed."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    # Keep every requested type as an x-axis slot, in the given order. A type
    # absent from the feature matrix (e.g. a VCBN whose VIC is below threshold
    # on this side, like pC1_5b on the left) becomes an all-NaN row and renders
    # as a blank bar instead of vanishing from the plot.
    sub = feature_df.reindex(list(cluster_types))[in_instances]
    plot_df = sub.div(sub.sum(1).replace(0, np.nan), axis=0)
    plot_df.columns = [s[:-2] for s in plot_df.columns]
    plot_df.index.name = None
    # dropna=False keeps the all-NaN row so its category stays on the axis.
    plot_df = plot_df.stack(0, dropna=False).reset_index(name="effective weight").rename(
        columns={"level_0": "out type", "level_1": "in type"}
    )
    fig = _stacked_bars(
        plot_df, "out type", "effective weight", "in type",
        colored_seed_df["color"].to_dict(),
        xrot=-45, height=450, width=550,
    )
    fig.update_layout(xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}{side_str}_cb_in_out_examples.pdf", html=True)
    return fig


# %%
def plot_cb_clusters_pathway_features(
    feature_df: pd.DataFrame, in_instances, cluster_df: pd.DataFrame,
    cluster_num: pd.Series, colored_seed_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
):
    """Stacked bar of per-cluster mean normalised CB feature vector.
    Labels right clusters as cc1..ccN and left clusters
    as cc'1..cc'N, with member count."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    sub = feature_df.loc[:, in_instances] / feature_df.loc[:, in_instances].sum(1).values[:, np.newaxis]
    df = sub.reset_index(names="instance").merge(cluster_df, on="instance")
    df = df.groupby("cluster")[in_instances].mean()
    df.columns = [s[:-2] for s in df.columns]
    df = df.stack(0).reset_index(name="mean effective weight").rename(
        columns={"level_0": "cluster", "level_1": "in type"}
    )
    fig = _stacked_bars(
        df, "cluster", "mean effective weight", "in type",
        colored_seed_df["color"].to_dict(), height=400, width=495,
    )
    n_clu = len(cluster_num)
    cluster_labels = _cb_cluster_labels(n_clu, _cb_cluster_side(cluster_df, side_str))
    fig.update_layout(
        xaxis_tickvals=np.arange(1, n_clu + 1),
        xaxis_ticktext=[
            f"{cluster_labels[i - 1]} ({cluster_num[i]})"
            for i in range(1, n_clu + 1)
        ],
        xaxis_tickangle=-45,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        font=dict(family="arial", size=16),
    )
    _save(fig, save_dir, f"{dataset}{side_str}_cb_in_out_pathways.pdf", html=True)
    return fig


# %%
def plot_ol_cb_cluster_connectivity(
    conn: pd.DataFrame, *, dataset: str = DATASET,
    save_dir: Path | None = None, anno_min_frac: float = 0.05,
):
    """Two heatmaps: OL-cluster × CB-cluster connectivity, input-normalised
    (each CB column sums to 1) and output-normalised (each OL row sums to 1).
    Annotations show raw total weight (K/M) where normalised ≥ `anno_min_frac`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"

    def _format_total(v):
        if v >= 1e6:
            return f"{v / 1e6:.0f}M"
        if v >= 1e3:
            return f"{v / 1e3:.0f}K"
        return f"{v:.0f}"

    def _one(norm_df, total_df, title, stem):
        anno = np.empty(total_df.shape, dtype=object)
        for j in range(total_df.shape[1]):
            for i in range(total_df.shape[0]):
                anno[i, j] = (
                    _format_total(total_df.values[i, j])
                    if norm_df.values[i, j] > anno_min_frac else ""
                )
        fig = go.Figure(data=go.Heatmap(
            z=norm_df.values, x=list(norm_df.columns), y=list(norm_df.index),
            colorscale="Greys", showscale=True, zmin=0, zmax=1,
            text=anno, texttemplate="%{text}", textfont=dict(size=18),
        ))
        fig.update_layout(
            font=dict(family="arial", size=18),
            title=title, xaxis_title="", yaxis_title="",
            xaxis_tickvals=list(norm_df.columns),
            xaxis_ticktext=[f"cc{c}" for c in norm_df.columns],
            yaxis_tickvals=list(norm_df.index),
            yaxis_ticktext=[f"oc{c}" for c in norm_df.index],
            paper_bgcolor="white", plot_bgcolor="white",
            height=550, width=550,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black",
                         mirror=True, title_standoff=0)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black",
                         mirror=True, title_standoff=2, autorange="reversed")
        _save(fig, save_dir, f"{dataset}_{stem}.pdf")
        return fig

    total = conn.copy()
    col_sum = np.abs(total).sum(axis=0).replace(0, 1)
    row_sum = np.abs(total).sum(axis=1).replace(0, 1)
    norm_in = total.div(col_sum, axis=1)
    norm_out = total.div(row_sum, axis=0)

    fig_in = _one(norm_in, total,
                  "input-normalised connectivity from VPN clusters to CB clusters",
                  "cb_clusters_in")
    fig_out = _one(norm_out, total,
                   "output-normalised connectivity from VPN clusters to CB clusters",
                   "cb_clusters_out")
    return fig_in, fig_out


# %%
_CB_ROI_ORDER = [
    "PVLP(R)", "PVLP(L)", "SLP(R)", "IB", "CRE(R)",
    "AVLP(R)", "AVLP(L)", "EB", "IPS(R)", "SIP(R)",
    "PLP(R)", "PLP(L)", "SMP(R)", "ICL(R)", "VES(R)",
    "SPS(R)", "SPS(L)", "GNG", "SCL(R)", "WED(R)",
    "LAL(R)", "LAL(L)", "AOTU(R)", "PB", "BU(R)",
]


# %%
def _fmt_km(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}"
    return f"{int(v)}"


# %%
def plot_cb_clusters_rois(
    tbar_counts: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "", save_dir: Path | None = None,
    roi_order: list[str] | None = None, anno_min: float = 0.1,
):
    """CB-cluster × ROI normalised-tbar heatmap."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    roi_order = roi_order if roi_order is not None else _CB_ROI_ORDER
    roi_keep = [r for r in roi_order if r in tbar_counts.columns]
    m = tbar_counts[roi_keep]
    total_all = tbar_counts.values.sum()
    total_shown = m.values.sum()
    frac_shown = total_shown / total_all if total_all > 0 else 0.0
    m_norm = m.div(m.sum(0), axis=1).fillna(0)

    show = m_norm.values > anno_min
    # Guarantee at least one value per row: for any cluster row with nothing
    # above the threshold, reveal its largest (non-zero) cell.
    missing = ~show.any(axis=1) & (m_norm.values.max(axis=1) > 0)
    if missing.any():
        rows = np.nonzero(missing)[0]
        cols = m_norm.values[rows].argmax(axis=1)
        show[rows, cols] = True
    anno = np.where(
        show, np.vectorize(_fmt_km)(m.values.astype(float)), "",
    )

    fig = go.Figure(data=go.Heatmap(
        z=m_norm.values, x=list(m_norm.columns), y=list(m_norm.index),
        colorscale="Greys", showscale=True, zmin=0, zmax=1,
        text=anno, texttemplate="%{text}", textfont=dict(size=20),
    ))
    fig.update_layout(
        font=dict(family="arial", size=21),
        title=f"frac. synapses per cluster (showing {frac_shown:.0%} of total)",
        xaxis_title="", yaxis_title="",
        xaxis_tickvals=list(range(len(m_norm.columns))),
        xaxis_ticktext=list(m_norm.columns),
        yaxis_tickvals=list(range(1, m_norm.shape[0] + 1)),
        yaxis_ticktext=[f"cc{i}" for i in m_norm.index],
        paper_bgcolor="white", plot_bgcolor="white",
        height=440, width=1188,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, tickangle=-90, title_standoff=0)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, autorange="reversed", title_standoff=2)
    _save(fig, save_dir, f"{dataset}{side_str}_cb_clusters_rois.pdf", html=True)
    return fig


# %%
def plot_cb_participation_n_clu_out(
    type_part_cb: pd.DataFrame, colored_main_groups_df: pd.DataFrame, *,
    side_char: str = SIDE_CHAR, dataset: str = DATASET,
    side_str: str = "", save_dir: Path | None = None,
):
    """Histogram of n_clusters (CB clusters with pi > 1/N) for VPN
    cell types. Output `…_in_out_n_clu_participation_out.pdf`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    side_word = _SIDE_WORD[side_char] if side_char in _SIDE_WORD else "right"
    sub = type_part_cb[
        type_part_cb["region"].isin([f"{side_word} OL"])
        & (type_part_cb["main_groups"] == "VPN")
    ]
    cluster_names = [c for c in type_part_cb.columns if bool(re.match(r"^c\d+$", str(c)))]
    n_clu = len(cluster_names)
    bins = np.linspace(0, n_clu + 1, n_clu + 2) - 0.5
    fig = _hist_lines(sub, "n_clusters", "main_groups", bins, colored_main_groups_df)
    fig.update_layout(
        xaxis_title=f"no. clusters with pi>1/{n_clu}",
        xaxis_range=[-.1, n_clu + 0.1], xaxis_tickvals=np.arange(n_clu + 1),
        yaxis_title="frac. cell types", yaxis_range=[-.01, 0.61],
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    _save(fig, save_dir, f"{dataset}{side_str}_in_out_n_clu_participation_out.pdf")
    return fig


# %%
def plot_cb_clusters_graph(
    type_part_cb: pd.DataFrame, ol_cluster_df: pd.DataFrame,
    colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, side_str: str = "",
    save_dir: Path | None = None, edge_color: str = "lightgray",
    ol_highlight: list[str] | None = None,
):
    """igraph network of VPN types + CB clusters, edges are
    participation ≥ 1/N."""
    import igraph as ig
    import matplotlib.pyplot as plt
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    save_dir.mkdir(parents=True, exist_ok=True)

    cluster_names_c = [c for c in type_part_cb.columns if bool(re.match(r"^c\d+$", str(c)))]
    cluster_names_no_c = [s[1:] for s in cluster_names_c]
    n_clu = len(cluster_names_c)

    base = type_part_cb[type_part_cb["main_groups"] == "VPN"].copy()
    long = base.melt(
        id_vars=["instance"], value_vars=cluster_names_c,
        var_name="post", value_name="weight",
    ).rename(columns={"instance": "pre"})
    long = long[long["weight"] >= 1 / n_clu].copy()
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

    # Color: CB cluster nodes share the CB region gray; VPN cell types
    # keep the per-cluster blue gradient.
    cb_gray = colored_region_df.loc["CB", "color"]
    cluster_to_color = {str(i + 1): cb_gray for i in range(n_clu)}

    ol_color = ol_cluster_df.copy()
    ol_color["cell_type"] = [s.split("_")[0] for s in ol_color["instance"]]
    ol_blues = [
        "#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
        "#6baed6", "#9ecae1", "#c6dbef", "#deebf7",
    ]
    n_ol = int(ol_color["cluster"].max())
    ol_color["color"] = [ol_blues[(c - 1) % len(ol_blues)] for c in ol_color["cluster"]]
    pre_color = ol_color.set_index("cell_type")["color"].to_dict()
    pre_color.update(cluster_to_color)
    cb_fallback = colored_region_df.loc["CB", "color"]
    vertex_colors = [pre_color.get(v["name"], cb_fallback) for v in g.vs]

    ol_types = ol_highlight if ol_highlight is not None else _CLUSTER_TYPES
    highlight = set(cluster_names_no_c) | set(ol_types)
    vertex_label = [s if s in highlight else "" for s in g.vs["name"]]
    vertex_frame_color = ["black" if v["name"] in highlight else "white" for v in g.vs]
    vertex_size = [25 if v["name"][0].isdigit() else 15 for v in g.vs]

    layout = g.layout("kk")
    fig, ax = plt.subplots(figsize=(8, 6))
    ig.plot(
        g, target=ax, layout=layout,
        vertex_label=vertex_label, vertex_label_size=14,
        vertex_color=vertex_colors, vertex_frame_color=vertex_frame_color,
        vertex_frame_width=1, vertex_size=vertex_size,
        edge_color=edge_color, edge_arrow_size=0.0,
        edge_width=[0.1 + 3 * w for w in g.es["weight"]],
    )
    ax.axis("off")
    out = save_dir / f"{dataset}{side_str}_graph_cb_pathways_v1.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _save_graph_html(
        g, layout, vertex_colors, vertex_size,
        out.with_suffix(".html"),
    )
    return out


# %%
def plot_out_cb_highres_sub(
    tb_clu_df: pd.DataFrame, sizes: pd.Series, *,
    thre_row: float = 1.0, thre_col: float = 0.8, thre_row2: float = 0.5,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Subset of VPN × CB-type high-res effective-weight heatmap,
    filtered by row/col/row2 score thresholds."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    out_score = tb_clu_df.sum(1)
    ol_major = out_score[out_score > thre_row].index
    cb_score = tb_clu_df.loc[ol_major].sum(0)
    cb_major = cb_score[cb_score > thre_col].index
    out_score2 = tb_clu_df.loc[:, cb_major].sum(1)
    ol_major2 = out_score2[out_score2 > thre_row2].index
    sub = tb_clu_df.loc[ol_major2, cb_major]

    xtt = [s.split("_")[0] + f" ({int(sizes.get(s, 0))})" for s in sub.columns]
    ytt = [s.split("_")[0] + f" ({int(sizes.get(s, 0))})" for s in sub.index]
    fig = go.Figure(data=go.Heatmap(
        z=sub.values, x=list(sub.columns), y=list(sub.index),
        colorscale="Greys", showscale=True, zmin=0, zmax=1,
    ))
    fig.update_layout(
        font=dict(family="arial", size=16),
        title="rel effective weight from VPN to high-res. CB cell types (major only)",
        xaxis_title="high-resolution VCBN types (size in col)",
        yaxis_title="high-resolution VPN types (size in col)",
        xaxis_tickvals=list(range(sub.shape[1])), xaxis_ticktext=xtt,
        yaxis_tickvals=list(range(sub.shape[0])), yaxis_ticktext=ytt,
        paper_bgcolor="white", plot_bgcolor="white",
        height=600, width=1500,
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, tickangle=-45, title_standoff=0)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black",
                     mirror=True, autorange="reversed", title_standoff=2)
    _save(fig, save_dir, f"{dataset}_out_cb_highres_in_sub.pdf")
    return fig


# %%
def plot_rf_example_pos_par(
    meta_ol: pd.DataFrame, stepsn, raw_ol: pd.DataFrame, bodyId: int,
    in_instances: list[str], *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Hex heatmap for a single bodyId with Gaussian ellipse overlay
    (output `…_rf_pos_par.pdf`)."""
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    row = raw_ol[raw_ol["bodyId"] == bodyId]
    if row.empty:
        return None
    params_single = row.iloc[[0]]

    inidx = meta_ol[meta_ol.instance.isin(in_instances)].idx.values
    outidx = meta_ol[meta_ol.bodyId == bodyId].idx.values
    from utils.ol_rf import compute_rf
    df = compute_rf(meta_ol, stepsn, inidx, outidx)
    if df.empty:
        return None
    plot_df = df.set_index("coords")[["effective weight"]]
    cmax = float(np.abs(plot_df["effective weight"]).max())

    fig = hex_heatmap(
        plot_df, custom_colorscale="Picnic",
        global_min=-1, global_max=1, dataset="mcns_right",
    )
    _add_gaussian_ellipses(fig, params_single, example_bid=bodyId, y_scale=np.sqrt(3))
    _save(fig, save_dir, f"{dataset}_example_{bodyId}_rf_pos_par.pdf")
    return fig


# %%
def plot_ol_layers_hit_diff(
    roi_adj: pd.DataFrame, rois: list[str], labels: list[str], *,
    filename_stem: str, dataset: str = DATASET, save_dir: Path | None = None,
    height: int = 800, width: int = 500,
):
    """Per-ROI distribution of connection layer differences (post - pre).

    For each anatomical OL ROI in `rois`, takes every (pre, post) connection
    with at least one synapse in that ROI, expands by `weight` so each
    individual synapse is a sample, and draws a horizontal violin of
    `hit_diff`. Y-axis labels are `labels` (e.g. M1..AME or LO1..LOP4),
    matching SI 1i. `roi_adj` comes from `get_ol_roi_adjacency`.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    sub = roi_adj[roi_adj["roi"].isin(rois)]
    sub = sub.loc[sub.index.repeat(sub["weight"])][["roi", "hit_diff"]]
    sub["roi"] = pd.Categorical(sub["roi"], categories=rois, ordered=True)
    sub = sub.sort_values("roi")
    # Downsample to avoid exceeding kaleido's 128 MB IPC message limit.
    _MAX_POINTS = 500_000
    if len(sub) > _MAX_POINTS:
        # Proportional per-ROI downsample. A plain groupby loop (rather than
        # groupby.apply) keeps the `roi` grouping column and avoids the
        # pandas >=2.2 "apply operated on the grouping columns" FutureWarning.
        parts = [
            g.sample(
                n=max(1, round(_MAX_POINTS * len(g) / len(sub))),
                random_state=0,
            )
            for _, g in sub.groupby("roi", observed=True)
        ]
        sub = pd.concat(parts)
    fig = _violin_median(
        sub, "hit_diff", "roi", labels=labels, height=height, width=width,
        xaxis_title="post - pre layer", xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
        xaxis_range=[-3.5, 3.5],
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_dir / f"{dataset}_{filename_stem}_diff.png", scale=2)
    return fig


# %%
def plot_cb_layer_hist_left(
    layer_lr_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Horizontal histogram of left-side hitting time for VPN + CB
    instances."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    cmap = colored_region_df.copy()
    cmap.loc["VPN", "color"] = colored_main_groups_df.loc["VPN", "color"]
    df = layer_lr_df.copy()
    id_col = "instance" if "instance" in df.columns else "cell_type"
    df = df[~df[id_col].str.contains("_unclear", regex=False)]
    ol = df[
        df["region"].isin(["left OL", "right OL"])
        & (df["main_groups"] == "VPN")
    ].copy()
    ol["region"] = "VPN"
    cb = df[df["region"] == "CB"]
    plot_df = pd.concat([ol, cb], axis=0, ignore_index=True)

    bins = np.arange(0.5 - 0.25 / 2, 7 + 0.25 / 2, 0.5 / 2)
    fig = _hist_lines_v(
        plot_df, "hitting_time_l", "region", bins, cmap,
        height=400, width=400,
    )
    fig.update_yaxes(range=[0.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6])
    _save(fig, save_dir, "layer_hist_left.pdf")
    return fig


# %%
def plot_cb_clusters_size_vs_cluster(
    cb_cluster_df: pd.DataFrame, cb_type_df: pd.DataFrame,
    colored_region_df: pd.DataFrame, *,
    size_col: str = "size", size_lo: float = 4.5, size_hi: float = 250,
    size_tickvals: tuple[float, ...] = (5, 10, 50, 100),
    dataset: str = DATASET, save_dir: Path | None = None, jitter: float = 0.2,
    highlight: list[str] | None = None,
):
    """Scatter ARF size × cluster for CB types (output
    `examples_size_vs_cluster_v2`).

    `cb_type_df` must have `instance`, `region` and a size column named
    `size_col`. Pass `size_col="ahull_size"` (e.g. `get_rf_ahull_type_cb()`) to
    use the alpha-hull size instead of the 2D-Gaussian `size`; `size_lo`/
    `size_hi`/`size_tickvals` set the (log) x-axis to fit the chosen metric."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    n_clu = int(cb_cluster_df["cluster"].max())
    cluster_names = _cb_cluster_labels(n_clu, _cb_cluster_side(cb_cluster_df))
    stars = list(highlight) if highlight is not None else _CB_CLUSTER_TYPES.get("r", [])

    merged = cb_cluster_df.merge(
        cb_type_df[["instance", size_col, "region"]], on="instance", how="inner",
    ).copy()
    rng = np.random.default_rng(0)
    merged["y"] = merged["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(merged))

    fig = _scatter(
        merged, size_col, "y", "region", "instance",
        colored_region_df, stars, height=400, width=500,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names,
    )
    fig.update_xaxes(type="log", tickformat="d",
                     tickvals=list(size_tickvals),
                     range=[np.log10(size_lo), np.log10(size_hi)])
    fig.update_layout(yaxis_title="cluster", xaxis_title="ARF size (col)",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}_examples_size_vs_cluster_v2.pdf", html=True)
    return fig


# %%
def plot_cb_clusters_scatter(
    cb_cluster_df: pd.DataFrame, layer_lr: pd.DataFrame,
    colored_region_df: pd.DataFrame, *,
    dataset: str = DATASET, side_str: str = "_cb",
    save_dir: Path | None = None, jitter: float = 0.2,
    highlight: list[str] | None = None,
):
    """Scatter layer (hitting_time) × cluster for CB instances, colored by
    region. Mirrors the OL `plot_clusters_scatter` layer-vs-cluster panel.
    `layer_lr` supplies per-instance `hitting_time_r` (right side layer)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    n_clu = int(cb_cluster_df["cluster"].max())
    cluster_names = _cb_cluster_labels(n_clu, _cb_cluster_side(cb_cluster_df, side_str))
    stars = list(highlight) if highlight is not None else _CB_CLUSTER_TYPES.get("r", [])

    base = cb_cluster_df.merge(
        layer_lr[["instance", "hitting_time_r", "region"]],
        on="instance", how="left",
    ).copy()
    base = base.dropna(subset=["hitting_time_r"])
    rng = np.random.default_rng(0)
    base["y"] = base["cluster"].astype(float) + rng.uniform(-jitter, jitter, size=len(base))

    fig = _scatter(
        base, "hitting_time_r", "y", "region", "instance",
        colored_region_df, stars, height=300, width=400,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickvals=np.arange(1, n_clu + 1), ticktext=cluster_names,
    )
    fig.update_xaxes(range=[1.5, 5.5])
    fig.update_layout(
        xaxis_title="layer", yaxis_title="cluster",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        font=dict(family="arial", size=20),
    )
    _save(fig, save_dir, f"{dataset}{side_str}_examples_layer_vs_cluster.pdf", html=True)
    return fig


# %%
def plot_cb_clusters_intra_connectivity(
    intra: dict, *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """CB-cluster × CB-cluster connectivity heatmap (output-normalised).
    Writes `{dataset}_cb_outrel_conn_matrix_clusters.pdf` in `CB_clustering/`
    — `cb_` prefix distinguishes it from the OL analogue in `pathways/`.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    n = intra["n_clu"]
    tick_labels = _cb_cluster_labels(n, "r")
    fig = _heatmatrix(
        intra["matrix"], tick_labels, cmap="Greys", height=600, width=600,
        font_size=18 * 1.2,  # 20% larger than the _heatmatrix default
    )
    fig.update_layout(
        title=f"all weights: {intra['total_weight']: .0f}",
        xaxis_title="post", yaxis_title="pre", xaxis_tickangle=0,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(
        colorbar=dict(tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"]),
        zmin=0, zmax=1,
    )
    _save(fig, save_dir, f"{dataset}_cb_outrel_conn_matrix_clusters.pdf")
    return fig


# %%
def plot_cb_lr_clustering_sweep(
    sweep_df: pd.DataFrame, n0: int, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Line panels for the CB L/R clustering threshold sweep.

    `n0` is the reference cluster count (from the chosen `frac_thre` in
    section 5). Panels: n_R vs frac, n_L vs n_R, ARI_LR vs n_R, ARI_fromR vs
    n_R, ARI_fromR vs ARI_LR.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    matches = np.where(sweep_df["n_clu_R"].values == n0)[0]
    idx0 = int(matches[0]) if len(matches) else int(np.argmin(
        np.abs(sweep_df["n_clu_R"].values - n0)
    ))
    figs = []

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["frac"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(xaxis_title="No. R clusters", yaxis_title="Clustering threshold")
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0, 2])
    fig.update_yaxes(type="log", tickvals=[0.05, 0.1, 0.5, 1], range=[-1.7, 0.1])
    figs.append((fig, f"{dataset}_cb_no_thre.pdf"))

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["n_clu_L"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.add_trace(go.Scatter(x=[1, 300], y=[1, 300], mode="lines",
                             line=dict(color="gray", dash="dash"), showlegend=False))
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="no. L clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    figs.append((fig, f"{dataset}_cb_LR_no_clusters.pdf"))

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_LR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_LR_ARI.pdf"))

    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_fromR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0])
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and L from R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_LR_fromR_ARI.pdf"))

    df = pd.DataFrame({"x": sweep_df["ari_fromR"].values, "y": sweep_df["ari_LR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=500)
    fig.add_trace(go.Scatter(x=[0.01, 1], y=[0.01, 1], mode="lines",
                             line=dict(color="gray", dash="dash"), showlegend=False))
    fig.update_layout(
        yaxis_title="ARI between L and R clusters",
        xaxis_title="ARI between L and L from R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_LR_ARI_comparison.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_cb_lr_cluster_confusion(
    match: dict, *,
    dataset: str = DATASET, save_dir: Path | None = None, anno_min: int = 10,
    filename_stem: str = "cb_left_to_right_confusion",
    xaxis_title: str = "right cluster", yaxis_title: str = "left cluster",
):
    """Heatmap of the CB L↔R cluster confusion matrix."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    conf = match["confusion"]
    x_tick_labels = match.get("x_tick_labels")
    y_tick_labels = match.get("y_tick_labels")
    if x_tick_labels is None or y_tick_labels is None:
        tick_labels = match.get("tick_labels")
        if tick_labels is not None:
            x_tick_labels = list(tick_labels)[:conf.shape[1]]
            y_tick_labels = list(tick_labels)[:conf.shape[0]]
        else:
            x_tick_labels = _cb_cluster_labels(conf.shape[1], "r")
            y_tick_labels = _cb_cluster_labels(conf.shape[0], "l")
    if len(x_tick_labels) != conf.shape[1]:
        x_tick_labels = _cb_cluster_labels(conf.shape[1], "r")
    if len(y_tick_labels) != conf.shape[0]:
        y_tick_labels = _cb_cluster_labels(conf.shape[0], "l")
    anno = np.where(conf >= anno_min, conf.astype(object), "")
    fig = _heatmatrix(
        conf, x_tick_labels, y_tick_labels=y_tick_labels,
        anno=anno, width=550,
    )
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf")
    return fig


# %%
def _plot_layer_lr_scatter(
    df: pd.DataFrame, id_col: str, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    highlight: list[str] | None = None,
):
    """Shared scatter builder for right-layer vs left-layer. Filters to OL
    output + CB rows, recolors VPN with `colored_main_groups_df`."""
    cmap = colored_region_df.copy()
    cmap.loc["VPN", "color"] = colored_main_groups_df.loc["VPN", "color"]
    df = df[~df[id_col].str.contains("_unclear", regex=False)]
    ol = df[
        df["region"].isin(["left OL", "right OL"])
        & (df["main_groups"] == "VPN")
    ].copy()
    ol["region"] = "VPN"
    cb = df[df["region"] == "CB"]
    plot_df = pd.concat([ol, cb], axis=0, ignore_index=True)
    stars = list(highlight) if highlight else []
    fig = _scatter(
        plot_df, "hitting_time_r", "hitting_time_l", "region", id_col,
        cmap, stars,
    )
    fig.add_trace(go.Scatter(
        x=[0.8, 7.5], y=[0.8, 7.5], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="right layer", yaxis_title="left layer",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    return fig


# %%
def plot_cb_layer_lr_homologue(
    layer_lr_homologue: pd.DataFrame, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
    highlight: list[str] | None = None,
):
    """Cell-type-matched right-layer vs left-layer scatter (output
    `layer_lr_comp`). Takes output of `get_cb_layer_lr_homologue`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    fig = _plot_layer_lr_scatter(
        layer_lr_homologue, "cell_type", colored_region_df,
        colored_main_groups_df, highlight=highlight,
    )
    fig.update_xaxes(range=[0.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6])
    fig.update_yaxes(range=[0.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6])
    _save(fig, save_dir, f"{dataset}_cb_layer_lr_comp.pdf", html=True)
    return fig


# %%
def plot_cb_layer_lr_comparison(
    layer_lr: pd.DataFrame, colored_region_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
    highlight: list[str] | None = None,
):
    """Per-instance right-layer vs left-layer scatter (output
    `layer_lr_idx_comp`)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    fig = _plot_layer_lr_scatter(
        layer_lr, "instance", colored_region_df,
        colored_main_groups_df, highlight=highlight,
    )
    fig.update_xaxes(range=[1.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6, 7])
    fig.update_yaxes(range=[1.8, 6.5], tickvals=[1, 2, 3, 4, 5, 6, 7])
    _save(fig, save_dir, f"{dataset}_cb_layer_lr_idx_comp.pdf", html=True)
    return fig


# %%
def plot_cb_vic_lr_homologue(
    homologue_df: pd.DataFrame, colored_region_df: pd.DataFrame, *,
    vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Per-cell-type right vs left CB VIC scatter + diff histogram.
    Region labels are based on summed VIC:
    `VIC_r+VIC_l < vic_thre` → VNC, `< 1e-2` → left OL, `>= 1e-2` → right OL.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    df = homologue_df.copy()
    df["VIC_diff"] = (df["VIC_r"] - df["VIC_l"]) / (df["VIC_r"] + df["VIC_l"])
    s = df["VIC_r"] + df["VIC_l"]
    sub1 = df[s < vic_thre].copy(); sub1["region"] = "VNC"
    sub2 = df[(s >= vic_thre) & (s < 1e-2)].copy(); sub2["region"] = "left OL"
    sub3 = df[s >= 1e-2].copy(); sub3["region"] = "right OL"
    plot_df = pd.concat([sub1, sub2, sub3], axis=0, ignore_index=True)

    fig = _scatter(plot_df, "VIC_r", "VIC_l", "region", "cell_type",
                   colored_region_df, [])
    fig.add_trace(go.Scatter(
        x=[5e-8, 2], y=[5e-8, 2], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    fig.update_xaxes(type="log", range=[np.log10(5e-8), np.log10(1.2)])
    fig.update_yaxes(type="log", range=[np.log10(5e-8), np.log10(1.2)])
    _save(fig, save_dir, f"{dataset}_cb_vic_lr_comp.pdf", html=True)

    bins = np.linspace(-1.05, 1.05, 22)
    centers = (bins[1:] + bins[:-1]) / 2
    hist_fig = go.Figure()
    for reg, sub in [("VNC", sub1), ("left OL", sub2), ("right OL", sub3)]:
        if sub.empty:
            continue
        counts, _ = np.histogram(sub["VIC_diff"].values, bins=bins)
        hist_fig.add_trace(go.Scatter(
            x=centers, y=counts, mode="lines+markers", name=reg,
            line=dict(color=colored_region_df.loc[reg, "color"], width=2),
        ))
    hist_fig.update_layout(
        xaxis_title="(right - left) / (right + left) VIC",
        yaxis_title="no. cell types",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        paper_bgcolor="white", plot_bgcolor="white",
        height=400, width=500, font=dict(family="arial", size=14),
    )
    hist_fig.update_xaxes(**_AXIS)
    hist_fig.update_yaxes(**_AXIS)
    _save(hist_fig, save_dir, f"{dataset}_cb_vic_lr_comp_diag.pdf")
    return fig, hist_fig


# %%
def plot_cb_vic_binocularity(
    vic_lr_type: pd.DataFrame, colored_region_df: pd.DataFrame, *,
    vic_thre: float = 5e-4, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-instance VIC_r vs VIC_l scatter, colored by binocularity region:
    'VNC' = both below `vic_thre`, 'right OL' = both above, 'left OL' = one
    above one below. Output `vic_lr_binoc`.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_clustering"
    df = vic_lr_type.copy()
    df["VIC_diff"] = (df["VIC_r"] - df["VIC_l"]) / (df["VIC_r"] + df["VIC_l"])
    both_below = (df["VIC_r"] < vic_thre) & (df["VIC_l"] < vic_thre)
    both_above = (df["VIC_r"] >= vic_thre) & (df["VIC_l"] >= vic_thre)
    sub1 = df[both_below].copy(); sub1["region"] = "VNC"
    sub2 = df[both_above].copy(); sub2["region"] = "right OL"
    sub3 = df[~both_below & ~both_above].copy(); sub3["region"] = "left OL"
    plot_df = pd.concat([sub1, sub2, sub3], axis=0, ignore_index=True)

    fig = _scatter(plot_df, "VIC_r", "VIC_l", "region", "instance",
                   colored_region_df, [])
    fig.add_trace(go.Scatter(
        x=[5e-8, 2], y=[5e-8, 2], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    fig.update_xaxes(type="log", range=[np.log10(8e-7), np.log10(1)])
    fig.update_yaxes(type="log", range=[np.log10(8e-7), np.log10(1)])
    _save(fig, save_dir, f"{dataset}_cb_vic_lr_binoc.pdf", html=True)

    bins = np.linspace(-1.05, 1.05, 22)
    hist_fig = go.Figure()
    for reg, sub in [("VNC", sub1), ("right OL", sub2), ("left OL", sub3)]:
        if sub.empty:
            continue
        counts, _ = np.histogram(sub["VIC_diff"].values, bins=bins)
        centers = (bins[1:] + bins[:-1]) / 2
        color = colored_region_df.loc[reg, "color"]
        hist_fig.add_trace(go.Scatter(
            x=centers, y=counts, mode="lines+markers", name=reg,
            line=dict(color=color, width=2),
        ))
    hist_fig.update_layout(
        xaxis_title="(right - left) / (right + left) VIC",
        yaxis_title="no. cell types",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        paper_bgcolor="white", plot_bgcolor="white",
        height=400, width=500, font=dict(family="arial", size=14),
    )
    hist_fig.update_xaxes(**_AXIS)
    hist_fig.update_yaxes(**_AXIS)
    _save(hist_fig, save_dir, f"{dataset}_cb_vic_lr_binoc_diag.pdf")
    return fig, hist_fig
