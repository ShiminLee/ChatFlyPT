"""Section 2 (Flow) plot_* functions: lateral-flow-matrix summaries by
main group / sign / cell type, and OL-layer flow.
"""


from pathlib import Path


import numpy as np
import pandas as pd
import plotly.graph_objects as go


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _box, _directedness_triangle, _heatmatrix, _hist_lines, _hist_sum_lines, _point_line, _save, _scatter, _violin_median


# %%
_FLOW_PLOT_TYPES = [
    "Mi1", "Mi9", "Tm5c", "T4a", "LPi2b", "C2",
    "Mi4", "TmY16", "TmY20", "Dm9", "Dm4", "Y13", "Cm11b",
    "MeTu1", "MeTu3c", "MeTu2a", "aMe4",
    "LPC2", "LPLC2", "LC4", "LC6", "LC9", "VS",
    "LPT111", "LC10a", "LPT23", "LoVP91",
]
_FLOW_STAR_TYPES = [
    "LC6", "LC4", "VS", "LPLC2", "Tm5c", "T4a",
    "C2", "Dm9", "TmY16", "aMe4", "Mi1",
]


# %%
def _plot_types(side_suffix="_R"):
    return [s + side_suffix for s in _FLOW_PLOT_TYPES], [s + side_suffix for s in _FLOW_STAR_TYPES]


# %%
def plot_coords(
    ol_meta: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Hex heatmap of visual-input column counts."""
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    df = ol_meta[ol_meta.main_groups == "visual input"]
    df = df.groupby("coords")["bodyId"].nunique().reset_index().rename(columns={"bodyId": "count"})
    df = df.set_index("coords")
    fig = hex_heatmap(
        df, custom_colorscale="RdBu_r", global_min=-6, global_max=6,
        dataset="mcns_right",
    )
    _save(fig, save_dir, f"{dataset}_coords.pdf")
    return fig


# %%
def plot_flow_sector_division(
    sector_map: pd.DataFrame, *,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Hex heatmap of the 5-sector retinotopic division."""
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    fig = hex_heatmap(
        sector_map.set_index("coords")[["sector"]],
        custom_colorscale="RdBu_r", global_min=-5, global_max=-5,
        dataset="mcns_right",
    )
    _save(fig, save_dir, f"{dataset}_division.pdf")
    return fig


# %%
def plot_flow_by_main_groups(
    flow_type_ol: pd.DataFrame, conn_ol: pd.DataFrame,
    type_dir_ol: pd.DataFrame, flow_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, hit_diff_thre: float = 0.5, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []

    # fraction of cell types per layer by main_groups
    bins = np.arange(-0.5, 6 + 0.5, 1)
    fig = _hist_lines(flow_type_ol, "hitting_time", "main_groups", bins, colored_main_groups_df)
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 5.1], xaxis_tickvals=[0, 1, 2, 3, 4, 5],
        yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_frac_groups_types.pdf"))

    # fraction of weights vs layer difference by main_groups_pre
    diff_bins = np.array(range(-4, 4)).astype(float) + .5
    fig = _hist_sum_lines(conn_ol, "hitting_time_diff", "weight", "main_groups_pre",
                          diff_bins, colored_main_groups_df)
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="frac. weights",
        xaxis_range=[-3.2, 3.2], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
        yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_frac_groups_conn.pdf"))

    # type-level ff/fb/la distributions by main_groups_pre
    ff_bins = np.arange(-0.1, 1.2, 0.2)
    for direct in ["ff", "fb", "la"]:
        fig = _hist_lines(type_dir_ol, f"frac_{direct}", "main_groups_pre",
                          ff_bins, colored_main_groups_df)
        fig.update_layout(
            xaxis_title=f"frac_{direct}", xaxis_range=[-.05, 1.05],
            yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
        )
        figs.append((fig, f"{dataset}_{direct}_frac_hist_all_groups.pdf"))

    # fine-grained layer histogram (raw counts)
    fine_bins = np.arange(-0.25 / 2, 6 + 0.25 / 2, .5 / 2)
    fig = _hist_lines(flow_ol, "hitting_time_median", "main_groups",
                      fine_bins, colored_main_groups_df, norm=False)
    for i in range(6):
        fig.add_vline(x=i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 3.3], xaxis_tickvals=[0, 1, 2, 3, 4, 5],
        yaxis_title="no. neurons",
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_fine_tot_groups_types.pdf"))

    # fine-grained diff histogram (sum weights, raw)
    diff_fine = np.arange(-4 - 0.25 / 2, 4 + 0.25 / 2, .5 / 2)
    fig = _hist_sum_lines(conn_ol, "hitting_time_diff", "weight", "main_groups_pre",
                          diff_fine, colored_main_groups_df, norm=False)
    for i in range(7):
        fig.add_vline(x=-3 + i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="no. synapses",
        xaxis_range=[-1.6, 1.6], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_fine_tot_groups_conn.pdf"))

    # fine-grained diff histogram (pair counts)
    conn_pairs = conn_ol.copy()
    conn_pairs["weight"] = 1
    fig = _hist_sum_lines(conn_pairs, "hitting_time_diff", "weight", "main_groups_pre",
                          diff_fine, colored_main_groups_df, norm=False)
    for i in range(7):
        fig.add_vline(x=-3 + i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="no. partners",
        xaxis_range=[-1.6, 1.6], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_fine_tot_groups_pairs.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_by_sign(
    flow_type_ol: pd.DataFrame, conn_ol: pd.DataFrame,
    type_dir_ol: pd.DataFrame, flow_ol: pd.DataFrame,
    colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []

    bins = np.arange(-0.5, 6 + 0.5, 1)
    fig = _hist_lines(flow_type_ol, "hitting_time", "sign", bins, colored_sign_df)
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 5.1],
        yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_frac_signs_types.pdf"))

    diff_bins = np.array(range(-4, 4)).astype(float) + .5
    fig = _hist_sum_lines(conn_ol, "hitting_time_diff", "weight", "sign_pre",
                          diff_bins, colored_sign_df)
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="frac. weights",
        xaxis_range=[-3.2, 3.2], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
        yaxis_range=[-.01, 1.01],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_frac_signs_conn.pdf"))

    ff_bins = np.arange(-0.1, 1.2, 0.2)
    for direct in ["ff", "fb", "la"]:
        fig = _hist_lines(type_dir_ol, f"frac_{direct}", "sign_pre",
                          ff_bins, colored_sign_df, width=450)
        fig.update_layout(
            xaxis_title=f"frac_{direct}", xaxis_range=[-.05, 1.05],
            yaxis_title="frac. cell types", yaxis_range=[-.01, 1.01],
        )
        figs.append((fig, f"{dataset}_{direct}_frac_hist_all_signs.pdf"))

    fine_bins = np.arange(-0.25 / 2, 6 + 0.25 / 2, .5 / 2)
    fig = _hist_lines(flow_ol[flow_ol.sign != 0], "hitting_time_median", "sign",
                      fine_bins, colored_sign_df, norm=False, width=450)
    for i in range(6):
        fig.add_vline(x=i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="layer", xaxis_range=[-.1, 3.3], xaxis_tickvals=[0, 1, 2, 3, 4, 5],
        yaxis_title="no. neurons", yaxis_range=[-1.5e3, 19e3],
    )
    figs.append((fig, f"{dataset}_0.1thre_hit_hist_fine_tot_signs_types.pdf"))

    diff_fine = np.arange(-4 - 0.25 / 2, 4 + 0.25 / 2, .5 / 2)
    fig = _hist_sum_lines(conn_ol[conn_ol.sign_pre != 0], "hitting_time_diff", "weight",
                          "sign_pre", diff_fine, colored_sign_df, norm=False, width=450)
    for i in range(7):
        fig.add_vline(x=-3 + i, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="post - pre layer", yaxis_title="no. synapses",
        xaxis_range=[-1.6, 1.6], xaxis_tickvals=[-3, -2, -1, 0, 1, 2, 3],
    )
    figs.append((fig, f"{dataset}_0.1thre_diff_hit_hist_fine_tot_signs_conn.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_type_scatter_and_triangle(
    flow_type_ol: pd.DataFrame, type_dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []
    _, star_types = _plot_types()

    fig = _scatter(flow_type_ol, "count", "hitting_time", "main_groups", "instance",
                   colored_main_groups_df, star_types)
    fig.update_layout(xaxis_title="no. neurons", yaxis_title="layer")
    fig.update_xaxes(type="log")
    figs.append((fig, f"{dataset}_examples_0.1thre_no_vs_layer.pdf"))

    eps = 5e-2
    d = type_dir_ol.copy()
    d[["frac_la", "frac_fb", "frac_ff"]] = d[["frac_la", "frac_fb", "frac_ff"]] * (1 - eps) + eps / 3
    fig = _directedness_triangle(d, "main_groups_pre", "instance_pre",
                                 colored_main_groups_df["color"].to_dict(), star_types)
    figs.append((fig, f"{dataset}_examples_0.1thre_dir_triangle.pdf"))

    d["sign_pre"] = d["sign_pre"].astype(str)
    cmap_sign = colored_sign_df.copy()
    cmap_sign.index = cmap_sign.index.astype(str)
    fig = _directedness_triangle(d, "sign_pre", "instance_pre",
                                 cmap_sign["color"].to_dict(), star_types)
    figs.append((fig, f"{dataset}_examples_0.1thre_dir_triangle_sign.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_flow_type_boxes(
    flow_df: pd.DataFrame, dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []
    plot_types, _ = _plot_types()

    flow_sub = flow_df[flow_df["instance"].isin(plot_types)].copy()
    flow_sub["cell_type"] = [s[:-2] for s in flow_sub["instance"]]
    color_map = dict(zip(
        flow_sub["cell_type"],
        colored_main_groups_df.loc[flow_sub["main_groups"], "color"].values,
    ))

    grouped = flow_sub.groupby(["instance", "cell_type"])["hitting_time"].median().reset_index()
    sorted_types = grouped.loc[grouped.sort_values(["hitting_time", "instance"]).index, "cell_type"].values
    flow_sub["cell_type"] = pd.Categorical(flow_sub["cell_type"], categories=sorted_types, ordered=True)
    flow_sub = flow_sub.sort_values("cell_type")

    dir_sub = dir_ol[dir_ol["instance_pre"].isin(plot_types)].copy()
    dir_sub["cell_type_pre"] = pd.Categorical(dir_sub["cell_type_pre"], categories=sorted_types, ordered=True)
    dir_sub = dir_sub.sort_values("cell_type_pre")

    # Pad cells with no outgoing connections as zero rows so frac medians
    # cover the full roster per type (matches plot_flow_numerous_type_boxes).
    type_totals = (
        flow_sub.groupby("cell_type", observed=True).size()
        .reindex(sorted_types, fill_value=0)
    )
    type_present = (
        dir_sub.groupby("cell_type_pre", observed=True).size()
        .reindex(sorted_types, fill_value=0)
    )
    n_missing = (type_totals - type_present).clip(lower=0).astype(int)
    pad_rows = pd.DataFrame({
        "cell_type_pre": pd.Categorical(
            np.repeat(sorted_types, n_missing.values),
            categories=sorted_types, ordered=True,
        ),
        "frac_ff": 0.0, "frac_fb": 0.0, "frac_la": 0.0,
    })
    dir_sub = pd.concat([dir_sub, pad_rows], ignore_index=True).sort_values("cell_type_pre")

    fig = _box(flow_sub, "hitting_time", "cell_type", "cell_type", color_map, height=650)
    for i in range(4):
        fig.add_vline(x=i + 1, line_width=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="layer", yaxis_title="")
    fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4), selector=dict(type="box"))
    figs.append((fig, f"{dataset}_examples_0.1thre_hit_hist.pdf"))

    for direct in ["ff", "fb", "la"]:
        fig = _box(dir_sub, f"frac_{direct}", "cell_type_pre", "cell_type_pre",
                   color_map, width=250, height=650)
        fig.update_layout(xaxis_range=[-0.05, 1.05])
        fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4), selector=dict(type="box"))
        figs.append((fig, f"{dataset}_examples_0.1thre_{direct}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_numerous_type_boxes(
    flow_ol: pd.DataFrame, dir_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    groups: list[tuple[int, int]] = ((100, 2000), (28, 100)),
    height: int = 3000,
):
    """Box plots across **all** numerous cell types.
    For each `(n_min, n_max)` group: hitting-time box + frac_ff/fb/la
    boxes with a black marker per type indicating its dominant direction.
    Default groups: (100, 2000] and (28, 100] per neuron count.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []
    flow_sub_base = flow_ol[flow_ol["main_groups"] != "other"]

    for j, (n_min, n_max) in enumerate(groups):
        numerous = flow_sub_base[
            (flow_sub_base["count"] > n_min) & (flow_sub_base["count"] <= n_max)
        ]["instance"].unique()
        if len(numerous) == 0:
            continue
        flow_sub = flow_sub_base[flow_sub_base["instance"].isin(numerous)].copy()
        flow_sub["cell_type"] = [s[:-2] for s in flow_sub["instance"]]
        color_map = dict(zip(
            flow_sub["cell_type"],
            colored_main_groups_df.loc[flow_sub["main_groups"], "color"].values,
        ))
        grouped = flow_sub.groupby("cell_type")["hitting_time"].median().reset_index()
        sorted_types = grouped.loc[grouped.sort_values("hitting_time").index, "cell_type"].values
        flow_sub["cell_type"] = pd.Categorical(
            flow_sub["cell_type"], categories=sorted_types, ordered=True,
        )
        flow_sub = flow_sub.sort_values("cell_type")

        dir_sub = dir_ol[dir_ol["instance_pre"].isin(numerous)].copy()
        dir_sub["cell_type_pre"] = pd.Categorical(
            dir_sub["cell_type_pre"], categories=sorted_types, ordered=True,
        )
        dir_sub = dir_sub.sort_values("cell_type_pre")

        fig = _box(flow_sub, "hitting_time", "cell_type", "cell_type",
                   color_map, height=height)
        for i in range(4):
            fig.add_vline(x=i + 1, line_width=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="layer", xaxis_range=[-0.1, 3.3])
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(sorted_types))),
            ticktext=list(sorted_types)[::-1],
            range=[-0.5, len(sorted_types) - 0.5],
        )
        fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4),
                          selector=dict(type="box"))
        figs.append((fig, f"{dataset}_numerous{j + 1}_0.1thre_hit_hist.pdf"))

        # Cells of a type with no outgoing connections are absent from dir_sub;
        # pad them as zero rows so the median is over the full roster per type.
        type_totals = (
            flow_sub.groupby("cell_type", observed=True).size()
            .reindex(sorted_types, fill_value=0)
        )
        type_present = (
            dir_sub.groupby("cell_type_pre", observed=True).size()
            .reindex(sorted_types, fill_value=0)
        )
        n_missing = (type_totals - type_present).clip(lower=0).astype(int)
        pad_rows = pd.DataFrame({
            "cell_type_pre": pd.Categorical(
                np.repeat(sorted_types, n_missing.values),
                categories=sorted_types, ordered=True,
            ),
            "frac_ff": 0.0, "frac_fb": 0.0, "frac_la": 0.0,
        })
        dir_sub = pd.concat([dir_sub, pad_rows], ignore_index=True).sort_values("cell_type_pre")
        dir_medians = dir_sub.groupby("cell_type_pre", observed=True)[
            ["frac_ff", "frac_fb", "frac_la"]
        ].median()
        frac_max = dir_medians.idxmax(axis=1).values
        for direct in ["ff", "fb", "la"]:
            fig = _box(dir_sub, f"frac_{direct}", "cell_type_pre", "cell_type_pre",
                       color_map, width=350, height=height)
            idx_sub = np.where(frac_max == f"frac_{direct}")[0]
            yvals = pd.Categorical(
                dir_medians.index[idx_sub],
                categories=sorted_types, ordered=True,
            )
            fig.add_trace(go.Scatter(
                x=dir_medians.iloc[idx_sub][f"frac_{direct}"].values,
                y=yvals, mode="markers",
                marker=dict(color="black", size=10),
                showlegend=False,
            ))
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(sorted_types))),
                ticktext=list(sorted_types)[::-1],
                range=[-0.5, len(sorted_types) - 0.5],
            )
            fig.update_layout(xaxis_range=[-0.05, 1.05])
            fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4),
                              selector=dict(type="box"))
            figs.append((fig, f"{dataset}_numerous{j + 1}_0.1thre_{direct}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_connectivity_heatmap(
    conn_ol: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    figs = []

    bins = np.arange(-0.5 / 2, 6 + 0.5, 1 / 2)
    centers = (bins[1:] + bins[:-1]) / 2
    n_plot = int((bins <= 4).sum())
    tick_labels = ["%.1f" % i for i in centers[:n_plot][::1]]

    cb = conn_ol.copy()
    cb["bin_pre"] = pd.cut(cb["hitting_time_pre"], bins=bins, labels=centers).astype(float)
    cb["bin_post"] = pd.cut(cb["hitting_time_post"], bins=bins, labels=centers).astype(float)
    full = cb.groupby(["bin_pre", "bin_post"], observed=False)["weight"].sum().unstack().fillna(0)
    mat = full.reindex(index=centers, columns=centers, fill_value=0)

    fig = _heatmatrix(mat, tick_labels, cmap="Greys")
    fig.update_yaxes(autorange="reversed")
    figs.append((fig, f"{dataset}_0.1thre_conn_matrix.pdf"))

    for group in colored_main_groups_df.index:
        sub = cb[cb["main_groups_pre"] == group]
        m = sub.groupby(["bin_pre", "bin_post"], observed=False)["weight"].sum().unstack().fillna(0)
        m = m.reindex(index=centers, columns=centers, fill_value=0)
        color = colored_main_groups_df.loc[group, "color"]
        cs = [[0.0, "white"], [1.0, color]]
        fig = _heatmatrix(m, tick_labels, cmap=cs)
        fig.update_layout(title=group)
        fig.update_yaxes(autorange="reversed")
        figs.append((fig, f"{dataset}_0.1thre_conn_matrix_{group}.pdf"))

    for sign in colored_sign_df.index.astype("int"):
        sub = cb[cb["sign_pre"] == sign]
        m = sub.groupby(["bin_pre", "bin_post"], observed=False)["weight"].sum().unstack().fillna(0)
        m = m.reindex(index=centers, columns=centers, fill_value=0)
        color = colored_sign_df.loc[sign, "color"]
        cs = [[0.0, "white"], [1.0, color]]
        fig = _heatmatrix(m, tick_labels, cmap=cs)
        fig.update_layout(title=sign)
        fig.update_yaxes(autorange="reversed")
        figs.append((fig, f"{dataset}_0.1thre_conn_matrix_{sign}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_flow_threshold_choice(
    conn_ol: pd.DataFrame,
    *, dataset: str = DATASET, hit_diff_thre: float = 0.5, save_dir: Path | None = None,
):
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    thresholds = np.linspace(-1, 1, 50)
    total_w = conn_ol["weight"].sum()
    fracs = np.array([
        conn_ol[conn_ol["hitting_time_diff"] >= thr]["weight"].sum() for thr in thresholds
    ]) / total_w
    y0 = fracs[np.where(thresholds > 0)[0][0]]
    plot_df = pd.DataFrame({"thre": thresholds, "frac": fracs})
    fig = _point_line(plot_df, "thre", "frac", x_thre=0, y_thre=y0)
    fig.update_layout(
        xaxis_title="post - pre layer threshold",
        yaxis_title="frac. weights above layer threshold",
        xaxis_range=[-1.05, 1.05], yaxis_range=[-0.05, 1.05],
    )
    _save(fig, save_dir, f"{dataset}_threshold_fraction_connections.pdf")
    return fig


# %%
def plot_ol_layers_violin(
    layers: pd.DataFrame, rois: list[str], labels: list[str],
    *, dataset: str = DATASET, filename_stem: str,
    save_dir: Path | None = None, height: int = 800, width: int = 500,
):
    """Horizontal violin of `hitting_time` across the given OL ROIs.

    `rois` is the ordered list of ROI names to display (top → bottom); `labels`
    are the short display names (e.g. `M1..M10, AME`). `filename_stem` goes
    into the output filename as `{dataset}_{filename_stem}.png`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "info_flow"
    sub = layers[layers["roi"].isin(rois)].copy()
    sub["roi"] = pd.Categorical(sub["roi"], categories=rois, ordered=True)
    sub = sub.sort_values("roi")
    fig = _violin_median(
        sub, "hitting_time", "roi", labels=labels, height=height, width=width,
        xaxis_title="layer", xaxis_tickvals=[0, 1, 2, 3, 4],
        xaxis_range=[-0.2, 4.1],
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_dir / f"{dataset}_{filename_stem}.png", scale=2)
    return fig
