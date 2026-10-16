"""Section 6 (Propagated RFs) plot_* functions: RF ellipse-fit examples,
estimated-overlay RFs, RF positions, input-RF examples, and RF summaries
by cell type / main group / cluster.
"""


import re
from pathlib import Path


import numpy as np
import pandas as pd
import plotly.graph_objects as go


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _add_gaussian_ellipses, _box, _box_scatter, _hist_cumsum_lines, _hist_lines, _hist_lines_v, _hist_sum_lines, _save, _scatter
from utils.plotting_clusters import _CLUSTER_TYPES, _OUT_TYPES


# %%
_RF_EXAMPLE_BID = 30134
_RF_STAR_EXAMPLES = [
    "LPT111", "VS", "aMe12",
    "LC4", "LC6", "LPLC2", "T4a", "Tm5a",
]
_RF_STAR_OUTPUT = _RF_STAR_EXAMPLES
_RF_STAR_INTERNAL = _RF_STAR_EXAMPLES


# %%
def plot_rf_example(
    ol_meta: pd.DataFrame, stepsn, params_all_df: pd.DataFrame,
    example_bid: int, in_instances: list[str],
    *, dataset: str = DATASET, save_dir: Path | None = None,
    per_input_threshold: float = 0.2,
):
    """Three figures for one example VPN neuron:
    (1) original effective-weight hex RF,
    (2) fitted 2D-Gaussian RF + overlay,
    (3) per-visual-input hex RFs for inputs above `per_input_threshold * tot_max`.
    """
    from connectome_interpreter.external_map import hex_heatmap
    from utils.external_rf import twoD_Gaussian
    from utils.ol_rf import compute_rf

    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    figs = []
    inidx = ol_meta[ol_meta.instance.isin(in_instances)].idx.values
    outidx = ol_meta[ol_meta.bodyId == example_bid].idx.values

    df = compute_rf(ol_meta, stepsn, inidx, outidx)
    if df.empty:
        return figs
    tot_max = float(np.abs(df["effective weight"]).max())
    orig = df.set_index("coords")[["effective weight"]]
    fig = hex_heatmap(
        orig, custom_colorscale="RdBu_r",
        global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
    )
    figs.append((fig, f"{dataset}_example_{example_bid}_rf_orig.pdf"))

    row = params_all_df.loc[params_all_df["bodyId"] == example_bid]
    if not row.empty:
        row = row.iloc[0]
        fit = twoD_Gaussian(df["x"].values, df["y"].values,
                            row["x0"], row["y0"], row["a"], row["b"], row["phi"])
        fit_df = pd.DataFrame(fit * tot_max, index=df["coords"], columns=["effective weight"])
        fit_df = fit_df[~fit_df.index.duplicated()]
        fig = hex_heatmap(
            fit_df, custom_colorscale="RdBu_r",
            global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
        )
        _add_gaussian_ellipses(
            fig, pd.DataFrame([row]), example_bid=example_bid, y_scale=np.sqrt(3),
        )
        figs.append((fig, f"{dataset}_example_{example_bid}_rf_fit.pdf"))

    for inst in in_instances:
        ii = ol_meta[ol_meta.instance == inst].idx.values
        d_i = compute_rf(ol_meta, stepsn, ii, outidx)
        if d_i.empty or np.abs(d_i["effective weight"]).max() < per_input_threshold * tot_max:
            continue
        plot_df = d_i.set_index("coords")[["effective weight"]]
        fig = hex_heatmap(
            plot_df, custom_colorscale="RdBu_r",
            global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
        )
        figs.append((fig, f"{dataset}_example_{example_bid}_rf_{inst[:-2]}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_rf_estimated_overlay(
    ol_meta: pd.DataFrame, stepsn, params_all_df: pd.DataFrame,
    example_bid: int, in_instances: list[str],
    *, dataset: str = DATASET, save_dir: Path | None = None,
    thr_mode: str = "cumsum", remove_frac: float = 0.6, min_remove_frac: float = 0.1,
    filename_stem: str | None = None,
):
    """Estimated (propagated) ARF for one body with the Gaussian ellipse AND the
    alpha-hull overlaid -- the gallery-style RF panel (cf.
    `gallery_vpn.make_rf_fig`), as opposed to `plot_rf_example`'s fitted-Gaussian
    reconstruction.

    Renders the raw effective-weight hex heatmap (signed `RdBu_r`, symmetric
    about 0), overlays the fitted 2D-Gaussian ellipse (red, from `params_all_df`)
    and the cumsum alpha-hull polygon (blue) with its centre-of-mass cross. The
    hull uses the same `thr_mode`/`remove_frac`/`min_remove_frac` as
    `ol_rf._compute_rf_params`, so the drawn hull matches the reported
    `ahull_size`. Saves `<dataset>_<filename_stem>.pdf` (+ `.html` sidecar) and
    returns the figure, or None if the RF is empty. If the hull is not a single
    polygon the ellipse panel is still saved without the hull (warning printed).
    """
    from connectome_interpreter.external_map import hex_heatmap
    from utils.ol_rf import compute_rf, count_col_ahull

    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    if filename_stem is None:
        filename_stem = f"example_{example_bid}_rf_fit"

    inidx = ol_meta[ol_meta.instance.isin(in_instances)].idx.values
    outidx = ol_meta[ol_meta.bodyId == example_bid].idx.values
    df = compute_rf(ol_meta, stepsn, inidx, outidx)
    if df.empty:
        return None
    tot_max = float(np.abs(df["effective weight"]).max())
    orig = df.set_index("coords")[["effective weight"]]
    orig = orig[~orig.index.duplicated()]
    fig = hex_heatmap(
        orig, custom_colorscale="RdBu_r",
        global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
    )

    row = params_all_df.loc[params_all_df["bodyId"] == example_bid]
    if not row.empty:
        _add_gaussian_ellipses(fig, row, example_bid=example_bid, y_scale=np.sqrt(3))

    (edge_points, edges, _area, com, _kept, _frac, _alpha) = count_col_ahull(
        orig, thr_mode=thr_mode, remove_frac=remove_frac,
        min_remove_frac=min_remove_frac,
    )
    if edges is not None and edge_points is not None and len(edge_points) == 1:
        edge_pt = edge_points[0]
        for j in range(len(edge_pt)):
            p1, p2 = edge_pt[j], edge_pt[(j + 1) % len(edge_pt)]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="lines",
                line=dict(color="blue", width=4), showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[com[0]], y=[com[1]], mode="markers",
            marker=dict(color="blue", size=10, symbol="cross"),
            showlegend=False, hoverinfo="skip",
        ))
    else:
        print(f"plot_rf_estimated_overlay: multiple/degenerate alpha-hull for "
              f"bodyId {example_bid}; hull overlay skipped")

    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf", html=True)
    return fig


# %%
def plot_rf_positions_across_neurons(
    params_all_df: pd.DataFrame, instance: str,
    *, dataset: str = DATASET, example_bid: int | None = None,
    save_dir: Path | None = None,
):
    """Gaussian-ellipse field for every fit in one instance, overlaid on a blank
    hex grid. Also draws an R^2 hex heatmap for the same instance.
    """
    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    figs = []
    sub = params_all_df[params_all_df["instance"] == instance].copy()
    if sub.empty:
        return figs

    blank = pd.DataFrame([0], index=["0,22"], columns=["value"])
    fig = hex_heatmap(
        blank, custom_colorscale="Picnic",
        global_min=-1, global_max=1, dataset="mcns_right",
    )
    _add_gaussian_ellipses(fig, sub, example_bid=example_bid, y_scale=np.sqrt(3))
    fig.update_layout(font=dict(family="arial", size=22))
    figs.append((fig, f"{dataset}_{instance}_rf_pos_par.pdf"))

    r2 = pd.DataFrame({
        "x": sub["x0"].values, "y": sub["y0"].values * np.sqrt(3), "value": sub["r2"].values,
    })
    r2["coords"] = r2.apply(lambda r: f"{r['x']:.0f},{r['y']:.0f}", axis=1)
    cmax = float(min(np.quantile(np.abs(r2["value"]), 0.95), 1))
    fig = hex_heatmap(
        r2.set_index("coords")[["value"]],
        custom_colorscale="RdBu", global_min=-cmax, global_max=cmax,
        dataset="mcns_right",
    )
    fig.update_layout(title="R2", title_x=0.1, title_y=0.95)
    figs.append((fig, f"{dataset}_{instance}_rf_r2.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_input_rf_example(
    roi_counts: pd.DataFrame, input_raw: pd.DataFrame,
    example_bid: int, target_instance: str,
    *, roi: str = "LO(R)", dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Three direct-input RF figures for one example body:
    (1) per-column post-synapse hex RF, max-normalised,
    (2) fitted 2D-Gaussian hex RF + ellipse overlay,
    (3) ellipse field over all `target_instance` fits in `roi`.

    `roi_counts` must have ROIs of the form `<ROI>_col_<h1>_<h2>` (e.g.
    `LO_R_col_18_19`) providing per-column post counts. `input_raw` is per-body
    direct-input fits from `get_input_rf_raw_ol`.
    """
    from connectome_interpreter.external_map import hex_heatmap, load_dataset
    from utils.external_rf import twoD_Gaussian

    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    figs = []

    roi_prefix = roi.replace("(", "_").replace(")", "")
    col_re = re.compile(rf"^{re.escape(roi_prefix)}_col_(-?\d+)_(-?\d+)$")

    hex_df = load_dataset("Nern2024")[["x", "y"]].drop_duplicates().reset_index(drop=True)

    body_rows = roi_counts[roi_counts["bodyId"] == example_bid]
    parsed = [
        (int(m.group(1)), int(m.group(2)), p)
        for r, p in zip(body_rows["roi"], body_rows["post"])
        if (m := col_re.match(str(r)))
    ]
    col_df = pd.DataFrame(parsed, columns=["hex1_id", "hex2_id", "post"])
    col_df["x"] = col_df["hex2_id"] - col_df["hex1_id"]
    col_df["y"] = col_df["hex2_id"] + col_df["hex1_id"]
    df = col_df.merge(hex_df, how="right", on=["x", "y"]).fillna({"post": 0})
    df["coords"] = df.apply(lambda r: f"{int(r['x'])},{int(r['y'])}", axis=1)
    amax = df["post"].max()
    df["effective weight"] = df["post"] / amax if amax > 0 else 0.0

    orig = df.set_index("coords")[["effective weight"]]
    orig = orig[~orig.index.duplicated()]
    fig = hex_heatmap(
        orig, custom_colorscale="RdBu_r",
        global_min=-1, global_max=1, dataset="mcns_right",
    )
    figs.append((fig, f"{dataset}_{target_instance}_input_rf_orig.pdf"))

    row = input_raw[(input_raw["bodyId"] == example_bid) & (input_raw["roi"] == roi)]
    if not row.empty:
        row = row.iloc[0]
        fit = twoD_Gaussian(
            df["x"].values.astype(float), (df["y"].values / np.sqrt(3)).astype(float),
            row["x0"], row["y0"], row["a"], row["b"], row["phi"],
        )
        fit_df = pd.DataFrame({"effective weight": fit}, index=df["coords"])
        fit_df = fit_df[~fit_df.index.duplicated()]
        fig = hex_heatmap(
            fit_df, custom_colorscale="RdBu_r",
            global_min=-1, global_max=1, dataset="mcns_right",
        )
        _add_gaussian_ellipses(
            fig, pd.DataFrame([row]), example_bid=example_bid, y_scale=np.sqrt(3),
        )
        figs.append((fig, f"{dataset}_{target_instance}_input_rf_fit.pdf"))

    sub = input_raw[(input_raw["instance"] == target_instance) & (input_raw["roi"] == roi)]
    if not sub.empty:
        blank = pd.DataFrame([0], index=["-16,38"], columns=["value"])
        fig = hex_heatmap(
            blank, custom_colorscale="Picnic",
            global_min=-1, global_max=1, dataset="mcns_right",
        )
        _add_gaussian_ellipses(fig, sub, example_bid=example_bid, y_scale=np.sqrt(3))
        figs.append((fig, f"{dataset}_{target_instance}_input_rf_pos_par.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def _rf_axis_update(fig, col, *, axis="x"):
    """Apply the RF-param-specific axis formatting used across every
    scatter/box/histogram: log scales for amp/size/n_col, clipped ranges for
    r2/ecc, degree ticks for phi."""
    upd = fig.update_xaxes if axis == "x" else fig.update_yaxes
    if col == "amp":
        upd(title=f"ARF {col}", type="log", range=[np.log10(1e-5), np.log10(1)], tickangle=0)
    elif col == "size":
        upd(title=f"ARF {col}", type="log", tickformat="d", dtick=1,
            range=[np.log10(0.6), np.log10(350)])
    elif col == "n_col":
        upd(title=f"ARF {col}", type="log", tickformat="d", dtick=1,
            range=[np.log10(2), np.log10(1200)])
    elif col == "r2":
        upd(title=f"ARF {col}", range=[-1.1, 1.1], tickangle=0)
    elif col == "ecc":
        upd(title=f"ARF {col}", range=[-0.1, 1.1])
    elif col == "phi":
        upd(title="ARF ori (deg)",
            tickvals=[-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
            ticktext=["-90", "-45", "0", "45", "90"],
            range=[-np.pi / 2 * 1.1, np.pi / 2 * 1.1])


# %%
def plot_rf_boxes_by_celltype(
    params_all_df: pd.DataFrame, plot_cell_types: list[str],
    colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-cell-type box plots of r2/amp/size/ecc/n_col/phi, sorted by median
    hitting_time, colored by main_groups."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    sub = params_all_df[params_all_df["instance"].isin(plot_cell_types)].copy()
    if sub.empty:
        return []
    color_map = dict(zip(
        sub["cell_type"],
        colored_main_groups_df.loc[sub["main_groups"], "color"].values,
    ))
    grouped = sub.groupby(["instance", "cell_type"])["hitting_time"].median().reset_index()
    sorted_types = grouped.loc[grouped.sort_values(["hitting_time", "instance"]).index, "cell_type"].values
    sub["cell_type"] = pd.Categorical(sub["cell_type"], categories=sorted_types, ordered=True)
    sub = sub.sort_values("cell_type")

    figs = []
    for val in ["r2", "amp", "size", "ecc", "n_col", "phi"]:
        fig = _box(sub, val, "cell_type", "cell_type", color_map, height=650)
        _rf_axis_update(fig, val, axis="x")
        fig.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4),
                          selector=dict(type="box"))
        figs.append((fig, f"{dataset}_examples_rf_{val}.pdf"))
    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_rf_scatters_by_main_groups(
    params_type_df: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *,
    star_types: list[str] | None = None,
    star_types_output: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-instance scatters colored by main_groups (and by sign): n_neurons vs
    each RF param, plus eight X-Y param pairs (amp/n_col/hitting_time/frac_ff
    vs size/r2/phi/amp). Returns list[(fig, name)]."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    stars_mg = list(star_types) if star_types is not None else [s + "_R" for s in _RF_STAR_INTERNAL]
    stars_sign = list(star_types_output) if star_types_output is not None else [s + "_R" for s in _RF_STAR_OUTPUT]

    sub = params_type_df[
        params_type_df["main_groups"].isin(["VIN", "VPN", "VCN"])
    ].copy()
    figs = []

    for val in ["r2", "amp", "size", "ecc", "n_col", "phi"]:
        fig = _scatter(
            sub, "n_neurons", val, "main_groups", "instance",
            colored_main_groups_df, stars_mg,
        )
        fig.update_layout(yaxis_title=f"ARF {val}", xaxis_title="no. neurons")
        fig.update_xaxes(type="log", tickformat="d", dtick=1,
                         range=[np.log10(0.6), np.log10(1500)])
        _rf_axis_update(fig, val, axis="y")
        figs.append((fig, f"{dataset}_examples_no_vs_{val}.pdf"))

    pairs = [("amp", "size"), ("n_col", "size"), ("amp", "r2"), ("ecc", "phi"),
             ("hitting_time", "size"), ("frac_ff", "size"),
             ("hitting_time", "amp"), ("frac_ff", "amp"),
             ("r2", "size"), ("phi", "size"), ("VIC", "size")]
    for xv, yv in pairs:
        # r2_vs_size is reproduced at the larger exp-comparison panel size
        # (600x600; font stays 18 as elsewhere) to match exp_*_comp.pdf.
        big = (xv, yv) == ("r2", "size")
        fig = _scatter(
            sub, xv, yv, "main_groups", "instance",
            colored_main_groups_df, stars_mg,
            height=600 if big else 400, width=600 if big else 400,
        )
        fig.update_layout(yaxis_title=f"ARF {yv}", xaxis_title=f"ARF {xv}")
        if xv == "hitting_time":
            fig.update_xaxes(title="layer", range=[0.8, 4.2])
        elif xv == "frac_ff":
            fig.update_xaxes(title="ff fraction", range=[-0.1, 1.1])
        elif xv == "VIC":
            fig.update_xaxes(
                title="VIC", type="log", tickvals=[0.01, 0.1],
                range=[np.log10(0.002), np.log10(0.6)],
            )
        else:
            _rf_axis_update(fig, xv, axis="x")
        _rf_axis_update(fig, yv, axis="y")
        fig.update_xaxes(tickangle=0)
        figs.append((fig, f"{dataset}_examples_{xv}_vs_{yv}.pdf"))

    sign_sub = sub[sub.sign != 0].copy()
    sign_sub["sign"] = sign_sub["sign"].astype(int)
    cmap_sign = colored_sign_df.copy()
    cmap_sign.index = cmap_sign.index.astype(int)
    for xv, yv in [("amp", "size"), ("n_col", "size"), ("amp", "r2"), ("ecc", "phi"),
                   ("hitting_time", "size"), ("frac_ff", "size"),
                   ("hitting_time", "amp"), ("frac_ff", "amp")]:
        fig = _scatter(
            sign_sub, xv, yv, "sign", "instance", cmap_sign, stars_sign,
        )
        fig.update_layout(yaxis_title=f"ARF {yv}", xaxis_title=f"ARF {xv}")
        if xv == "hitting_time":
            fig.update_xaxes(title="layer", range=[0.8, 4.2])
        elif xv == "frac_ff":
            fig.update_xaxes(title="ff fraction", range=[-0.1, 1.1])
        else:
            _rf_axis_update(fig, xv, axis="x")
        _rf_axis_update(fig, yv, axis="y")
        fig.update_xaxes(tickangle=0)
        figs.append((fig, f"{dataset}_examples_{xv}_vs_{yv}_sign.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_scatters_by_cluster(
    params_type_df: pd.DataFrame, in_out_instance_df: pd.DataFrame,
    *, cluster_names: list[str],
    colored_main_groups_df: pd.DataFrame,
    colored_cluster_df: pd.DataFrame | None = None,
    star_types_output: list[str] | None = None,
    star_types_internal: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
    jitter: float = 0.2,
):
    """Per-instance RF params jittered against cluster index. Two panels:
    VPN types (by `idxmax` cluster) and VIN types (filtered to
    max-participation > 0.4)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    n_clu = len(cluster_names)
    # Display names add "o" prefix (e.g. "c1" -> "oc1") for this panel only
    display_names = ["o" + n for n in cluster_names]
    if colored_cluster_df is None:
        blues = ["#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
                 "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]
        colored_cluster_df = pd.DataFrame(blues[:n_clu], index=display_names, columns=["color"])
    else:
        colored_cluster_df = colored_cluster_df.copy()
        colored_cluster_df.index = display_names
    stars_out = list(star_types_output) if star_types_output is not None else list(
        set([s + "_R" for s in _OUT_TYPES]) | set([s + "_R" for s in _CLUSTER_TYPES])
    )
    stars_int = list(star_types_internal) if star_types_internal is not None else [s + "_R" for s in _RF_STAR_INTERNAL]

    figs = []
    rng = np.random.default_rng(0)

    # VPN: y = cluster
    out = (
        params_type_df[params_type_df["main_groups"] == "VPN"]
        .merge(in_out_instance_df[["instance", "idxmax"]], how="inner", on="instance")
    )
    out["cluster_name"] = "o" + out["idxmax"]
    out["cluster_idx"] = out["idxmax"].str.slice(1).astype(int)
    out = out.dropna(subset=["size"]).reset_index(drop=True)
    out["y"] = out["cluster_idx"].astype(float) + rng.uniform(-jitter, jitter, len(out))

    for val in ["r2", "size", "amp", "ecc", "phi"]:
        fig = _scatter(out, val, "y", "cluster_name", "instance",
                       colored_cluster_df, stars_out, height=400, width=550)
        fig.update_layout(yaxis_title="cluster", xaxis_title=f"ARF {val}")
        fig.update_yaxes(autorange="reversed",
                         tickvals=list(range(1, n_clu + 1)),
                         ticktext=display_names, tickangle=0)
        if val == "size":
            fig.update_layout(font=dict(family="arial", size=22), yaxis_title="class")
            fig.update_xaxes(type="log", tickformat="d",
                             tickvals=[5, 10, 50, 100], range=[np.log10(3), np.log10(250)])
        elif val == "amp":
            fig.update_xaxes(type="log", range=[np.log10(1e-4), np.log10(0.05)])
        elif val == "phi":
            fig.update_xaxes(title="ARF ori (deg)",
                             tickvals=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
                             ticktext=["-90", "-45", "0", "45", "90"])
        elif val == "r2":
            fig.update_xaxes(range=[-1.1, 1.1])
        else:
            fig.update_xaxes(range=[-0.1, 1.1])
        figs.append((fig, f"{dataset}_examples_pathway_vs_{val}_output.pdf"))

    # VIN: x = cluster
    inside = (
        params_type_df[params_type_df["main_groups"] == "VIN"]
        .merge(in_out_instance_df, how="left", on="instance")
    )
    inside = inside[inside["max"] > 0.4].dropna(subset=["size"]).reset_index(drop=True)
    inside["cluster_idx"] = inside["idxmax"].str.slice(1).astype(int)
    inside["x"] = inside["cluster_idx"].astype(float) + rng.uniform(-jitter, jitter, len(inside))

    for val in ["r2", "size", "amp", "ecc", "phi"]:
        fig = _scatter(inside, "x", val, "main_groups", "instance",
                       colored_main_groups_df, stars_int, height=400, width=550)
        fig.update_layout(xaxis_title="cluster", yaxis_title=f"ARF {val}")
        fig.update_xaxes(tickvals=list(range(1, n_clu + 1)),
                         ticktext=cluster_names, tickangle=0)
        if val == "size":
            fig.update_yaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(0.6), np.log10(350)])
        elif val == "amp":
            fig.update_yaxes(type="log", range=[np.log10(1e-4), np.log10(0.8)])
        elif val == "phi":
            fig.update_yaxes(title="ARF ori (deg)",
                             tickvals=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
                             ticktext=["-90", "-45", "0", "45", "90"])
        elif val == "r2":
            fig.update_yaxes(range=[-1.1, 1.1])
        else:
            fig.update_yaxes(range=[0, 1])
        figs.append((fig, f"{dataset}_examples_pathway_vs_{val}_internal.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_histograms(
    params_type_df: pd.DataFrame, in_out_instance_df: pd.DataFrame,
    *, cluster_names: list[str],
    colored_main_groups_df: pd.DataFrame,
    colored_sign_df: pd.DataFrame,
    colored_cluster_df: pd.DataFrame | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Size-histogram (by main_groups / sign) + size cumsum by cluster +
    amp/ecc/phi/r2 histograms (by main_groups / sign)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    n_clu = len(cluster_names)
    if colored_cluster_df is None:
        blues = ["#041f4a", "#08306b", "#08519c", "#2171b5", "#4292c6",
                 "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]
        colored_cluster_df = pd.DataFrame(blues[:n_clu], index=cluster_names, columns=["color"])
    figs = []

    size_bins = 2 ** np.linspace(np.log2(0.6), np.log2(350), 12)
    base = params_type_df[
        params_type_df["main_groups"].isin(["VIN", "VPN", "VCN"])
    ].copy()

    cmaps = {"main_groups": colored_main_groups_df, "sign": colored_sign_df}
    for grp, cmap in cmaps.items():
        df = base.copy()
        if grp == "sign":
            df = df[df.sign != 0]
            df["sign"] = df["sign"].astype(int)
            cmap = cmap.copy()
            cmap.index = cmap.index.astype(int)
        fig = _hist_lines_v(df, "size", grp, size_bins, cmap, height=400,
                            width=405 if grp == "sign" else 450)
        fig.update_layout(yaxis_title="RF size", xaxis_title="frac. cell types",
                          xaxis_range=[-.01, 0.45])
        fig.update_yaxes(type="log", tickformat="d", dtick=1,
                         range=[np.log10(0.6), np.log10(350)])
        figs.append((fig, f"{dataset}_hist_rf_size_{grp}_types.pdf"))

    # cumsum by cluster (VPN)
    out = (
        params_type_df[params_type_df["main_groups"] == "VPN"]
        .merge(in_out_instance_df[["instance", "idxmax"]], how="inner", on="instance")
    )
    out["cluster_name"] = out["idxmax"]
    cs_bins = 2 ** np.linspace(np.log2(2.5), np.log2(280), 12)
    fig = _hist_cumsum_lines(out, "size", "cluster_name", cs_bins,
                             colored_cluster_df, height=400, width=450)
    fig.update_layout(xaxis_title="ARF size", yaxis_title="cum. frac. cell types",
                      yaxis_range=[-.05, 1.05], yaxis_tickvals=[0, 0.5, 1],
                      xaxis_tickvals=[5, 10, 50, 100],
                      font=dict(family="arial", size=20))
    fig.update_xaxes(type="log", tickformat="d", dtick=1,
                     range=[np.log10(3), np.log10(250)])
    figs.append((fig, f"{dataset}_rf_size_cumsum_types.pdf"))

    # other histograms
    hist_bins = {
        "amp": 10 ** np.linspace(-5, 0, 12),
        "ecc": np.linspace(-0.05, 1.05, 12),
        "phi": np.pi / 180 * np.arange(-105, 120, 30),
        "r2": np.linspace(-1.05, 1.05, 12),
    }
    widths = {"main_groups": 450, "sign": 400}
    for grp, cmap in cmaps.items():
        for col in ["amp", "ecc", "phi", "r2"]:
            df = base.copy()
            if grp == "sign":
                df = df[df.sign != 0]
                df["sign"] = df["sign"].astype(int)
                cmap_use = cmap.copy()
                cmap_use.index = cmap_use.index.astype(int)
            else:
                cmap_use = cmap
            fig = _hist_lines(df, col, grp, hist_bins[col], cmap_use,
                              height=400, width=widths[grp])
            fig.update_layout(xaxis_title=f"RF {col}", yaxis_title="frac. cell types",
                              yaxis_range=[-.01, 0.45])
            if col == "amp":
                fig.update_xaxes(type="log", range=[np.log10(1e-5), np.log10(1)])
            elif col == "phi":
                fig.update_xaxes(title="RF ori (deg)",
                                 tickvals=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
                                 ticktext=["-90", "-45", "0", "45", "90"])
            figs.append((fig, f"{dataset}_hist_rf_{col}_{grp}_types.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
_SIZE_DIFF_BINS = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
_SIZE_DIFF_TICKS = (_SIZE_DIFF_BINS[1:] + _SIZE_DIFF_BINS[:-1]) / 2
_SIZE_DIFF_LABELS = [-1, 0] + [int(2 ** i) for i in range(0, 8)]


# %%
def _bin_size_diff(df):
    d = df.copy()
    d["val"] = np.nan
    d.loc[d["size_diff"] < -0.5, "val"] = -2
    d.loc[(d["size_diff"] >= -0.5) & (d["size_diff"] <= 0.5), "val"] = -1
    d.loc[(d["size_diff"] >= 0.5) & (d["size_diff"] <= 1), "val"] = 0
    pos = d["size_diff"] >= 1
    d.loc[pos, "val"] = np.log2(d.loc[pos, "size_diff"])
    return d


# %%
def plot_rf_sizediff_internal(
    conn_edges: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Histogram of `size_diff` weighted by weight for VIN →
    (VIN, VPN) edges. Needs only OL RFs (step 6a)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    d = _bin_size_diff(conn_edges[
        (conn_edges["main_groups_pre"] == "VIN")
        & conn_edges["main_groups_post"].isin(["VIN", "VPN"])
    ])
    fig = _hist_sum_lines(d, "val", "weight", "main_groups_post",
                          _SIZE_DIFF_BINS, colored_main_groups_df,
                          height=400, width=550)
    fig.update_layout(
        yaxis_title="frac. connections", xaxis_title="(post - pre)/pre ARF size",
        xaxis_tickvals=_SIZE_DIFF_TICKS, xaxis_tickangle=0,
        xaxis_ticktext=_SIZE_DIFF_LABELS, yaxis_range=[-0.01, 0.51],
        font=dict(family="arial", size=20),
    )
    _save(fig, save_dir, f"{dataset}_sizediff_internal_conn_post.pdf")
    return fig


# %%
def plot_rf_sizediff_output(
    conn_edges: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Histogram of `size_diff` weighted by weight for VPN →
    (right OL, CB) edges. Needs CB RFs (step 8) for the CB-post curve;
    otherwise only the right-OL line appears."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    d = _bin_size_diff(conn_edges[
        (conn_edges["main_groups_pre"] == "VPN")
        & conn_edges["region_post"].isin(["right OL", "CB"])
    ])
    if d.empty:
        return None
    fig = _hist_sum_lines(d, "val", "weight", "region_post",
                          _SIZE_DIFF_BINS, colored_region_df,
                          height=400, width=550)
    fig.update_layout(
        yaxis_title="frac. connections", xaxis_title="(post - pre)/pre ARF size",
        xaxis_tickvals=_SIZE_DIFF_TICKS, xaxis_tickangle=0,
        xaxis_ticktext=_SIZE_DIFF_LABELS, yaxis_range=[-0.01, 0.51],
        font=dict(family="arial", size=20),
    )
    _save(fig, save_dir, f"{dataset}_sizediff_output_conn_post.pdf")
    return fig


# %%
def plot_rf_connectivity_histograms(
    conn_edges: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, colored_region_df: pd.DataFrame | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Both size-diff histograms (internal + output) in one call. Pass
    `colored_region_df` to get the CB curve in the output panel."""
    figs = [plot_rf_sizediff_internal(conn_edges, colored_main_groups_df,
                                       dataset=dataset, save_dir=save_dir)]
    region_df = (
        colored_region_df if colored_region_df is not None
        else pd.DataFrame(
            [colored_main_groups_df.loc["VPN", "color"]],
            index=["right OL"], columns=["color"],
        )
    )
    out = plot_rf_sizediff_output(conn_edges, region_df,
                                   dataset=dataset, save_dir=save_dir)
    if out is not None:
        figs.append(out)
    return figs


# %%
def plot_rf_connectivity_boxes(
    conn_edges: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_sign_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
):
    """Size-diff binned (log2 after +2 shift) × layer-diff box plots, colored
    by main_groups_pre and by sign_pre. Only VIN + VPN pres
    with aggregated weight > 5."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "rf_fits"
    df = conn_edges.copy()
    col_bins = np.arange(0, 10, 1, dtype=float) - .5
    col_labels = (col_bins[1:] + col_bins[:-1]) / 2
    df["col_bin_diff"] = pd.cut(
        np.log2(df["size_diff"].values + 2), bins=col_bins, labels=col_labels,
        include_lowest=True,
    ).astype(float)
    df = df.groupby(
        ["instance_pre", "instance_post", "main_groups_pre", "sign_pre"], dropna=False,
    ).agg({"layer_diff": "median", "col_bin_diff": "median", "weight": "median"}).reset_index()
    df = df[df["main_groups_pre"].isin(["VIN", "VPN"]) & (df["weight"] > 5)]

    figs = []
    color_val = {"main_groups_pre": colored_main_groups_df, "sign_pre": colored_sign_df}
    for grp, cmap in color_val.items():
        sub = df.copy()
        if grp == "sign_pre":
            sub["sign_pre"] = sub["sign_pre"].astype(int)
            cmap = cmap.copy()
            cmap.index = cmap.index.astype(int)
        fig = _box_scatter(sub, "col_bin_diff", "layer_diff", grp, cmap,
                           height=400, width=500)
        fig.update_layout(
            xaxis_title="(post - pre)/pre RF size", yaxis_title="post - pre layer",
            xaxis_tickvals=col_labels,
            xaxis_ticktext=["%d" % (2 ** i - 2) for i in col_labels],
        )
        figs.append((fig, f"{dataset}_layerdiff_vs_sizediff_{grp}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs
