"""Section 7 (RF size experimental comparisons) plot_* functions: RF
size/aspect vs. published experimental measurements.
"""


from pathlib import Path


import numpy as np
import pandas as pd
import plotly.graph_objects as go


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _save, _scatter
from utils.plotting_rf import _RF_STAR_OUTPUT


# %%
def plot_rf_size_example_comparison(
    comparison_body: pd.DataFrame, *,
    star_instance: str, example_bid: int,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Scatter of direct-vs-propagated RF centre (x0, y0) and size per body for
    one star instance."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    figs = []
    sub = comparison_body[comparison_body["instance"] == star_instance].copy()
    if sub.empty:
        return figs
    sub["bodyId"] = sub["bodyId"].astype(str)
    # subtract 22 from y so y-range is centered around 0
    sub["y0_dir"] = sub["y0_dir"] - 22
    sub["y0_prop"] = sub["y0_prop"] - 22
    color_gray = pd.DataFrame(["lightgray"], columns=["color"], index=[star_instance])
    color_dark = pd.DataFrame(["darkgray"], columns=["color"], index=[star_instance])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-60, 60], y=[-60, 60], mode="lines",
                             line=dict(color="gray", dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig2 = _scatter(sub, "y0_dir", "y0_prop", "instance", "bodyId",
                    color_gray, t_stars=[str(example_bid)],
                    height=350, width=350, marker="cross")
    fig1 = _scatter(sub, "x0_dir", "x0_prop", "instance", "bodyId",
                    color_dark, t_stars=[str(example_bid)],
                    height=350, width=350)
    for tr in fig1.data:
        fig.add_trace(tr)
    for tr in fig2.data:
        fig.add_trace(tr)
    fig.update_layout(fig1.layout)
    fig.update_layout(
        xaxis_title="direct RF x0 / y0", yaxis_title="propagated RF x0 / y0",
        xaxis_range=[-30, 30], yaxis_range=[-30, 30], height=350, width=350,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    figs.append((fig, f"{dataset}_{star_instance}_rf_center_comp.pdf"))

    fig = _scatter(sub, "size_dir", "size_prop", "instance", "bodyId",
                   color_dark, t_stars=[str(example_bid)],
                   height=350, width=350)
    fig.add_trace(go.Scatter(x=[0, 20], y=[0, 20], mode="lines",
                             line=dict(color="gray", dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        xaxis_title="direct ARF size", yaxis_title="indirect ARF size",
        xaxis_range=[-1, 20], yaxis_range=[-1, 20],
    )
    figs.append((fig, f"{dataset}_{star_instance}_rf_size_comp.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_size_type_scatters(
    comparison_type: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, star_types: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Per-instance direct-vs-propagated scatters for each RF param."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    stars = list(star_types) if star_types is not None else [s + "_R" for s in _RF_STAR_OUTPUT]
    sub = comparison_type[comparison_type["main_groups"].isin(["VIN", "VPN"])].copy()
    figs = []
    for val in ["size", "n_col", "r2", "ecc", "phi"]:
        fig = _scatter(
            sub, f"{val}_dir", f"{val}_prop", "main_groups", "instance",
            colored_main_groups_df, stars,
        )
        fig.add_trace(go.Scatter(
            x=[-1000, 1000], y=[-1000, 1000], mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(
            yaxis_title=f"ARF {val}", xaxis_title=f"direct {val}",
            xaxis_title_standoff=0, yaxis_title_standoff=2,
        )
        if val == "size":
            fig.update_xaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(0.6), np.log10(350)])
            fig.update_yaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(0.6), np.log10(350)])
        elif val == "n_col":
            fig.update_xaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(2), np.log10(1200)])
            fig.update_yaxes(type="log", tickformat="d", dtick=1,
                             range=[np.log10(2), np.log10(1200)])
        elif val == "r2":
            fig.update_xaxes(range=[-1.1, 1.1])
            fig.update_yaxes(range=[-1.1, 1.1])
        elif val == "ecc":
            fig.update_xaxes(range=[-0.1, 1.1])
            fig.update_yaxes(range=[-0.1, 1.1])
        elif val == "phi":
            ticks = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
            labels = ["-90", "-45", "0", "45", "90"]
            fig.update_xaxes(title="direct ori (deg)", tickvals=ticks, ticktext=labels)
            fig.update_yaxes(title="ARF ori (deg)", tickvals=ticks, ticktext=labels)
        figs.append((fig, f"{dataset}_type_{val}_dir_vs_prop.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def _mixed_color_df(colored_main_groups_df, colored_region_df):
    return pd.concat(
        (colored_main_groups_df.iloc[:-1], colored_region_df.iloc[2:3]),
        axis=0,
    )


# %%
def _add_2x_guide_lines(fig, lo: float, hi: float, *, color: str = "gray", width: float = 2):
    """Add three dashed guide lines: y=x, y=2x, y=x/2 over the log-range [lo, hi]."""
    for xs, ys in (
        ([lo, hi], [lo, hi]),
        ([lo, hi / 2], [2 * lo, hi]),
        ([2 * lo, hi], [lo, hi / 2]),
    ):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=width, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))


# %%
def plot_rf_size_vs_experiment(
    comparison_type: pd.DataFrame, exp_sizes: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    col_to_deg_size: float = 25.0, col_to_deg_axis: float = 5.0,
):
    """Experimental vs predicted (propagated + direct) RF size scatter, plus
    sx/sy variants.

    `col_to_deg_size`: degrees² per column (default 25).
    `col_to_deg_axis`: degrees per column-radius (default 5).
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    mixed = _mixed_color_df(colored_main_groups_df, colored_region_df)
    instances_comp = [s + "_R" for s in exp_sizes["cell_type"].values]

    prop = comparison_type[[
        "instance", "cell_type_prop", "main_groups", "sign",
        "size_prop", "sx_prop", "sy_prop",
        "size_dir", "sx_dir", "sy_dir",
    ]].copy().rename(columns={"cell_type_prop": "cell_type"})
    sub = prop[prop["instance"].isin(instances_comp)].copy()
    sub = sub.merge(
        exp_sizes[["cell_type", "sx", "sy", "size"]].rename(
            columns={"size": "size_exp_deg", "sx": "sx_exp_deg", "sy": "sy_exp_deg"},
        ),
        on="cell_type", how="left",
    )
    sub["size_pred_deg"] = sub["size_prop"] * col_to_deg_size
    sub["sx_pred_deg"] = sub["sx_prop"] * col_to_deg_axis
    sub["sy_pred_deg"] = sub["sy_prop"] * col_to_deg_axis
    sub["size_dir_deg"] = sub["size_dir"] * col_to_deg_size
    sub["sx_dir_deg"] = sub["sx_dir"] * col_to_deg_axis
    sub["sy_dir_deg"] = sub["sy_dir"] * col_to_deg_axis

    from scipy.stats import pearsonr
    for val in ("size_pred_deg", "size_dir_deg"):
        nn = sub[["size_exp_deg", val]].dropna()
        if len(nn):
            corr, pval = pearsonr(nn["size_exp_deg"].values, nn[val].values)
            print(f"Pearson corr exp vs {val}: {corr:.3f} (p={pval:.2e}, n={len(nn)})")

    figs = []
    panels = [
        ("size_exp_deg", "size_pred_deg", "size_dir_deg", "RF size (deg²)",
         "exp_size_comp", (np.log10(5), np.log10(1000))),
        ("sx_exp_deg", "sx_pred_deg", "sx_dir_deg", "sx (deg)",
         "exp_sx_comp", (np.log10(0.7), np.log10(20))),
        ("sy_exp_deg", "sy_pred_deg", "sy_dir_deg", "sy (deg)",
         "exp_sy_comp", (np.log10(0.7), np.log10(20))),
    ]
    for x_col, y_prop, y_dir, label, stem, rng in panels:
        plot_df = sub.dropna(subset=[x_col, y_prop]).copy()
        if plot_df.empty:
            continue
        all_stars = list(plot_df["instance"].values)
        fig = _scatter(plot_df, x_col, y_prop, "main_groups", "instance",
                       mixed, all_stars, height=600, width=600, size=12)
        _add_2x_guide_lines(fig, 10 ** rng[0], 10 ** rng[1])
        for _, r in plot_df.iterrows():
            if pd.notna(r.get(y_dir)):
                fig.add_trace(go.Scatter(
                    x=[r[x_col], r[x_col]], y=[r[y_prop], r[y_dir]],
                    mode="lines", line=dict(color="black", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
        dir_df = plot_df.dropna(subset=[y_dir])
        fig2 = _scatter(dir_df, x_col, y_dir, "main_groups", "instance",
                        mixed, list(dir_df["instance"].values),
                        height=600, width=600, marker="square", size=12,
                        show_labels=False)
        for tr in fig2.data:
            fig.add_trace(tr)
        fig.update_layout(
            xaxis_title=f"measured {label}", yaxis_title=f"predicted {label}",
            xaxis_title_standoff=0, yaxis_title_standoff=2,
        )
        fig.update_xaxes(type="log", tickformat="d", dtick=1, range=list(rng))
        fig.update_yaxes(type="log", tickformat="d", dtick=1, range=list(rng))
        figs.append((fig, f"{dataset}_{stem}.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_rf_aspect_vs_experiment(
    comparison_type: pd.DataFrame, exp_sizes: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    dev: float = 0.2,
):
    """Measured vs predicted RF aspect ratio (sy/sx), analogous to
    `plot_rf_size_vs_experiment` (Main Fig. 3e) but for the same cell types'
    elongation rather than area. Prediction is the propagated ARF only.

    The sy/sx ratio is scale-free, so the column->degree axis conversion
    cancels and is irrelevant. Cell types whose measured *or* predicted ratio
    deviates from 1.0 by more than `dev` are highlighted (solid marker) and
    labelled; the rest stay faint and unlabelled. Hovering shows the cell type.

    Writes an HTML sidecar to `save_dir` (default the exp_comparison topic
    folder, so `paper_figs.distribute()` can rename it to the panel file) and
    returns the figure so a caller (e.g. a notebook) can also export a PDF.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    mixed = _mixed_color_df(colored_main_groups_df, colored_region_df)
    instances_comp = [s + "_R" for s in exp_sizes["cell_type"].values]

    prop = comparison_type[[
        "instance", "cell_type_prop", "main_groups", "sx_prop", "sy_prop",
    ]].copy().rename(columns={"cell_type_prop": "cell_type"})
    sub = prop[prop["instance"].isin(instances_comp)].copy()
    sub = sub.merge(
        exp_sizes[["cell_type", "sx", "sy"]].rename(
            columns={"sx": "sx_exp", "sy": "sy_exp"},
        ),
        on="cell_type", how="left",
    )
    sub["ratio_exp"] = sub["sy_exp"] / sub["sx_exp"]
    sub["ratio_pred"] = sub["sy_prop"] / sub["sx_prop"]
    plot_df = sub.dropna(subset=["ratio_exp", "ratio_pred"]).copy()
    plot_df["cell_type_label"] = plot_df["instance"].str.replace(
        "_R", "", regex=False
    )

    deviates = (
        (plot_df["ratio_exp"] - 1.0).abs() > dev
    ) | (
        (plot_df["ratio_pred"] - 1.0).abs() > dev
    )
    labelled = list(plot_df.loc[deviates, "cell_type_label"].values)

    rng = (np.log10(0.4), np.log10(2.0))
    fig = _scatter(plot_df, "ratio_exp", "ratio_pred", "main_groups",
                   "cell_type_label", mixed, t_stars=labelled,
                   height=600, width=600, size=12)
    fig.add_trace(go.Scatter(
        x=[10 ** rng[0], 10 ** rng[1]], y=[10 ** rng[0], 10 ** rng[1]],
        mode="lines", line=dict(color="gray", width=2, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    for sel in (dict(mode="markers"), dict(mode="markers+text")):
        fig.update_traces(
            hovertemplate="%{text}<extra></extra>", selector=sel,
        )
    fig.update_layout(
        xaxis_title="measured sy/sx", yaxis_title="predicted sy/sx",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
        font=dict(family="arial", size=27),  # 50% larger than the _scatter default (18)
    )
    ticks = [0.5, 0.7, 1.0, 1.5, 2.0]
    tick_text = ["0.5", "0.7", "1", "1.5", "2"]
    fig.update_xaxes(type="log", tickvals=ticks, ticktext=tick_text, range=list(rng))
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=tick_text, range=list(rng))

    stem = f"{dataset}_exp_aspect_comp"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_dir / f"{stem}.html")
    return fig


# %%
def plot_rf_size_vs_experiment_cc(
    type_ol: pd.DataFrame, cc_sizes: pd.DataFrame,
    colored_main_groups_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    col_to_deg_size: float = 25.0,
):
    """CurrierClandinin experimental (ellipse-area) vs propagated RF size.
    The CC sheet carries only `size_ellipse` (no separate `size_center`
    measure), so a single ellipse-area series is plotted."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    mixed = _mixed_color_df(colored_main_groups_df, colored_region_df)

    prop = type_ol[["instance", "cell_type", "main_groups", "sign", "size"]].copy()
    prop = pd.concat([
        prop,
        pd.DataFrame({
            "instance": ["L1_R", "L2_R", "L3_R"],
            "cell_type": ["L1", "L2", "L3"],
            "size": [1.0, 1.0, 1.0],
            "main_groups": ["visual input"] * 3,
            "sign": ["-1", "1", "1"],
        }),
    ], axis=0, ignore_index=True)
    prop["size_pred_deg"] = prop["size"] * col_to_deg_size

    sub = cc_sizes[["cell_type", "size_ellipse"]].merge(
        prop, on="cell_type", how="left",
    )

    ellipse_df = sub.dropna(subset=["size_ellipse", "size_pred_deg"])
    fig = _scatter(ellipse_df, "size_ellipse", "size_pred_deg",
                   "main_groups", "instance", mixed,
                   list(ellipse_df["instance"].values),
                   height=600, width=600, marker="square", size=12)
    _add_2x_guide_lines(fig, 5, 1000)
    fig.update_layout(
        xaxis_title="measured RF size (deg²) [CC]",
        yaxis_title="predicted ARF size (deg²)",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickvals=[10, 20, 50, 100, 200, 500],
                     range=[np.log10(15), np.log10(500)])
    fig.update_yaxes(type="log", tickvals=[10, 20, 50, 100, 200, 500],
                     range=[np.log10(15), np.log10(500)])
    _save(fig, save_dir, f"{dataset}_exp_size_comp_cc.pdf", html=True)
    return [(fig, f"{dataset}_exp_size_comp_cc.pdf")]
