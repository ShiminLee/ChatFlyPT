"""Section 4 (LR clustering) plot_* functions: left/right clustering-sweep
and cluster-confusion summaries.
"""


from pathlib import Path


import numpy as np
import pandas as pd
import plotly.graph_objects as go


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _heatmatrix, _point_line, _save


# %%
def plot_lr_clustering_sweep(
    sweep_df: pd.DataFrame, n0: int, *,
    dataset: str = DATASET, save_dir: Path | None = None,
    diag_font_size: int | None = None,
):
    """Six line/scatter PDFs for the L/R clustering threshold sweep.

    `n0` is the reference cluster count (from the chosen FRAC in section 3);
    used to draw dashed reference lines on each panel. `diag_font_size`, if set,
    overrides the font of the `clustering_threshold_diag` panel only (SI 3g);
    the other panels (incl. SI 3d/3f) keep the default size.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "clustering"
    idx0 = int(np.where(sweep_df["n_clu_R"].values == n0)[0][0])
    figs = []

    # panel 1: n_R vs frac
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["frac"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(xaxis_title="No. R clusters", yaxis_title="Clustering threshold")
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0, 2])
    fig.update_yaxes(type="log", tickvals=[0.05, 0.1, 0.5, 1], range=[-1.7, 0.1])
    figs.append((fig, f"{dataset}_no_thre.pdf"))

    # panel 2: diag vs n_R
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["diag"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(
        xaxis_title="no. clusters",
        yaxis_title="frac. diagonal cluster-cluster connectivity",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(tickvals=[0, 0.5, 1], range=[0, 1])
    if diag_font_size is not None:
        fig.update_layout(font=dict(family="arial", size=diag_font_size))
    figs.append((fig, f"{dataset}_clustering_threshold_diag.pdf"))

    # panel 3: n_L vs n_R
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
    figs.append((fig, f"{dataset}_LR_no_clusters.pdf"))

    # panel 4: ARI_LR vs n_R
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_LR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0], width=550)
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_LR_ARI.pdf"))

    # panel 5: ARI_fromR vs n_R
    df = pd.DataFrame({"x": sweep_df["n_clu_R"].values, "y": sweep_df["ari_fromR"].values})
    fig = _point_line(df, "x", "y", x_thre=df["x"].iloc[idx0], y_thre=df["y"].iloc[idx0])
    fig.update_layout(
        xaxis_title="no. R clusters", yaxis_title="ARI between L and L from R clusters",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1, range=[0.2, 2])
    fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_LR_fromR_ARI.pdf"))

    # panel 6: ARI comparison (ari_fromR x, ari_LR y)
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
    figs.append((fig, f"{dataset}_LR_ARI_comparison.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_lr_cluster_confusion(
    match: dict, *,
    dataset: str = DATASET, save_dir: Path | None = None, anno_min: int = 10,
    filename_stem: str = "left_to_right_confusion",
    xaxis_title: str = "right cluster", yaxis_title: str = "left cluster",
):
    """Heatmap of the L↔R cluster confusion matrix (anno only for cells ≥ anno_min)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "clustering"
    conf = match["confusion"]
    tick_labels = match["tick_labels"]
    anno = np.where(conf >= anno_min, conf.astype(object), "")
    fig = _heatmatrix(conf, tick_labels, anno=anno)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf")
    return fig
