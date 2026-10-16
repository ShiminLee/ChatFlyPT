"""Section 8 (CB VIC) plot_* functions: OL-to-CB visual-input-contribution
(VIC) cumulative fraction, per-ROI, and per-layer summaries.
"""


from pathlib import Path


import numpy as np
import pandas as pd


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _hist_cumsum_lines_v, _hist_lines, _save, _scatter


# %%
def plot_vic_cumsum(
    vic_type: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Cumulative VIC histogram per region with the CB threshold reference."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    bins = 10 ** np.linspace(-6.5, -0.2, 31)
    fig = _hist_cumsum_lines_v(
        vic_type, "VIC", "region", bins, colored_region_df,
        height=400, width=300,
    )
    fig.add_hline(y=vic_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(type="log", range=[np.log10(2e-7), np.log10(0.8)])
    _save(fig, save_dir, f"{dataset}_vic_cumsum.pdf")
    return fig


# %%
def plot_cumfrac_by_roi(
    curves, *, ascending: bool, xlabel: str, xlim,
    save_dir: Path | None = None, filename: str | None = None,
    drop_zero: bool = False, figsize=(4, 4),
):
    """Cumulative-fraction-vs-value curves, one line per ROI (or ROI group).

    `curves` is a list of ``(label, color, values, total)``: `values` is the
    array of the plotted quantity for that ROI's synapses and `total` its
    normaliser (the ROI's full synapse count). Each curve is the cumulative
    fraction of synapses at or below (`ascending=True`, a CDF) or at or above
    (`ascending=False`, a survival curve) each observed value, ending in a black
    dot. `drop_zero` omits a zero-valued final threshold (used by the DOB
    panel). ROI selection / grouping is the caller's job; this only draws.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for label, color, values, total in curves:
        vals = np.asarray(values, dtype=float)
        vals = np.sort(vals[~np.isnan(vals)])
        if vals.size == 0 or total <= 0:
            continue
        thr = np.unique(vals)                       # ascending unique values
        if ascending:
            cum = np.searchsorted(vals, thr, side="right")
        else:
            thr = thr[::-1]                          # descending thresholds
            cum = vals.size - np.searchsorted(vals, thr, side="left")
        if drop_zero and thr[-1] == 0:
            thr, cum = thr[:-1], cum[:-1]
        y = cum / total
        ax.plot(thr, y, linewidth=2, color=color, label=label)
        ax.plot(thr[-1], y[-1], "o", color="black", markersize=2)

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Fraction", fontsize=12, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    if save_dir is not None and filename is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / filename, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return fig


# %%
def plot_vic_layer_histogram(
    vic_type: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Layer histogram of VCBN types (VIC > threshold) by region.
    Requires `vic_type` to have a `hitting_time` column."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    bins = np.arange(0.5 - 0.25 / 2, 7 + 0.25 / 2, 0.5 / 2)
    sub = vic_type[vic_type.VIC > vic_thre]
    fig = _hist_lines(
        sub, "hitting_time", "region", bins, colored_region_df,
        height=230, width=450,
    )
    fig.update_xaxes(range=[0.5, 6], tickvals=[1, 2, 3, 4, 5, 6])
    fig.update_yaxes(tickvals=[0, 0.3])
    fig.update_layout(xaxis_title="layer", yaxis_title="frac.")
    fig.update_layout(xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}_vic_layer_hist.pdf")
    return fig


# %%
def plot_vic_vs_layer_scatter(
    vic_type: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, vic_thre: float = 5e-4, dataset: str = DATASET,
    save_dir: Path | None = None, highlight: list[str] | None = None,
    show_highlight_labels: bool = True,
):
    """Scatter of VIC vs hitting_time (layer), one point per instance, colored
    by region. Optional `highlight` list for labeled callouts."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_pathways"
    highlight = list(highlight) if highlight is not None else []

    fig = _scatter(
        vic_type, "hitting_time", "VIC", "region", "instance",
        colored_region_df, highlight, show_labels=show_highlight_labels,
    )
    fig.add_hline(y=vic_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(type="log", range=[np.log10(2e-7), np.log10(0.8)])
    fig.update_xaxes(range=[0.5, 6], tickvals=[1, 2, 3, 4, 5, 6])
    fig.update_layout(xaxis_title="layer")
    fig.update_layout(xaxis_title_standoff=0, yaxis_title_standoff=2)
    _save(fig, save_dir, f"{dataset}_vic_vs_layer.pdf", html=True, rasterize=True)
    return fig
