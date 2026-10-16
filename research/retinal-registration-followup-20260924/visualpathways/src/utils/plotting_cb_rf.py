"""Section 10 (CB RFs) plot_* functions: CB RF r2/size/alpha-hull summaries,
propagated/combined ARF examples, OL-to-CB heatmaps, and ROI coverage.
"""


from pathlib import Path


import numpy as np
import pandas as pd
import plotly.graph_objects as go


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _AXIS, _hist_cumsum_lines_v, _hist_lines_v, _point_line, _save, _scatter
from utils.plotting_rf import _RF_STAR_EXAMPLES
from utils.plotting_cb_lr_clustering import _CB_CLUSTER_TYPES


# %%
def plot_cb_rf_r2_summary(
    type_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, r2_thre: float = 0.05, dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """r2 histogram (vertical), r2 cumulative, and r2-vs-hitting_time scatter
    across regions."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    figs = []
    bins = np.linspace(-1.05, 1.05, 31)

    fig = _hist_lines_v(type_df, "r2", "region", bins, colored_region_df,
                        height=400, width=300)
    fig.update_layout(
        yaxis_title="ARF r2", xaxis_title="frac. cell types",
        xaxis_range=[-.01, 0.5], xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_yaxes(range=[-1.1, 1.1])
    figs.append((fig, f"{dataset}_cb_r2_hist.pdf"))

    bins2 = np.linspace(-1.05, 1.05, 22)
    fig = _hist_cumsum_lines_v(type_df, "r2", "region", bins2,
                               colored_region_df, height=400, width=300)
    fig.add_hline(y=r2_thre, line_dash="dash", line_color="gray")
    fig.update_layout(
        yaxis_title="ARF r2", xaxis_title="cum. frac. cell types",
        xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_yaxes(range=[-1.1, 1.1])
    fig.update_xaxes(range=[-0.1, 1.1], tickvals=[0, 0.5, 1])
    figs.append((fig, f"{dataset}_cb_r2_cumsum.pdf"))

    fig = _scatter(type_df, "hitting_time", "r2", "region", "instance",
                   colored_region_df, [], height=600, width=600)
    fig.add_hline(y=r2_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(range=[-1.1, 1.1])
    fig.update_xaxes(range=[0.8, 5.2])
    fig.update_layout(xaxis_title="layer", yaxis_title="ARF r2",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_layer_vs_r2.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_cb_rf_size_summary(
    type_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, r2_thre: float = 0.05, vic_thre: float = 5e-4,
    star_types: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """VIC-vs-size and VIC-vs-amp scatters. Only
    includes rows with `r2 > r2_thre`.

    NOTE: this function used to also emit `cb_size_hist` / `cb_layer_vs_size`,
    but those are the 2D-Gaussian analogues of the SI 13a alpha-hull panels
    produced by `plot_cb_ahull_size_summary` and shared the same output
    filenames. They were dropped here to avoid clobbering the (manifest-owned)
    alpha-hull SI 13a outputs — see the name-clash note in that function."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    stars = [s + "_R" for s in _RF_STAR_EXAMPLES] if star_types is None else list(star_types)
    figs = []
    sub = type_df[type_df["r2"] > r2_thre].copy()

    fig = _scatter(sub, "VIC", "size", "region", "instance",
                   colored_region_df, stars, height=600, width=600)
    fig.add_vline(x=vic_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=[5, 10, 50, 100], range=[np.log10(3), np.log10(350)])
    fig.update_xaxes(type="log", tickvals=[0.001, 0.01, 0.1],
                     range=[np.log10(4e-4), np.log10(0.6)])
    fig.update_layout(yaxis_title="ARF size", xaxis_title="VIC",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_vic_vs_size.pdf"))

    fig = _scatter(sub, "VIC", "amp", "region", "instance",
                   colored_region_df, stars, height=600, width=600)
    fig.add_vline(x=vic_thre, line_dash="dash", line_color="gray")
    fig.update_yaxes(type="log", range=[np.log10(1e-6), np.log10(1e-1)])
    fig.update_xaxes(type="log", tickvals=[0.001, 0.01, 0.1],
                     range=[np.log10(4e-4), np.log10(0.6)])
    fig.update_layout(yaxis_title="ARF amp", xaxis_title="VIC",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_vic_vs_amp.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name, html=True)
    return figs


# %%
def plot_cb_ahull_size_summary(
    ahull_type_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, size_col: str = "ahull_size", size_lo: float = 5, size_hi: float = 1000,
    size_tickvals: tuple[float, ...] = (10, 100, 1000),
    star_types: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Alpha-hull ARF size histogram + size-vs-layer scatter, coloured by
    region (SI Fig. 10b).

    Alpha-hull counterpart of the 2D-Gaussian `plot_cb_rf_size_summary` size
    panels: there is no fit-quality (`r2`) gate because the alpha hull is
    defined for every neuron, and the (log) size axis is widened via
    `size_lo`/`size_hi`/`size_tickvals` to fit the larger hull values.
    `ahull_type_df` is per-instance with `instance`, `region`, `hitting_time`
    and a size column named `size_col` — e.g. `get_rf_ahull_type_cb()` (filter
    to `region == 'CB'` for VCBN-only). `star_types` defaults to the CB cluster
    example types (`_CB_CLUSTER_TYPES['r']`, the same set highlighted in Main
    Fig. 4c and SI 10c). Writes `{dataset}_cb_size_hist.pdf` and
    `{dataset}_cb_layer_vs_size.pdf`."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    stars = list(star_types) if star_types is not None else _CB_CLUSTER_TYPES.get("r", [])
    sub = ahull_type_df.dropna(subset=[size_col, "hitting_time"]).copy()
    figs = []

    bins = 2 ** np.linspace(np.log2(size_lo), np.log2(size_hi), 12)
    fig = _hist_lines_v(sub, size_col, "region", bins, colored_region_df,
                        height=400, width=300)
    fig.update_layout(
        yaxis_title="ARF size", xaxis_title="frac. cell types",
        xaxis_range=[-.01, 0.5], xaxis_title_standoff=0, yaxis_title_standoff=2,
    )
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=list(size_tickvals),
                     range=[np.log10(size_lo), np.log10(size_hi)])
    figs.append((fig, f"{dataset}_cb_size_hist.pdf"))

    fig = _scatter(sub, "hitting_time", size_col, "region", "instance",
                   colored_region_df, stars, height=400, width=400)
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=list(size_tickvals),
                     range=[np.log10(size_lo), np.log10(size_hi)])
    fig.update_xaxes(range=[0.8, 5.2])
    fig.update_layout(xaxis_title="layer", yaxis_title="ARF size",
                      xaxis_title_standoff=0, yaxis_title_standoff=2)
    figs.append((fig, f"{dataset}_cb_layer_vs_size.pdf"))

    # This function OWNS the `cb_size_hist` / `cb_layer_vs_size` filenames for
    # the (manifest) SI 13a panels; `plot_cb_rf_size_summary` no longer writes
    # them (name clash removed). Only the layer-vs-size scatter (SI 13a_1) gets
    # an interactive sidecar; the size histogram (SI 13a_2) stays PDF-only.
    for fig, name in figs:
        _save(fig, save_dir, name, html=name.endswith("cb_layer_vs_size.pdf"))
    return figs


# %%
def plot_cb_vcbn_ahull_size_comparison(
    cmp_df: pd.DataFrame, colored_region_df: pd.DataFrame,
    *, size_lo: float = 1, size_hi: float = 1000,
    star_types: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """VCBN alpha-hull arbor size vs propagated RF size (SI Fig. 10a) — the
    VCBN analogue of the OL `plot_ol_ahull_size_comparison` panel (SI Fig. 7g).

    `cmp_df` is per-instance with `instance`, `region`, `size` (propagated RF
    size, col) and `ahull_size` (alpha-hull arbor size, col). Writes two
    scatters: `{dataset}_cb_examples_size_vs_ahull.pdf` (ahull vs RF with a
    y=x diagonal) and `{dataset}_cb_examples_ahull_reldiff_vs_size.pdf`
    (relative size difference `(ahull - RF) / RF` vs RF size). Both scatters
    highlight `star_types` (default the CB cluster example types
    `_CB_CLUSTER_TYPES['r']`, the same set highlighted in SI 10b)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    df = cmp_df.dropna(subset=["size", "ahull_size"]).copy()
    stars = list(star_types) if star_types is not None else _CB_CLUSTER_TYPES.get("r", [])

    fig = _scatter(df, "ahull_size", "size", "region", "instance",
                   colored_region_df, stars, height=400, width=400)
    fig.add_trace(go.Scatter(
        x=[size_lo, size_hi], y=[size_lo, size_hi], mode="lines",
        line=dict(color="gray", width=2, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title="ahull size (col)", yaxis_title="propagated RF size (col)",
    )
    fig.update_xaxes(type="log", tickformat="d", dtick=1,
                     range=[np.log10(size_lo), np.log10(size_hi)])
    fig.update_yaxes(type="log", tickformat="d", dtick=1,
                     range=[np.log10(size_lo), np.log10(size_hi)])
    _save(fig, save_dir, f"{dataset}_cb_examples_size_vs_ahull.pdf", html=True)

    df["rel_diff"] = (df["ahull_size"] - df["size"]) / df["size"]
    fig_rd = _scatter(df, "size", "rel_diff", "region", "instance",
                      colored_region_df, stars, height=600, width=600)
    fig_rd.update_layout(
        xaxis_title="propagated RF size (col)",
        yaxis_title="(ahull size - prop RF size) / prop RF size",
    )
    fig_rd.update_xaxes(type="log", tickformat="d", dtick=1)
    _save(fig_rd, save_dir,
          f"{dataset}_cb_examples_ahull_reldiff_vs_size.pdf", html=True)
    return fig, fig_rd


# %%
def plot_ol_ahull_size_comparison(
    cmp_df: pd.DataFrame, colored_main_groups_df: pd.DataFrame,
    *, star_types: list[str] | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """OL alpha-hull ARF size vs propagated RF size (SI Fig. 7g) — the OL
    analogue of `plot_cb_vcbn_ahull_size_comparison`, computed over all
    right-OL instances instead of VCBNs and coloured by `main_groups`
    (VPN / VIN) rather than region.

    `cmp_df` is per-instance with `instance`, `main_groups`, `size`
    (propagated RF size, col) and `ahull_size` (alpha-hull ARF size, col).
    Writes `{dataset}_examples_ahull_reldiff_vs_size.pdf` — the relative size
    difference `(ahull - RF) / RF` vs RF size — into the exp_comparison topic
    dir. `star_types` (default none) are highlighted with labels."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    df = cmp_df.dropna(subset=["size", "ahull_size"]).copy()
    stars = list(star_types) if star_types is not None else []

    df["rel_diff"] = (df["ahull_size"] - df["size"]) / df["size"]
    fig_rd = _scatter(df, "size", "rel_diff", "main_groups", "instance",
                      colored_main_groups_df, stars, height=600, width=600)
    fig_rd.update_layout(
        xaxis_title="propagated RF size (col)",
        yaxis_title="(ahull size - prop RF size) / prop RF size",
    )
    fig_rd.update_xaxes(type="log", tickformat="d", dtick=1)
    _save(fig_rd, save_dir,
          f"{dataset}_examples_ahull_reldiff_vs_size.pdf", html=True)
    return fig_rd


# %%
def plot_rf_size_per_layer(
    type_df: pd.DataFrame,
    *, size_col: str = "ahull_size", color: str = "#333333",
    col_to_deg_size: float = 25.0,
    dataset: str = DATASET, save_dir: Path | None = None,
    width: int = 896, height: int = 280, font_size: float = 19.656,
):
    """Single-box-per-layer box plot of ARF (arbor alpha-hull) size vs
    hitting-time layer, one box per half-integer layer. Size in deg² (col² ×
    `col_to_deg_size`, default 25 — same factor used by
    `plot_rf_size_vs_experiment`). Bin width 0.5, centers at multiples of 0.5
    (0, 0.5, ..., 6). Each x tick shows the layer and, on a second line in
    parentheses, the number of neurons (rows) in that bin. Pass a per-neuron df
    with `hitting_time` and `size_col` columns — e.g. `get_rf_ahull_neuron()`,
    whose `ahull_size` is the per-neuron alpha-hull (VIN + VPN + VCBN
    pooled)."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    bins = np.arange(-0.25, 6 + 0.25 + 1e-9, 0.5)
    labels = (bins[1:] + bins[:-1]) / 2
    df = type_df.copy()
    # Fold the high tail into the last visible layer so nothing is dropped:
    # clip the hitting time to <= 5 before binning, so every neuron with
    # layer > 4.75 (incl. layer > 5.25) goes into layer 5.
    ht = df["hitting_time"].clip(upper=5.0)
    df["bin"] = pd.cut(ht, bins=bins, labels=labels).astype(float)
    df = df.dropna(subset=["bin", size_col])
    df["size_deg"] = df[size_col] * col_to_deg_size

    # All half-integer layer ticks in view; second line = neuron count per bin
    # (one row per neuron).
    counts = df.groupby("bin").size()
    tickvals = [v for v in labels if 1.0 <= v <= 5.0]
    ticktext = [f"{v:g}<br>({int(counts.get(v, 0))})" for v in tickvals]

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df["bin"], y=df["size_deg"], boxpoints=False,
        marker=dict(color=color), line=dict(color=color),
    ))
    fig.update_layout(
        xaxis_title="layer", yaxis_title="ARF size (deg²)",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="arial", size=font_size),
        height=height, width=width, showlegend=False,
        xaxis_range=[0.5, 5.5], boxmode="group",
    )
    fig.update_xaxes(title_standoff=0, tickvals=tickvals, ticktext=ticktext, **_AXIS)
    fig.update_yaxes(type="log", tickformat="d",
                     tickvals=[100, 1000, 10000],
                     range=[np.log10(25), np.log10(25000)],
                     title_standoff=2, **_AXIS)
    _save(fig, save_dir, f"{dataset}_size_per_layer.pdf", html=True)
    return fig


# %%
def plot_propagated_arf_example(
    meta: pd.DataFrame, stepsn, raw_params: pd.DataFrame, cell_type: str,
    in_instances: list[str], *, bodyId: int | None = None, rng=None,
    dataset: str = DATASET, save_dir: Path | None = None,
):
    """Propagated effective-weight RF (hex heatmap) for one randomly-selected
    neuron of `cell_type`, saved as a PDF. Shows the raw propagated ARF only —
    no 2D-Gaussian or alpha-hull fit is overlaid. Companion to
    `plot_rf_size_per_layer` (Main Fig. 6b).

    One body is drawn at random from the `raw_params` rows of `cell_type`
    (pass `rng` for reproducibility, or an explicit `bodyId`). `meta`/`stepsn`
    must be the propagation matrix the RF came from: the OL-subset pair
    (`get_ol_meta` / `get_ol_stepsn_sum`, `get_rf_raw_ol`) for OL-region types,
    the full-brain pair (`get_meta` / `<dataset>_<side>_lat_flow_sum.npz`,
    `get_rf_raw_cb`) for CB types. Returns (fig, bodyId), or (None, bodyId) if
    the type/body has no propagated RF.
    """
    from connectome_interpreter.external_map import hex_heatmap
    from utils.ol_rf import compute_rf

    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    sub = raw_params[raw_params["cell_type"] == cell_type]
    if sub.empty:
        return None, None
    if bodyId is None:
        rng = rng if rng is not None else np.random.default_rng(0)
        bodyId = int(rng.choice(sub["bodyId"].to_numpy()))
    elif bodyId not in set(sub["bodyId"]):
        return None, bodyId

    inidx = meta[meta["instance"].isin(in_instances)].idx.values
    outidx = meta[meta["bodyId"] == bodyId].idx.values
    df = compute_rf(meta, stepsn, inidx, outidx)
    if df.empty:
        return None, bodyId
    tot_max = float(np.abs(df["effective weight"]).max())
    orig = df.set_index("coords")[["effective weight"]]
    orig = orig[~orig.index.duplicated()]
    fig = hex_heatmap(
        orig, custom_colorscale="RdBu_r",
        global_min=-tot_max, global_max=tot_max, dataset="mcns_right",
    )
    fig.update_traces(marker_line_width=0)  # drop the gray hexagon outlines
    fig.update_layout(title=f"{cell_type} ({bodyId}) — propagated ARF")
    _save(fig, save_dir, f"{dataset}_propagated_arf_{cell_type}_{bodyId}.pdf")
    return fig, bodyId


# %%
def plot_combined_arf(
    meta: pd.DataFrame, stepsn, bodyIds, in_instances: list[str], *,
    weights=None, label: str = "combined", title: str | None = None,
    cmax: float | None = None,
    dataset: str = DATASET, save_dir: Path | None = None,
    save_name: str | None = None,
):
    """Combined propagated effective-weight ARF (hex heatmap) over a set of
    neurons. For each visual-input hex column the effective weights from
    `in_instances` onto each body in `bodyIds` are combined across neurons:
    a plain sum by default, or — if `weights` is given — a weighted sum, with
    each body's ARF scaled by `weights[bodyId]` before summing (e.g. the body's
    `inprop` weight onto a downstream target). Companion to
    `plot_propagated_arf_example`, which draws a single neuron.

    `meta`/`stepsn` must be the propagation matrix the bodies live in: the
    full-brain pair (`get_meta` / `<dataset>_<side>_lat_flow_sum.npz`) or the
    OL-subset pair (`get_ol_meta` / `get_ol_stepsn_sum`). `weights` is an
    optional `{bodyId: float}` mapping (or array aligned to `bodyIds`); bodies
    absent from it get weight 0. `cmax` sets the symmetric colour-scale bound
    (`global_min=-cmax`, `global_max=cmax`); pass the corresponding
    single-neuron ARF's `max(|effective weight|)` to share its scale. Defaults
    to this set's own max. `save_name` (if given) is the PDF filename. Returns
    (fig, combined_df) where `combined_df` is indexed by hex `coords` with a
    single 'effective weight' column, or (None, None) if no body has a
    propagated RF.
    """
    from connectome_interpreter.compress_paths import result_summary
    from connectome_interpreter.external_map import hex_heatmap

    bids = [int(b) for b in (bodyIds.tolist() if hasattr(bodyIds, "tolist") else bodyIds)]
    inidx = meta[meta["instance"].isin(in_instances)].idx.values
    outidx = meta[meta["bodyId"].isin(bids)].idx.values
    if len(inidx) == 0 or len(outidx) == 0:
        return None, None

    idx_to_coords = dict(zip(meta.idx, meta.coords))
    idx_to_body = dict(zip(meta.idx, meta.bodyId))
    m = result_summary(
        stepsn, inidx, outidx,
        inidx_map=idx_to_coords, outidx_map=idx_to_body,
        combining_method="sum", display_threshold=0, display_output=False,
    )
    # m: rows = coords strings, one column per body (str(bodyId)) = its ARF
    m = m[(m.index != "nan") & (m.index != "None") & (~m.index.isnull())]
    if m.empty:
        return None, None
    if weights is None:
        combined = m.sum(axis=1)  # plain pooled ARF
    else:
        wmap = weights if hasattr(weights, "items") else dict(zip(bids, weights))
        wser = pd.Series({str(int(b)): float(w) for b, w in wmap.items()})
        combined = m.mul(wser.reindex(m.columns).fillna(0.0), axis=1).sum(axis=1)
    orig = combined.to_frame("effective weight")
    orig = orig[~orig.index.duplicated()]
    tot_max = float(np.abs(orig["effective weight"]).max())
    if tot_max == 0:
        return None, None
    scale_max = float(cmax) if cmax is not None else tot_max
    fig = hex_heatmap(
        orig, custom_colorscale="RdBu_r",
        global_min=-scale_max, global_max=scale_max, dataset="mcns_right",
    )
    fig.update_traces(marker_line_width=0)  # drop the gray hexagon outlines
    fig.update_layout(title=title or f"{label} — combined ARF (n={len(bids)})")
    if save_name is not None:
        _save(fig, save_dir, save_name)
    return fig, orig


# %%
def plot_ol_to_cb_heatmap(
    tb_clu_df: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None,
    height: int = 500, width: int = 1050, title_suffix: str = "",
    xlabel: str = "CB types (small <---> medium)",
    ylabel: str = "VPN types (medium <---> small)",
    show_ticks: bool = False,
):
    """Heatmap of normalised VPN → CB-type effective weights."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "CB_rf_fits"
    fig = go.Figure(data=go.Heatmap(
        z=tb_clu_df.values,
        x=tb_clu_df.columns.tolist(), y=tb_clu_df.index.tolist(),
        colorscale="Greys", reversescale=False, showscale=True,
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        title=f"rel. effective weight from VPN to CB types{title_suffix}",
        xaxis_title=xlabel, yaxis_title=ylabel,
        xaxis_tickangle=-45 if show_ticks else 0,
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    if show_ticks:
        fig.update_xaxes(
            showline=True, title_standoff=0, linewidth=1, linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            showline=True, title_standoff=2, linewidth=1, linecolor="black",
            mirror=True, autorange="reversed",
        )
    else:
        fig.update_xaxes(
            showline=True, title_standoff=2, linewidth=1, linecolor="black",
            mirror=True, tickvals=list(range(tb_clu_df.shape[1])),
            ticktext=["" for _ in tb_clu_df.columns],
        )
        fig.update_yaxes(
            showline=True, title_standoff=2, linewidth=1, linecolor="black",
            mirror=True, tickvals=list(range(tb_clu_df.shape[0])),
            ticktext=["" for _ in tb_clu_df.index],
            autorange="reversed",
        )
    _save(fig, save_dir, f"{dataset}_ol_to_cb_heatmap.pdf")
    return fig


# %%
def _draw_sector_polar(sec_vals, *, cov0_max: float = 0.4):
    """Retinotopic-coverage "pokemon" pictogram.

    Draw the five sector fractions (`sec_vals` = sec1..sec5 = A, D, P, V, C) as
    a polar annulus of four 90-degree outer wedges (A/D/P/V) around a central
    disk (C), coloured on ``RdBu_r`` scaled to ``cov0_max``. ``sec_vals`` is
    renormalised to sum to 1 so the wedges show per-sector coverage fractions.
    Returns a matplotlib ``Figure`` (save with ``fig.savefig``; this is a
    matplotlib figure, not a plotly one, so it does not go through ``_save``).
    """
    import matplotlib.pyplot as plt

    sec_vals = np.asarray(sec_vals, dtype=float)
    total = np.nansum(sec_vals)
    if total > 0:
        sec_vals = sec_vals / total

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
    ax.set_ylim(0, 1)
    ax.set_theta_zero_location("SW")
    ax.set_theta_direction(-1)
    r0 = 1 / np.sqrt(5)  # inner radius: centre disk area == one outer wedge

    for sector, wt in enumerate(sec_vals, start=1):
        color = plt.cm.RdBu_r(wt / cov0_max)
        if sector == 5:  # central disk (C)
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.fill(theta, np.full_like(theta, r0), color=color,
                    edgecolor="k", linewidth=0)
        else:            # outer 90-degree wedges (A, D, P, V)
            a0 = np.deg2rad((sector - 1) * 90)
            a1 = np.deg2rad(sector * 90)
            ax.bar(x=(a0 + a1) / 2, height=1 - r0, width=a1 - a0, bottom=r0,
                   color=color, edgecolor="k", linewidth=0, align="center")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


# %%
def plot_ol_roi_coverage_per_roi(
    coverage: dict, sectors: pd.DataFrame,
    *, dataset: str = DATASET, save_dir: Path | None = None, cov0_max: float = 0.4,
):
    """For each ROI in `coverage['per_roi']`, save three PDFs: full hex heatmap,
    per-sector relative heatmap, and a "pokemon" polar-annulus pictogram of the
    5 sector fractions (`..._coverage_symbol.pdf`; replaces the former
    eye-symbol). Iterates ~78 ROIs. `cov0_max`: upper bound for the
    relative-coverage colorscale (values above are clipped to full colour)."""
    import matplotlib.pyplot as plt

    from connectome_interpreter.external_map import hex_heatmap
    save_dir = save_dir if save_dir is not None else FIG_DIR / "coverage"
    summary = coverage["summary"].set_index("roi")
    valid_sector_coords = sectors.loc[sectors["coords"].notna() & (sectors["coords"] != "None"), "coords"].values
    sector_coords = {
        i + 1: sectors.loc[(sectors.sector == i + 1) & sectors["coords"].notna() & (sectors["coords"] != "None"), "coords"].values
        for i in range(5)
    }

    figs = []
    for roi, roi_df in coverage["per_roi"].items():
        w = roi_df.set_index("coords")[["effective weight"]]
        w = w[w.index.notna() & (w.index.astype(str) != "None")]
        if w.empty:
            continue

        cmax = float(np.quantile(np.abs(w["effective weight"].values), 0.995))
        fig = hex_heatmap(
            w, custom_colorscale="RdBu_r",
            global_min=-cmax, global_max=cmax, dataset="mcns_right",
        )
        figs.append((fig, f"{dataset}_{roi}_coverage_full.pdf"))

        rel = w.reindex(valid_sector_coords).fillna(0).copy()
        for sec_id, coords in sector_coords.items():
            rel.loc[coords, "effective weight"] = rel.loc[coords, "effective weight"].sum()
        fig = hex_heatmap(
            rel, custom_colorscale="RdBu_r",
            global_min=0, global_max=cov0_max, dataset="mcns_right",
        )
        figs.append((fig, f"{dataset}_{roi}_coverage_relative.pdf"))

        # "pokemon" pictogram: the 5 sector fractions as a polar annulus
        # (replaces the former eye-symbol). This is a matplotlib figure, so it
        # is saved directly rather than through the plotly-only `_save`.
        sec_vals = summary.loc[roi, ["sec1", "sec2", "sec3", "sec4", "sec5"]].values.astype(float)
        pfig = _draw_sector_polar(sec_vals, cov0_max=cov0_max)
        save_dir.mkdir(parents=True, exist_ok=True)
        pfig.savefig(
            save_dir / f"{dataset}_{roi}_coverage_symbol.pdf",
            bbox_inches="tight", transparent=True,
        )
        plt.close(pfig)

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_ol_roi_coverage_summary(
    coverage: dict, *, dataset: str = DATASET, save_dir: Path | None = None,
    top_n_coverage: int = 60, top_n_pval: int = 30, cov0_max: float = 0.4,
):
    """Cross-ROI summary: ranked p-value line plot, relative-coverage heatmap
    (sectors × top-N ROIs, ordered D/V/C/A/P), and p-value heatmap (single row).
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "coverage"
    summary = coverage["summary"].copy()
    figs = []

    ranked = summary.sort_values("pval").reset_index(drop=True)
    ranked["index"] = np.arange(len(ranked))
    fig = _point_line(ranked, "index", "pval", height=300, width=1200)
    fig.update_xaxes(tickvals=ranked["index"], ticktext=ranked["roi"], tickangle=-45)
    fig.update_layout(xaxis_title="ROI", yaxis_title="p-value")
    fig.update_yaxes(type="log", range=[-4.2, -2])
    figs.append((fig, f"{dataset}_pval_ranked.pdf"))

    top = summary.iloc[:top_n_coverage]
    cov = top[["sec2", "sec4", "sec5", "sec1", "sec3"]].T
    cov.index = ["D", "V", "C", "A", "P"]
    cov.columns = top["roi"].values
    fig = go.Figure(data=go.Heatmap(
        z=cov.values, x=cov.columns.tolist(), y=cov.index.tolist(),
        colorscale="RdBu_r", showscale=True, zmin=0, zmax=cov0_max,
    ))
    fig.add_hline(y=1.5, line_color="black", line_width=0.5)
    fig.add_hline(y=2.5, line_color="black", line_width=0.5)
    fig.update_layout(
        font=dict(family="arial", size=16),
        title="relative coverage", xaxis_title="", yaxis_title="",
        paper_bgcolor="white", plot_bgcolor="white",
        height=300, width=1400,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(tickangle=-45, title_standoff=0, **_AXIS)
    fig.update_yaxes(autorange="reversed", title_standoff=2, **_AXIS)
    figs.append((fig, f"{dataset}_rel_coverage_per_roi.pdf"))

    top_p = summary.iloc[:top_n_pval]
    pv = np.log10(top_p["pval"].values.astype(float)).reshape(1, -1)
    fig = go.Figure(data=go.Heatmap(
        z=pv, x=top_p["roi"].values.tolist(), y=["log10 p"],
        colorscale="Viridis_r", showscale=True, zmin=-4, zmax=0,
    ))
    fig.update_layout(
        font=dict(family="arial", size=16),
        title="p-values maxT", xaxis_title="", yaxis_title="",
        paper_bgcolor="white", plot_bgcolor="white",
        height=200, width=800,
    )
    fig.update_xaxes(tickangle=-45, title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    figs.append((fig, f"{dataset}_pval_per_roi.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs
