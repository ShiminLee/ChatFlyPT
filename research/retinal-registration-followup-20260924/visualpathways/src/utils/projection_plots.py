"""
ROI-outline generation and 2D/3D projection plotting.

Generate 2D alpha-shape outlines of neuprint ROI meshes (``get_roi_outline``)
and render synapse-value heatmaps or neuron skeletons/meshes on top of those
outlines: ``plot_extreme_projection_xy`` / ``plot_extreme_projection_xz``
(matplotlib maximum/minimum-projection heatmaps) and
``plot_neuron_with_outlines`` (plotly 3D). The outlines produced by
``get_roi_outline`` feed directly into the plotting functions here.
"""

import warnings

import alphashape
import matplotlib as mpl
import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import Polygon


def get_roi_outline(roi_names, alpha=0.002, lop=None, zranges=None):
    """
        Generate 2D outlines for a list of ROIs using alpha shapes.

        Parameters:
        roi_names: list of str
            List of ROI names to process.
        alpha: float
            Alpha value for the alpha shape algorithm.
        lop: str, optional
            Line of projection, assuming z-axis, rotate [x y z] if it's not z-axis.
        zranges: list of float, optional
            List of z-ranges to filter the ROIs. Assuming after rotation to have aligned with lop.

        Returns:
        list:
            List of outlines for the given ROIs.
        """
    if not isinstance(roi_names, list) or not all(isinstance(roi, str) for roi in roi_names):
        raise ValueError("roi_names must be a list of strings.")

    # check lop
    if lop is not None:
        if lop not in ["x", "y", "z"]:
            raise ValueError("lop must be 'x', 'y', or 'z'.")

    # outlines_roi = []
    outlines_roi = [[] for _ in range(len(roi_names))]
    for i, roi in enumerate(roi_names):
        try:
            roi_data = neu.fetch_roi(roi)
        except Exception as e:
            raise RuntimeError(f"Error fetching ROI '{roi}': {e}")
        # [x y z] coordinates roi mesh points
        xyz = roi_data.vertices

        # rotate such that the line of projection is z-axis
        if lop == "x":            # rotate [x y z] to [y z x]
            xyz = xyz[:, [1, 2, 0]]
        elif lop == "y":            # rotate [x y z] to [x z y]
            xyz = xyz[:, [0, 2, 1]]

        if zranges is not None:
            if not isinstance(zranges, list) or not all(isinstance(section, (int, float)) for section in zranges):
                raise ValueError("zranges must be a list of numeric values (integers or floats).")
            if zranges != sorted(zranges) and zranges != sorted(zranges, reverse=True):
                raise ValueError("zranges must be a monotonic (sorted) list of integers (either increasing or decreasing).")
            # section the roi mesh points
            outlines_tmp = [[] for _ in range(len(zranges)-1)]
            for j in range(len(zranges)-1):
                xy = xyz[(xyz[:, 2] > zranges[j]) & (xyz[:, 2] < zranges[j+1]), :2]  # Select the first two columns (x and y)
                # check if xy is not empty
                if xy.shape[0] != 0:
                    alpha_shape = alphashape.alphashape(xy, alpha=alpha)
                    # Handle possible geometry types
                    if alpha_shape is None:
                        outline_coords = []
                    elif hasattr(alpha_shape, 'geoms'):  # MultiPolygon or collection
                        warnings.warn(f"Multi-part geometry for ROI '{roi}' in section {j}; using largest polygon.")
                        outline_coords = []
                        # pick largest polygon exterior to represent outline
                        largest = max(alpha_shape.geoms, key=lambda g: g.area)
                        if not largest.is_empty:
                            outline_coords = [[xv, yv] for xv, yv in largest.exterior.coords]
                    else:  # Single Polygon
                        if alpha_shape.is_empty:
                            outline_coords = []
                        else:
                            outline_coords = [[xv, yv] for xv, yv in alpha_shape.exterior.coords]
                    outlines_tmp[j] = outline_coords
            outlines_roi[i] = outlines_tmp
        else:
            alpha_shape = alphashape.alphashape(points=xyz[:, :2], alpha=alpha)
            if alpha_shape is None:
                roi_outline = []
            elif hasattr(alpha_shape, 'geoms'):
                warnings.warn(f"Multi-part geometry for ROI '{roi}'; using largest polygon.")
                largest = max(alpha_shape.geoms, key=lambda g: g.area)
                roi_outline = [] if largest.is_empty else [[xv, yv] for xv, yv in largest.exterior.coords]
            else:
                roi_outline = [] if alpha_shape.is_empty else [[xv, yv] for xv, yv in alpha_shape.exterior.coords]
            outlines_roi[i] = roi_outline
    return outlines_roi


def _rotate_outlines(outlines, rot_mat, center):
    if isinstance(outlines[0], pd.DataFrame):
        # store the column name to recover later
        col_names = outlines[0].columns
    rotated = []
    for poly in outlines:
        pts = np.asarray(poly, dtype=float)
        if pts.size == 0:
            rotated.append([])
            continue
        pts_rot = (rot_mat @ (pts - center).T).T + center
        if col_names is not None:
            rotated.append(pd.DataFrame(pts_rot, columns=col_names))
        else:
            rotated.append(pts_rot.tolist())
    return rotated


def plot_extreme_projection_xy(
    df: pd.DataFrame,
    outlines_bkgd: list,
    xrange: list | tuple = [0, 104000],
    yrange: list | tuple = [4000, 56000],
    *,
    rotation_deg: float = 3.0,
    xnbins: int = 208,
    ynbins: int = 104,
    agg: str = 'largest',
    cmap_div: bool = False,
    agg_frac: float = 0.1,
    agg_num: float = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    im_norm: str ='linear',
    clip: bool = True,
    colorbar_label: str = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, mpl.image.AxesImage]:
    """Plot a synapse-value heatmap with ROI outlines."""

    # check df has required columns, x ,y, val
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    required_cols = {'x', 'y', 'val'}
    if not all(col in df.columns for col in ['x', 'y', 'val']):
        raise ValueError(f"df must contain columns 'x', 'y', and 'val'.")
    df = df[['x', 'y', 'val']].dropna().copy()
    df = df.astype({'x': float, 'y': float}, copy=False)
    if df.empty:
        print("No data available after filtering for heatmap plotting.")
        return None, None, None

    if agg not in ['largest', 'smallest']:
        raise ValueError("agg must be either 'largest' or 'smallest'.")

    # rotate [x y]
    if rotation_deg:
        center = np.array([(xrange[0] + xrange[1]) / 2, (yrange[0] + yrange[1]) / 2])
        theta = np.radians(rotation_deg)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords = df[['x', 'y']].to_numpy()
        df.loc[:, ['x', 'y']] = ((rot_mat @ (coords - center).T).T + center)
        outlines_bkgd = _rotate_outlines(outlines_bkgd, rot_mat, center)

    x_bins = np.linspace(xrange[0], xrange[1], xnbins + 1)
    y_bins = np.linspace(yrange[0], yrange[1], ynbins + 1)
    in_bounds = (
        (df['x'] >= x_bins[0]) & (df['x'] <= x_bins[-1]) &
        (df['y'] >= y_bins[0]) & (df['y'] <= y_bins[-1])
    )
    df = df.loc[in_bounds].copy()
    if df.empty:
        raise ValueError('No data falls inside the requested plotting bounds.')

    x_idx = np.digitize(df['x'].to_numpy(), x_bins) - 1
    y_idx = np.digitize(df['y'].to_numpy(), y_bins) - 1
    vals = df['val'].to_numpy(dtype=float)

    # Per-bin statistic: mean of the top (agg='largest') or bottom (agg='smallest')
    # `agg_frac` fraction of values in each bin. Vectorized equivalent of the former
    # groupby(['x_grid','y_grid']).agg(_grid_stat) + iterrows fill (~10x faster: no
    # per-bin Python call). Sort every point by (bin, value), keep the first
    # ceil(agg_frac * count) within each bin, and average them; bins with
    # count <= agg_num stay NaN (rendered black via cmap.set_bad).
    heatmap_data = np.full((ynbins, xnbins), np.nan, dtype=float)
    valid = (
        ~np.isnan(vals)
        & (x_idx >= 0) & (x_idx < xnbins)
        & (y_idx >= 0) & (y_idx < ynbins)
    )
    if valid.any():
        xi, yi, vv = x_idx[valid], y_idx[valid], vals[valid]
        bin_id = yi * xnbins + xi
        # sort within each bin: descending for 'largest', ascending for 'smallest'
        signed = -vv if agg == 'largest' else vv
        order = np.lexsort((signed, bin_id))
        bin_sorted = bin_id[order]
        val_sorted = vv[order]
        nbin = xnbins * ynbins
        counts = np.bincount(bin_sorted, minlength=nbin)
        starts = np.zeros(nbin, dtype=np.int64)
        starts[1:] = np.cumsum(counts)[:-1]
        rank = np.arange(bin_sorted.size) - starts[bin_sorted]   # 0-based rank in bin
        n_keep = np.minimum(
            counts, np.maximum(1, np.ceil(agg_frac * counts).astype(np.int64)))
        keep = rank < n_keep[bin_sorted]
        sums = np.bincount(bin_sorted[keep], weights=val_sorted[keep], minlength=nbin)
        occupied = counts > agg_num
        flat = np.full(nbin, np.nan)
        flat[occupied] = sums[occupied] / n_keep[occupied]
        heatmap_data = flat.reshape(ynbins, xnbins)

    if vmin is None or vmax is None:
        finite_vals = heatmap_data[np.isfinite(heatmap_data)]
        if finite_vals.size == 0:
            raise ValueError('Heatmap contains only NaNs; cannot infer color limits.')
        if vmin is None:
            vmin = float(np.nanpercentile(finite_vals, 5))
        if vmax is None:
            vmax = float(np.nanpercentile(finite_vals, 95))

    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    if agg == 'largest':
        if cmap_div:
            cmap_obj = mpl.colormaps.get_cmap('RdYlBu_r')
        else:
            cmap_obj = mpl.colormaps.get_cmap('plasma')
    elif agg == 'smallest':
        if cmap_div:
            cmap_obj = mpl.colormaps.get_cmap('RdYlBu_r')
        else:
            cmap_obj = mpl.colormaps.get_cmap('plasma_r')
    cmap_obj = cmap_obj.copy() if hasattr(cmap_obj, 'copy') else cmap_obj
    cmap_obj.set_bad(color='black')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    im = ax.imshow(
        heatmap_data,
        extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
        origin='lower',
        aspect='equal',
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        norm=im_norm,
        interpolation='none'
    )
    # Force a compact raster in vector exports (PDF/SVG). Without this, a clipped
    # imshow with many filled bins makes matplotlib's PDF backend fall back to
    # ~10k+ vector line segments per panel (dense heatmaps blow up to ~17 kB each);
    # rasterizing keeps it one small image while colorbar/outline/text stay vector.
    im.set_rasterized(True)

    if clip:
        pts = np.asarray(outlines_bkgd[0], dtype=float)
        clip_patch = Polygon(pts, transform=ax.transData, facecolor='none')
        im.set_clip_path(clip_patch)
        pts = np.asarray(outlines_bkgd[1], dtype=float)
        clip_patch = Polygon(pts, transform=ax.transData, facecolor='white')
        ax.add_patch(clip_patch)
        outline_color = 'black'
    else:
        outline_color = 'white'

    # add outlines
    for poly in outlines_bkgd:
        pts = np.asarray(poly, dtype=float)
        if pts.size >= 4:
            ax.plot(pts[:, 0], pts[:, 1], color=outline_color, linewidth=1)

    # add scale bar, 8nm, 10um scale bar
    ax.plot([60000, 60000+12500], [42000, 42000], color='black', linewidth=2)
    ax.text(66250, 43000, '100 µm', ha='center', va='top', fontsize=10, color='black')

    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.invert_yaxis()

    # remove axes
    ax.axis('off')

    # set colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.75)
    if colorbar_label is not None:
        cbar.ax.set_xlabel(colorbar_label, rotation=0, ha='center')
        cbar.ax.xaxis.set_label_position('bottom')

    return fig, ax, im


# ── visual-input contribution panels ──────────────────────────────────────
# Shared plotting window for every contribution / resolution panel.
_CONTRIB_XRANGE = [20000, 80000]
_CONTRIB_YRANGE = [0, 50000]
_CONTRIB_XNBINS = 240
_CONTRIB_YNBINS = 200

# main_groups -> hitting-time layer edges (each consecutive pair is one panel).
DEFAULT_CONTRIB_LAYERS = {"VPN": [1, 5], "nonOL": [2, 3.5, 4, 5]}


def _resolve_vmax(vmax, gp, layer_i, vals):
    """Per-panel colour ceiling. ``'p99'`` -> 99th percentile of this panel;
    a ``{group: [per-layer, ...]}`` dict -> the per-group/per-layer entry;
    a scalar -> the same value for every panel."""
    if isinstance(vmax, str):
        if vmax == "p99":
            return float(np.nanpercentile(vals, 99))
        raise ValueError(f"unknown vmax spec {vmax!r}")
    if isinstance(vmax, dict):
        return vmax[gp][layer_i]
    return vmax


def plot_contribution_layers(
    syn, val_col, *, plot_dir, fname, outlines,
    gp_layers=DEFAULT_CONTRIB_LAYERS, vmax="p99", vmin=0,
    agg="largest", im_norm="linear", cmap_div=False, agg_frac=0.1,
    top_n=6, show_ninst=False, annotate=True,
):
    """Extreme-value projection panels split by main-group and hitting-time layer.

    For each ``main_groups`` in ``gp_layers`` and each consecutive ``ht``
    interval, filter ``syn``, colour by ``val_col``, and render one
    :func:`plot_extreme_projection_xy` heatmap saved to
    ``plot_dir / fname(gp, lo, hi)``. Pass ``gp_layers=None`` to render a single
    panel over the whole ``syn`` (``fname`` is then called as
    ``fname(None, None, None)``).

    ``vmax`` accepts ``'p99'`` (per-panel 99th percentile), a scalar (same
    ceiling for every panel), or a ``{group: [per-layer, ...]}`` dict. Returns
    ``(top_rois_ls, tbar_counts)``: per panel, the top-``top_n`` ROIs by synapse
    count (a Series) and the panel's synapse count.
    """
    if gp_layers is None:
        panels = [(None, 0, None, None)]
    else:
        panels = [
            (gp, i, edges[i], edges[i + 1])
            for gp, edges in gp_layers.items()
            for i in range(len(edges) - 1)
        ]

    top_rois_ls, tbar_counts = [], []
    for gp, layer_i, lo, hi in panels:
        if gp is None:
            df_plot = syn.copy()
        else:
            mask = (
                (syn["main_groups"] == gp)
                & (syn["ht"] > lo) & (syn["ht"] <= hi)
            )
            df_plot = syn[mask].copy()
        if df_plot.empty:
            continue
        df_plot["val"] = df_plot[val_col]

        top_rois_ls.append(
            df_plot.groupby("roi").size().sort_values(ascending=False).iloc[:top_n]
        )
        tbar_counts.append(len(df_plot))

        fig, ax, _ = plot_extreme_projection_xy(
            df_plot, outlines,
            xrange=_CONTRIB_XRANGE, yrange=_CONTRIB_YRANGE,
            xnbins=_CONTRIB_XNBINS, ynbins=_CONTRIB_YNBINS,
            agg=agg, im_norm=im_norm, cmap_div=cmap_div, agg_frac=agg_frac,
            vmin=vmin, vmax=_resolve_vmax(vmax, gp, layer_i, df_plot["val"]),
        )
        if annotate:
            parts = []
            if show_ninst:
                parts.append(f"n_inst={df_plot.instance.nunique()}")
            parts += [f"n_n={df_plot.bodyId.nunique()}", f"n_syn={len(df_plot)}"]
            if lo is not None:
                parts.append(f"ht={lo}_{hi}")
            ax.text(
                0.5, 0.95, ", ".join(parts), fontsize=10, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes,
            )
        fig.savefig(plot_dir / fname(gp, lo, hi), bbox_inches="tight")
        plt.close(fig)

    return top_rois_ls, tbar_counts


# todo, combine with above
def plot_extreme_projection_xz(
    df: pd.DataFrame,
    outlines_bkgd: list,
    xrange: list | tuple = [26000, 70000],
    zrange: list | tuple = [50000, 138000],
    *,
    rotation_deg: float = 0.0,
    xnbins: int = 208,
    znbins: int = 416,
    agg: str = 'largest',
    agg_frac: float = 0.1,
    vmin: float | None = None,
    vmax: float | None = None,
    im_norm: str ='linear',
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, mpl.image.AxesImage]:
    """Plot a synapse-value heatmap with ROI outlines."""

    def _grid_stat(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        n_keep = max(1, int(np.ceil(agg_frac * arr.size)))
        arr_sorted = np.sort(arr)
        sel = arr_sorted[-n_keep:] if agg == 'largest' else arr_sorted[:n_keep]
        return float(np.mean(sel))

    # check df has required columns, x ,y, val
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    required_cols = ['x', 'z', 'val']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df must contain columns {required_cols}.")
    df = df[required_cols].dropna().copy()
    df = df.astype({'x': float, 'z': float}, copy=False)
    # if no data after filtering, return a msg and stop the function
    if df.empty:
        print("No data available after filtering for heatmap plotting.")
        return None, None, None

    if agg not in ['largest', 'smallest']:
        raise ValueError("agg must be either 'largest' or 'smallest'.")

    # rotate [x z]
    if rotation_deg != 0.0:
        center = np.array([(xrange[0] + xrange[1]) / 2, (zrange[0] + zrange[1]) / 2])
        theta = np.radians(rotation_deg)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords = df[['x', 'z']].to_numpy()
        df.loc[:, ['x', 'z']] = ((rot_mat @ (coords - center).T).T + center)
        outlines_bkgd = _rotate_outlines(outlines_bkgd, rot_mat, center)

    x_bins = np.linspace(xrange[0], xrange[1], xnbins + 1)
    z_bins = np.linspace(zrange[0], zrange[1], znbins + 1)
    in_bounds = (
        (df['x'] >= x_bins[0]) & (df['x'] <= x_bins[-1]) &
        (df['z'] >= z_bins[0]) & (df['z'] <= z_bins[-1])
    )
    df = df.loc[in_bounds].copy()
    if df.empty:
        warnings.warn('No data falls inside the requested plotting bounds.')
        return None, None, None

    x_idx = np.digitize(df['x'], x_bins) - 1
    z_idx = np.digitize(df['z'], z_bins) - 1
    df = df.assign(x_grid=x_idx, z_grid=z_idx, val=df['val'])

    grid_stats = (
        df.groupby(['x_grid', 'z_grid'])['val']
        .agg(_grid_stat)
        .reset_index()
    )

    heatmap_data = np.full((znbins, xnbins), np.nan, dtype=float)
    for _, row in grid_stats.iterrows():
        heatmap_data[int(row['z_grid']), int(row['x_grid'])] = row['val']

    if vmin is None or vmax is None:
        finite_vals = heatmap_data[np.isfinite(heatmap_data)]
        if finite_vals.size == 0:
            raise ValueError('Heatmap contains only NaNs; cannot infer color limits.')
        if vmin is None:
            vmin = float(np.nanpercentile(finite_vals, 5))
        if vmax is None:
            vmax = float(np.nanpercentile(finite_vals, 95))

    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    if agg == 'largest':
        cmap_obj = mpl.colormaps.get_cmap('plasma')
    else:
        cmap_obj = mpl.colormaps.get_cmap('plasma_r')
    cmap_obj = cmap_obj.copy() if hasattr(cmap_obj, 'copy') else cmap_obj
    cmap_obj.set_bad(color='black')

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 8))
    else:
        fig = ax.figure

    im = ax.imshow(
        heatmap_data,
        extent=[x_centers[0], x_centers[-1], z_centers[0], z_centers[-1]],
        origin='lower',
        aspect='equal',
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        norm=im_norm,
        interpolation='none'
    )
    # Force a compact raster in vector exports (PDF/SVG). Without this, a clipped
    # imshow with many filled bins makes matplotlib's PDF backend fall back to
    # ~10k+ vector line segments per panel (dense heatmaps blow up to ~17 kB each);
    # rasterizing keeps it one small image while colorbar/outline/text stay vector.
    im.set_rasterized(True)

    for poly in outlines_bkgd:
        pts = np.asarray(poly, dtype=float)
        if pts.size >= 4:
            ax.plot(pts[:, 0], pts[:, 1], color='white', linewidth=1)

    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(zrange[0], zrange[1])
    ax.invert_yaxis()

    # if ax is not None and ax.figure is fig:
    fig.colorbar(im, ax=ax, orientation='vertical')

    return fig, ax, im


def plot_neuron_with_outlines(n_ls, outlines_bkgd,
                            neuron_color='black',
                            outline_color='gray',
                            xrange: list | tuple = [20000, 80000],
                            yrange: list | tuple = [000, 50000],
                            rotation_deg: float = 3.0,
                            width=360,
                            height=300,
                            show_legend=False):
    """
    Create a 3D plotly figure with neuron(s) and background outlines.

    Parameters
    ----------
    n_ls : list of navis neuron mesh/skeleton
        List of neurons (mesh or skeleton) to plot
    outlines_bkgd : list of pd.DataFrame
        List of dataframes containing outline coordinates with columns 'x1', 'y1', and optionally 'z'
    neuron_color : str or list of str, optional
        Color(s) for neuron(s). If n is a list and neuron_color is a single color,
        all neurons will have the same color. Default: 'black'
    outline_color : str, optional
        Color for outline traces. Default: 'gray'
    xrange : list or tuple, optional
        X-axis range for rotation center. Default: [0, 104000]
    yrange : list or tuple, optional
        Y-axis range for rotation center. Default: [4000, 56000]
    rotation_deg : float, optional
        Rotation angle in degrees to apply to the neuron and outlines around the Z-axis. Default: 3.0
    width : int, optional
        Width of the figure in pixels. Default: 400
    height : int, optional
        Height of the figure in pixels. Default: 400
    show_legend : bool, optional
        Whether to show legend in the figure. Default: False

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure with neuron(s) and outlines
    """
    fig = go.Figure()

    # Handle single neuron vs list of neurons
    if not isinstance(n_ls, navis.core.neuronlist.NeuronList):
        n_ls = [n_ls]

    # Handle neuron colors
    if isinstance(neuron_color, str):
        neuron_colors = [neuron_color] * len(n_ls)
    elif isinstance(neuron_color, list):
        if len(neuron_color) != len(n_ls):
            raise ValueError(f"Length of neuron_color ({len(neuron_color)}) must match number of neurons ({len(n_ls)})")
        neuron_colors = neuron_color
    else:
        raise ValueError("neuron_color must be a string or list of strings")

    # rotate around z-axis
    if rotation_deg:
        center = np.array([(xrange[0] + xrange[1]) / 2, (yrange[0] + yrange[1]) / 2])
        theta = np.radians(rotation_deg)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        outlines_bkgd = _rotate_outlines(outlines_bkgd, rot_mat, center)
        if isinstance(neuron_colors[0], navis.MeshNeuron):
            for i in range(len(n_ls)):
                coords = n_ls[i].vertices
                n_ls[i].vertices = ((rot_mat @ (coords - center).T).T + center)
        elif isinstance(neuron_colors[0], navis.TreeNeuron):
            for i in range(len(n_ls)):
                coords = n_ls[i].nodes[['x','y','z']]
                n_ls[i].skeleton = ((rot_mat @ (coords - center).T).T + center)

    # Add neuron traces
    fig_n = navis.plot3d(n_ls, inline=False, color=neuron_color, backend='plotly')
    for trace in fig_n.data:
        fig.add_trace(trace)

    # fig = go.Figure(data= fig_n.data)

    # Add outline traces
    for outline in outlines_bkgd:
        if len(outline) > 0:
            fig.add_trace(go.Scatter3d(
                x=outline['x1'],
                y=outline['y1'],
                z=outline['z'] if 'z' in outline.columns else [30000] * len(outline),
                mode='lines',
                line=dict(color=outline_color, width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Update layout
    fig.update_layout(
        scene=dict(
            camera=dict(
                projection=dict(type='orthographic'),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z= -1),
                up=dict(x=0, y=-1, z=0)
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor="rgba(0, 0, 0, 0)"
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        autosize=False,
        width=width,
        height=height,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=show_legend
    )

    return fig
