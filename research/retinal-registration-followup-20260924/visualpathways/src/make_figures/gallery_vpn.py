# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### make gallery for VPNs, Document S2

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")


# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
import pandas as pd
import numpy as np
import scipy.sparse

import navis
from neuprint import fetch_adjacencies
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages


import connectome_interpreter as ci

# %%
from utils.projection_plots import plot_extreme_projection_xy
from utils.ol_rf import compute_rf, count_col_ahull
from utils import core_data, ol_rf, vic
from utils.figure_helpers import plot_gaussian_params
from utils.plotter import plot_cns, save_figure_anterior
from utils.pdf_report import (
    natural_sort_key, safe_name, _fmt, split_rect_lr, insert_title_italic_f,
    spread_path_labels, mpl_to_pdf_bytes, plotly_to_pdf_bytes,
    place, place_image, place_plotly_raster, mpl_to_svg_b64, index_pdf_images,
)
from utils.gallery_common import make_bar_fig

# %%
from utils.config import (CACHE_DIR, CACHE_PROC_DIR, DATA_DIR, DATASET, PARAMS_DIR)

result_dir = Path(PROJECT_ROOT, 'results')
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = DATA_DIR
cache_dir.mkdir(parents=True, exist_ok=True)

gallery_dir = CACHE_DIR / 'gallery'  # scratch dir for transient anterior-view renders
gallery_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load data

# %% [markdown]
# ## Get meta

# %%
# Load meta, this with AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
print(meta.shape)  

# %%
# make dictionaries that map indices to meta info
idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))
idx_to_instance = dict(zip(meta.idx, meta.instance))
idx_to_coords = dict(zip(meta.idx, meta.coords))
bodyId_to_idx = dict(zip(meta.bodyId, meta.idx))

# %% [markdown]
# ## brain outline 

# %%
# load outline
outline_1 = pd.read_csv(Path(PARAMS_DIR, 'outline_cb_1.csv'), index_col=False)
outline_2 = pd.read_csv(Path(PARAMS_DIR, 'outline_cb_2.csv'), index_col=False)

outlines_bkgd = [outline_1,  outline_2]

# %% [markdown]
# ## syn

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %% [markdown]
# ## ht, vic, size
#

# %%
# lr = 'r'
# vp_cb_vic = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))

# meta_VP_CB_ahull replacement: per-neuron alpha hull (VPN + VCBN) from the
# combined RF parquet, joined with VIC / hitting_time from the assembled VP+CB table.
_ah = pd.concat([
    ol_rf.get_rf_raw_ol().query("main_groups == 'VPN'"),
    ol_rf.get_rf_raw_cb(),
], ignore_index=True)[['bodyId', 'instance', 'cell_type',
                       'ahull_size', 'ahull_kept_points', 'ahull_rmfrac']]
_vp = vic.load_vp_cb('r').rename(columns={'vision': 'VIC', 'ht': 'hitting_time'})
meta_ahull = _ah.merge(_vp[['bodyId', 'VIC', 'hitting_time', 'main_groups']],
                       on='bodyId', how='inner')
print(meta_ahull.shape)

meta_cb_vpn = meta_ahull.loc[meta_ahull['ahull_rmfrac'] >= 0.6,
                            ['bodyId', 'instance', 'ahull_size', 'ahull_kept_points', 'VIC', 'hitting_time', 'main_groups']]
meta_cb_vpn.rename(columns={'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape, meta_cb_vpn.instance.nunique())

# %%
# ellipse fit + ahull, per neuron (VPN + VCBN), from the combined RF parquet
rf_fit = pd.concat([
    ol_rf.get_rf_raw_ol().query("main_groups == 'VPN'"),
    ol_rf.get_rf_raw_cb(),
], ignore_index=True)
print(rf_fit.shape)
# meta_ahull = pd.read_pickle(Path(cache_dir, 'meta_ahull_scan.pkl'))

rf_size = pd.merge(
    rf_fit[['instance','bodyId', 'size', 'r2']],
    # meta_ahull[['bodyId', 'ahull_max_0.4', 'ahull_cumsum_0.6']],
    meta_ahull[['bodyId', 'ahull_size']],
    on='bodyId',
    how='inner'
)
print(rf_fit.shape)

# %%
# compute instance-level median
rf_size_median = rf_size.groupby('instance').agg({
    'size': 'median',
    'r2': 'median',
    'ahull_size': 'median',
    'bodyId': 'first'
}).reset_index()
rf_size_median['bodyId'] = rf_size_median['bodyId'].astype(int)

# page stats (cell count, median layer, median VIC) come from the same
# type-level sources TableS1 uses, so the gallery pages and
# results/tables/TableS1_VPN_in_out_summary.csv always agree.
_ol_flow_type = core_data.get_ol_flow_type()
ht_median = (_ol_flow_type[['instance', 'hitting_time']]
             .rename(columns={'hitting_time': 'ht'}))
vic_median = vic.get_ol_type_vic()[['instance', 'VIC']]

# %%

# %% [markdown]
# ## Load inprop

# %%
step0 = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_r_lat_flow_0.npz').tocsr()

# %% [markdown]
# ## ol types
#
# R-dominant ?

# %%
# cell type table
oltypes0 = pd.read_excel(Path(PROJECT_ROOT, 'params', 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx'))
oltypes0.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes0.shape)

oltypes_vpn_0 = oltypes0[oltypes0['main_groups'].str.contains('VPN')]

oltypes = oltypes0.sort_values(by='instance', key=lambda x: x.map(natural_sort_key)).reset_index(drop=True)
oltypes_vpn = oltypes_vpn_0.sort_values(by='instance', key=lambda x: x.map(natural_sort_key)).reset_index(drop=True)
oltypes_vpn.shape

# %%
# inst for gallery
inst_vp = pd.merge(oltypes_vpn, rf_size_median, on='instance', how='inner')
print(inst_vp.shape)
inst_vp = pd.merge(inst_vp, ht_median[['instance','ht']], on='instance', how='left')
print(inst_vp.shape)
inst_vp = pd.merge(inst_vp, vic_median[['instance','VIC']], on='instance', how='left')
print(inst_vp.shape)
# displayed cell count = connectome flow count (same as TableS1), not the
# 'no. of cells' column of the Nern et al. supplementary table
inst_vp = pd.merge(inst_vp, _ol_flow_type[['instance','count']], on='instance', how='left')
print(inst_vp.shape)

# %% [markdown]
# ## barcodes

# %%
from utils.palettes import load_colors
# from utils import polarity
# from utils import plotting as plot

# the seed color map: a DataFrame indexed by bare seed name (e.g. "R7", "L1")
#    with a "color" column. 4th element returned by load_colors().
colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()



# %%
# load the lateral flow sum matrix
from utils.config import (DATA_DIR, DATASET, SIDE_CHAR)
stepsn = scipy.sparse.load_npz(Path(DATA_DIR, f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz'))
print(stepsn.shape)

# %%
from utils.core_data import grouped_effweight
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R'])]
outidx = meta.idx[meta.instance.isin(oltypes_vpn['instance'])]

feat = grouped_effweight(stepsn, inidx, outidx, idx_to_instance, idx_to_instance, combining_method="median")

inidx_rf = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])]
outidx_rf = meta.idx[meta.bodyId.isin(inst_vp['bodyId'])]
effwt_visr_7 = grouped_effweight(stepsn, inidx_rf, outidx_rf, idx_to_coords, idx_to_bodyId, combining_method="mean").T

# normalize the feature vectors by row sum
row_sum = feat.sum(axis=1)
feat_norm = feat.div(row_sum.where(row_sum > 0, 1), axis=0)
feat_norm = feat_norm.reset_index(names="instance")

# remove '_R' from column names
feat_norm.columns = [col.replace('_R', '') for col in feat_norm.columns]
# rename HBeyelet to H-B
feat_norm.columns = [col.replace('HBeyelet', 'H-B') for col in feat_norm.columns]
feat.shape

feat_norm_long = feat_norm.melt(id_vars="instance", var_name="VI", value_name="effective weight")
cmap = colored_seed["color"].to_dict()
# change HBeyelet to H-B in the cmap dictionary
cmap = {k.replace('HBeyelet', 'H-B'): v for k, v in cmap.items()}

# %% [markdown]
# ## VP cluster

# %%
from utils import ol_data
FRAC = 0.19
clusters  = ol_data.get_ol_clusters(frac_thre=FRAC)

# %% [markdown]
# ## VCBN ids

# %%
# central-brain (non-OL) candidate target bodyIds
cb_ids = meta_cb_vpn.loc[meta_cb_vpn['main_groups'] == 'nonOL', 'bodyId'].unique()

# %% [markdown]
# # Rendering

# %% [markdown]
# ## settings

# %%
# =======================================================================
#  GALLERY CONFIG  --  every tunable knob in one place.
#  Edit values here, then re-run THIS cell -> the build cell -> the index
#  cell. (Re-running this cell is enough; no kernel restart needed.)
# =======================================================================
import io
import matplotlib as mpl
from PIL import Image
import fitz  # PyMuPDF
from utils.plotting import _stacked_bars

# --- A. FILE SIZE  (size <-> quality trade-offs) -----------------------
# The 3 VIC heatmaps (bottom row) are rasterised; their bytes scale with the
# savefig DPI (NOT the bin count, sec. C). The single-PDF full-gallery size grows
# ~(dpi/40)^2 from the ~19 MB it was at dpi 40 -- fine now that HTML is the
# primary output (per-page SVGs, not one giant PDF).
HEATMAP_DPI = 150       # raster dpi for the VIC heatmaps (was 40); higher = crisper

# Anterior-view (top-left) render size in px, before the white-margin crop.
# Match gallery_vcbn, which uses save_figure_anterior's default 1523x850.
ANTERIOR_WIDTH, ANTERIOR_HEIGHT = 1523, 850

# Palette size for the colour-indexing in the size-shrink cell (PDF modes only).
# 256 = full colormap fidelity (matches gallery_vcbn).
PALETTE_COLORS = 256

# PDF-only: rasterise the RF hex panel at this dpi in add_pdf_page (None -> vector).
RF_RASTER_DPI = 150

# --- B. PAGE LAYOUT  (points; 72 pt = 1 inch) --------------------------
W, H         = 1080, 620     # page width / height
M, TITLE_H   = 12, 28        # panel margin / top title-bar height
NCOLS, NROWS = 3, 2          # panel grid: TL TM TR / BL BM BR

# --- C. CONTENT / ANALYSIS  (change what's shown, not the file size) ----
HEATMAP_XNBINS, HEATMAP_YNBINS = 240, 200  # VIC heatmap spatial bins (granularity)
CONN_THR       = 100         # min summed synaptic weight for the T0->T1 / T1->T2 hop gates
PATHWAY_TOP_N  = 5           # number of strongest upstream paths drawn (top-middle)
MESH_LOD       = 2           # anterior neuron mesh level-of-detail (higher = coarser/faster)
RF_THR_MODE    = 'cumsum'    # RF alpha-hull threshold mode (top-right)
RF_REMOVE_FRAC = 0.6         # RF alpha-hull cumulative-mass cutoff

# --- apply settings ----------------------------------------------------
mpl.rcParams['savefig.dpi'] = HEATMAP_DPI

# PyMuPDF fonts and PDF export/tiling helpers now live in utils.pdf_report.


# %% [markdown]
# ## helper functions

# %%
# _fmt imported from utils.pdf_report.

def make_stats_table_fig(row, figsize=(3.2, 2.6)):
    """Build a matplotlib table of per-instance stats (vector, for `place`)."""
    rows = [
        ("instance",           str(row['instance'])),
        ("cell count",         _fmt(row['count'], "{:.0f}")),
        ("pred. NT",           str(row['predicted neurotransmitter'])),
        ("median hitting time",    _fmt(row['ht'])),
        ("median VIC",         _fmt(row['VIC'], "{:.4f}")),
        ("ellipse size (#col)",    _fmt(row['size'], "{:.1f}")),
        ("median r²",     _fmt(row['r2'])),
        ("α-hull size (#col)",  _fmt(row['ahull_size'], "{:.1f}")),
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['metric', 'value'],
                   cellLoc='left', colLoc='left', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('0.85')
        if r == 0:                       # header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('0.92')
    fig.tight_layout(pad=0.2)
    return fig

def make_vic_hop_figs(
    inst, *, step0, syn0, meta_cb_vpn, cb_ids,
    bodyId_to_idx, idx_to_bodyId, outlines_bkgd,
    conn_thr=100, min_weight=1, top_n_max=5,
):
    """Build three central-brain VIC max-projection heatmaps for one instance.

    Panels (left -> right on the gallery's bottom row):
      * T0 : the VPN population itself, colored by per-cell VIC.
      * T1 : its direct (1-hop) central-brain targets.
      * T2 : the next hop -- targets of the *curated* T1 population.

    Both hops reuse a single propagation matrix `step0` (the 1-hop in-proportion
    matrix A): T1 is seeded from T0, and T2 is seeded from the T1 result and has
    `step0` applied once more. This chains the populations actually shown in the
    panels and avoids materializing A @ A. Consequently T2 only counts 2-hop
    paths through the displayed T1 set (central-brain, multi-synapse, past the
    >conn_thr connection gate); paths relaying through OL or pruned neurons are
    intentionally excluded.

    PERF: up to 2 live neuprint `fetch_adjacencies` calls per instance (to drop
    single-synapse partners and enforce the >conn_thr connection gate). For large
    galleries this dominates runtime; caching partner weights keyed on
    (sources, targets) is where to optimize later.

    Heatmap spatial resolution is HEATMAP_XNBINS x HEATMAP_YNBINS (CONFIG cell).

    Returns a dict:
      t0/t1/t2        : matplotlib Figures (or None if that hop has no data)
      n_t0/n_t1/n_t2  : cell counts for the panel titles
      top_inst_t1/t2  : up to `top_n_max` "instance: median-value" strings
    """
    out = dict(t0=None, t1=None, t2=None, n_t0=0, n_t1=None, n_t2=None,
               top_inst_t1=[], top_inst_t2=[])

    # bodyIds of the VPN instance itself (the T0 seed population)
    ids_t0 = meta_cb_vpn.loc[meta_cb_vpn.instance == inst, 'bodyId'].unique()
    out['n_t0'] = len(ids_t0)
    if len(ids_t0) == 0:
        return out

    def _project(df_plot, pct_hi, label):
        """Render one max-projection heatmap; `pct_hi` sets the upper color clip
        (color floor is fixed at 0)."""
        df_plot['val'] = df_plot['value'] if 'value' in df_plot else df_plot['VIC']
        vmax = np.nanpercentile(df_plot['val'], pct_hi) if len(df_plot) else 0
        fig, _, _ = plot_extreme_projection_xy(
            df_plot, outlines_bkgd,
            xrange=[20000, 80000], yrange=[0, 50000],
            xnbins=HEATMAP_XNBINS, ynbins=HEATMAP_YNBINS,
            agg='largest', agg_frac=0.1, im_norm='linear',
            vmin=0, vmax=vmax, colorbar_label=label)
        return fig

    def _downstream(seed_ids, seed_weights, exclude_ids):
        sidx = [bodyId_to_idx[i] for i in seed_ids]
        summed = seed_weights @ step0[sidx, :]      # row-vec × matrix -> 1-D ndarray
        dn = summed.nonzero()[0]
        d = pd.DataFrame({'bodyId': [idx_to_bodyId[i] for i in dn],
                          'value': summed[dn]})
        d = d[d['bodyId'].isin(cb_ids) & ~d['bodyId'].isin(exclude_ids)]
        return d.sort_values('value', ascending=False)

    def get_adj_with_thr(sources, targets):
        """Live neuprint check: keep only targets whose summed synaptic weight
        from `sources` exceeds `min_weight` (drops single-synapse partners)."""
        _, c = fetch_adjacencies(sources=sources, targets=targets,
                                 include_nonprimary=True, rois=['CentralBrain'])
        c = c.groupby('bodyId_post').agg({'weight': 'sum'}).reset_index()
        return c[c['weight'] > min_weight]

    # bodyId -> instance lookup, built once. Replaces a per-row
    # meta_cb_vpn.bodyId == x scan (O(n_targets x n_meta)); identical result
    # because bodyId is unique within meta_cb_vpn.
    _bid_to_inst = dict(zip(meta_cb_vpn['bodyId'], meta_cb_vpn['instance']))

    def _add_instance(df):
        """Attach each target's instance and build the 'instance: value' list
        (ranked by median propagated value) used for the panel annotation."""
        df = df.copy()
        df['instance'] = df['bodyId'].map(_bid_to_inst)
        df_inst = (df.groupby('instance').agg({'value': 'median'})
                   .reset_index().sort_values('value', ascending=False))
        top = [f"{r['instance']}: {r['value']:.2g}"
               for _, r in df_inst.head(min(top_n_max, df_inst.shape[0])).iterrows()]
        return df, top

    # --- T0: the VPN population, colored by per-cell VIC ---
    df_plot = pd.merge(
        syn0,
        meta_cb_vpn.loc[meta_cb_vpn.instance == inst, ['instance', 'bodyId', 'VIC']],
        on='bodyId', how='inner')
    out['t0'] = _project(df_plot, 95, 'VIC')

    # first-hop seed weights = per-cell VIC of the VPN population
    vision_values = (meta_cb_vpn[meta_cb_vpn.bodyId.isin(ids_t0)]
                     .set_index('bodyId')['VIC'].to_dict())
    w_t0 = np.array([vision_values[b] for b in ids_t0], dtype=np.float32)

    # --- T1: 1-hop downstream of T0 ---
    df = _downstream(ids_t0, w_t0, ids_t0)
    c1 = get_adj_with_thr(ids_t0, df['bodyId'].unique()) if df.shape[0] > 0 \
        else pd.DataFrame(columns=['bodyId_post', 'weight'])

    if c1['weight'].sum() > conn_thr:                    # gate: skip weak projections
        df = df[df['bodyId'].isin(c1['bodyId_post'].unique())]
        if df.shape[0] > 0:
            df, out['top_inst_t1'] = _add_instance(df)
            out['n_t1'] = df.shape[0]
            out['t1'] = _project(pd.merge(syn0, df, on='bodyId', how='inner'),
                                 95, 'VIC')

            # --- T2: reseed from the curated T1 set, propagate one more hop ---
            ids_t1 = df['bodyId'].values
            w_t1 = df['value'].values.astype(np.float32)   # T1 propagated VIC as weights
            df = _downstream(ids_t1, w_t1, np.concatenate([ids_t0, ids_t1]))
            if df.shape[0] > 0:
                c1 = get_adj_with_thr(ids_t1, df['bodyId'].unique())
                df = df[df['bodyId'].isin(c1['bodyId_post'].unique())]
                if df.shape[0] > 0:
                    df, out['top_inst_t2'] = _add_instance(df)
                    out['n_t2'] = df.shape[0]
                    out['t2'] = _project(pd.merge(syn0, df, on='bodyId', how='inner'),
                                         95, 'VIC')
    return out


# %%
def grid_cells(ncols=NCOLS, nrows=NROWS):
    """Return a list of fitz.Rect tiles in row-major order (TL, TM, TR, BL, BM, BR).
    Uses the page-layout knobs (W, H, M, TITLE_H, NCOLS, NROWS) from the CONFIG cell."""
    cells = []
    col_w = W / ncols
    row_h = (H - TITLE_H) / nrows
    for r in range(nrows):
        for col in range(ncols):
            x0 = col * col_w       + (M if col == 0          else M / 2)
            x1 = (col + 1) * col_w - (M if col == ncols - 1  else M / 2)
            y0 = TITLE_H + r * row_h       + (M if r == 0         else M / 2)
            y1 = TITLE_H + (r + 1) * row_h - (M if r == nrows - 1 else M / 2)
            cells.append(fitz.Rect(x0, y0, x1, y1))
    return cells



# %%
# =======================================================================
#  MEDIUM-AGNOSTIC HELPERS  --  shared by the PDF and HTML consumers.
#  (Kept consistent with gallery_vcbn.ipynb; will be factored out later.)
# =======================================================================
import base64
import kaleido   # imported once here so the pdf/html driver cells below don't
                 # depend on each other (either can be run standalone)

# stacked-bar strip aspect (PDF points; also the px aspect for HTML)
BAR_W, BAR_H = 36, 260

# %%
# --- HTML layout knobs ------------------------------------------------------
TOP_H          = 520   # top-row panel height in px (raised so the bigger RF isn't clipped)
CAP_FONT       = 16    # panel-title (caption) font size in px  (was 18; -2)
ROW_H          = 400   # shared height of the pathway plot AND the bar in the TR panel (bigger)
CAP_FONT       = 16
BAR_HTML_W     = 80    # HTML bar GRAPH-DIV width in px  (smaller -> more room for the pathway)
BAR_SIDE_PAD   = 25    # L/R margin inside the bar fig   (was 80)
BAR_HOVER_FONT = 14    # hover-tooltip font size in px


# %%
# RF hex panel sizing -------------------------------------------------------
# The upper-right RF lives in the FIXED-height top row, so it must be GENERATED
# at panel size (markers scale with the figure -> no overlap) rather than shrunk
# by CSS afterwards. RF_SIZING is a uniform downscale of hex_heatmap's
# default_sizing (see external_map.py); keeping every size key in the same ratio
# preserves the marker-to-plot spacing the native (PDF) figure has.
RF_HTML_SCALE = 0.72          # tune: bigger = larger RF, less right-margin whitespace (may clip on narrow screens)
RF_SIZING = {
    "fig_width":   round(260 * RF_HTML_SCALE),   # mm (260 = default WITH colorbar)
    "fig_height":  round(220 * RF_HTML_SCALE),   # mm
    "markersize":  18 * RF_HTML_SCALE,           # default 18 (mcns_right)
    "fsize_ticks_pt": round(20 * RF_HTML_SCALE),
    "fsize_title_pt": round(20 * RF_HTML_SCALE),
    "ticklen":     round(15 * RF_HTML_SCALE),
    "tickwidth":   max(1, round(5 * RF_HTML_SCALE)),
    "axislinewidth": max(1, round(3 * RF_HTML_SCALE)),
    "cbar_thickness": round(20 * RF_HTML_SCALE),
}   # -> ~531 x 449 px at dpi 72; PDF passes sizing=None for the native 737 x 624 px


def make_rf_fig(inst, bodyId, *, sizing=None):
    """RF hex heatmap + ellipse fit + alpha-hull overlay (plotly), or None.

    `sizing` is forwarded to ci.hex_heatmap's default_sizing so the figure is
    generated at the wanted on-screen size (markers scale with it -> no overlap).
    Pass None for the native PDF size. Returns None if the alpha-hull is not a
    single polygon (panel is then left blank).
    """
    df = effwt_visr_7[str(bodyId)].copy()
    fig_rf0 = ci.hex_heatmap(
        df,
        custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
        global_min=0,
        sizing=sizing,
    )
    fig_rf0.update_traces(
        selector=lambda t: t.marker.colorbar.thickness is not None,
        marker_colorbar_title_text="effective weight",
        marker_colorbar_title_side="bottom",
    )
    params_single_df = rf_fit.loc[rf_fit.instance == inst]
    fig1 = plot_gaussian_params(params_single_df, example_bid=bodyId, fac=np.sqrt(3))
    for trace in fig1.data[1:]:
        fig_rf0.add_trace(trace)

    edge_points, edges, area, com, kept_points, used_frac, used_alpha = count_col_ahull(
        df, thr_mode=RF_THR_MODE, remove_frac=RF_REMOVE_FRAC)
    if edges is None or len(edge_points) != 1:
        print(f"no single alpha-hull for bodyId {bodyId} "
              f"({RF_THR_MODE}, {RF_REMOVE_FRAC}), skipping panel")
        return None

    fig_rf = go.Figure(fig_rf0)
    edge_pt = edge_points[0]
    for j in range(len(edge_pt)):
        p1, p2 = edge_pt[j], edge_pt[(j + 1) % len(edge_pt)]
        fig_rf.add_trace(go.Scatter(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines',
            line=dict(color='blue', width=4), showlegend=False, hoverinfo='skip'))
    fig_rf.add_trace(go.Scatter(
        x=[com[0]], y=[com[1]], mode='markers',
        marker=dict(color='blue', size=10, symbol='cross'),
        name='center of mass', hoverinfo='skip'))
    return fig_rf


# %%
# 2D gaussian, prob mass (cumsum) within one std contour, is ~0.39
# PDF height (max) at one std is ~0.61

# check ahull area calculation using bodyid = 28862.
# check edge_points using 531732
# thr_2dGaussian = 0.4

# pathway plot
from utils import ol_data, core_data
from utils import plotting as plot
from utils.palettes import load_colors
ol_flow = core_data.get_ol_flow_type()
ol_meta     = core_data.get_ol_meta()
colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()
type_to_sign = dict(zip(ol_meta.cell_type, ol_meta.sign))
sign_to_color = colored_sign['color'].to_dict()
type_to_mg = dict(zip(ol_meta.cell_type, colored_main_groups.loc[ol_meta.main_groups, 'color']))

# 'LC6_R', 
sampled_meta = inst_vp[inst_vp['instance'].isin(['LC6_R', 'LC4_R','VS_R', 'LC41_R'])].reset_index(drop=True)


# %%
def build_panels(row):
    """Compute the medium-agnostic panels for one VPN instance (or None to skip).

    The bar AND the RF are consumer-specific (size differs for PDF vs HTML), so
    they are built in the consumers via make_bar_fig / make_rf_fig, not here.
    """
    bodyId    = int(row['bodyId'])
    inst      = row['instance']
    cell_type = row['cell_type']
    count     = row['count']

    clu = clusters[clusters.instance == inst]['cluster'].values
    if len(clu) == 0:
        print(f"instance {inst} has no cluster, skipping")
        return None
    if inst not in vic_median['instance'].values:
        print(f"instance {inst} not in vic_median, skipping")
        return None
    if str(bodyId) not in effwt_visr_7.columns:
        print(f"bodyId {bodyId} ({inst}) not in effwt_visr_7, skipping")
        return None

    title_tl = f"anterior view; bodyId: {bodyId}"
    title_tr = (f"ARF; median ellipse size: {_fmt(row['size'], '{:.1f}')} cols; "
                f"median hull size: {_fmt(row['ahull_size'], '{:.1f}')} cols")

    # --- top-left: anterior view (PNG file) ---
    fig_ant = plot_cns(
        int(bodyId), cell_type,
        show_meshes=True, show_outline=True, mesh_lod=MESH_LOD,
        color=(0, 0, 0, 1), annotate=False,
    )
    img_fn = save_figure_anterior(fig_ant, name="tmp_anterior", path=gallery_dir,
                                  width=ANTERIOR_WIDTH, height=ANTERIOR_HEIGHT)

    # --- top-middle: upstream pathway (matplotlib) ---
    frac, fig_path = plot.plot_pathway(
        ol_data.get_paths_to_instance(inst), inst, ol_flow,
        top_n=PATHWAY_TOP_N,
        neuron_to_color=type_to_mg,
        neuron_to_sign=type_to_sign, sign_color_map=sign_to_color,
        figsize=(8, 8),
        save_path=None,
        show_frac=False,
        show=False,
    )
    spread_path_labels(fig_path)

    # --- bottom row: central-brain VIC max-projection heatmaps (T0/T1/T2) ---
    vic_figs = make_vic_hop_figs(
        inst, step0=step0, syn0=syn0,
        meta_cb_vpn=meta_cb_vpn, cb_ids=cb_ids,
        bodyId_to_idx=bodyId_to_idx, idx_to_bodyId=idx_to_bodyId,
        outlines_bkgd=outlines_bkgd, conn_thr=CONN_THR)
    # plot_extreme_projection_xy uses figsize=(8,4) with aspect='equal', which leaves
    # a wide empty strip on the right (axes wider than the 1.2-ratio content). Narrow
    # the figure so 'tight' export trims that strip in both the PDF and HTML panels.
    for _k in ('t0', 't1', 't2'):
        if vic_figs[_k] is not None:
            vic_figs[_k].set_size_inches(5.7, 4)   # was 8x4; tune for less right margin

    # RF (make_rf_fig) is built per-consumer for medium-specific sizing.
    return dict(inst=inst, bodyId=bodyId, cell_type=cell_type, count=count,
                clu=clu[0], frac=frac,
                title_tl=title_tl, title_tr=title_tr,
                anterior_png=img_fn, fig_path=fig_path,
                vic_figs=vic_figs, row=row)



# %%
def add_pdf_page(out_doc, p):
    """Tile one VPN instance's panels onto a new 3x2 page of `out_doc` (needs kaleido)."""
    page = out_doc.new_page(width=W, height=H)
    tl, tm, tr, bl, bm, br = grid_cells()       # row-major: TL TM TR / BL BM BR

    page.insert_text(
        (M, 18),
        f"instance: {p['inst']}; neuron count: {p['count']}; VPN class: oc{p['clu']}",
        fontsize=13, fontname="hebo")

    # --- top-left: anterior view (raster PNG) ---
    place_image(page, p['anterior_png'], tl, title=p['title_tl'])

    # --- top-middle: left strip = stacked bar, right = pathway ---
    bar_rect, tm_path = split_rect_lr(tm, left_frac=0.2)
    bar_rect = fitz.Rect(
        bar_rect.x0, bar_rect.y0 + 50,
        bar_rect.x0 + BAR_W, bar_rect.y0 + BAR_H,
    )
    fig_bar = make_bar_fig(p['inst'], feat_norm_long, cmap, height=1200, font_size=55)   # tall -> crisp when scaled
    place(page, plotly_to_pdf_bytes(fig_bar), bar_rect, title="")

    place(page, mpl_to_pdf_bytes(p['fig_path'], tight=False), tm_path, title="")
    _tx, _fs, _lh = tm_path.x0 + 4, 9, 11
    insert_title_italic_f(
        page, (_tx, tm_path.y0 + 11),
        prefix="upstream paths; ",
        suffix=f" = {p['frac']:.2f}",
    )
    page.insert_text(
        (_tx, tm_path.y0 + 11 + _lh),
        f"median VIC= {_fmt(p['row']['VIC'], '{:.4f}')}; median layer= {_fmt(p['row']['ht'])}",
        fontsize=_fs,
    )

    # --- top-right: RF hex heatmap (native size; raster if RF_RASTER_DPI) ---
    fig_rf = make_rf_fig(p['inst'], p['bodyId'])      # sizing=None -> native (PDF-good)
    if fig_rf is not None:
        if RF_RASTER_DPI:
            place_plotly_raster(page, fig_rf, tr, title=p['title_tr'],
                                target_dpi=RF_RASTER_DPI)
        else:
            place(page, plotly_to_pdf_bytes(fig_rf), tr, title=p['title_tr'])


    # --- bottom row: central-brain VIC max-projection heatmaps (T0/T1/T2) ---
    figs = p['vic_figs']
    if figs['t0'] is not None:
        place(page, mpl_to_pdf_bytes(figs['t0']), bl,
              title=f"maximum VIC projection; {figs['n_t0']} cells")
    if figs['t1'] is not None:
        place(page, mpl_to_pdf_bytes(figs['t1']), bm,
              title=f"1-hop downstream; {figs['n_t1']} cells ( > {CONN_THR} connections)",
              subtext=["top inst: VIC"] + figs['top_inst_t1'])
    if figs['t2'] is not None:
        place(page, mpl_to_pdf_bytes(figs['t2']), br,
              title=f"2-hop downstream; {figs['n_t2']} cells ( > {CONN_THR} connections)",
              subtext=["top inst: VIC"] + figs['top_inst_t2'])


# %%
def html_from_panels(p):
    """Interactive HTML page for a VPN instance (no kaleido needed).

    Layout mirrors the PDF 3x2 grid. The RF is generated at panel size via
    make_rf_fig(sizing=RF_SIZING) and rendered at that FIXED size (no responsive),
    so its hex markers never overlap. The pathway plot and the (smaller) bar share
    one fixed height (ROW_H).
    """
    # bar: same height as the pathway plot (ROW_H); font scales with height
    fig_bar = make_bar_fig(p['inst'], feat_norm_long, cmap, height=ROW_H,
                           font_size=round(55 * ROW_H / 1200), width=BAR_HTML_W)
    fig_bar.update_layout(
        margin=dict(l=BAR_SIDE_PAD, r=BAR_SIDE_PAD, t=2, b=2),
        hoverlabel=dict(font_size=BAR_HOVER_FONT),
    )
    bar_div = fig_bar.to_html(full_html=False, include_plotlyjs='cdn')

    # RF generated at panel size; render at its native (fixed) size -> no overlap
    fig_rf = make_rf_fig(p['inst'], p['bodyId'], sizing=RF_SIZING)
    rf_div = (fig_rf.to_html(full_html=False, include_plotlyjs=False)
              if fig_rf is not None
              else "<em>multiple alpha-hulls — RF panel skipped</em>")

    path_src = mpl_to_svg_b64(p['fig_path'])
    ant_src  = ("data:image/png;base64,"
                + base64.b64encode(Path(p['anterior_png']).read_bytes()).decode())

    figs = p['vic_figs']

    def _vic_html(fig, label, n, top_inst=None):
        if fig is None:
            return f"<p class='cap'>{label}</p><em>no data</em>"
        sub = ""
        if top_inst:
            items = "".join(f"<li>{x}</li>" for x in top_inst)
            sub = f"<p class='sub'>top inst: VIC</p><ul class='sub'>{items}</ul>"
        return (f"<p class='cap'>{label}; {n} cells</p>"
                f"<img src='{mpl_to_svg_b64(fig)}'>{sub}")

    t0_html = _vic_html(figs['t0'], "maximum VIC projection", figs['n_t0'])
    t1_html = _vic_html(figs['t1'], f"1-hop downstream (&gt; {CONN_THR} conn.)",
                        figs['n_t1'], figs['top_inst_t1'])
    t2_html = _vic_html(figs['t2'], f"2-hop downstream (&gt; {CONN_THR} conn.)",
                        figs['n_t2'], figs['top_inst_t2'])

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{p['inst']}</title>
<style>
  :root {{ --top-h: {TOP_H}px; --row-h: {ROW_H}px; }}
  body {{ font-family: arial, sans-serif; margin: 12px; }}
  h2 {{ font-size: 16px; margin: 0 0 8px 0; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; align-items: start; }}
  .panel {{ border: 1px solid #ddd; padding: 6px; min-width: 0; box-sizing: border-box; }}
  /* top row: fixed height; clamp content so nothing overflows onto the bottom row */
  .top {{ height: var(--top-h); overflow: hidden; }}
  .top.anterior {{ display: flex; flex-direction: column; }}
  .top.anterior img.ant {{ height: calc(100% - 24px); width: auto; max-width: 100%; object-fit: contain; }}
  /* RF rendered at its own fixed px size (no responsive) -> markers don't overlap */
  /* tight padding on the top-middle/right panels -> more room for the bigger plots */
  .top.rf {{ display: flex; flex-direction: column; padding: 2px; }}
  /* TR: caption on top, then a fixed-height row holding pathway + bar at equal height */
  .top.tr {{ display: flex; flex-direction: column; padding: 2px; }}
  .tr .plotrow {{ display: flex; gap: 8px; height: var(--row-h); align-items: flex-start; min-width: 0; }}
  .tr .plotrow > img {{ height: 100%; width: auto; max-width: 100%; }}   /* pathway */
  .tr .barbox {{ height: 100%; flex: 0 0 auto; }}
  /* bottom panels: fill the column width (kills the right-side gap) */
  .vic img {{ width: 100%; height: auto; }}
  img {{ max-width: 100%; height: auto; }}
  .cap {{ font-size: {CAP_FONT}px; font-weight: 600; color: #333; margin: 0 0 4px 0; }}
  .sub {{ font-size: 11px; color: #666; margin: 2px 0 0 0; }}
  ul.sub {{ margin: 0; padding-left: 16px; }}
</style></head>
<body>
  <h2>instance: {p['inst']}; neuron count: {p['count']}; VPN class: oc{p['clu']}</h2>
  <div class='grid'>
    <div class='panel top anterior'><p class='cap'>{p['title_tl']}</p><img class='ant' src='{ant_src}'></div>
    <div class='panel top tr'>
      <p class='cap'>upstream paths; <i>f</i> = {p['frac']:.2f}</p>
      <div class='plotrow'>
        <img src='{path_src}'>
        <div class='barbox'>{bar_div}</div>
      </div>
      <p class='sub'>median VIC = {_fmt(p['row']['VIC'], '{:.4f}')}; median layer = {_fmt(p['row']['ht'])}</p>
    </div>
    <div class='panel top rf'><p class='cap'>{p['title_tr']}</p>{rf_div}</div>
    <div class='panel vic'>{t0_html}</div>
    <div class='panel vic'>{t1_html}</div>
    <div class='panel vic'>{t2_html}</div>
  </div>
</body></html>"""



# %% [markdown]
# ## pdf driver

# %%
# --- pdf driver ---------------------------------------------------------
# ~60 min
# 'single_pdf'  -> one multi-page PDF for all instances (original behavior)
# 'pdf_per_row' -> one PDF file per instance
# (kaleido is imported once in the helper functions section above)

PDF_MODE = 'single_pdf'        # 'single_pdf' | 'pdf_per_row'
OUT_STEM = 'gallery_vpn'
# rows     = sampled_meta          # row selector; use inst_vp for the full gallery
rows = inst_vp

out_dir = Path(result_dir, OUT_STEM)
if PDF_MODE == 'pdf_per_row':
    out_dir.mkdir(parents=True, exist_ok=True)

saved_pdfs = []                  # consumed by the shrink-file-size cell below

# Persistent Kaleido server: one Chrome reused across every fig.to_image(), so each
# export is ~0.3s instead of ~5s cold-start. build_panels always rasterizes the
# anterior view via save_figure_anterior (fig.to_image()).
try:
    kaleido.stop_sync_server()
except Exception:
    pass
kaleido.start_sync_server()

out = fitz.open() if PDF_MODE == 'single_pdf' else None

for i, row in rows.iterrows():
    print(row['instance'])
    p = build_panels(row)
    if p is None:
        continue

    if PDF_MODE == 'single_pdf':
        add_pdf_page(out, p)
    else:
        doc = fitz.open()
        add_pdf_page(doc, p)
        fp = out_dir / f"{safe_name(p['inst'])}.pdf"
        doc.save(str(fp), garbage=4, deflate=True, deflate_images=True,
                 deflate_fonts=True, clean=True)
        doc.close()
        saved_pdfs.append(fp)

    print(f"Added {p['inst']} (bodyId {p['bodyId']})")

kaleido.stop_sync_server()             # release the persistent Chrome

if PDF_MODE == 'single_pdf':
    pdf_path = Path(result_dir, f'{OUT_STEM}.pdf')
    out.save(str(pdf_path), garbage=4, deflate=True, deflate_images=True,
             deflate_fonts=True, clean=True)
    saved_pdfs.append(pdf_path)
    print(f"Saved {out.page_count} pages to {pdf_path}")
else:
    print(f"Saved {len(saved_pdfs)} PDF(s) to {out_dir}")


# %% [markdown]
# ## shrink pdf

# %%
# --- shrink file size: rewrite embedded RGB rasters to 8-bit /Indexed colour -----
# The heatmaps come from a <=256-colour matplotlib colormap and the anterior render
# is near 2-tone, so palette-indexing is visually lossless yet ~2x smaller than RGB.
# Palette size = PALETTE_COLORS (CONFIG cell). All vector content (pathway, RF hexes,
# colorbar/outline/axis text) is left untouched.
import os
import re
import zlib
from PIL import Image

# _expected_len, _image_issues, _repair_images and index_pdf_images are imported
# from utils.pdf_report.

_mb = lambda f: round(Path(f).stat().st_size / 1e6, 2)
# Shrink every PDF the pdf driver above wrote (one file for 'single_pdf', many for
# 'pdf_per_row').
for _f in saved_pdfs:
    _before = _mb(_f)
    _n = index_pdf_images(_f, colors=PALETTE_COLORS)
    print(f"{Path(_f).name}: indexed {_n} images, {_before} MB -> {_mb(_f)} MB  (integrity check passed)")

# %% [markdown]
# ## html driver

# %%
# --- html driver ---------------------------------------------------------
# ~60 min
# one interactive HTML page per instance (plotly bar + RF stay zoomable/hoverable;
# matplotlib panels as SVG, anterior as PNG)
OUT_STEM = 'gallery_vpn'
rows = inst_vp

out_dir = Path(result_dir, OUT_STEM)
pages_dir = out_dir / 'pages'    # writes pages here; index.html sits in out_dir
pages_dir.mkdir(parents=True, exist_ok=True)

# Persistent Kaleido server: build_panels always rasterizes the anterior view via
# save_figure_anterior (fig.to_image()), even though the RF/bar panels here stay
# native HTML (no kaleido export needed for those).
try:
    kaleido.stop_sync_server()
except Exception:
    pass
kaleido.start_sync_server()

import warnings

# Both warnings below are expected side effects of this cell's design (persistent
# Kaleido server; RF hull-fitting on column-less instances) — suppress just here so
# the pdf driver cell above still surfaces the hull-fitting warning when relevant.
_prev_pbar_hide = navis.config.pbar_hide
navis.config.pbar_hide = True   # silence navis's per-mesh/skeleton progress bars from plot_cns
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The kopts argument is ignored",
                                 category=UserWarning)
        warnings.filterwarnings("ignore", message=r"Dropping \d+ entries without finite coordinates",
                                 category=UserWarning)
        for i, row in rows.iterrows():
            print(row['instance'])
            p = build_panels(row)
            if p is None:
                continue
            (pages_dir / f"{safe_name(p['inst'])}.html").write_text(
                html_from_panels(p), encoding='utf-8')
            print(f"Added {p['inst']} (bodyId {p['bodyId']})")
finally:
    navis.config.pbar_hide = _prev_pbar_hide

kaleido.stop_sync_server()             # release the persistent Chrome

# build the searchable index over the pages just written
from utils.gallery_index import build_gallery_index
build_gallery_index(out_dir, move=False)
print(f"Saved HTML file(s) to {pages_dir}; open {out_dir / 'index.html'}")

# %% [markdown]
# # End
