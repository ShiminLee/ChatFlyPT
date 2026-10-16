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

# %% [markdown]
# ### make gallery for VCBNs, Document S3
#
# This takes time!

# %%
# %load_ext autoreload
# %autoreload 2

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
# print(f"Project root directory: {PROJECT_ROOT}")


# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
import pandas as pd
import numpy as np
import scipy.sparse

import plotly.graph_objects as go
import matplotlib.pyplot as plt

import connectome_interpreter as ci

from utils.ol_rf import count_col_ahull
from utils import core_data, ol_rf, vic
from utils.figure_helpers import plot_gaussian_params
from utils.plotter import plot_cns, save_figure_anterior
from utils.pdf_report import (
    natural_sort_key, safe_name, _fmt, split_rect_lr, insert_title_italic_f,
    spread_path_labels, mpl_to_pdf_bytes, plotly_to_pdf_bytes,
    place, place_image, mpl_to_svg_b64, index_pdf_images,
)
from utils.gallery_common import make_bar_fig

from utils.config import (CACHE_DIR, CACHE_PROC_DIR, DATA_DIR, DATASET, PARAMS_DIR)

# %%
result_dir = Path(PROJECT_ROOT, 'results')
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = DATA_DIR
cache_dir.mkdir(parents=True, exist_ok=True)

gallery_dir = CACHE_DIR / 'gallery'  # scratch dir for transient anterior-view renders
gallery_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# # Load data

# %% [markdown]
# ## brain outline 

# %%
# load outline
outline_1 = pd.read_csv(Path(PARAMS_DIR, 'outline_cb_1.csv'), index_col=False)
outline_2 = pd.read_csv(Path(PARAMS_DIR, 'outline_cb_2.csv'), index_col=False)

outlines_bkgd = [outline_1,  outline_2]

# %% [markdown]
# ## ht, vic, size
#

# %%
# ellipse fit + ahull, per neuron (VPN + VCBN), from the combined RF parquet
rf_fit = pd.concat([
    ol_rf.get_rf_raw_ol().query("main_groups == 'VPN'"),
    ol_rf.get_rf_raw_cb(),
], ignore_index=True)


# %%
# meta_VP_CB_ahull replacement: per-neuron alpha hull (VPN + VCBN) reusing the
# rf_fit frame above, joined with VIC / hitting_time from the assembled VP+CB table.
_ah = rf_fit[['bodyId', 'instance', 'cell_type',
              'ahull_size', 'ahull_kept_points', 'ahull_rmfrac']]
_vp = vic.load_vp_cb('r').rename(columns={'vision': 'VIC', 'ht': 'hitting_time'})
meta_ahull = _ah.merge(_vp[['bodyId', 'VIC', 'hitting_time', 'main_groups']],
                       on='bodyId', how='inner')
print(meta_ahull.shape)

meta_cb_vpn = meta_ahull.loc[meta_ahull['ahull_rmfrac'] >= 0.6,
                            ['bodyId', 'instance', 'cell_type', 'ahull_size', 'ahull_kept_points', 'VIC', 'hitting_time', 'main_groups']]
meta_cb_vpn.rename(columns={'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)

meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['ahull_size'].notna()]
meta_cb_vpn['area_to_count'] = meta_cb_vpn['ahull_size'] / meta_cb_vpn['ahull_kept_points']

print(meta_cb_vpn.shape, meta_cb_vpn.instance.nunique())

# %%
# median values
# The rmfrac-filtered frame only picks the representative bodyId (3D/ARF
# panels) and the size-to-count ratio; the printed stats (cell count, median
# layer/VIC, median alpha-hull size) are merged from the same sources TableS3
# uses, so gallery pages and results/tables/TableS3_VCBN_in_out_summary.csv
# always agree.
inst_median = meta_cb_vpn.groupby(['instance', 'cell_type', 'main_groups']).agg(
    area_to_count=('area_to_count', 'median'),
    bodyId=('bodyId', 'first'),
).reset_index()
_flow_all = core_data.get_flow()
inst_stats = (
    _flow_all[['instance', 'count']].drop_duplicates('instance')
    .merge(_flow_all.groupby('instance')['hitting_time'].median()
           .rename('ht').reset_index(), on='instance', how='left')
    .merge(vic.get_ol_cb_vic_type().query("region == 'CB'")[['instance', 'VIC']],
           on='instance', how='left')
    .merge(ol_rf.get_rf_ahull_neuron().groupby('instance')['ahull_size'].median()
           .reset_index(), on='instance', how='left')
)
inst_median = inst_median.merge(inst_stats, on='instance', how='left')
inst_median.shape

# %% [markdown]
# ## Get meta

# %%
# LOAD meta, this has AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
print(meta.shape)  

# %%
# make dictionaries that map indices to meta info
idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))
idx_to_instance = dict(zip(meta.idx, meta.instance))
idx_to_coords = dict(zip(meta.idx, meta.coords))
bodyId_to_idx = dict(zip(meta.bodyId, meta.idx))

# %% [markdown]
# ## vcbn types

# %%
from utils.vic import get_vcbn_types

# right-side vpn and cb
vcbn = get_vcbn_types(side_char='r')

# %%
# oltypes
oltypes = pd.read_excel(Path(PROJECT_ROOT, 'params', 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx'))
oltypes.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes.shape)

# %%
# # inst for gallery
# # For each (instance, cell_type)
# inst_cb = vp_cb_vic.groupby(['instance', 'cell_type']).agg({'bodyId': 'count'}).reset_index()
# inst_cb.rename(columns={'bodyId': 'bodyId_count'}, inplace=True)

# add nt
inst_cb = inst_median.merge(
    meta.groupby('instance')['nt'].first().reset_index(),
    on='instance',
    how='left'
)
print(inst_cb.shape)

# keep vcbn
inst_cb = inst_cb[inst_cb['instance'].isin(vcbn['instance'])]
print(inst_cb.shape)

# %%
# sort by ht
# inst_cb = inst_cb.sort_values(by='ht').reset_index(drop=True)

# natural sort
inst_cb = inst_cb.sort_values(by='instance', key=lambda x: x.map(natural_sort_key)).reset_index(drop=True)

# %% [markdown]
# ## barcodes

# %%
from utils.palettes import load_colors
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
outidx = meta.idx[meta.instance.isin(inst_cb['instance'])]

feat = grouped_effweight(stepsn, inidx, outidx, idx_to_instance, idx_to_instance, combining_method="median")

inidx_rf = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])]
outidx_rf = meta.idx[meta.bodyId.isin(inst_cb['bodyId'])]
effwt_visr_7 = grouped_effweight(stepsn, inidx_rf, outidx_rf, idx_to_coords, idx_to_bodyId, combining_method="mean").T

# stepsn (~10 GB in RAM) is only needed for the two grouped_effweight calls above —
# free it before the render loop. Re-run the load cell above if this cell is re-run.
del stepsn

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
# ## VCBN cluster

# %%
# VCBN cluster
from utils import cb_data
FRAC_CB = 0.17
# NB: force=True would recompute get_cb_features, which loads its own 10 GB copy
# of lat_flow_sum — rely on the disk cache (keyed by frac/vic) unless inputs changed.
cl_r = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB)
cl_r.shape

# %% [markdown]
# # Rendering

# %%
import io
from PIL import Image
import fitz  # PyMuPDF
from utils.plotting import _stacked_bars

# --- A. FILE SIZE  (size <-> quality trade-offs) -----------------------
# Palette size for the colour-indexing in the size-shrink cell (after the build).
# 256 = full colormap fidelity; 128 ~= -1.5 MB with faint heatmap-gradient banding.
PALETTE_COLORS = 256

# --- B. PAGE LAYOUT  (points; 72 pt = 1 in) -----------------------------
W, H = 720, 620          # page width and height in points
M, TITLE_H = 12, 28      # margin and title bar height in points
gx, gy = W / 2, (H - TITLE_H) / 2  # grid center x; grid cell height (below title)

# --- C. CONTENT / ANALYSIS  (change what's shown, not the file size) ----
PATHWAY_TOP_N   = 5           # number of strongest upstream paths drawn (top-middle)
RF_THR_MODE     = 'cumsum'    # RF alpha-hull threshold mode (top-right)
RF_REMOVE_FRAC  = 0.6         # RF alpha-hull cumulative-mass cutoff
SHOW_RF_ELLIPSE = False       # overlay the 2D-Gaussian RF-ellipse fit + its center on the ARF panel

# split_rect_lr, insert_title_italic_f and add_segment_labels are imported from
# utils.pdf_report.


# %% [markdown]
# ## helper functions

# %%
# Export helpers (mpl/plotly -> PDF bytes, PDF tiling) are imported from
# utils.pdf_report; only the gallery-specific stats table stays here.

def make_stats_table_fig(row, cl, figsize=(3.2, 2.6)):
    """Build a matplotlib table of per-instance stats (vector, for `place`)."""
    rows = [
        ("instance",           str(row['instance'])),
        ("cell count",         _fmt(row['count'], "{:.0f}")),
        ("pred. NT",           str(row['nt'])),
        ("VCBN cluster",            f'cc{cl}'),
        ("median layer",    _fmt(row['ht'])),
        ("median VIC",         _fmt(row['VIC'], "{:.4f}")),
        ("median α-hull size",  f"{_fmt(row['ahull_size'], '{:.1f}')} columns"),
        ("size-to-count ratio",  f"{_fmt(row['area_to_count'], '{:.1f}')} columns"),
        # ("median ellipse size",    f"{_fmt(row['size'], '{:.1f}')} columns"),
        # ("median R²",     _fmt(row['r2'])),
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=None,
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


def quadrants():
    """Return the four fitz.Rect quadrants (TL, TR, BL, BR) using the page-layout
    knobs (W, H, M, TITLE_H, gx, gy) from the settings cell."""
    tl = fitz.Rect(M,         TITLE_H + M,        gx - M/2, TITLE_H + gy - M/2)
    tr = fitz.Rect(gx + M/2,  TITLE_H + M,        W - M,    TITLE_H + gy - M/2)
    bl = fitz.Rect(M,         TITLE_H + gy + M/2, gx - M/2, H - M)
    br = fitz.Rect(gx + M/2,  TITLE_H + gy + M/2, W - M,    H - M)
    return tl, tr, bl, br


# %%
# 2D gaussian, prob mass (cumsum) within one std contour, is ~0.39
# PDF height (max) at one std is ~0.61

# check ahull area calculation using bodyid = 28862.
# check edge_points using 531732
# thr_2dGaussian = 0.4

# pathway plot
from utils import core_data, cb_data
from utils import plotting as plot
from utils.palettes import load_colors
# cb_flow = core_data.get_cb_flow_type()
cb_flow = core_data.get_flow_per_group()
cb_meta     = core_data.get_meta()
colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()
type_to_sign = dict(zip(cb_meta.cell_type, cb_meta.sign))
sign_to_color = colored_sign['color'].to_dict()
# type_to_mg = dict(zip(cb_meta.cell_type, colored_main_groups.loc[cb_meta.main_groups, 'color']))
mg = cb_meta.main_groups.where(cb_meta.main_groups.isin(colored_main_groups.index), 'other')
type_to_mg = dict(zip(cb_meta.cell_type, colored_main_groups.loc[mg, 'color']))


# %%
# Build the gallery. Output MODE controls how pages are written:
#   'single_pdf'   -> one multi-page PDF for all instances (original behavior)
#   'pdf_per_row'  -> one PDF file per instance
#   'html_per_row' -> one interactive HTML per instance (plotly bar + RF stay
#                     zoomable/hoverable; matplotlib panels as SVG, anterior as PNG)
import base64
import kaleido

BAR_W, BAR_H = 36, 260         # stacked-bar strip aspect (PDF points; also px aspect)
PATH_TEXT_SIZE = 12            # pathway node-label font (smaller -> fewer collisions)
PATH_FIGSIZE   = (11, 8)       # pathway fig size; wider -> more x room between layers


# safe_name and spread_path_labels are imported from utils.pdf_report;
# make_bar_fig is imported from utils.gallery_common.


def build_panels(row):
    """Compute the medium-agnostic panels for one instance.

    Returns a dict of LIVE figure objects / image paths + metadata (the consumers
    convert+close them), or None if the instance should be skipped. The stacked bar is
    NOT built here: it is consumer-specific (size/font differ for PDF vs HTML) and is
    created in add_pdf_page / html_from_panels via make_bar_fig.
    """
    bodyId = int(row['bodyId'])
    inst      = row['instance']
    cell_type = row['cell_type']
    cl = cl_r.loc[cl_r['instance'] == inst, 'cluster'].values
    cl = cl[0] if len(cl) > 0 else 'n/a'

    if str(bodyId) not in effwt_visr_7.columns:
        print(f"bodyId {bodyId} ({inst}) not in effwt_visr_7, skipping")
        return None

    # --- stats table (matplotlib) ---
    stats_fig = make_stats_table_fig(row, cl)

    # NB: the stacked VI bar is consumer-specific (different size/font for PDF vs HTML),
    # so it is built in add_pdf_page / html_from_panels via make_bar_fig, not here.

    # --- upstream pathway plot (matplotlib) ---
    frac, fig_path = plot.plot_pathway(
        cb_data.get_cb_paths_to_instance(inst), inst, cb_flow,
        top_n=PATHWAY_TOP_N,
        neuron_to_color=type_to_mg,
        neuron_to_sign=type_to_sign, sign_color_map=sign_to_color,
        figsize=PATH_FIGSIZE,
        node_text_size=PATH_TEXT_SIZE,
        save_path=None,
        show_frac=False,
        show=False,
    )
    spread_path_labels(fig_path)            # fan overlapping labels apart vertically

    # --- anterior view (PNG file) ---
    fig_ant = plot_cns(
        int(bodyId), cell_type,
        show_meshes=True, show_outline=True, mesh_lod=2,
        color=(0, 0, 0, 1), annotate=False,
    )
    img_fn = save_figure_anterior(fig_ant, name=f"tmp_anterior", path=gallery_dir)

    # --- RF heatmap + ellipse + alpha-hull (plotly), or None if multiple ahulls ---
    df = effwt_visr_7[str(bodyId)].copy()
    fig_rf0 = ci.hex_heatmap(
        df,
        custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
        global_min=0,
    )
    fig_rf0.update_traces(
        selector=lambda t: t.marker.colorbar.thickness is not None,
        marker_colorbar_title_text="effective weight",
        marker_colorbar_title_side="bottom",
    )
    if SHOW_RF_ELLIPSE:
        params_single_df = rf_fit.loc[rf_fit.instance == inst]
        fig1 = plot_gaussian_params(params_single_df, example_bid=bodyId, fac=np.sqrt(3))
        for trace in fig1.data[1:]:
            fig_rf0.add_trace(trace)

    thr_mode, remove_frac = RF_THR_MODE, RF_REMOVE_FRAC
    edge_points, edges, area, com, kept_points, used_frac, used_alpha = count_col_ahull(
        df, thr_mode=thr_mode, remove_frac=remove_frac)
    fig_rf = None
    if edges is not None and len(edge_points) == 1:
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
    else:
        print(f"no single alpha-hull for bodyId {bodyId} ({thr_mode}, {remove_frac}), skipping panel")

    return dict(inst=inst, bodyId=bodyId, cell_type=cell_type, cl=cl, frac=frac,
                stats_fig=stats_fig, fig_path=fig_path,
                anterior_png=img_fn, fig_rf=fig_rf)


def add_pdf_page(out_doc, p):
    """Tile one instance's panels onto a new 2x2 page of `out_doc` (needs kaleido)."""
    page = out_doc.new_page(width=W, height=H)
    tl, tr, bl, br = quadrants()

    # - top-left: stats table — matplotlib -> PDF
    place(page, mpl_to_pdf_bytes(p['stats_fig']), tl, title="")

    # - top-right cell: left strip = bar, right = pathway
    bar_rect, tm_path = split_rect_lr(tr, left_frac=0.2)
    bar_rect = fitz.Rect(
        bar_rect.x0, bar_rect.y0 + 50,
        bar_rect.x0 + BAR_W, bar_rect.y0 + BAR_H,
    )
    fig_bar = make_bar_fig(p['inst'], feat_norm_long, cmap, height=1200, font_size=55)   # tall -> crisp when scaled down
    place(page, plotly_to_pdf_bytes(fig_bar), bar_rect, title="")

    # --- right sub-rect: pathway plot ---
    place(page, mpl_to_pdf_bytes(p['fig_path'], tight=False), tm_path, title="")
    insert_title_italic_f(
        page, (tm_path.x0 + 4, tm_path.y0 + 11),
        prefix=f"upstream paths; ",
        suffix=f" = {p['frac']:.2f}",
    )

    # - bottom-left: anterior view (raster PNG)
    title_bl = f"anterior view - bodyId: {p['bodyId']}"
    place_image(page, p['anterior_png'], bl, title=title_bl)

    # - bottom-right: RF heatmap (plotly), if a single alpha-hull was found
    if p['fig_rf'] is not None:
        place(page, plotly_to_pdf_bytes(p['fig_rf']), br, title="ARF")


# --- HTML layout knobs ------------------------------------------------------
TOP_H          = 450   # top-row panel height in px (~10% shorter than before)
CAP_FONT       = 18    # panel-title (caption) font size in px
BAR_HTML_W     = 220   # HTML bar GRAPH-DIV width in px (keep wide so tooltips don't clip)
BAR_SIDE_PAD   = 80    # L/R margin inside the bar fig: narrows the VISIBLE bar, leaves
                       # the div wide so hover tooltips still have room
BAR_HOVER_FONT = 16    # hover-tooltip font size in px


def html_from_panels(p):
    """Assemble one interactive HTML page for an instance (no kaleido needed).

    The two plotly figures (bar + RF) stay interactive; the matplotlib panels embed as
    vector SVG and the anterior render as a PNG. Layout is a 2x2 grid mirroring the PDF:
    TL stats, TR = pathway + bar, BL anterior, BR RF. Top row is a fixed (shorter)
    height; the stats table is scaled to fill its panel.
    """
    # bar fits inside the fixed-height top panel (minus padding/caption); font scales
    # with height (~55 * h/1200). The graph div stays BAR_HTML_W wide (room for the
    # tooltip) while BAR_SIDE_PAD margins narrow the actual bar.
    bar_h = TOP_H - 20
    fig_bar = make_bar_fig(p['inst'], feat_norm_long, cmap, height=bar_h,
                           font_size=round(55 * bar_h / 1200), width=BAR_HTML_W)
    fig_bar.update_layout(
        margin=dict(l=BAR_SIDE_PAD, r=BAR_SIDE_PAD, t=2, b=2),  # narrow the visible bar
        hoverlabel=dict(font_size=BAR_HOVER_FONT),
    )
    bar_div = fig_bar.to_html(full_html=False, include_plotlyjs='cdn')
    rf_div  = (p['fig_rf'].to_html(full_html=False, include_plotlyjs=False)
               if p['fig_rf'] is not None else "<em>no single alpha-hull — RF panel skipped</em>")
    stats_src = mpl_to_svg_b64(p['stats_fig'])
    path_src  = mpl_to_svg_b64(p['fig_path'])
    ant_src   = ("data:image/png;base64,"
                 + base64.b64encode(Path(p['anterior_png']).read_bytes()).decode())
    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{p['inst']}</title>
<style>
  :root {{ --top-h: {TOP_H}px; }}
  body {{ font-family: arial, sans-serif; margin: 12px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .panel {{ border: 1px solid #ddd; padding: 6px; min-width: 0; box-sizing: border-box; }}
  /* top row: fixed (shorter) height so both top panels align */
  .top {{ height: var(--top-h); }}
  /* stats table scaled to fill its panel */
  .top.stats {{ display: flex; align-items: center; justify-content: center; }}
  .top.stats img {{ height: 100%; width: auto; max-width: 100%; }}
  .tr {{ display: flex; align-items: flex-start; gap: 8px; }}
  .tr .pathway {{ flex: 1; min-width: 0; display: flex; flex-direction: column; height: 100%; }}
  .tr .pathway img {{ max-height: calc(100% - 24px); width: auto; max-width: 100%; object-fit: contain; }}
  img {{ max-width: 100%; height: auto; }}
  .cap {{ font-size: {CAP_FONT}px; font-weight: 600; color: #333; margin: 0 0 4px 0; }}
</style></head>
<body>
  <div class='grid'>
    <div class='panel top stats'><img src='{stats_src}'></div>
    <div class='panel top tr'>
      <div class='pathway'><p class='cap'>upstream paths; <i>f</i> = {p['frac']:.2f}</p><img src='{path_src}'></div>
      <div>{bar_div}</div>
    </div>
    <div class='panel'><p class='cap'>anterior view - bodyId: {p['bodyId']}</p><img src='{ant_src}'></div>
    <div class='panel'><p class='cap'>ARF</p>{rf_div}</div>
  </div>
</body></html>"""



# %% [markdown]
# ## pdf driver

# %%
# for pdf, divide inst_cb into 5 parts to render 5 pdf files
rows_ls = []
n = len(inst_cb)
for i in range(5):
    rows_ls.append(inst_cb[i*n//5:(i+1)*n//5])

print(len(rows_ls[0]), len(rows_ls[1]), len(rows_ls[2]), len(rows_ls[3]), len(rows_ls[4]))

# %%
# --- pdf driver ---------------------------------------------------------
# ~10h
# 'single_pdf'  -> one multi-page PDF for all instances (original behavior)
# 'pdf_per_row' -> one PDF file per instance
PDF_MODE = 'single_pdf'        # 'single_pdf' | 'pdf_per_row'
OUT_STEM = 'gallery_vcbn'
saved_pdfs = []                # consumed by the shrink-file-size cell below

# select instances
# rows = inst_cb            

for i, rows in enumerate(rows_ls):
    print(f"Processing part {i+1} of {len(rows_ls)}, {len(rows)} instances")

    out_dir = Path(result_dir, OUT_STEM)
    if PDF_MODE == 'pdf_per_row':
        out_dir.mkdir(parents=True, exist_ok=True)

    # Persistent Kaleido server: one Chrome reused across every fig.to_image(), so each
    # export is ~0.3s instead of paying ~5s Chrome cold-start per call. build_panels
    # always rasterizes the anterior view via save_figure_anterior (fig.to_image()).
    try:
        kaleido.stop_sync_server()
    except Exception:
        pass
    kaleido.start_sync_server()

    out = fitz.open() if PDF_MODE == 'single_pdf' else None

    for _, row in rows.iterrows():
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
        pdf_path = Path(result_dir, f'{OUT_STEM}_{i+1}.pdf')
        # deflate_images Flate-compresses the (otherwise uncompressed) anterior-view rasters
        # that dominate file size; lossless, ~13x smaller. garbage/clean prune orphans.
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
# ~7 h
# one interactive HTML page per instance (plotly bar + RF stay zoomable/hoverable;
# matplotlib panels as SVG, anterior as PNG)
OUT_STEM = 'gallery_vcbn'
rows = inst_cb            # row selector

out_dir = Path(result_dir, OUT_STEM)
pages_dir = out_dir / 'pages'          # writes pages here; index.html sits in out_dir
pages_dir.mkdir(parents=True, exist_ok=True)

# Persistent Kaleido server: build_panels always rasterizes the anterior view via
# save_figure_anterior (fig.to_image()), even though the RF/bar panels here stay
# native HTML (no kaleido export needed for those).
try:
    kaleido.stop_sync_server()
except Exception:
    pass
kaleido.start_sync_server()

import navis
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
            (pages_dir / f"{safe_name(p['inst'])}.html").write_text(html_from_panels(p), encoding='utf-8')
            print(f"Added {p['inst']} (bodyId {p['bodyId']})")
finally:
    navis.config.pbar_hide = _prev_pbar_hide

kaleido.stop_sync_server()             # release the persistent Chrome

# build the searchable index over the pages just written (pages already in pages/)
from utils.gallery_index import build_gallery_index
build_gallery_index(out_dir, move=False)
print(f"Saved HTML file(s) to {pages_dir}; open {out_dir / 'index.html'}")

# %% [markdown]
# # End


