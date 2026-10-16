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
# # Paper figures
#
# Single-pass preprocessing + plotting for every panel in `paper_figs.xlsx`.
# Each section runs its own getters (cached, so re-runs are cheap), invokes
# the relevant plots from `plotting.py`, then calls `pf.distribute('<folder>')`
# to copy the produced files into `results/<folder>/<folder>_<panel>...<ext>`.
# Plot calls that span multiple figures live in the earliest section per
# xlsx order; later sections re-run them too so each section is self-contained.
# Final cell runs `pf.parity_check(exclude_folders=['si_5', 'si_6'], expected_missing=['si_12/si_12_25.pdf', 'si_12/si_12_26.pdf'])`.
#
# The **flywire Main 2d/4d** section runs last on purpose: it monkeypatches the
# shared `core_data` / `polarity` loaders for the flywire dataset, so it must
# come after every malecns polarity panel.
#
# Most cells run under ~10 min.

# %% [markdown]
# # Load shared modules

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().resolve().parent if Path.cwd().name == 'make_figures' else Path.cwd().resolve()))

import importlib
import pandas as pd
import numpy as np

from utils import config
from utils.palettes import load_colors
from utils import cb_data, core_data, ol_data, ol_rf, vic, polarity
from utils import plotting as plot
plot = importlib.reload(plot)
from make_figures import figures as pf

plot.set_html_output(True)

# %% [markdown]
# # Table of contents
#
# *Document-order map of every section; links jump to the section anchors below.*
#
# **Main figures**
#
# - [Main Fig. 1](#main-1)
# - [Main Fig. 2](#main-2)
# - [Main Fig. 3](#main-3)
# - [Main Fig. 4](#main-4)
# - [Main Fig. 5](#main-5)
# - [Main Fig. 6](#main-6)
# - [Main Fig. 7](#main-7)
#
# **SI figures** (document order)
#
# - [SI Fig. 1](#si-1)
# - [SI Fig. 2](#si-2)
# - [SI Fig. 3](#si-3)
# - [SI Fig. 4](#si-4)
# - [SI Fig. 5 — hitting-time variants](#si-5)
# - [SI Fig. 6 — alternative-model comparison](#si-6)
# - [SI Fig. 7](#si-7)
# - [SI Fig. 8](#si-8)
# - [SI Fig. 9](#si-9)
# - [SI Fig. 10](#si-10)
# - [SI Fig. 11 & 12 — VIC projections](#si-11)
# - [SI Fig. 13](#si-13)
# - [SI Fig. 14 — flywire](#si-14)
#
# - [Parity check](#parity-check)

# %% [markdown]
# <a id="main-1"></a>
# ## Main Fig. 1

# %%
# Preprocessing
inv     = core_data.get_inventory_meta()
ol_meta = core_data.get_ol_meta()

colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()
colors = {
    'region':      colored_region['color'].to_dict(),
    'main_groups': colored_main_groups['color'].to_dict(),
    'sign':        colored_sign['color'].to_dict(),
}

# Inventory pies (1d full brain, 1e right OL)
plot.plot_inventory_full_brain(inv, colors)                                     # 1d
plot.plot_inventory_right_ol(inv, colors)                                       # 1e

# Fig 1d / 1e inventory totals
_inv_all      = inv[inv.region != "other"]
_inv_all_sign = _inv_all[~_inv_all["sign"].isna()]
_inv_ol       = inv[(inv.region == "right OL") & (inv.main_groups != "nonOL")]
_inv_ol_sign  = _inv_ol[~_inv_ol["sign"].isna()]
print(f"Fig 1d panel 1 (neurons by region):        {_inv_all['bodyId'].nunique():,} neurons")
print(f"Fig 1d panel 2 (instances by region):      {_inv_all['cell_type'].nunique():,} instances")
print(f"Fig 1d panel 3 (neurons by sign):          {_inv_all_sign['bodyId'].nunique():,} neurons")
print(f"Fig 1d panel 4 (instances by sign):        {_inv_all_sign['cell_type'].nunique():,} instances")
print(f"Fig 1e panel 1 (neurons by main_groups):   {_inv_ol['bodyId'].nunique():,} neurons")
print(f"Fig 1e panel 2 (instances by main_groups): {_inv_ol['cell_type'].nunique():,} instances")
print(f"Fig 1e panel 3 (neurons by sign):          {_inv_ol_sign['bodyId'].nunique():,} neurons")
print(f"Fig 1e panel 4 (instances by sign):        {_inv_ol_sign['cell_type'].nunique():,} instances")

# Fig 1b right: OL coordinate map
plot.plot_coords(ol_meta)                                                       # 1b right

pf.distribute('main_1')


# %% [markdown]
# <a id="main-2"></a>
# ## Main Fig. 2

# %%
# Preprocessing
meta        = core_data.get_meta()
ol_meta     = core_data.get_ol_meta()
ol_flow     = core_data.get_ol_flow_type()
flow_df     = core_data.get_flow().merge(meta[['idx','main_groups','sign']], on='idx', how='right')
flow_ol     = core_data.get_flow_ol()
conn_ol     = ol_data.get_ol_connectivity()
dir_ol      = ol_data.get_ol_directedness()
type_dir_ol = ol_data.get_ol_type_directedness()

colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()

plot.plot_flow_by_main_groups(ol_flow, conn_ol, type_dir_ol, flow_ol, colored_main_groups)  # 2b_left hit_hist, 2c_1, 2d_1

# Connection-level ff/la/fb breakdown for Fig 2d_1
_thr = 0.5
_ff = conn_ol['hitting_time_diff'] >  _thr
_la = conn_ol['hitting_time_diff'].abs() <= _thr
_fb = conn_ol['hitting_time_diff'] < -_thr
_W = conn_ol.weight.sum(); _N = len(conn_ol)
print(f'Total weight fractions:      ff {conn_ol[_ff].weight.sum()/_W:.3f}, la {conn_ol[_la].weight.sum()/_W:.3f}, fb {conn_ol[_fb].weight.sum()/_W:.3f}')
print(f'Connection-count fractions:  ff {_ff.sum()/_N:.3f}, la {_la.sum()/_N:.3f}, fb {_fb.sum()/_N:.3f}')
print(f'E/I fractions:               ff is E {conn_ol[_ff & (conn_ol.sign_pre == 1)].weight.sum()/conn_ol[_ff].weight.sum():.3f}, '
      f'la is E {conn_ol[_la & (conn_ol.sign_pre == 1)].weight.sum()/conn_ol[_la].weight.sum():.3f}, '
      f'fb is I {conn_ol[_fb & (conn_ol.sign_pre == -1)].weight.sum()/conn_ol[_fb].weight.sum():.3f}')
plot.plot_flow_by_sign(ol_flow, conn_ol, type_dir_ol, flow_ol, colored_sign)               # 2c_2, 2d_2, 2e_left fb/la/ff
plot.plot_flow_type_boxes(flow_df, dir_ol, colored_main_groups)                            # 2f ff_frac_hist
plot.plot_flow_type_scatter_and_triangle(ol_flow, type_dir_ol, colored_main_groups, colored_sign)  # 2f dir_triangle

# Mi4 / T4a pathway examples (2b right)
type_to_sign = dict(zip(ol_meta.cell_type, ol_meta.sign))
sign_to_color = colored_sign['color'].to_dict()
type_to_mg = dict(zip(ol_meta.cell_type, colored_main_groups.loc[ol_meta.main_groups, 'color']))
examples_dir = config.FIG_DIR / 'examples'
_layer_cumsum = {'T4a_R': 0.8, 'Mi4_R': 0.8}
for inst in core_data.load_layer_example_types():
    plot.plot_pathway(
        ol_data.get_paths_to_instance(inst), inst, ol_flow,
        thre_cumsum=_layer_cumsum[inst],
        neuron_to_color=type_to_mg,
        neuron_to_sign=type_to_sign, sign_color_map=sign_to_color,
        save_path=examples_dir / f'{config.FIG_DATASET}_{inst[:-2]}_pathway.pdf',
    )

pf.distribute('main_2')


# %% [markdown]
# <a id="main-3"></a>
# ## Main Fig. 3

# %%
FRAC = 0.19

# Preprocessing
feat        = ol_data.get_ol_features()
clusters    = ol_data.get_ol_clusters(frac_thre=FRAC)
type_part   = ol_data.get_ol_type_participation(frac_thre=FRAC)
intra       = ol_data.get_ol_clusters_intra()
pathways    = ol_data.get_ol_connectivity_pathways(frac_thre=FRAC)
sankey      = ol_data.get_ol_sankey_connectivity(frac_thre=FRAC, output_normalized=False, weight_thre=100, min_plot_neurons=20)
ol_meta     = core_data.get_ol_meta()
ol_flow     = core_data.get_ol_flow_type()
type_dir_ol = ol_data.get_ol_type_directedness()
cmp_ol      = polarity.get_ol_polarity_comparison()

n_clu = int(clusters.cluster.max())
cluster_num = clusters['cluster'].value_counts().sort_index()
in_instances = list(ol_meta[ol_meta.main_groups=='visual input'].instance.unique())

colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()

# Clustering / pathway plots; 
Z, thre, dendr = plot.plot_clusters_dendrograms(feat, frac_thre=FRAC)
plot.plot_clusters_correlation_matrix(feat, dendr, cluster_num)
plot.plot_clusters_example_features(feat, in_instances, [s+'_R' for s in plot._CLUSTER_TYPES], colored_seed)
# Cluster assignments for example types
_ex_types_2d = [s + '_R' for s in plot._CLUSTER_TYPES]
_ex_cls_2d = clusters[clusters['instance'].isin(_ex_types_2d)][['instance', 'cluster']].sort_values('cluster')
print('main_2_d cluster assignments:')
print(_ex_cls_2d.to_string(index=False))
print('Largest cell type per OL cluster (by neuron count):')
_top_2d = clusters.loc[clusters.groupby('cluster')['count'].idxmax()][['cluster', 'instance', 'count']].sort_values('cluster')
print(_top_2d.to_string(index=False))
plot.plot_clusters_pathway_features(feat, in_instances, clusters, cluster_num, colored_seed)
plot.plot_clusters_scatter(clusters, ol_flow, type_dir_ol, colored_main_groups, colored_sign)
plot.plot_clusters_participation(type_part, ol_meta, ol_flow, colored_main_groups, colored_sign)
plot.plot_clusters_pathway_connectivity(pathways, n_clu, font_size=27)
plot.plot_clusters_sankey(sankey, colored_main_groups, colored_seed, colored_region)
plot.plot_clusters_graph(type_part, ol_meta, colored_main_groups, colored_sign, colored_seed)

# pathway examples and participation 
type_to_sign  = dict(zip(ol_meta.cell_type, ol_meta.sign))
sign_to_color = colored_sign['color'].to_dict()
type_to_mg    = dict(zip(ol_meta.cell_type, colored_main_groups.loc[ol_meta.main_groups, 'color']))
examples_dir  = config.FIG_DIR / 'examples'
_out_cumsum = {'LC6_R': 0.4, 'LC4_R': 0.7, 'LPLC2_R': 0.6}
for inst, tc in _out_cumsum.items():
    plot.plot_pathway(
        ol_data.get_paths_to_instance(inst), inst, ol_flow,
        thre_cumsum=tc, neuron_to_color=type_to_mg,
        neuron_to_sign=type_to_sign, sign_color_map=sign_to_color,
        save_path=examples_dir / f'{config.FIG_DATASET}_{inst[:-2]}_pathway.pdf',
    )
_blues = ['#041f4a','#08306b','#08519c','#2171b5','#4292c6',
          '#6baed6','#9ecae1','#c6dbef','#deebf7']
type_to_color_part = type_to_mg | {
    s.split('_')[0]: _blues[c - 1]
    for s, c in zip(clusters['instance'], clusters['cluster'])
}
for inst in core_data.load_participation_example_types():
    plot.plot_pathway(
        ol_data.get_participation_paths(inst), inst, ol_flow,
        thre_cumsum=0.85, neuron_to_color=type_to_color_part,
        neuron_to_sign=type_to_sign, sign_color_map=sign_to_color,
        node_size=800, node_text_size=21,
        figsize=(9, 10), frac_text_pos=(0.2, 0.8), frac_text_size=21,
        save_path=examples_dir / f'{config.FIG_DATASET}_{inst[:-2]}_participation_pathway.pdf',
    )
    # Print participation fractions to all clusters
    _cc = [c for c in type_part.columns if c.startswith('c') and c[1:].isdigit()]
    _fr = type_part.loc[type_part.instance == inst, _cc]
    if len(_fr):
        _fr = _fr.iloc[0].sort_values(ascending=False)
        print(f'{inst} participation fractions to {len(_cc)} clusters (sum={_fr.sum():.3f}):')
        print(_fr.to_string(float_format=lambda x: f'{x:.4f}'))

# OL polarity bars
cmp_vpn   = cmp_ol[cmp_ol['main_groups'] == 'VPN'].reset_index(drop=True)
cmp_other = cmp_ol[cmp_ol['main_groups'] != 'VPN'].reset_index(drop=True)
_w_vpn = int(max(400, 35 * len(cmp_vpn) + 150) * 0.9 * 0.95)
plot.plot_polarity_stacked_bars(cmp_vpn,   colored_seed, filename_stem='polarity_predictions_vpn', width=_w_vpn)
_w_other = int(max(400, 35 * len(cmp_other) + 150) * 0.9 * 0.8)
plot.plot_polarity_stacked_bars(cmp_other, colored_seed, filename_stem='polarity_predictions_other', width=_w_other)

# Mismatched cell types per panel
_miss_other = cmp_other[~cmp_other.match][['instance', 'Contrast selectivity', 'predicted']]
_miss_vpn   = cmp_vpn  [~cmp_vpn.match  ][['instance', 'Contrast selectivity', 'predicted']]
print(f'Fig 2b_1 (other OL): {cmp_other.match.mean():.1%} match ({len(cmp_other)} types); {len(_miss_other)} mismatches:')
print(_miss_other.to_string(index=False))
print(f'\nFig 2b_2 (VPN / VPN): {cmp_vpn.match.mean():.1%} match ({len(cmp_vpn)} types); {len(_miss_vpn)} mismatches:')
print(_miss_vpn.to_string(index=False))

pf.distribute('main_3')

# %%
# --- OL neurons summary CSV (results/tables/) ---
_tables_dir = pf.RESULTS / 'tables'
_tables_dir.mkdir(parents=True, exist_ok=True)
_vic_ol_type = vic.get_ol_type_vic()
_raw_ol_rf   = ol_rf.get_rf_raw_ol()
# per-instance dimorphism: per-neuron property, constant within nearly every
# instance; for the few that disagree take the mode (ties -> alphabetical).
_dimorph = (
    core_data.get_meta().dropna(subset=['dimorphism'])
    .groupby('instance')['dimorphism'].agg(lambda s: s.mode().iloc[0]).reset_index()
)
_ol_csv = (
    ol_flow[['instance', 'count', 'sign', 'main_groups', 'hitting_time']]
    .copy()
    .rename(columns={'hitting_time': 'layer', 'main_groups': 'main_group'})
)
_ol_csv = _ol_csv.merge(
    type_dir_ol[['instance_pre', 'frac_ff', 'frac_la', 'frac_fb']].rename(
        columns={'instance_pre': 'instance', 'frac_la': 'frac_lat'}
    ),
    on='instance', how='left',
)
_ol_csv = _ol_csv.merge(_vic_ol_type, on='instance', how='left')
_ol_csv = _ol_csv.merge(clusters[['instance', 'cluster']], on='instance', how='left')
# ellipse_size/r2 are per-neuron medians (same aggregation as hull_size below
# and as the gallery pages), not the single type-level fit from get_rf_type_ol
_ol_csv = _ol_csv.merge(
    _raw_ol_rf.groupby('instance')[['size', 'r2']].median().reset_index(),
    on='instance', how='left',
)
_ol_csv = _ol_csv.merge(
    ol_rf.get_rf_ahull_neuron().groupby('instance')['ahull_size'].median().reset_index(),
    on='instance', how='left',
)
_ol_csv = _ol_csv.merge(_dimorph, on='instance', how='left')
_ol_csv = _ol_csv[['instance', 'count', 'sign', 'main_group', 'layer',
                    'frac_fb', 'frac_lat', 'frac_ff', 'VIC', 'cluster', 'size', 'r2',
                    'ahull_size', 'dimorphism']]
_ol_csv = _ol_csv.rename(columns={'cluster': 'oc_class', 'size': 'ellipse_size',
                                  'ahull_size': 'hull_size'})
_ol_csv.to_csv(_tables_dir / 'TableS1_VPN_in_out_summary.csv', index=False)
print(f"Saved OL neurons CSV ({len(_ol_csv)} rows) -> results/tables/TableS1_VPN_in_out_summary.csv")

# %% [markdown]
# <a id="main-4"></a>
# ## Main Fig. 4

# %%
FRAC = 0.19

raw       = ol_rf.get_rf_raw_ol()
type_ol   = ol_rf.get_rf_type_ol()
edges_ol  = ol_rf.get_rf_connectivity_edges_ol(include_cb_post=False)
edges_cb  = ol_rf.get_rf_connectivity_edges_ol(include_cb_post=True)
ol_meta   = core_data.get_ol_meta()
stepsn    = ol_data.get_ol_stepsn_sum()
type_part = ol_data.get_ol_type_participation(frac_thre=FRAC)
clusters  = ol_data.get_ol_clusters(frac_thre=FRAC)
cmp_body  = ol_rf.get_rf_comparison_body()
cmp_type  = ol_rf.get_rf_comparison_type()
exp_sz    = ol_rf.get_experimental_rf_sizes()
exp_cc    = ol_rf.get_experimental_rf_sizes(sheet='RF size CC')
vic_type  = vic.get_ol_cb_vic_type().merge(
    core_data.get_flow().groupby('instance')['hitting_time'].median().reset_index(),
    on='instance', how='left',
)
rf_all   = ol_rf.get_rf_types_combined()
rf_vpn_vcbn = rf_all[(rf_all['main_groups'] == 'VPN') | (rf_all['region'] == 'CB')]

colored_region, colored_main_groups, colored_sign, _ = load_colors()
n_clu = int(clusters['cluster'].max())
cluster_names = [f'c{i}' for i in range(1, n_clu + 1)]
in_instances  = ['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R']
EXAMPLE_BID = 30134

# Main 4a: example RF positions/fits 
plot.plot_rf_example(ol_meta, stepsn, raw, EXAMPLE_BID, in_instances)
plot.plot_rf_example_pos_par(ol_meta, stepsn, raw, EXAMPLE_BID, in_instances)
# Main 4a_5 
plot.plot_rf_positions_across_neurons(raw, 'LC6_R', example_bid=EXAMPLE_BID)
# Main 4c: direct-input RFs for LC6_R
roi_counts = pd.read_pickle(
    config.DATA_DIR / f'{config.DATASET}_OL_{config.SIDE_CHAR}_roi_counts.pkl'
)
input_raw = ol_rf.get_input_rf_raw_ol()
plot.plot_input_rf_example(roi_counts, input_raw, EXAMPLE_BID, 'LC6_R')
# Main 4d, 4e: experimental size comparisons
plot.plot_rf_size_example_comparison(cmp_body, star_instance='LC6_R', example_bid=EXAMPLE_BID)
plot.plot_rf_size_type_scatters(cmp_type, colored_main_groups)
plot.plot_rf_size_vs_experiment(cmp_type, exp_sz, colored_main_groups, colored_region)
plot.plot_rf_size_vs_experiment_cc(type_ol, exp_cc, colored_main_groups, colored_region)
# Main 4f: scatters + histograms by main groups / by sign
plot_cell_types = [s + '_R' for s in plot._FLOW_PLOT_TYPES]
plot.plot_rf_boxes_by_celltype(raw, plot_cell_types, colored_main_groups)
# Highlight the default RF example types plus three extra example types
# (Pm12, MeVPMe12, Dm8a) as labelled dots in the Main 4f/g scatter panels.
_rf_scatter_stars = (
    [s + '_R' for s in plot._RF_STAR_EXAMPLES]
    + ['Pm12_R', 'MeVPMe12_R', 'Dm8a_R']
)
plot.plot_rf_scatters_by_main_groups(
    type_ol, colored_main_groups, colored_sign,
    star_types=_rf_scatter_stars, star_types_output=_rf_scatter_stars,
)
plot.plot_rf_scatters_by_cluster(
    type_ol, type_part, cluster_names=cluster_names,
    colored_main_groups_df=colored_main_groups,
)
plot.plot_rf_histograms(
    type_ol, type_part, cluster_names=cluster_names,
    colored_main_groups_df=colored_main_groups,
    colored_sign_df=colored_sign,
)
# Main 4g: hitting/no/amp vs size + cb_vic_vs_size
# (cb_vic_vs_size produced by plot_cb_rf_size_summary in CB-RF block)
plot.plot_cb_rf_size_summary(rf_vpn_vcbn, colored_region)
# Main 4h: sizediff internal + output
plot.plot_rf_sizediff_internal(edges_ol, colored_main_groups)
plot.plot_rf_sizediff_output(edges_cb, colored_region)
# Main 4i: connectivity boxes
plot.plot_rf_connectivity_boxes(edges_ol, colored_main_groups, colored_sign)

pf.distribute('main_4')

# %% [markdown]
# <a id="main-5"></a>
# ## Main Fig. 5

# %%
FRAC_CB = 0.17
FRAC_OL = 0.19

feat_r       = cb_data.get_cb_features(side_char='r')
cl_r         = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB)
conn         = cb_data.get_ol_cb_cluster_connectivity(ol_frac_thre=FRAC_OL, cb_frac_thre=FRAC_CB)
layer_lr     = cb_data.get_cb_layer_lr()
vic_type_cb  = vic.get_ol_cb_vic_type().merge(
    core_data.get_flow().groupby('instance')['hitting_time'].median().reset_index(),
    on='instance', how='left',
)
tbar_counts  = cb_data.get_cb_cluster_tbars_per_roi(side_char='r', cb_frac_thre=FRAC_CB)
_tbar_roi_keep = [r for r in plot._CB_ROI_ORDER if r in tbar_counts.columns]
_tbar_total_all = tbar_counts.values.sum()
_tbar_total_shown = tbar_counts[_tbar_roi_keep].values.sum()
_tbar_frac_shown = _tbar_total_shown / _tbar_total_all if _tbar_total_all > 0 else 0.0
print(f"Main 5g top displayed connection fraction: {_tbar_frac_shown:.3f} ({_tbar_total_shown:,.0f}/{_tbar_total_all:,.0f})")
type_part_cb_out = cb_data.get_ol_in_cb_type_participation(side_char='r', cb_frac_thre=FRAC_CB)
ol_clu       = ol_data.get_ol_clusters(side_char='r', frac_thre=FRAC_OL)
_meta        = core_data.get_meta()
flow_type_full = core_data.get_flow().groupby('instance')['hitting_time'].median().reset_index()

colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()
cluster_num   = cl_r['cluster'].value_counts().sort_index()
in_instances  = [c for c in feat_r.columns if c.endswith('_R')]
cb_example_types = plot._CB_CLUSTER_TYPES[config.SIDE_CHAR]

VIC_THRE = 5e-4
_vic_raw = vic.get_ol_cb_vic_raw()
_vpn_above_raw  = _vic_raw[(_vic_raw.main_groups == 'VPN') & (_vic_raw.VIC > VIC_THRE)]
_cb_above_raw   = _vic_raw[(_vic_raw.region == 'CB') & (_vic_raw.VIC > VIC_THRE)]
_vpn_above_type = vic_type_cb[(vic_type_cb.region == 'right OL') & (vic_type_cb.VIC > VIC_THRE)]
_cb_above_type  = vic_type_cb[(vic_type_cb.region == 'CB') & (vic_type_cb.VIC > VIC_THRE)]
print(f"VPN above VIC threshold ({VIC_THRE:.0e}): {_vpn_above_raw['bodyId'].nunique():,} neurons, {_vpn_above_type['instance'].nunique():,} cell types")
print(f"CB  above VIC threshold ({VIC_THRE:.0e}): {_cb_above_raw['bodyId'].nunique():,} neurons, {_cb_above_type['instance'].nunique():,} cell types")

# Visual-sensitivity labels (TableS2 'CBN visual sensitivity' tab): 'Y...' =
# visually sensitive (highlighted in 5a), 'N...' = not visually sensitive (used
# only for the VIC-threshold mismatch check below). Names with '?' are skipped;
# '*' is a prefix wildcard (e.g. ER3* -> ER3a..ER3w).
_lit_df = pd.read_excel(
    config.PARAMS_DIR / 'TableS2_experimental_data.xlsx',
    sheet_name='CBN visual sensitivity',
)
_vic_instances = set(vic_type_cb['instance'].values)
_suffix = f'_{config.SIDE_CHAR.upper()}'
_bare_types = {i[:-len(_suffix)] for i in _vic_instances if isinstance(i, str) and i.endswith(_suffix)}

def _resolve_light_types(_field):
    _out = []
    for _t in str(_field).split(','):
        _t = _t.strip()
        if not _t or '?' in _t:
            continue
        if '*' in _t:  # wildcard: prefix match, exclude '_unclear'
            _pre = _t.split('*', 1)[0]
            _out += sorted(b for b in _bare_types if b.startswith(_pre) and 'unclear' not in b)
        else:
            _out.append(_t)
    return _out

_light_pos, _light_neg = [], []
for _, _row in _lit_df.iterrows():
    _bucket = _light_neg if str(_row['Visual sensitivity']).strip().upper().startswith('N') else _light_pos
    _bucket += _resolve_light_types(_row['Neuron type'])
_light_pos = list(dict.fromkeys(_light_pos))  # deduplicate, preserve order
_light_neg = list(dict.fromkeys(_light_neg))
_light_highlight = [t + _suffix for t in _light_pos if t + _suffix in _vic_instances]
# 4a highlights light-sensitive VCBNs only. A few CBN-tab types (s-LNv, l-LNv,
# 5thsLNv_LNd6, SLP249) are classified VPN/right-OL in the connectome, so drop
# any highlight that is not a CB-region (VCBN) instance.
_cb_instances = set(vic_type_cb.loc[vic_type_cb['region'] == 'CB', 'instance'])
_light_highlight = [i for i in _light_highlight if i in _cb_instances]
print(f"Visual-sensitive highlight types found in VIC data: {len(_light_highlight)}")

# VIC vs visual-sensitivity mismatches (per-instance median VIC vs VIC_THRE)
_vic_by_inst = vic_type_cb.groupby('instance')['VIC'].max()
_neg_inst = [t + _suffix for t in _light_neg if t + _suffix in _vic_instances]
_mm_fp = sorted(((i, _vic_by_inst[i]) for i in _neg_inst if _vic_by_inst[i] > VIC_THRE), key=lambda x: -x[1])
_mm_fn = sorted(((i, _vic_by_inst[i]) for i in _light_highlight if _vic_by_inst[i] < VIC_THRE), key=lambda x: x[1])
print(f"Mismatch VIC > {VIC_THRE:.0e} but labeled 'N': {[f'{i} ({v:.1e})' for i, v in _mm_fp]}")
print(f"Mismatch VIC < {VIC_THRE:.0e} but labeled 'Y': {[f'{i} ({v:.1e})' for i, v in _mm_fn]}")

# VIC scatter / layer histogram / cumsum
# VPNs colored by VPN blue (main_groups), CB stays dark gray (region).
_vic_colors = colored_region.copy()
_vic_colors.loc['right OL', 'color'] = colored_main_groups.loc['VPN', 'color']
plot.plot_vic_vs_layer_scatter(vic_type_cb, _vic_colors,
    highlight=_light_highlight,
    show_highlight_labels=False)
plot.plot_vic_layer_histogram(vic_type_cb, _vic_colors)
plot.plot_vic_cumsum(vic_type_cb, _vic_colors)
# Main 4c, 4d top: in-out examples + pathways for CB clusters
plot.plot_cb_clusters_example_features(feat_r, in_instances, cb_example_types, colored_seed)
# Cluster assignments for example types
_ex_cls_5c = cl_r[cl_r['instance'].isin(cb_example_types)][['instance', 'cluster']].sort_values('cluster')
print('main_5_c cluster assignments:')
print(_ex_cls_5c.to_string(index=False))
print('Largest cell type per CB cluster (by neuron count):')
_top_5c = cl_r.loc[cl_r.groupby('cluster')['count'].idxmax()][['cluster', 'instance', 'count']].sort_values('cluster')
print(_top_5c.to_string(index=False))
plot.plot_cb_clusters_pathway_features(feat_r, in_instances, cl_r, cluster_num, colored_seed)
# Main 5d bottom + SI: CB dendrograms
plot.plot_clusters_dendrograms(feat_r, frac_thre=FRAC_CB, side_str='_cb', save_dir=config.FIG_DIR / 'CB_pathways')
# Main 5e: CB cluster vs layer scatter
plot.plot_cb_clusters_scatter(cl_r, layer_lr, colored_region, highlight=cb_example_types)
# Main 5f: CB cluster graph
_fig4f_types = ["aMe4", "LPT111", "VS", "aMe12", "LC4", "LC6", "LPLC2", "T4a", "Tm5a"]
plot.plot_cb_clusters_graph(type_part_cb_out, ol_clu, colored_region, ol_highlight=_fig4f_types)
# CB cluster x ROI heatmap
plot.plot_cb_clusters_rois(tbar_counts)

# pC1_2a / pC1_5b CB pathway examples (Main 5b right)
type_to_sign_full  = dict(zip(_meta.cell_type, _meta.sign))
sign_to_color_full = colored_sign['color'].to_dict()
_cb_color_df = pd.concat([
    pd.DataFrame(colored_region.loc['CB', 'color'], index=['nonOL'], columns=['color']),
    colored_main_groups,
])
type_to_color_cb = dict(zip(_meta.cell_type, _cb_color_df.loc[_meta.main_groups, 'color']))
examples_dir_cb = config.FIG_DIR / 'examples'
for inst, thre_cumsum, h_scale in [('pC1_2a_R', 0.6, 0.64), ('pC1_5b_R', 0.2, 0.65)]:
    paths = cb_data.get_cb_paths_to_instance(inst)
    hit_inst = paths['hit_inst']
    plot.plot_pathway(
        paths, inst, flow_type_full,
        thre_cumsum=thre_cumsum,
        figsize=(1.5 + hit_inst * 3, 8 * h_scale),
        neuron_to_color=type_to_color_cb,
        neuron_to_sign=type_to_sign_full, sign_color_map=sign_to_color_full,
        node_size=1400, node_text_size=18, highlight=False,
        save_path=examples_dir_cb / f'{config.FIG_DATASET}_{inst[:-2]}_pathway.pdf',
    )

# CB polarity prediction bars
cmp_cb = polarity.get_cb_polarity_comparison()
plot.plot_polarity_stacked_bars(
    cmp_cb, colored_seed, filename_stem='polarity_predictions_cb',
    width=int(max(400, 35 * len(cmp_cb) + 150) * 0.9), font_size=12,
)
_miss_cb = cmp_cb[~cmp_cb.match][['instance', 'Contrast selectivity', 'predicted']]
print(f'Fig 4d (CB): {cmp_cb.match.mean():.1%} match ({len(cmp_cb)} types); {len(_miss_cb)} mismatches:')
print(_miss_cb.to_string(index=False))

# Main 5d (2nd file): CB experimental RF size vs. arbor alpha-hull size
# (companion to Main 3e, which compares measured 2D-Gaussian ellipse RF size to
# propagated RF size for OL/VPN types). Here, for literature cell types whose
# connectome region is CB:
#   x = measured RF size (deg^2; TableS2 'RF size' sheet, 'size' column)
#   y = per-type median arbor alpha-hull size (col^2) x 25 -> deg^2
#       (right-side neurons in meta_VP_CB_ahull_cumsum60.pkl, region == 'CB')
# Comma-separated spreadsheet entries are split and `*` wildcards expanded.
from scipy.stats import pearsonr
from utils.polarity import _expand_wildcards

# get_rf_raw_cb is all-VCBN (== all CB region), carrying cell_type + per-body ahull_size
_ahull_cb = ol_rf.get_rf_raw_cb()
_ahull_cb = _ahull_cb[_ahull_cb['instance'].str.endswith('_R')].dropna(subset=['ahull_size'])
_ahull_cb_type = _ahull_cb.groupby('cell_type')['ahull_size'].median()

_exp_rf = pd.read_excel(
    config.PARAMS_DIR / 'TableS2_experimental_data.xlsx', sheet_name='RF size',
)
_exp_rf = _exp_rf[_exp_rf['Neuron type'].notna()]
_exp_split = pd.DataFrame([
    {'cell_type': _ct, 'size_exp_deg': _row['size']}
    for _, _row in _exp_rf.iterrows()
    for _part in str(_row['Neuron type']).split(',')
    for _ct in _expand_wildcards([_part.strip()], _ahull_cb_type.index)
])

# Keep only CB-region types; convert ahull col^2 -> deg^2 (x 25).
_cb_size = _exp_split[_exp_split['cell_type'].isin(_ahull_cb_type.index)].copy()
_cb_size['ahull_size_deg'] = _cb_size['cell_type'].map(_ahull_cb_type) * 25.0
_cb_size = _cb_size.dropna(subset=['size_exp_deg', 'ahull_size_deg'])
_cb_size = _cb_size[(_cb_size['size_exp_deg'] > 0) & (_cb_size['ahull_size_deg'] > 0)]
_cb_size['region'] = 'CB'

# Scatter mirroring: log-log, CB region color, y=x +/- 2x guides.
_lo = 10 ** np.floor(np.log10(
    min(_cb_size['size_exp_deg'].min(), _cb_size['ahull_size_deg'].min())))
_hi = 892 * 25
_fig_cb_ahull = plot._scatter(
    _cb_size, 'size_exp_deg', 'ahull_size_deg', 'region', 'cell_type',
    colored_region, list(_cb_size['cell_type'].unique()),
    height=600, width=600, size=12,
)
plot._add_2x_guide_lines(_fig_cb_ahull, _lo, _hi)
_fig_cb_ahull.update_layout(
    xaxis_title='measured RF size (deg^2)',
    yaxis_title='predicted ARF (deg^2)',
    xaxis_title_standoff=0, yaxis_title_standoff=2,
)
_decade_ticks = [10 ** _k for _k in range(round(np.log10(_lo)), round(np.log10(_hi)) + 1)]
_fig_cb_ahull.update_xaxes(type='log', tickmode='array', tickvals=_decade_ticks,
                           tickformat='d', range=[np.log10(_lo), np.log10(_hi)])
_fig_cb_ahull.update_yaxes(type='log', tickmode='array', tickvals=_decade_ticks,
                           tickformat='d', range=[np.log10(_lo), np.log10(_hi)])

_r_lin, _p_lin = pearsonr(_cb_size['size_exp_deg'], _cb_size['ahull_size_deg'])
_r_log, _p_log = pearsonr(np.log10(_cb_size['size_exp_deg']), np.log10(_cb_size['ahull_size_deg']))
print(f'Fig 5d (CB) exp-size vs ahull: {len(_cb_size)} points, '
      f'{_cb_size["cell_type"].nunique()} CB types; '
      f'Pearson r={_r_lin:.3f} (p={_p_lin:.1e}), log10 r={_r_log:.3f}')
_fig_cb_ahull.show()
plot._save(_fig_cb_ahull, config.FIG_DIR / 'exp_comparison',
           f'{config.FIG_DATASET}_exp_size_ahull_comp_cb.pdf', html=True)



# %%
# --- VCBNs summary CSV (results/tables/) ---
_tables_dir = pf.RESULTS / 'tables'
_tables_dir.mkdir(parents=True, exist_ok=True)
_type_dir_cb_4 = cb_data.get_cb_type_directedness()
_rf_cb_raw     = ol_rf.get_rf_raw_cb()
_vcbn_meta     = _meta[core_data.is_cb_neuron(_meta)][['instance', 'sign']].drop_duplicates('instance')
_count_all     = core_data.get_flow()[['instance', 'count']].drop_duplicates('instance')
# per-instance dimorphism: per-neuron property, constant within nearly every
# instance; for the few that disagree take the mode (ties -> alphabetical).
_dimorph = (
    core_data.get_meta().dropna(subset=['dimorphism'])
    .groupby('instance')['dimorphism'].agg(lambda s: s.mode().iloc[0]).reset_index()
)
_vcbn_csv = (
    _vcbn_meta
    .merge(_count_all, on='instance', how='left')
    .merge(flow_type_full.rename(columns={'hitting_time': 'layer'}), on='instance', how='left')
    .merge(
        _type_dir_cb_4[['instance_pre', 'frac_ff', 'frac_la', 'frac_fb']].rename(
            columns={'instance_pre': 'instance', 'frac_la': 'frac_lat'}
        ),
        on='instance', how='left',
    )
)
_vic_vcbn = vic_type_cb[vic_type_cb['region'] == 'CB'][['instance', 'VIC']]
_vcbn_csv = _vcbn_csv.merge(_vic_vcbn, on='instance', how='left')
_vcbn_csv = _vcbn_csv.merge(cl_r[['instance', 'cluster']], on='instance', how='left')
# ellipse_size/r2 are per-neuron medians (same aggregation as hull_size below
# and as TableS1), not the single type-level fit from get_rf_type_cb
_vcbn_csv = _vcbn_csv.merge(
    _rf_cb_raw.groupby('instance')[['size', 'r2']].median().reset_index(),
    on='instance', how='left',
)
_vcbn_csv = _vcbn_csv.merge(
    ol_rf.get_rf_ahull_neuron().groupby('instance')['ahull_size'].median().reset_index(),
    on='instance', how='left',
)
_vcbn_csv = _vcbn_csv.merge(_dimorph, on='instance', how='left')
_vcbn_csv = _vcbn_csv[['instance', 'count', 'sign', 'layer',
                        'frac_fb', 'frac_lat', 'frac_ff', 'VIC', 'cluster', 'size', 'r2',
                        'ahull_size', 'dimorphism']]
# full CB table (all instances); name kept (the dimorphism notebook reads it)
_vcbn_csv.to_csv(_tables_dir / f'{config.DATASET}_VCBNs.csv', index=False)
print(f"Saved VCBNs CSV ({len(_vcbn_csv)} rows) -> results/tables/{config.DATASET}_VCBNs.csv")
# TableS3 = visual subset (VIC > 5e-4)
_vcbn_visual = _vcbn_csv[_vcbn_csv['VIC'] > 5e-4].reset_index(drop=True)
_vcbn_visual = _vcbn_visual.rename(columns={'cluster': 'cc_class', 'size': 'ellipse_size',
                                            'ahull_size': 'hull_size'})
_vcbn_visual.to_csv(_tables_dir / 'TableS3_VCBN_in_out_summary.csv', index=False)
print(f"Saved visual VCBNs CSV ({len(_vcbn_visual)} rows, VIC > 5e-4) -> results/tables/TableS3_VCBN_in_out_summary.csv")

# %%
# ==========================================================================
# Visual-input contribution / resolution -- shared setup for the VIC
# projection panels (Main Fig. 5g/5h, 6a, 7c-right and SI Fig. 11/12).
# Loaded once here, the first consumer; the Main Fig. 6/7 and SI Fig. 11/12
# panels reuse `syn0`, `outlines_bkgd`, `plot_dir`, `roi_col` and `olc` from
# this cell, so run this section before them. Their PDFs go to
# results/quan_propagation/ and are distributed by pf.distribute() like any
# other panel; only the results/html_figures/ viewer entries are added by
# hand, since these panels are PDF-only (no .html sidecar).
#
# Heavy run-once neuprint caches (built by setup_data): tbar
# locations cache/data/tbar_vp_cb.pkl (~100 min, s5_tbar) + per-ROI tbar
# counts results/quan_propagation/tbar_count_roi_cb.csv (~9 h); the effwt /
# ahull / vp_cb tables come from s4_assemble_vpn_vcbn.
# ==========================================================================
from utils import olc_client
from utils.config import DATA_DIR, FIG_DIR, PARAMS_DIR
from utils.palettes import load_roi_colors
from utils.projection_plots import plot_contribution_layers

olc = olc_client.connect(verbose=False)

plot_dir = FIG_DIR / 'quan_propagation'
plot_dir.mkdir(parents=True, exist_ok=True)
roi_col = load_roi_colors()

outline_1 = pd.read_csv(PARAMS_DIR / 'outline_cb_1.csv', index_col=False)
outline_2 = pd.read_csv(PARAMS_DIR / 'outline_cb_2.csv', index_col=False)
outlines_bkgd = [outline_1, outline_2]

# tbar synapses (heavy query built by setup_data/s5_tbar)
syn0 = pd.read_pickle(DATA_DIR / 'tbar_vp_cb.pkl')

# --- Main Fig. 5g: VIC, all 5 input channels, right side ---
lr = 'r'
meta_cb_vpn = vic.load_vp_cb(lr)
syn = syn0.merge(
    meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'main_groups']], on='bodyId')

top_rois_ls, tbar_counts = plot_contribution_layers(
    syn, 'vision', plot_dir=plot_dir, outlines=outlines_bkgd,
    vmax={'VPN': [0.26], 'nonOL': [0.11, 0.06, 0.02]},
    fname=lambda gp, lo, hi: f'vic_ht_{gp}_{lo}_{hi}_{lr}.pdf')

# top-N ROI accounting: fraction of tbars captured by the top-6 ROIs per layer
tbar_count_roi_cb = pd.read_csv(DATA_DIR / 'tbar_count_roi_cb.csv')
top_rois_df = pd.DataFrame(top_rois_ls).fillna(0)
top_rois_perc_df = top_rois_df.copy()
for roi in top_rois_df.columns:
    cnt = tbar_count_roi_cb.loc[tbar_count_roi_cb['roi'] == roi, 'tbar_count'].values
    top_rois_perc_df[roi] = top_rois_df[roi] / cnt[0]
top_rois_df.to_csv(plot_dir / f'top_rois{lr}.csv', index=False)
top_rois_perc_df.round(3).to_csv(plot_dir / f'top_rois_perc{lr}.csv', index=False)
top_rois_df.sum(axis=1) / tbar_counts

# --- Main Fig. 5h: 2D extreme-value projection, binocular (DOB) ---
# combine right + left VIC into per-neuron DOB (degree of binocularity)
r = vic.load_vp_cb('r')
l = vic.load_vp_cb('l')
meta_cb_vpn = r[['bodyId', 'instance', 'main_groups', 'ht', 'vision']].merge(
    l[['bodyId', 'instance', 'main_groups', 'ht', 'vision']],
    on=['bodyId', 'instance'], how='outer', suffixes=('_r', '_l'))
# CB only
meta_cb_vpn = meta_cb_vpn[(meta_cb_vpn.main_groups_r == 'nonOL')
                          | (meta_cb_vpn.main_groups_l == 'nonOL')]
meta_cb_vpn['vision'] = meta_cb_vpn[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
# dob = min/max of the two sides (0 = monocular, 1 = fully binocular)
meta_cb_vpn['dob'] = (
    meta_cb_vpn[['vision_l', 'vision_r']].min(axis=1, skipna=False)
    / meta_cb_vpn[['vision_l', 'vision_r']].max(axis=1, skipna=False)).fillna(0)
meta_cb_vpn['ht'] = meta_cb_vpn[['ht_l', 'ht_r']].min(axis=1, skipna=True)  # earliest side
meta_cb_vpn['main_groups'] = meta_cb_vpn['main_groups_r'].combine_first(meta_cb_vpn['main_groups_l'])

syn = syn0.merge(
    meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'dob', 'main_groups']], on='bodyId')

plot_contribution_layers(
    syn, 'dob', plot_dir=plot_dir, outlines=outlines_bkgd,
    gp_layers={'nonOL': [2, 3.5, 4, 5]}, vmax=1, cmap_div=True,
    fname=lambda gp, lo, hi: f'dob_union_max_ht_{gp}_{lo}_{hi}.pdf')

# cumulative DOB per ROI, per layer (top-6 roi_bases by high-DOB tbar count)
N = 6
perc_shown = []
for lo, hi in [(2, 3.5), (3.5, 4), (4, 5)]:
    df_plot = syn[(syn.main_groups == 'nonOL') & (syn.ht > lo) & (syn.ht <= hi)].copy()
    df_plot['roi_base'] = (df_plot.roi.str.replace('(R)', '', regex=False)
                                     .str.replace('(L)', '', regex=False))
    hi_bases = (df_plot[df_plot.dob > 0.5].groupby('roi_base').size()
                .sort_values(ascending=False).index[:N])
    curves = []
    for rb in hi_bases:
        sub = df_plot[df_plot.roi_base == rb]
        color = roi_col.loc[roi_col.ROI == rb, 'rgba'].values[0]
        curves.append((f'{rb}_n={len(sub) / 1e3:.0f}k', color, sub.dob.values, len(sub)))
    plot.plot_cumfrac_by_roi(
        curves, ascending=False, xlabel='DOB', xlim=(1, 0), drop_zero=True,
        save_dir=plot_dir, filename=f'dob_cum_top{N}_roi_{lo}_{hi}.pdf')
    perc_shown.append(sum(len(df_plot[df_plot.roi_base == rb]) for rb in hi_bases) / len(df_plot))
print('DOB f=', perc_shown)

pf.distribute('main_5')

# %% [markdown]
# <a id="main-6"></a>
# ## Main Fig. 6

# %%
# ~16 min
coverage = cb_data.get_ol_roi_coverage(seed=0)
sectors  = core_data.get_sector_map()

# Loops over all ROIs producing {full, relative, symbol}.pdf each.
# Main 6b left = ME(R)_{full,relative,symbol}; Main 6b right = symbol files of every other ROI.
plot.plot_ol_roi_coverage_per_roi(coverage, sectors)


# %%
# --- Main Fig. 6a: resolution (arbor alpha-hull area) ---
# Reuses the VIC-projection setup (syn0, outlines_bkgd, plot_dir, roi_col)
# loaded in the Main Fig. 5 section above.
lr = 'r'
# resolution panel uses no VIC-median filter (vic_filter=False)
vp_cb = vic.load_vp_cb(lr, vic_filter=False)
# per-neuron alpha hull for VPN + VCBN, from the combined RF parquet
ahull = pd.concat([
    ol_rf.get_rf_raw_ol().query("main_groups == 'VPN'"),
    ol_rf.get_rf_raw_cb(),
], ignore_index=True)
ahull = ahull[(ahull.ahull_rmfrac >= 0.6) & ahull.ahull_size.notna()]
ahull = ahull[ahull.ahull_size / ahull.ahull_kept_points < 3]
meta_cb_vpn = (
    ahull[['bodyId', 'instance', 'ahull_size']]
    .merge(vp_cb[['bodyId', 'vision', 'ht', 'main_groups']], on='bodyId', how='inner')
    .rename(columns={'ahull_size': 'area_fit'}))
syn = syn0.merge(
    meta_cb_vpn[['bodyId', 'instance', 'vision', 'ht', 'area_fit', 'main_groups']], on='bodyId')

plot_contribution_layers(
    syn, 'area_fit', plot_dir=plot_dir, outlines=outlines_bkgd,
    vmin=5, vmax=400, agg='smallest', im_norm='log',
    fname=lambda gp, lo, hi: f'res_ht_{gp}_{lo}_{hi}_{lr}.pdf')

# cumulative area_fit per ROI, per layer (top-6 ROIs by tbar count)
N = 6
top_rois_ls, tbar_counts = [], []
for gp, edges in {'VPN': [1, 5], 'nonOL': [2, 3.5, 4, 5]}.items():
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        df_plot = syn[(syn.main_groups == gp) & (syn.ht > lo) & (syn.ht <= hi)]
        top = df_plot.groupby('roi').size().sort_values(ascending=False).iloc[:N]
        top_rois_ls.append(top)
        tbar_counts.append(len(df_plot))
        curves = []
        for roi in top.index:
            sub = df_plot[df_plot.roi == roi]
            color = roi_col.loc[roi_col.ROI == roi.replace('(R)', ''), 'rgba'].values[0]
            curves.append((f'{roi}_n={len(sub) / 1e3:.0f}k', color, sub.area_fit.values, len(sub)))
        plot.plot_cumfrac_by_roi(
            curves, ascending=True, xlabel='Area Fit', xlim=(0, 400),
            save_dir=plot_dir, filename=f'res_cum_top{N}_roi_{lo}_{hi}.pdf')
top_rois_df = pd.DataFrame(top_rois_ls).fillna(0)
top_rois_df.sum(axis=1) / tbar_counts

pf.distribute('main_6')

# %% [markdown]
# <a id="main-7"></a>
# ## Main Fig. 7

# %%
# Main Fig. 7a: full visual-system Sankey (visual inputs -> OL/VPN & CB/VCBN groups)
FRAC_OL = 0.19
FRAC_CB = 0.17
visual_sankey = cb_data.get_visual_system_sankey_connectivity(
    ol_frac_thre=FRAC_OL, cb_frac_thre=FRAC_CB,
    weight_thre=1000, min_plot_neurons=120, output_normalized=False,
)
colored_region, colored_main_groups, _, colored_seed = load_colors()
plot.plot_visual_system_sankey(visual_sankey, colored_main_groups, colored_seed, colored_region)

# Main Fig. 7b: RF size per layer (per-neuron alpha-hull ARF from the ahull pkls)
rf_ahull = ol_rf.get_rf_ahull_neuron()
plot.plot_rf_size_per_layer(rf_ahull)

# Per layer bin, the cell type whose median ahull ARF is closest to the
# per-neuron bin median (mirrors the binning in plot.plot_rf_size_per_layer).
_b6 = rf_ahull.copy()
_bins6 = np.arange(-0.25, 6 + 0.25 + 1e-9, 0.5)
_labels6 = (_bins6[1:] + _bins6[:-1]) / 2
_b6["bin"] = pd.cut(_b6["hitting_time"].clip(upper=5.0), bins=_bins6, labels=_labels6).astype(float)
_b6 = _b6.dropna(subset=["bin", "ahull_size"])
_b6["size_deg"] = _b6["ahull_size"] * 25.0
for _lyr in [v for v in _labels6 if 1.0 <= v <= 5.0]:
    _g6 = _b6[_b6["bin"] == _lyr]
    if _g6.empty:
        continue
    _med6 = _g6["size_deg"].median()
    _tmed6 = _g6.groupby("cell_type")["size_deg"].median()
    _ct6 = _tmed6.index[(_tmed6 - _med6).abs().argmin()]
    print(f"layer {_lyr:g}: median ARF {_med6:.0f} deg2 -> closest type "
          f"{_ct6} ({_tmed6[_ct6]:.0f} deg2)")

# --- Main Fig. 7c (left): sexual dimorphism panel (CB neurons) ---
# Paper panel `dimorphic_neurons_paper_panel.pdf`: per dimorphism category
# (male-specific, sexually dimorphic), the fraction of CB neurons that are
# visual (VIC > 5e-4, lighter) vs non-visual (darker). `dimorphism` comes
# from the meta table; per-neuron VIC from `vic.get_ol_cb_vic_raw`.
# Exploratory outputs from the former dimorphism_analysis notebook were
# dropped -- only this panel is in paper_figs.xlsx; the flywire analogue is
# the SI Fig. 14 section.

import matplotlib.pyplot as plt
from utils import vic

VIC_THRE = 5e-4
CATEGORIES = ["male-specific", "sexually dimorphic"]
DIM_DIR = pf.RESULTS / "dimorphism"
DIM_DIR.mkdir(parents=True, exist_ok=True)

# per-neuron CB table: dimorphism (from meta) + VIC -> visual flag
_meta = core_data.get_meta()
nrn = _meta.loc[
    (_meta["region"] == "CB") & _meta["dimorphism"].isin(CATEGORIES),
    ["bodyId", "dimorphism"],
].copy()
nrn = nrn.merge(vic.get_ol_cb_vic_raw()[["bodyId", "VIC"]], on="bodyId", how="left")
nrn["VIC"] = nrn["VIC"].fillna(0.0)            # CB body absent from VIC table -> 0
nrn["visual"] = nrn["VIC"] > VIC_THRE

n_by_cat = nrn["dimorphism"].value_counts()
vis_frac = nrn.groupby("dimorphism")["visual"].mean().reindex(CATEGORIES)
print(f"per-neuron visual fraction (VIC > {VIC_THRE:g}):")
for cat in CATEGORIES:
    n_vis = int(nrn[(nrn.dimorphism == cat) & nrn.visual].shape[0])
    print(f"  {cat}: {vis_frac[cat] * 100:.1f}% visual ({n_vis}/{int(n_by_cat[cat])})")

# male-specific = pink, sexually dimorphic = teal; visual = lighter, non-visual = darker
CAT_BAR = {
    "male-specific":      {"visual": "#f1b6da", "nonvisual": "#c51b7d"},
    "sexually dimorphic": {"visual": "#80cdc1", "nonvisual": "#01665e"},
}
xpos = np.arange(len(CATEGORIES))


from utils.figure_helpers import _wrap


panel_xlabels = [f"{_wrap(c)}\n({int(n_by_cat[c])})" for c in CATEGORIES]
FS = 13                    # 30% larger than matplotlib's default font size of 10
PANEL_W = 13 / 3 * 0.8     # 20% narrower than its share of the original 1x3 figure

fig, ax = plt.subplots(figsize=(PANEL_W, 4.2))
for i, cat in enumerate(CATEGORIES):
    vf = vis_frac[cat]
    ax.bar(i, vf, color=CAT_BAR[cat]["visual"], edgecolor="white")
    ax.bar(i, 1 - vf, bottom=vf, color=CAT_BAR[cat]["nonvisual"], edgecolor="white")
    ax.text(i, vf / 2, f"{vf * 100:.0f}%\nvisual", ha="center", va="center",
            color="black", fontsize=FS - 1, fontweight="bold")
    ax.text(i, vf + (1 - vf) / 2, "non-visual", ha="center", va="center",
            color="white", fontsize=FS - 3)
ax.set(xticks=xpos, ylim=(0, 1))
ax.set_ylabel("fraction of neurons", fontsize=FS)
ax.set_xticklabels(panel_xlabels, fontsize=FS)
ax.tick_params(axis="y", labelsize=FS)
fig.tight_layout()
out = DIM_DIR / f"{config.DATASET}_dimorphic_neurons_paper_panel.pdf"
fig.savefig(out, bbox_inches="tight")
plt.show()


# %%
# --- Main Fig. 7b companion: propagated ARF examples ---
# For each layer bin of the RF-size-per-layer panel above, save a PDF of the
# raw propagated effective-weight RF (hex heatmap) for one random neuron of a
# chosen representative cell type (`CHOICE`), with no 2D-Gaussian / alpha-hull
# fit overlaid (fixed seed -> reproducible). The CB types need the full-brain
# propagation matrix (~10 GB resident) -- run this on a compute cluster. PDFs
# go to results/CB_rf_fits/propagated_arf/ (needs the kaleido PDF backend).

import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils import config, core_data, ol_data, ol_rf, plotting as plot
from utils.ol_rf import _default_visual_input_instances

# Explicit per-layer cell type for the Fig. 6b examples (layer -> cell type).
# Picked by hand from the closest-to-median candidates; see the per-bin lists.
CHOICE = {
    1.0: 'Dm4', 1.5: 'Tm16', 2.0: 'Tm40', 2.5: 'LPLC2', 3.0: 'MeTu4b',
    3.5: 'PS324', 4.0: 'GNG496', 4.5: 'PLP025', 5.0: 'AVLP298',
}
nrn = ol_rf.get_rf_ahull_neuron()
_bins = np.arange(-0.25, 6 + 0.25 + 1e-9, 0.5)
_labels = (_bins[1:] + _bins[:-1]) / 2
_b = nrn.copy()
_b['bin'] = pd.cut(_b['hitting_time'].clip(upper=5.0), bins=_bins, labels=_labels).astype(float)
_b = _b.dropna(subset=['bin', 'ahull_size'])
_b['size_deg'] = _b['ahull_size'] * 25.0
reps = []
for _lyr, _ct in CHOICE.items():
    _g = _b[_b['bin'] == _lyr]
    _sel = _g[_g['cell_type'] == _ct]
    if _sel.empty:
        print(f'WARNING: {_ct} has no neurons in layer {_lyr}')
        continue
    reps.append({'layer': _lyr, 'cell_type': _ct, 'region': _sel['region'].iloc[0],
                 'ahull_deg': round(float(_sel['size_deg'].median()), 1),
                 'median_deg': round(float(_g['size_deg'].median()), 1)})
reps = pd.DataFrame(reps)
reps

# OL-subset propagation (OL-region types) — cheap.
ol_meta   = core_data.get_ol_meta()
ol_stepsn = ol_data.get_ol_stepsn_sum()
raw_ol    = ol_rf.get_rf_raw_ol()

# Full-brain propagation (CB types). The matrix (~9 GiB resident) loads lazily
# on the first CB plot, so OL-only (re)generations never pay for it. Shared
# (not reloaded) by SI Fig. 7f and SI Fig. 9, which need this same full-brain
# matrix -- see release_full_stepsn(), called at the end of SI Fig. 9 once
# nothing later needs it.
full_meta = core_data.get_meta()
raw_cb    = ol_rf.get_rf_raw_cb()
_full_stepsn = [None]
def load_full_stepsn():
    if _full_stepsn[0] is None:
        _full_stepsn[0] = sp.load_npz(config.DATA_DIR / f'{config.DATASET}_{config.SIDE_CHAR}_lat_flow_sum.npz')
    return _full_stepsn[0]
def release_full_stepsn():
    _full_stepsn[0] = None

# Visual-input instances that seed the propagated RF (same set used for the fits).
vis_in = _default_visual_input_instances(config.SIDE_CHAR)

# One PDF per representative: the raw propagated RF (no fit overlay), for a
# random neuron of each type. Fixed seed (1) across the (layer-ordered) loop.
# Knobs: bump a type in REROLL to swap in a different neuron (others unchanged);
# ONLY restricts which types are (re)plotted — set ONLY = None to do all.
import zlib
REROLL = {'Dm4': 5, 'Tm16': 7, 'LPLC2': 1}
ONLY = None #{'Tm40'}
arf_dir = config.FIG_DIR / 'CB_rf_fits' / 'propagated_arf'
rng = np.random.default_rng(1)
for _, r in reps.iterrows():
    ct, region = r['cell_type'], r['region']
    raw = raw_ol if 'OL' in str(region) else raw_cb
    _bodies = raw.loc[raw['cell_type'] == ct, 'bodyId'].to_numpy()
    bid = int(rng.choice(_bodies))                     # base pick keeps other examples fixed
    _n = REROLL.get(ct, 0)
    if _n:                                             # independent reroll -> a different neuron
        _alt = _bodies[_bodies != bid]
        if len(_alt):
            bid = int(np.random.default_rng(zlib.crc32(f'{ct}:{_n}'.encode())).choice(_alt))
    if ONLY is not None and ct not in ONLY:
        continue                                       # rng already advanced; others stay fixed
    if 'OL' in str(region):
        meta, stepsn = ol_meta, ol_stepsn
    else:
        meta, stepsn = full_meta, load_full_stepsn()   # lazy ~9 GiB load, CB only, shared/freed (see above)
    fig, bid = plot.plot_propagated_arf_example(
        meta, stepsn, raw, ct, vis_in, bodyId=bid, save_dir=arf_dir,
    )
    print(f"layer {r['layer']:g}: {ct} [{region}] -> body {bid}  "
          f"{arf_dir / f'{plot.DATASET}_propagated_arf_{ct}_{bid}.pdf'}")

# %%
# --- Main Fig. 7c (right): VIC for sexually dimorphic CB neurons ---
# Reuses the VIC-projection setup (syn0, outlines_bkgd, plot_dir, olc)
# loaded in the Main Fig. 5 section above.
cypher = '''
MATCH (n:Neuron)
RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance,
       n.dimorphism AS dimorphism
ORDER BY n.type, n.instance
'''
dim = olc.fetch_custom(cypher)
dim = dim[~dim['dimorphism'].str.contains('potentially', na=True)]

lr = 'r'
meta_cb_vpn = vic.load_vp_cb(lr)
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(dim.instance)]
syn = syn0.merge(
    meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'main_groups']], on='bodyId')

plot_contribution_layers(
    syn, 'vision', plot_dir=plot_dir, outlines=outlines_bkgd,
    gp_layers=None, vmax='p99',
    fname=lambda gp, lo, hi: f'sexdim_vic_ht_{lr}.pdf')

pf.distribute('main_7')

# %% [markdown]
# <a id="si-1"></a>
# ## SI Fig. 1

# %%
# Preprocessing (cached, no recompute if Main 1 ran)
ol_meta     = core_data.get_ol_meta()
ol_flow     = core_data.get_ol_flow_type()
meta        = core_data.get_meta()
flow_df     = core_data.get_flow().merge(meta[['idx','main_groups','sign']], on='idx', how='right')
flow_ol     = core_data.get_flow_ol()
conn_ol     = ol_data.get_ol_connectivity()
dir_ol      = ol_data.get_ol_directedness()
type_dir_ol = ol_data.get_ol_type_directedness()
layers      = ol_data.get_ol_layers()
roi_adj     = ol_data.get_ol_roi_adjacency()

colored_region, colored_main_groups, colored_sign, _ = load_colors()

# SI 1a/1b: fractional hit-hist groups & signs (additional outputs of by_main_groups / by_sign)
plot.plot_flow_by_main_groups(ol_flow, conn_ol, type_dir_ol, flow_ol, colored_main_groups)  # also frac
plot.plot_flow_by_sign(ol_flow, conn_ol, type_dir_ol, flow_ol, colored_sign)               # also frac, fb/la/ff signs

# Per-main-group connection-level ff/la/fb breakdown for SI 1b_1 / SI 1b_2
_thr = 0.5
print('SI 1b_1 / SI 1b_2 (by main_groups_pre):')
for _g in conn_ol['main_groups_pre'].dropna().unique():
    _sub = conn_ol[conn_ol['main_groups_pre'] == _g]
    _ff = _sub['hitting_time_diff'] >  _thr
    _la = _sub['hitting_time_diff'].abs() <= _thr
    _fb = _sub['hitting_time_diff'] < -_thr
    _W, _N = _sub.weight.sum(), len(_sub)
    _wE = lambda m, s: _sub[m & (_sub.sign_pre == s)].weight.sum() / _sub[m].weight.sum() if _sub[m].weight.sum() else float('nan')
    print(f'  {_g}:')
    print(f'    Total weight fractions:      ff {_sub[_ff].weight.sum()/_W:.3f}, la {_sub[_la].weight.sum()/_W:.3f}, fb {_sub[_fb].weight.sum()/_W:.3f}')
    print(f'    Connection-count fractions:  ff {_ff.sum()/_N:.3f}, la {_la.sum()/_N:.3f}, fb {_fb.sum()/_N:.3f}')
    print(f'    E/I fractions:               ff is E {_wE(_ff, 1):.3f}, la is E {_wE(_la, 1):.3f}, fb is I {_wE(_fb, -1):.3f}')
# SI 1c, 1d: ff/fb/la frac hist
plot.plot_flow_type_boxes(flow_df, dir_ol, colored_main_groups)                 # also fb_frac_hist_all_groups
# SI 1e: examples_0.1thre_no_vs_layer (extra output of plot_flow_type_scatter_and_triangle)
plot.plot_flow_type_scatter_and_triangle(ol_flow, type_dir_ol, colored_main_groups, colored_sign)
# SI 1f, 1g: connectivity heatmaps (visual input / VIN / VPN / VCN + signed)
plot.plot_flow_connectivity_heatmap(conn_ol, colored_main_groups, colored_sign)
# SI 1h: threshold-fraction
plot.plot_flow_threshold_choice(conn_ol)
_y0 = conn_ol[conn_ol['hitting_time_diff'] > 0]['weight'].sum() / conn_ol['weight'].sum()
print(f'SI 1h: at post-pre layer threshold 0, fraction of weights above is {_y0:.3f}')
# SI 1i, 1j: per-layer violin (hitting_time) + per-layer hit-diff (post - pre)
side = config.SIDE_CHAR.upper()
me_rois = [f'ME_{side}_layer_{i:02d}' for i in range(1, 11)] + [f'AME({side})']
me_labels = [f'M{i}' for i in range(1, 11)] + ['AME']
lo_rois = (
    [f'LO_{side}_layer_{i}' for i in range(1, 8)]
    + [f'LOP_{side}_layer_{i}' for i in range(1, 5)]
)
lo_labels = ['LO1','LO2','LO3','LO4','LO5A','LO5B','LO6','LOP1','LOP2','LOP3','LOP4']
plot.plot_ol_layers_violin(layers, me_rois, me_labels, filename_stem='ME_layers_hit')
plot.plot_ol_layers_violin(layers, lo_rois, lo_labels, filename_stem='LOPLO_layers_hit')
plot.plot_ol_layers_hit_diff(roi_adj, me_rois, me_labels, filename_stem='ME_layers_hit')
plot.plot_ol_layers_hit_diff(roi_adj, lo_rois, lo_labels, filename_stem='LOPLO_layers_hit')

pf.distribute('si_1')


# %% [markdown]
# <a id="si-2"></a>
# ## SI Fig. 2

# %%
ol_flow = core_data.get_ol_flow_type()
dir_ol  = ol_data.get_ol_directedness()
_, colored_main_groups, _, _ = load_colors()

# numerous1/numerous2 hit_hist + fb/la/ff (SI 2a, 2b)
plot.plot_flow_numerous_type_boxes(flow_ol, dir_ol, colored_main_groups)

pf.distribute('si_2')


# %% [markdown]
# <a id="si-3"></a>
# ## SI Fig. 3

# %%
# --- Polarity (signed): excitation − inhibition ---
# For each path length `k=1..4`: `find_paths_of_length` → `group_paths` (instance-level) → `signed_effective_conn_from_paths` → `tmp_e - tmp_i`. Sum across `k`, row-normalize by absolute-value sum (so Σ|bars| = 1 per target), predict `ON if |L1_R| > |L2_R| else OFF`. Sign overrides: `L1_R → +1` (flip), `sign == 0 → +1`. Plots all three panels (VPN, non-VPN OL, CB) as signed stacked bars.

import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils import config
from utils.palettes import load_colors
from utils import core_data, polarity
from utils.figure_helpers import (
    build_sign_map, compute_signed_totals, build_cmp,
    group_cmp_like_figures, plot_signed_bars,
)

_, _, _, colored_seed = load_colors()

# --- OL subset — VPN (VPN) + non-VPN (`_other`) ---

ol_meta = core_data.get_ol_meta()
flow_type = core_data.get_ol_flow_type()
labels_ol = polarity.get_polarity_experiments()
labels_ol = labels_ol[labels_ol['in_ol']].reset_index(drop=True)

A_OL = sp.load_npz(config.DATA_DIR / f'{config.DATASET}_OL_r_lat_flow_0.npz').tocsr()

in_instances_ol = flow_type[flow_type.hitting_time == 0].instance.unique()
inidx_ol  = ol_meta[ol_meta.instance.isin(in_instances_ol)].idx.values
outidx_ol = ol_meta[ol_meta.instance.isin(labels_ol['instance'].values)].idx.values
idx_to_instance_ol = dict(zip(ol_meta.idx, ol_meta.instance))
inst_to_sign_ol = build_sign_map(ol_meta)

total_ol = compute_signed_totals(
    A_OL, inidx_ol, outidx_ol, idx_to_instance_ol, inst_to_sign_ol,
)
print('total_ol shape:', total_ol.shape)

cmp_ol = build_cmp(total_ol, labels_ol, in_instances_ol, ol_meta)
cmp_ol = cmp_ol[cmp_ol['main_groups'] != 'visual input'].reset_index(drop=True)
cmp_ol = group_cmp_like_figures(cmp_ol, sort='ol')   # one averaged bar per entry

cmp_vpn   = cmp_ol[cmp_ol['main_groups'] == 'VPN'].reset_index(drop=True)
cmp_other = cmp_ol[cmp_ol['main_groups'] != 'VPN'].reset_index(drop=True)

for label, df in [('VPN', cmp_vpn), ('non-VPN OL', cmp_other)]:
    miss = df[~df.match][['instance', 'Contrast selectivity', 'predicted', 'main_groups']]
    print(f'{label}: {df.match.mean():.1%} match ({len(df)} types, {len(miss)} mismatches)')
    if len(miss):
        print(miss.to_string(index=False))

plot_signed_bars(cmp_vpn,   colored_seed, 'polarity_predictions_vpn_signed', width_scale=1.1)
plot_signed_bars(cmp_other, colored_seed, 'polarity_predictions_other_signed', width_scale=0.9)

# --- CB (full brain) ---

meta = core_data.get_meta()
labels_cb = polarity.get_polarity_experiments()
labels_cb = labels_cb[~labels_cb['in_ol']].reset_index(drop=True)

A_full = sp.load_npz(config.DATA_DIR / f'{config.DATASET}_r_lat_flow_0.npz').tocsr()
print(f'full-brain A: {A_full.shape}, nnz={A_full.nnz}')

in_instances_cb = meta[(meta.main_groups == 'visual input') & (meta.side == 'right')].instance.unique()
inidx_cb  = meta[meta.instance.isin(in_instances_cb)].idx.values
outidx_cb = meta[meta.instance.isin(labels_cb['instance'].values)].idx.values
idx_to_instance_full = dict(zip(meta.idx, meta.instance))
inst_to_sign_full = build_sign_map(meta)

total_cb = compute_signed_totals(
    A_full, inidx_cb, outidx_cb, idx_to_instance_full, inst_to_sign_full,
)
print('total_cb shape:', total_cb.shape)

cmp_cb = build_cmp(total_cb, labels_cb, in_instances_cb, meta)
cmp_cb = group_cmp_like_figures(cmp_cb, sort='cb')

miss_cb = cmp_cb[~cmp_cb.match][['instance', 'Contrast selectivity', 'predicted', 'main_groups']]
print(f'CB: {cmp_cb.match.mean():.1%} match ({len(cmp_cb)} types, {len(miss_cb)} mismatches)')
if len(miss_cb):
    print(miss_cb.to_string(index=False))

plot_signed_bars(cmp_cb, colored_seed, 'polarity_predictions_cb_signed', width_scale=1.2)

# %%
# --- SI Fig. 3a — polarity-mismatch pathway diagrams ---

# SI 3a: polarity-mismatch pathway diagrams. For every cell type whose
# propagated-polarity prediction disagrees with the experiment (unsigned
# comparison, same set as Main 3d / Main 5d), draw its input pathway. Saves to
# results/examples/.
colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()

# --- OL mismatches ---
cmp_ol_6a = polarity.get_ol_polarity_comparison()
_miss_ol = cmp_ol_6a[~cmp_ol_6a.match][['instance', 'main_groups']]
ol_meta = core_data.get_ol_meta()
ol_flow = core_data.get_ol_flow_type()
_t2s = dict(zip(ol_meta.cell_type, ol_meta.sign))
_s2c = colored_sign['color'].to_dict()
_t2mg = dict(zip(ol_meta.cell_type, colored_main_groups.loc[ol_meta.main_groups, 'color']))
_ex_dir = config.FIG_DIR / 'examples'
_ol_override = {'DN1a_R': 0.6, 'Tm29_R': 0.6}
for _inst, _mg in zip(_miss_ol.instance, _miss_ol.main_groups):
    _tc = _ol_override.get(_inst, 0.4 if _mg == 'VPN' else 0.8)
    plot.plot_pathway(
        ol_data.get_paths_to_instance(_inst), _inst, ol_flow, thre_cumsum=_tc,
        neuron_to_color=_t2mg, neuron_to_sign=_t2s, sign_color_map=_s2c,
        save_path=_ex_dir / f'{config.FIG_DATASET}_{_inst[:-2]}_pathway.pdf',
    )
print(f'SI 3a: {len(_miss_ol)} OL mismatch pathways {list(_miss_ol.instance)}')

# --- CB mismatches ---
cmp_cb_6a = polarity.get_cb_polarity_comparison()
_miss_cb = cmp_cb_6a[~cmp_cb_6a.match][['instance']]
_meta_6a = core_data.get_meta()
_t2s_full = dict(zip(_meta_6a.cell_type, _meta_6a.sign))
_cb_color_df = pd.concat([
    pd.DataFrame(colored_region.loc['CB', 'color'], index=['nonOL'], columns=['color']),
    colored_main_groups,
])
_t2c_cb = dict(zip(_meta_6a.cell_type, _cb_color_df.loc[_meta_6a.main_groups, 'color']))
_flow_full = core_data.get_flow().groupby('instance')['hitting_time'].median().reset_index()
for _inst in _miss_cb.instance:
    plot.plot_pathway(
        cb_data.get_cb_paths_to_instance(_inst), _inst, _flow_full, thre_cumsum=0.4,
        neuron_to_color=_t2c_cb, neuron_to_sign=_t2s_full, sign_color_map=_s2c,
        save_path=_ex_dir / f'{config.FIG_DATASET}_{_inst[:-2]}_pathway.pdf',
    )
print(f'SI 3a: {len(_miss_cb)} CB mismatch pathways {list(_miss_cb.instance)}')

pf.distribute('si_3')

# %% [markdown]
# <a id="si-4"></a>
# ## SI Fig. 4

# %%
FRAC = 0.19

feat        = ol_data.get_ol_features()
clusters    = ol_data.get_ol_clusters(frac_thre=FRAC)
ol_meta     = core_data.get_ol_meta()
intra       = ol_data.get_ol_clusters_intra()
intra_7     = ol_data.get_ol_clusters_intra(frac_thre=0.23)   # SI 4g
sweep       = ol_data.get_ol_lr_sweep()                       # SI 4f
match       = ol_data.get_ol_lr_cluster_match(frac_thre=FRAC) # SI 4j
_, colored_main_groups, _, colored_seed = load_colors()

n_clu = int(clusters.cluster.max())
cluster_num = clusters['cluster'].value_counts().sort_index()
in_instances = list(ol_meta[ol_meta.main_groups=='visual input'].instance.unique())

# SI 4a (full dendrogram), 4b (clustering matrix), 4c (outrel matrices)
Z, thre, dendr = plot.plot_clusters_dendrograms(feat, frac_thre=FRAC)
plot.plot_clusters_correlation_matrix(feat, dendr, cluster_num)
plot.plot_clusters_intra_connectivity(intra, font_size=23)             # SI 4g (outrel_13)
plot.plot_clusters_intra_connectivity(intra_7, font_size=23)           # SI 4g (outrel_7)

# SI 4d (no_thre clustering sweep), 4f (LR ARI), 4j (LR confusion)
n0 = int(ol_data.get_ol_clusters(frac_thre=FRAC, side_char='r')['cluster'].max())
plot.plot_lr_clustering_sweep(sweep, n0, diag_font_size=23)                 # SI 4d, 4f, 4g
plot.plot_lr_cluster_confusion(match)                    # SI 4j
print(f'SI 4j OL L->R Hungarian-optimal cluster assignment (n_L={match["n_clu_L"]}, n_R={match["n_clu_R"]}):')
for r, c in zip(match['row_ind'], match['col_ind']):
    print(f'  L cluster {r + 1}  ->  R cluster {c + 1}')

# SI 4e: 13-cluster pathway features
FRAC_13 = 0.15
clusters_13 = ol_data.get_ol_clusters(frac_thre=FRAC_13)
cluster_num_13 = clusters_13['cluster'].value_counts().sort_index()
plot.plot_clusters_pathway_features(feat, in_instances, clusters_13, cluster_num_13, colored_seed, side_str='_13', xtick_fontsize=12)

# SI 4h, 4i: left-side cluster pathway features
ol_meta_L = core_data.get_ol_meta(side_char='l')
feat_L    = ol_data.get_ol_features(side_char='l')
cl_L      = ol_data.get_ol_clusters(side_char='l', frac_thre=FRAC)
cluster_num_L = cl_L['cluster'].value_counts().sort_index()
in_L = list(ol_meta_L[ol_meta_L.main_groups=='visual input'].instance.unique())
plot.plot_clusters_example_features(feat_L, in_L, [s+'_L' for s in plot._CLUSTER_TYPES], colored_seed, side_str='_left')
plot.plot_clusters_pathway_features(feat_L, in_L, cl_L, cluster_num_L, colored_seed, side_str='_left')

pf.distribute('si_4')


# %% [markdown]
# <a id="si-5"></a>
# ## SI Fig. 5 — model-comparison hitting-time variants
#
# SI 5a/5b/5c (participation) come from Main Fig. 3 and are distributed by the
# live `pf.distribute('si_5')` below. SI 5d/5e compare the standard right-flow
# model (0.1 info-flow threshold) against four variants: 5d = 0.05 threshold,
# 0.2 threshold, and the >2-synapse (`2syn`) model; 5e = the idealized-R1-6
# (`r16`) model. **The 5d/5e generating code is commented out** — it is opt-in
# and reads the four variant `_hit.csv` flow tables from `cache/model_comparison`
# (built by the commented cells in `setup_data/s2_layers`), writing the scatters
# to `results/model_comparison/`. `si_5` is excluded from the parity check until
# those tables are regenerated.

# %%
# SI Fig. 5d / 5e — standard vs. threshold/parameter flow variants — COMMENTED OUT.
# Opt-in: needs the four variant `_hit.csv` tables under cache/model_comparison
# (see the commented cells in setup_data/s2_layers). Reads them from
# MODEL_COMPARISON_DIR, writes the scatters to results/model_comparison/.
# import pandas as pd
# from scipy.stats import spearmanr
#
# from utils import config, core_data
# from utils import plotting as plot
# from utils.palettes import load_colors
#
# _, colored_main_groups, _, _ = load_colors()
#
#
# def _median_hit(fname):
#     """Per-instance median hitting time from a flow `_hit.csv` table."""
#     return pd.read_csv(config.MODEL_COMPARISON_DIR / fname).groupby('instance')['hitting_time'].median()
#
#
# # right-OL instances + main_groups for colouring
# right_ol = (
#     core_data.get_meta()
#     .query("region == 'right OL'")[['instance', 'main_groups']]
#     .drop_duplicates()
# )
#
# _flow = f'{config.FIG_DATASET}_r_flow_{config.N_FLOW}step'
# _std_hit = _median_hit(f'{_flow}_{config.HIT_THRE}thre_hit.csv')
#
# # (alt file, y-axis label, output stem) -- standard 0.1 threshold on x, variant on y
# _hit_comparisons = [
#     (f'{_flow}_0.05thre_hit.csv',                   '0.05 thre', f'{config.FIG_DATASET}_hitting_0.1_vs_0.05_thre'),
#     (f'{_flow}_0.2thre_hit.csv',                    '0.2 thre',  f'{config.FIG_DATASET}_hitting_0.1_vs_0.2_thre'),
#     (f'{_flow}_{config.HIT_THRE}thre_hit_2syn.csv', '2syn',      f'{config.FIG_DATASET}_hitting_0.1_thre_vs_2syn'),
#     (f'{_flow}_{config.HIT_THRE}thre_hit_r16.csv',  'r16',       f'{config.FIG_DATASET}_hitting_0.1_thre_vs_r16'),
# ]
#
# for _alt_file, _alt_label, _out_stem in _hit_comparisons:
#     df = (
#         pd.DataFrame({'hit_std': _std_hit, 'hit_alt': _median_hit(_alt_file)})
#         .dropna().reset_index()
#         .merge(right_ol, on='instance', how='inner')
#     )
#     rho, pval = spearmanr(df['hit_std'], df['hit_alt'])
#     print(f'{_out_stem}: Spearman r = {rho:.3f}  (p = {pval:.2e}, n = {len(df)})')
#     plot.plot_comparison_scatter(
#         df, 'hit_std', 'hit_alt',
#         'main_groups', colored_main_groups,
#         xaxis_title='hitting time (0.1 thre)', yaxis_title=f'hitting time ({_alt_label})',
#         diag_range=[0.95, 5.0], axis_range=[0.9, 5.0],
#         filename_stem=_out_stem,
#         save_dir=config.FIG_DIR / 'model_comparison',
#     )

# SI 5a/b/c (participation, from Main Fig. 3) are distributed here regardless; d/e
# are skipped until regenerated (distribute reports them missing, non-fatal).
pf.distribute('si_5')

# %% [markdown]
# <a id="si-6"></a>
# ## SI Fig. 6 — alternative-model comparison
#
# **COMMENTED OUT — needs the per-variant propagated `_sum` matrices (heavy
# compute); `si_6` is excluded from the parity check.**
#
# The code cell below compares the default model against an alternative model
# across five quantities (layer/hitting time, VPN clusters, OL RF size, VIC,
# VCBN clusters). All alt inputs are read **flat from `cache/model_comparison/`
# with a per-variant suffix** (no `<ALT>` subfolders), and outputs go to
# `results/model_comparison/`. It emits comparison *diagnostics*
# (`_vs_<variant>_hitting`, `vpn_confusion`, …), which are NOT themselves the
# SI 6b–6e manifest panels.
#
# To generate SI Fig. 6, for EACH alternative model (the same variants as SI 5:
# 0.05thre / 0.2thre / 2syn / r16) you must:
#   1. Build that variant's propagated `_sum` matrix — the expensive step: run
#      the variant flow (`setup_data/s2_layers`) then trim + `compress_paths` +
#      sum (`setup_data/s3_effconn`); both write into `cache/model_comparison/`
#      with the variant suffix. ~15 min GPU each, plus the < 2 h flow.
#   2. Derive the alt inputs from that `_sum` (clustering/feature helpers in
#      `utils.ol_data` / `utils.cb_data`, RF in `utils.ol_rf`, VIC in
#      `utils.vic`) — either on the fly here or cached alongside the `_sum`
#      under `cache/model_comparison/` with the variant suffix.
#   3. Add a driver that re-runs the cluster + pathway-example plotters
#      (`plotting.plot_clusters_example_features` / `plot_clusters_pathway_features`
#      and the `cb_` variants) per variant, saving into
#      `results/model_comparison/` under the `in_out_*` names the manifest
#      expects. (Do NOT run here — heavy.)

# %%
# --- Model comparison ---
# Compares the default model (`config.DATASET` on `config.SIDE_CHAR`) against an alternative model across five quantities:
# 1. per-instance layer (hitting time)
# 2. VPN clusters — ARI + optimal matching
# 3. OL propagated RF size
# 4. VIC for VPN and CB neurons
# 5. VCBN clusters — ARI + optimal matching
# Alt inputs live flat under `cache/model_comparison/` (MODEL_COMPARISON_DIR) with a per-variant suffix; derived from each variant's `_sum` matrix (s3_effconn).

# import numpy as np
# import pandas as pd
# from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import adjusted_rand_score, confusion_matrix

# from utils import config
# from utils.palettes import load_colors
# from utils import cb_data, core_data, ol_data, ol_rf, vic
# from utils import plotting as plot
# from utils.ol_rf import _clean_rf_params

# ALT = '0.05thre'   # a SI-5 variant: '0.05thre' / '0.2thre' / '2syn' / 'r16'
# ALT_DIR = config.MODEL_COMPARISON_DIR   # flat input cache; alt files carry the {ALT} suffix (no <ALT> subfolder)
# SAVE_DIR = config.FIG_DIR / 'model_comparison' / ALT   # per-variant output subfolder (matches the manifest: SI 6b–6e = 0.05thre/0.2thre/2syn/r16)
# colored_region, colored_main_groups, _, colored_seed = load_colors()

# meta_ol = core_data.get_ol_meta()

# def _cluster_match(cl_a, cl_b, tick_prefix='c'):
#     """Merge two cluster frames on `instance`; return ARI + confusion + Hungarian assignment.
#     Shape matches `ol_data.get_ol_lr_cluster_match` so `plot_lr_cluster_confusion` can consume it."""
#     m = cl_a[['instance', 'cluster']].merge(
#         cl_b[['instance', 'cluster']], on='instance', how='inner', suffixes=('_a', '_b'),
#     )
#     ari = adjusted_rand_score(m['cluster_a'].values, m['cluster_b'].values)
#     conf = confusion_matrix(m['cluster_a'].values, m['cluster_b'].values)
#     row_ind, col_ind = linear_sum_assignment(-conf)
#     n_a = int(m['cluster_a'].max())
#     n_b = int(m['cluster_b'].max())
#     n = max(n_a, n_b)
#     return {
#         'ari': ari, 'n_common': len(m),
#         'confusion': conf, 'row_ind': row_ind, 'col_ind': col_ind,
#         'n_clu_L': n_a, 'n_clu_R': n_b,
#         'tick_labels': [f'{tick_prefix}{i}' for i in range(1, n + 1)],
#     }


# def _relabel_alt_to_match(cl_alt, match, max_def):
#     """Rename alt clusters so Hungarian-matched pairs share the default's label.
#     Unmatched alt clusters get labels `max_def+1, max_def+2, ...`."""
#     alt_to_def = {c + 1: r + 1 for r, c in zip(match['row_ind'], match['col_ind'])}
#     extra = sorted(set(cl_alt['cluster'].unique()) - set(alt_to_def))
#     for i, c in enumerate(extra, start=max_def + 1):
#         alt_to_def[c] = i
#     out = cl_alt.copy()
#     out['cluster'] = out['cluster'].map(alt_to_def).astype(int)
#     return out

# --- 1. Layer (hitting time) ---

# flow_name = f'{config.DATASET}_{config.SIDE_CHAR}_flow_{config.N_FLOW}step_{config.HIT_THRE}thre_hit_per_group.csv'
# flow_def = core_data.get_flow_per_group()
# flow_alt = pd.read_csv(ALT_DIR / flow_name)

# df = (
#     flow_def.merge(flow_alt, on='instance', suffixes=('_def', '_alt'))
#             .merge(meta_ol[['instance', 'main_groups']].drop_duplicates(),
#                    on='instance', how='inner')
# )

# plot.plot_comparison_scatter(
#     df, 'hitting_time_def', 'hitting_time_alt',
#     'main_groups', colored_main_groups,
#     xaxis_title=f'{config.DATASET} layer', yaxis_title=f'{ALT} layer',
#     diag_range=[0.9, 5], axis_range=[0.9, 5],
#     filename_stem=f'{config.FIG_DATASET}_vs_{ALT}_hitting',
#     save_dir=SAVE_DIR,
# )

# --- 2. VPN clusters ---

# cl_def = ol_data.get_ol_clusters(frac_thre=0.19)
# cl_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_in_out_clusters.csv')

# match = _cluster_match(cl_def, cl_alt, tick_prefix='c')
# print(f'Number of consistent VPNs: {match["n_common"]}')
# print(f'ARI: {match["ari"]:.2f}')

# plot.plot_lr_cluster_confusion(
#     match,
#     dataset=f'{config.FIG_DATASET}_vs_{ALT}',
#     filename_stem='vpn_confusion',
#     xaxis_title=f'{ALT} cluster', yaxis_title=f'{config.DATASET} cluster',
#     save_dir=SAVE_DIR,
# )

# print(f'Optimal matching (n_def={match["n_clu_L"]}, n_alt={match["n_clu_R"]}):')
# for r, c in zip(match['row_ind'], match['col_ind']):
#     print(f'  default c{r + 1}  ->  alt c{c + 1}')

# Stacked bar of the alt's per-cluster pathway features, re-numbered to match default
# feat_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_in_out_features.csv', index_col=0)
# feat_alt.index.name = 'instance'
# cl_alt_matched = _relabel_alt_to_match(cl_alt, match, max_def=match['n_clu_L'])
# cluster_num_alt = cl_alt_matched['cluster'].value_counts().sort_index()

# plot.plot_clusters_pathway_features(
#     feat_alt, list(feat_alt.columns), cl_alt_matched, cluster_num_alt, colored_seed,
#     dataset=f'{config.FIG_DATASET}_vs_{ALT}', save_dir=SAVE_DIR,
# )

# --- 3. OL propagated RF size ---

# rf_def = ol_rf.get_rf_type_ol().groupby('instance')['size'].median().reset_index()
# rf_alt = (
#     _clean_rf_params(pd.read_csv(ALT_DIR / f'{config.DATASET}_types_fit_rf.csv'))
#     .groupby('instance')['size'].median().reset_index()
# )

# df = (
#     rf_def.merge(rf_alt, on='instance', suffixes=('_def', '_alt'))
#           .merge(meta_ol[['instance', 'main_groups']].drop_duplicates(),
#                  on='instance', how='inner')
# )

# plot.plot_comparison_scatter(
#     df, 'size_def', 'size_alt',
#     'main_groups', colored_main_groups,
#     xaxis_title=f'{config.DATASET} RF size (col)', yaxis_title=f'{ALT} RF size (col)',
#     diag_range=[0.5, 1000], axis_range=[np.log10(0.5), np.log10(1000)],
#     log_axes=True,
#     filename_stem=f'{config.FIG_DATASET}_vs_{ALT}_rf_size',
#     save_dir=SAVE_DIR,
# )

# %%
# --- 4. VIC (VPN + CB) ---

# vic_def = vic.get_ol_cb_vic_type()
# vic_alt = pd.read_csv(ALT_DIR / 'vp_cb_vic.csv')

# df = vic_def.merge(vic_alt[['instance', 'VIC']], on='instance', suffixes=('_def', '_alt'))

# plot.plot_comparison_scatter(
#     df, 'VIC_def', 'VIC_alt',
#     'region', colored_region,
#     xaxis_title=f'{config.DATASET} VIC', yaxis_title=f'{ALT} VIC',
#     diag_range=[1e-5, 1], log_axes=True,
#     filename_stem=f'{config.FIG_DATASET}_vs_{ALT}_vic',
#     save_dir=SAVE_DIR, size=3,
# )

# --- 5. VCBN clusters ---

# cl_def = cb_data.get_cb_clusters(frac_thre=0.19)
# cl_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_cb_in_out_clusters.csv')

# match = _cluster_match(cl_def, cl_alt, tick_prefix='d')
# print(f'Number of consistent VCBNs: {match["n_common"]}')
# print(f'ARI: {match["ari"]:.2f}')

# plot.plot_cb_lr_cluster_confusion(
#     match,
#     dataset=f'{config.FIG_DATASET}_vs_{ALT}',
#     filename_stem='vcbn_confusion',
#     xaxis_title=f'{ALT} cluster', yaxis_title=f'{config.DATASET} cluster',
#     save_dir=SAVE_DIR,
# )

# print(f'Optimal matching (n_def={match["n_clu_L"]}, n_alt={match["n_clu_R"]}):')
# for r, c in zip(match['row_ind'], match['col_ind']):
#     print(f'  default d{r + 1}  ->  alt d{c + 1}')

# Stacked bar of the alt's per-cluster pathway features, re-numbered to match default
# feat_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_cb_in_out_features.csv', index_col=0)
# feat_alt.index.name = 'instance'
# cl_alt_matched = _relabel_alt_to_match(cl_alt, match, max_def=match['n_clu_L'])
# cluster_num_alt = cl_alt_matched['cluster'].value_counts().sort_index()

# plot.plot_cb_clusters_pathway_features(
#     feat_alt, list(feat_alt.columns), cl_alt_matched, cluster_num_alt, colored_seed,
#     dataset=f'{config.FIG_DATASET}_vs_{ALT}', save_dir=SAVE_DIR,
# )

pf.distribute('si_6')

# %% [markdown]
# <a id="si-7"></a>
# ## SI Fig. 7

# %%
# SI Fig. 7 - RF / experimental size comparisons.
import numpy as np
import plotly.graph_objects as go

cmp_type = ol_rf.get_rf_comparison_type()
exp_sz   = ol_rf.get_experimental_rf_sizes()
exp_cc   = ol_rf.get_experimental_rf_sizes(sheet='RF size CC')
colored_region, colored_main_groups, colored_sign, _ = load_colors()

# SI 7a: propagated vs direct ARF size (columns)
_rf_stars = [s + '_R' for s in plot._RF_STAR_OUTPUT]
_sz = cmp_type[cmp_type['main_groups'].isin(['VIN', 'VPN'])].copy()
fig = plot._scatter(
    _sz, 'size_dir', 'size_prop', 'main_groups', 'instance',
    colored_main_groups, _rf_stars, height=600, width=600,
)
fig.add_trace(go.Scatter(
    x=[-1000, 1000], y=[-1000, 1000], mode='lines',
    line=dict(color='gray', width=2, dash='dash'),
    showlegend=False, hoverinfo='skip',
))
fig.update_layout(
    xaxis_title='direct ARF size (col)', yaxis_title='propagated ARF size (col)',
    xaxis_title_standoff=0, yaxis_title_standoff=2,
)
fig.update_xaxes(type='log', tickformat='d', dtick=1, range=[np.log10(0.6), np.log10(350)])
fig.update_yaxes(type='log', tickformat='d', dtick=1, range=[np.log10(0.6), np.log10(350)])
plot._save(fig, config.FIG_DIR / 'rf_fits',
           f'{config.FIG_DATASET}_prop_vs_direct_size.pdf', html=True)
print(f"SI 7a prop vs direct size: {_sz.dropna(subset=['size_dir', 'size_prop']).shape[0]} types")

# SI 7b: exp_sx_comp / exp_sy_comp (also re-emits exp_size_comp used by Main 4e)
plot.plot_rf_size_vs_experiment(cmp_type, exp_sz, colored_main_groups, colored_region)

# SI 7d: CC-paper RF-size comparison using ellipse area (deg^2)
cc_size_col = 'size_ellipse' if 'size_ellipse' in exp_cc.columns else 'size ellipse'
cc_comp = (
    cmp_type[['instance', 'cell_type_prop', 'main_groups', 'size_prop']]
    .rename(columns={'cell_type_prop': 'cell_type'})
    .drop_duplicates(['instance', 'cell_type'])
    .merge(
        exp_cc[['cell_type', cc_size_col]].rename(columns={cc_size_col: 'size_exp_deg'}),
        on='cell_type', how='inner',
    )
    .dropna(subset=['size_exp_deg', 'size_prop'])
    .copy()
)
cc_comp['size_prop_deg'] = cc_comp['size_prop'] * 25.0
rf_size_highlight = [
    f'{cell_type}_R'
    for cell_type in exp_sz['cell_type'].dropna().astype(str)
    if f'{cell_type}_R' in set(cc_comp['instance'])
]
cc_color_df = pd.concat((colored_main_groups.iloc[:-1], colored_region.iloc[2:3]), axis=0)
fig = plot._scatter(
    cc_comp, 'size_exp_deg', 'size_prop_deg', 'main_groups', 'instance',
    cc_color_df, rf_size_highlight, height=600, width=600, size=12,
)
plot._add_2x_guide_lines(fig, 5, 1000)
fig.update_layout(
    xaxis_title='measured CC RF size, ellipse area (deg^2)',
    yaxis_title='propagated RF size (deg^2)',
    xaxis_title_standoff=0, yaxis_title_standoff=2,
)
fig.update_xaxes(type='log', tickformat='d', dtick=1, range=[0.7, 3.0])
fig.update_yaxes(type='log', tickformat='d', dtick=1, range=[0.7, 3.0])
plot._save(
    fig, config.FIG_DIR / 'exp_comparison',
    f'{config.FIG_DATASET}_exp_size_ellipse_comp_cc.pdf', html=True,
)
print(f'SI 7d CC ellipse comparison: {len(cc_comp)} types; highlighted {len(rf_size_highlight)} types')

# --- RF aspect-ratio (sy/sx): measured vs predicted ---
# Companion to Main Fig. 4e (RF *size*), but plotting RF *elongation*
# (`sy/sx`) for the same cell types. Prediction is the propagated ARF only.
# Cell types whose measured **or** predicted ratio deviates from 1.0 by more
# than `DEV` are highlighted and labelled; hovering shows the cell type.
# Running this notebook (re)writes the interactive HTML and exports a PDF.

from utils import config
from utils import ol_rf
from utils import plotting as plot
from utils.palettes import load_colors

# --- Load cached data ---
# Loads from cache; no NeuPrint query is needed unless the caches are missing.

colored_region, colored_main_groups, colored_sign, _ = load_colors()
cmp_type = ol_rf.get_rf_comparison_type()
exp_sz = ol_rf.get_experimental_rf_sizes()

# --- Build the figure (writes HTML to `results/html_figures/`) ---

DEV = 0.2  # deviation from 1.0 (measured or predicted) above which to label
fig = plot.plot_rf_aspect_vs_experiment(
    cmp_type, exp_sz, colored_main_groups, colored_region, dev=DEV,
)
fig.show()

# --- Export PDF ---
# Saved next to the Fig. 4e size panel, under `results/exp_comparison/`.

pdf_path = (
    config.FIG_DIR / "exp_comparison" / f"{config.FIG_DATASET}_exp_aspect_comp.pdf"
)
pdf_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(pdf_path)

# %%
# --- SI Fig. 7e / 7f — r2-vs-size (relocated) + Nod5_R example fit ---

# SI 7e: examples_r2_vs_size is produced by plot_rf_scatters_by_main_groups
# (Main Fig. 4) into results/rf_fits/; the manifest expects it under
# exp_comparison/, so copy it there (Main Fig. 4 runs earlier, so it exists).
import shutil
for _ext in ('.pdf', '.html'):
    _src = config.FIG_DIR / 'rf_fits' / f'{config.FIG_DATASET}_examples_r2_vs_size{_ext}'
    if _src.exists():
        _dst = config.FIG_DIR / 'exp_comparison' / _src.name
        _dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_src, _dst)
    else:
        print(f'SI 7e: {_src.name} not in rf_fits/ (run Main Fig. 4 first)')

# SI 7f: estimated ARF for Nod5_R (bodyId 12328), a central-brain example,
# shown with its fitted Gaussian ellipse AND alpha-hull overlaid (gallery style),
# NOT the fitted-Gaussian reconstruction that plot_rf_example draws.
# Added anew (the panel had no figure code; the original 12328 example lived in a
# since-removed exploratory notebook). Nod5 is a CB neuron, so use the full-brain
# propagation matrix and fit its RF explicitly -- Nod5_R is not in the standard
# VCBN RF cache, so plot_rf_estimated_overlay needs an explicit params row for
# bodyId 12328 to draw the Gaussian ellipse over the estimated ARF.
import scipy.sparse as sp
from utils.ol_rf import _compute_rf_params, _clean_rf_params, _default_visual_input_instances

_BID_7F = 12328  # Nod5_R
_meta_7f = core_data.get_meta()
# shared with Main Fig. 7 / SI Fig. 9 (load_full_stepsn(), same underlying
# matrix) -- requires the Main Fig. 7 cell above to have run in this session.
_stepsn_7f = load_full_stepsn()
_in_7f = _default_visual_input_instances(config.SIDE_CHAR)
_params_7f = _clean_rf_params(_compute_rf_params(_meta_7f, _stepsn_7f, _in_7f, ['Nod5_R']))
plot.plot_rf_estimated_overlay(
    _meta_7f, _stepsn_7f, _params_7f, _BID_7F, _in_7f,
    save_dir=config.FIG_DIR / 'exp_comparison',
)
print(f'SI 7f: Nod5_R estimated ARF + ellipse + hull '
      f'(fit rows for Nod5_R: {len(_params_7f)})')

# --- SI Fig. 7g — OL ahull-vs-propagated RF relative size ---

# SI 7g: OL alpha-hull ARF size vs propagated RF size -- the OL analogue of
# the VCBN cb_examples_ahull_reldiff_vs_size panel (SI 13), but over all
# right-OL instances instead of VCBNs and coloured by main_groups. ahull
# sizes are the per-neuron cumsum-60 alpha hulls (get_rf_ahull_neuron),
# aggregated to a per-instance median and merged with the per-instance
# propagated RF size (get_rf_type_ol).
_, colored_main_groups, *_ = load_colors()
_ahull_ol = ol_rf.get_rf_ahull_neuron()
_ahull_ol = _ahull_ol[_ahull_ol['region'] == 'right OL']
_ahull_ol_type = (_ahull_ol.groupby('instance')
                  .agg(ahull_size=('ahull_size', 'median')).reset_index())
_rf_size_ol = (ol_rf.get_rf_type_ol().groupby('instance')
               .agg(size=('size', 'median'),
                    main_groups=('main_groups', 'first')).reset_index())
_ol_cmp = _rf_size_ol.merge(_ahull_ol_type, on='instance', how='inner')
plot.plot_ol_ahull_size_comparison(_ol_cmp, colored_main_groups)
print(f'SI 7g: {len(_ol_cmp.dropna(subset=["size", "ahull_size"]))} OL instances')

pf.distribute('si_7')

# %% [markdown]
# <a id="si-8"></a>
# ## SI Fig. 8
#
# (Also produces SI 10b/10c/10d, distributed in SI 10.)

# %%
FRAC_CB = 0.17
FRAC_OL = 0.19

feat_r       = cb_data.get_cb_features(side_char='r')
cl_r         = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB)
sweep        = cb_data.get_cb_lr_sweep()
match        = cb_data.get_cb_lr_cluster_match(frac_thre=FRAC_CB)  # threshold-based; left side has 11 clusters
layer_lr     = cb_data.get_cb_layer_lr()
layer_lr_hom = cb_data.get_cb_layer_lr_homologue()
vic_homologue  = vic.get_cb_vic_lr_homologue()
vic_binoc_type = vic.get_cb_vic_binocular_type()
intra_cb     = cb_data.get_cb_clusters_intra(frac_thre=FRAC_CB)
type_part_cb_out = cb_data.get_ol_in_cb_type_participation(side_char='r', cb_frac_thre=FRAC_CB)

colored_region, colored_main_groups, _, colored_seed = load_colors()
n0 = int(cl_r['cluster'].max())

# SI 8a: CB layer L/R comparison + horizontal left layer histogram
plot.plot_cb_layer_lr_comparison(layer_lr, colored_region, colored_main_groups)
plot.plot_cb_layer_hist_left(layer_lr_hom, colored_region, colored_main_groups)
plot.plot_cb_layer_lr_homologue(layer_lr_hom, colored_region, colored_main_groups)  # SI 8a cb_layer_lr_comp
# SI 8b: CB VIC L/R comparison
plot.plot_cb_vic_lr_homologue(vic_homologue, colored_region)
# SI 8c: full CB dendrogram
plot.plot_clusters_dendrograms(feat_r, frac_thre=FRAC_CB, side_str='_cb', save_dir=config.FIG_DIR / 'CB_pathways')
# SI 8d: CB intra-cluster connectivity (cb_outrel_conn_matrix_clusters)
plot.plot_cb_clusters_intra_connectivity(intra_cb)
# SI 8e, 8f: CB no_thre + LR_ARI sweep + L/R confusion
plot.plot_cb_lr_clustering_sweep(sweep, n0)
plot.plot_cb_lr_cluster_confusion(match)
print(f'SI 8i CB L->R Hungarian-optimal cluster assignment (n_L={match["n_clu_L"]}, n_R={match["n_clu_R"]}):')
for r, c in zip(match['row_ind'], match['col_ind']):
    print(f'  L cluster {r + 1}  ->  R cluster {c + 1}')
# SI 8g, 8h: left-side CB pathway examples + pathways + dendrogram (9 clusters to parallel right side)
feat_cb_L = cb_data.get_cb_features(side_char='l')
cl_cb_L   = cb_data.get_cb_clusters(side_char='l', frac_thre=FRAC_CB)
cluster_num_cb_L = cl_cb_L['cluster'].value_counts().sort_index()
in_cb_L = [c for c in feat_cb_L.columns if c.endswith('_L')]
plot.plot_cb_clusters_example_features(feat_cb_L, in_cb_L, plot._CB_CLUSTER_TYPES['l'], colored_seed, side_str='_left')
plot.plot_cb_clusters_pathway_features(feat_cb_L, in_cb_L, cl_cb_L, cluster_num_cb_L, colored_seed, side_str='_left')
# SI 8h bottom: left-CB dendrogram (parallels Main 4d bottom cb_dendrogram_clusters.pdf)
plot.plot_clusters_dendrograms(feat_cb_L, frac_thre=FRAC_CB, side_str='_left_cb', save_dir=config.FIG_DIR / 'CB_pathways', compact_lastp=11)
# SI 10b: CB cluster output / input graphs (uses ol_cb_cluster_connectivity)
conn = cb_data.get_ol_cb_cluster_connectivity(ol_frac_thre=FRAC_OL, cb_frac_thre=FRAC_CB)
plot.plot_ol_cb_cluster_connectivity(conn)
# SI 10c: CB participation n_clu out
plot.plot_cb_participation_n_clu_out(type_part_cb_out, colored_main_groups)
# SI 10d: binoc VIC L/R
plot.plot_cb_vic_binocularity(vic_binoc_type, colored_region)

pf.distribute('si_8')


# %% [markdown]
# <a id="si-9"></a>
# ## SI Fig. 9
#

# %%
import importlib

import pandas as pd
import scipy.sparse as sp

from utils import config
from utils.palettes import load_colors
from utils import cb_data, core_data
from utils import plotting as plot
plot = importlib.reload(plot)

plot.set_html_output(False)

# full-brain propagation matrix (~9 GiB resident) -- shared with Main Fig. 7 /
# SI Fig. 7f via load_full_stepsn(), not reloaded here (requires the Main
# Fig. 7 cell to have run in this session). Freed at the end of this section
# (see release_full_stepsn() below) since nothing after SI Fig. 9 needs it.
full_stepsn = load_full_stepsn()

# --- 1. Pathway diagrams (like Main Fig. 5b) ---
# Effective-weight pathway diagrams from OL visual inputs to ellipsoid-body
# ring neurons, rendered with `plot.plot_pathway` (same machinery as the
# central-brain example pathways in Main Fig. 4b). Nodes are coloured by
# main group (ring neurons / non-OL in the CB colour) and outlined by sign;
# the number in the corner is the fraction of total input weight captured by
# the displayed paths.

# The five SI Fig. 9 central-neuron example pathways (order matches the xlsx manifest).
RING_INSTANCES = ['ER2_c_R', 'ER4d_R', 'PVLP008_a1_R', 'PVLP133_R', 'DNp03_R']
THRE_DICT = {
    'ER4d_R': 0.8,
    'ER2_c_R': 0.8,
    'DNp03_R': 0.4,
    'PVLP008_a1_R': 0.4,
    'PVLP133_R': 0.4,
}

colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()

_meta = core_data.get_meta()
type_to_sign_full = dict(zip(_meta.cell_type, _meta.sign))
sign_to_color_full = colored_sign['color'].to_dict()

# Colour CB / nonOL nodes in the CB region colour, OL nodes by main group.
_cb_color_df = pd.concat([
    pd.DataFrame(colored_region.loc['CB', 'color'], index=['nonOL'], columns=['color']),
    colored_main_groups,
])
type_to_color_cb = dict(zip(_meta.cell_type, _cb_color_df.loc[_meta.main_groups, 'color']))

flow_type_full = core_data.get_flow().groupby('instance')['hitting_time'].median().reset_index()

ring_dir = config.FIG_DIR / 'ring_neurons'

for inst in RING_INSTANCES:
    frac, _ = plot.plot_pathway(
        cb_data.get_cb_paths_to_instance(inst), inst, flow_type_full,
        node_text_size=24,
        node_size=2000,
        thre_cumsum=THRE_DICT[inst],
        neuron_to_color=type_to_color_cb,
        neuron_to_sign=type_to_sign_full, sign_color_map=sign_to_color_full,
        save_path=ring_dir / f'{config.FIG_DATASET}_{inst[:-2]}_pathway.pdf',
    )
    print(f'{inst}: displayed-path weight fraction = {frac:.2f}')

# %%
# --- 2. ARF plots for central-field example neurons ---
# For each TuBu (`TuBu01_R`–`TuBu10_R`) and ring (`ER4d_R`, `ER2_c_R`) cell type,
# pick the single neuron whose **2D-Gaussian ARF centre** (`x0`, `y0`, read from
# the RF-fit file) is closest to a **fixed centre of the eye** at
# `(x0, y0) = (0, 25)`. Draw that neuron's propagated effective-weight ARF (hex
# heatmap, same machinery as the Main Fig. 7b companion) and report its
# 2D-Gaussian size and alpha-hull size against the per-type medians. Sizes are in
# column² (×25 → deg²).

import re
import numpy as np
from utils import ol_rf

# Fixed centre of the eye, in hex-isotropic column coords (compute_rf scales y
# by 1/sqrt(3)), matching the x0/y0 stored in the 2D-Gaussian RF-fit file.
EYE_X, EYE_Y = 0.0, 25.0

raw_cb = ol_rf.get_rf_raw_cb()              # per-body 2D-Gaussian ARF fits (x0/y0/size/...)
# raw_cb = ol_rf.get_rf_raw_ol()
ahull  = ol_rf.get_rf_ahull_neuron()        # per-neuron alpha-hull RF size

# The five SI Fig. 9 central neurons (same set/order as RING_INSTANCES above);
# §2 picks one body per type (nearest ARF centre to the eye centre) and §3 draws
# the combined VPN ARFs onto it. Picked bodies match the xlsx body IDs:
# ER2_c=20945, ER4d=19323, PVLP008_a1=70604, PVLP133=42804, DNp03=10989.
TARGET_TYPES = ['ER2_c_R', 'ER4d_R', 'PVLP008_a1_R', 'PVLP133_R', 'DNp03_R']

ring = raw_cb[raw_cb['instance'].isin(TARGET_TYPES)].copy()
ring['dist_to_center'] = np.hypot(ring['x0'] - EYE_X, ring['y0'] - EYE_Y)

# nearest neuron to the fixed eye centre, per cell type (kept in TARGET order)
picks = ring.loc[ring.groupby('instance')['dist_to_center'].idxmin()].copy()
picks['instance'] = pd.Categorical(picks['instance'], categories=TARGET_TYPES, ordered=True)
picks = picks.sort_values('instance').reset_index(drop=True)

print(f'fixed eye centre: x0 = {EYE_X:.1f}, y0 = {EYE_Y:.1f}\n')
for _, r in picks.iterrows():
    g = ring[ring['instance'] == r['instance']]
    print(f"{r['instance']}: nearest body {int(r['bodyId'])} "
          f"@ ARF centre ({r['x0']:.2f}, {r['y0']:.2f}), dist {r['dist_to_center']:.2f}")
    print(f"    selected neuron  : gaussian size = {r['size']:7.2f} col2 "
          f"({r['size'] * 25:6.1f} deg2)   ahull size = {r['ahull_size']:7.2f} col2 "
          f"({r['ahull_size'] * 25:6.1f} deg2)")
    print(f"    {r['instance']} median (n={len(g)}): "
          f"gaussian size = {g['size'].median():7.2f} col2   "
          f"ahull size = {g['ahull_size'].median():7.2f} col2")

picks[['instance', 'bodyId', 'x0', 'y0', 'dist_to_center', 'size', 'ahull_size']]

# --- Propagated ARF heatmaps ---
# The cell below recomputes each picked neuron's propagated effective-weight RF
# from the **full-brain** propagation matrix (`<dataset>_<side>_lat_flow_sum.npz`,
# ~6 GB resident) and saves a PDF (needs the `kaleido` backend) — run on a
# compute node. PDFs are written to `results/ring_neurons/`.

import scipy.sparse as sp
from utils.ol_rf import _default_visual_input_instances

full_meta   = core_data.get_meta()
vis_in      = _default_visual_input_instances(config.SIDE_CHAR)
# vis_in = ['L1_R', 'L2_R', 'L3_R']


arf_dir = config.FIG_DIR / 'ring_neurons'
for _, r in picks.iterrows():
    fig, bid = plot.plot_propagated_arf_example(
        full_meta, full_stepsn, raw_cb, r['cell_type'], vis_in,
        bodyId=int(r['bodyId']), save_dir=arf_dir,
    )
    fig.show()
from utils import ol_rf

_inidx = full_meta[full_meta.instance.isin(vis_in)].idx.values
vcbn_arf_cmax = {}
for _, _r in picks.iterrows():
    _oidx = full_meta[full_meta.bodyId == int(_r['bodyId'])].idx.values
    _cdf = ol_rf.compute_rf(full_meta, full_stepsn, _inidx, _oidx)
    vcbn_arf_cmax[_r['instance']] = (
        float(_cdf['effective weight'].abs().max()) if not _cdf.empty else None
    )
print(vcbn_arf_cmax)

# %%
# --- 3. Combined VPN ARFs onto each central neuron ---
# For each VCBN central neuron (the per-type pick from §2), pool the VPN neurons
# that **directly** synapse onto it and draw their **combined ARF** — the
# propagated effective weight from visual inputs, summed across those neurons in
# each visual column (`plot.plot_combined_arf`).
# Two cells: the first lists, per central neuron, the VPN cell types that directly
# connect (raw synapse counts from `prop.npz`), to ground the manual choice; the
# second takes a hand-picked `VPN_TYPES` subset, selects the neurons of each type
# with at least `MIN_SYN` direct synapses onto the central neuron, and plots their
# combined ARF (one hex heatmap per central neuron × VPN type).

# Direct presynaptic VPN cell types for each central neuron, from the raw
# synapse-count matrix `prop.npz` (idx_pre -> idx_post). Use these tables to
# fill in `VPN_TYPES` below.
prop = core_data.get_full_prop().tocsc()          # raw synapse counts
side_word = {'r': 'right', 'l': 'left'}[config.SIDE_CHAR]
vpn_mask = ((full_meta.main_groups == 'VPN') & (full_meta.region == f'{side_word} OL')) |\
            (full_meta.region=='CB')


def vpn_direct_inputs(central_bid):
    """Per-VPN-type direct-input summary onto `central_bid`, sorted by synapses."""
    cidx = int(full_meta[full_meta.bodyId == central_bid].idx.values[0])
    w = np.asarray(prop[:, cidx].todense()).ravel()
    df = full_meta[vpn_mask].copy()
    df['syn'] = w[df.idx.values]
    df = df[df['syn'] > 0]
    return (
        df.groupby('instance')
          .agg(n_neurons=('syn', 'size'), tot_syn=('syn', 'sum'), max_syn=('syn', 'max'))
          .sort_values('tot_syn', ascending=False)
    )


for _, r in picks.iterrows():
    print(f"\n=== {r['instance']}  central body {int(r['bodyId'])}: direct VPN inputs ===")
    print(vpn_direct_inputs(int(r['bodyId'])).head(12).to_string())

# --- Combined ARF per VPN type ---
# Edit `VPN_TYPES` (keyed by VCBN instance) to the VPN cell types of interest for
# each central neuron, picked from the candidate tables above. `MIN_SYN` sets the
# minimum number of direct synapses for a VPN neuron to count as connected to the
# central neuron. One hex heatmap per (central neuron, VPN type); PDFs go to
# `results/ring_neurons/`.

# Manually chosen VPN cell types feeding each VCBN central neuron (see the
# candidate tables above). Keys are VCBN instances; values are VPN instances.
VPN_TYPES = {
    'PVLP008_a1_R': ['LC6_R', 'LC16_R', 'LT62_R', 'LC26_R'],
    'PVLP133_R':    ['LC16_R', 'LC6_R', 'LT79_R', 'LC25_R'],
    'ER4d_R':      ['TuBu08_R', 'ER4m_R', 'AOTU046_R'],
    'ER2_c_R':     ['TuBu10_R'],
    'DNp03_R':     ['LPLC4_R', 'LPLC1_R', 'LC4_R', 'LC22_R']
}
MIN_SYN = 1   # min direct synapses (VPN neuron -> central neuron) to count as connected


# input-fraction matrix: inprop[pre, post] = fraction of post's input coming from pre
if 'inprop' not in globals():
    # inprop = sp.load_npz(config.DATA_DIR / f'{config.DATASET}_inprop.npz').tocsc()
    inprop = sp.load_npz(config.DATA_DIR / f'{config.DATASET}_r_lat_flow_0.npz').tocsc()

central_of = dict(zip(picks['instance'], picks['bodyId']))

for vcbn, vpn_types in VPN_TYPES.items():
    if vcbn not in central_of:
        print(f'skip {vcbn}: no central-neuron pick in `picks`')
        continue
    central_bid = int(central_of[vcbn])
    cidx = int(full_meta[full_meta.bodyId == central_bid].idx.values[0])
    syn = np.asarray(prop[:, cidx].todense()).ravel()     # direct synapses onto central (selection)
    inw = np.asarray(inprop[:, cidx].todense()).ravel()   # inprop weight onto central (per-neuron weight)
    vcbn_cmax = vcbn_arf_cmax.get(vcbn)                    # share this VCBN's central-ARF scale (§2)
    for vpn in vpn_types:
        ridx = full_meta[full_meta.instance == vpn].idx.values
        sel_idx = ridx[syn[ridx] >= MIN_SYN]              # VPN neurons that connect
        if len(sel_idx) == 0:
            print(f'{vcbn} <- {vpn}: no neuron with >= {MIN_SYN} direct syn onto central')
            continue
        sel_bids = full_meta.loc[sel_idx, 'bodyId'].astype('int64').values
        weights = {int(b): float(inw[i]) for b, i in zip(sel_bids, sel_idx)}  # inprop weight per neuron
        # combined ARF = sum_neurons inprop(neuron -> central) * ARF(neuron)
        fig, _ = plot.plot_combined_arf(
            full_meta, full_stepsn, sel_bids, vis_in,
            weights=weights, label=vpn, cmax=vcbn_cmax,    # inprop-weighted, VCBN colour scale
            title=f'{vpn} -> {vcbn} ({central_bid}): inprop-weighted ARF (n={len(sel_bids)})',
            save_dir=arf_dir,
            save_name=f'{config.FIG_DATASET}_{vcbn[:-2]}_{central_bid}_{vpn[:-2]}_combined_arf.pdf',
        )
        print(f'{vcbn} <- {vpn}: {len(sel_bids)} neurons, inprop weight sum = {sum(weights.values()):.4f}')
        if fig is not None:
            fig.show()

pf.distribute('si_9')

# Nothing after SI Fig. 9 needs the full-brain propagation matrix -- release it.
for _name in ('full_stepsn', '_stepsn_7f'):
    globals().pop(_name, None)
release_full_stepsn()
import gc
gc.collect()

plot.set_html_output(True)  # section disabled HTML sidecars; re-enable for SI 10/13/14

# %% [markdown]
# <a id="si-10"></a>
# ## SI Fig. 10
#

# %%
# SI 10a: effective number of visual inputs per instance (VPN + VCBN) + the
# VPN-vs-VCBN statistical test. Per instance, normalise its effective visual-input
# vector to sum 1 and take effective N = exp(Shannon entropy): 1 = one input
# (selective), n_inputs = uniform (fully mixed). Box per cluster = distribution of
# that per-instance selectivity (line = median).
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

FRAC_OL = 0.19
FRAC_CB = 0.17
colored_region, colored_main_groups, _, _ = load_colors()


from utils.figure_helpers import _instance_effective_n


feat_vpn = ol_data.get_ol_features()
cl_vpn   = ol_data.get_ol_clusters(frac_thre=FRAC_OL)
feat_cb  = cb_data.get_cb_features(side_char='r')
cl_cb    = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB)

_eff = {}
for _name, _feat, _clu, _labels_fn, _dir, _color in [
    ('VPN',  feat_vpn, cl_vpn, lambda n: [f'oc{i}' for i in range(1, n + 1)],
     config.FIG_DIR / 'clustering', colored_main_groups.loc['VPN', 'color']),
    ('VCBN', feat_cb,  cl_cb,  lambda n: plot._cb_cluster_labels(n, 'r'),
     config.FIG_DIR / 'CB_clustering', colored_region.loc['CB', 'color']),
]:
    _df, _n_in = _instance_effective_n(_feat, _clu)
    _eff[_name] = (_df, _n_in)
    _n_clu = int(_df['cluster'].max())
    _labels = _labels_fn(_n_clu)
    _fig = go.Figure()
    for _c in range(1, _n_clu + 1):
        _sub = _df[_df.cluster == _c]
        _fig.add_trace(go.Box(
            y=_sub['eff_n'], name=_labels[_c - 1],
            marker_color=_color, line_color=_color,
        ))
    _fig.add_hline(y=_n_in, line_dash='dash', line_color='gray',
                   annotation_text=f'{_n_in} inputs (fully mixed)')
    _fig.update_layout(
        title=f'{_name} per-instance input mixing',
        xaxis_title='cluster', yaxis_title='effective # visual inputs (per instance)',
        yaxis_range=[0, _n_in + 0.3], xaxis_tickangle=0, showlegend=False,
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='arial', size=18), height=500, width=455,
    )
    _fig.update_xaxes(title_standoff=0, **plot._AXIS)
    _fig.update_yaxes(title_standoff=2, **plot._AXIS)
    plot._save(_fig, _dir, f'{config.FIG_DATASET}_{_name}_cluster_effective_n.pdf', html=True)
    _med = _df.groupby('cluster')['eff_n'].median()
    print(f'{_name}: per-instance median effective # inputs per cluster (of {_n_in}): '
          + ', '.join(f'{_labels[c - 1]}={_med[c]:.1f}' for c in range(1, _n_clu + 1)))

# --- Statistical test: is N_eff smaller for VCBNs than for VPNs? ---
# One-sided Mann-Whitney U on the pooled per-instance N_eff values.
(_vpn_df, _vpn_n), (_vcbn_df, _vcbn_n) = _eff['VPN'], _eff['VCBN']
assert _vpn_n == _vcbn_n, f"input-channel ceiling differs: VPN={_vpn_n}, VCBN={_vcbn_n}"
_vpn = _vpn_df['eff_n'].to_numpy()
_vcbn = _vcbn_df['eff_n'].to_numpy()
_U, _p_less = stats.mannwhitneyu(_vcbn, _vpn, alternative='less')
_n1, _n2 = len(_vcbn), len(_vpn)
_cliffs = 2 * _U / (_n1 * _n2) - 1                    # negative => VCBN typically smaller
_, _p_two = stats.mannwhitneyu(_vcbn, _vpn, alternative='two-sided')
print(f'\nEffective # visual inputs (of {_vpn_n} channels), per instance:')
print(f'  VPN : n={_n2:4d}  median={np.median(_vpn):.2f}  '
      f'IQR=[{np.percentile(_vpn, 25):.2f}, {np.percentile(_vpn, 75):.2f}]')
print(f'  VCBN: n={_n1:4d}  median={np.median(_vcbn):.2f}  '
      f'IQR=[{np.percentile(_vcbn, 25):.2f}, {np.percentile(_vcbn, 75):.2f}]')
print('Mann-Whitney U (H1: VCBN N_eff < VPN N_eff):')
print(f'  U={_U:.1f}  one-sided p={_p_less:.3e}  two-sided p={_p_two:.3e}  '
      f"Cliff's delta={_cliffs:+.3f} (negative => VCBN typically smaller)")

# --- SI Fig. 10e — VCN input pie ---
# Total synaptic input into the right-side VCN cell types, split by
# presynaptic category using the same decomposition as the Fig. 6a Sankey
# (VIN, VPN clusters `oc1..oc9`, VCBN clusters `cc1..cc10`, non-visual CB).
# Raw synapse counts from the full `prop` matrix; colours match the Sankey;
# slices below 1% are omitted. PDF written to `results/CB_pathways/` (needs
# the `kaleido` PDF backend).

# Pie of total synapses into right VCNs by presynaptic category (Fig. 6a
# decomposition + colours). Independent of the heavy propagation cells
# above: needs only the (cached) meta, prop matrix, and 6a Sankey grouping.
from utils import cb_data
from utils.palettes import load_colors

pie_meta   = core_data.get_meta()
pie_prop   = core_data.get_full_prop()
pie_sankey = cb_data.get_visual_system_sankey_connectivity(
    ol_frac_thre=0.19, cb_frac_thre=0.17,
    weight_thre=1000, min_plot_neurons=120, output_normalized=False,
)
colored_region, colored_main_groups, _, _ = load_colors()

fig_pie, pie_breakdown = plot.plot_vcn_input_pie(
    pie_sankey, pie_meta, pie_prop, colored_region, colored_main_groups,
    side_char=config.SIDE_CHAR, min_pct=1.0,
    save_dir=config.FIG_DIR / 'CB_pathways',
)
print(f"Total synapses into {config.SIDE_CHAR}-VCNs: {pie_breakdown['synapses'].sum():,.0f}")
print('Omitted (<1%):', ', '.join(pie_breakdown.loc[pie_breakdown['pct'] < 1.0, 'category']))
fig_pie.show()
pie_breakdown

pf.distribute('si_10')

# %% [markdown]
# <a id="si-11"></a>
# ## SI Fig. 11 & 12 — VIC projections
#
# These SI panels reuse the shared VIC-projection setup loaded in the Main Fig. 5 section, so they
# need to be run after that section.

# %%
# --- VIC, all 5 channels, left side — SI Fig. 11a ---

lr = 'l'
meta_cb_vpn = vic.load_vp_cb(lr)
syn = syn0.merge(
    meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'main_groups']], on='bodyId')

plot_contribution_layers(
    syn, 'vision', plot_dir=plot_dir, outlines=outlines_bkgd,
    vmax={'VPN': [0.26], 'nonOL': [0.11, 0.06, 0.02]},
    fname=lambda gp, lo, hi: f'vic_ht_{gp}_{lo}_{hi}_{lr}.pdf')

# --- VIC, individual input channels — SI Fig. 11b ---

meta_cb_vpn = vic.load_vp_cb('r')
vi_ls = ['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R']
for vi in vi_ls:
    df = vic.get_effweight_channel(vi, meta_cb_vpn)
    syn = syn0.merge(df, on='bodyId')
    plot_contribution_layers(
        syn, 'VIC', plot_dir=plot_dir, outlines=outlines_bkgd, vmax='p99',
        fname=lambda gp, lo, hi: f'vic_{vi}_ht_{gp}_{lo}_{hi}.pdf')

# --- VIC by CB (VCBN) cluster — SI Fig. 12 ---

from utils import cb_data
FRAC_CB = 0.17
cl_r = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB, force=True)

meta_cb_vpn = vic.load_vp_cb('r').merge(cl_r, on='instance', how='left')
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.cluster.notna()]  # drop OL cells
meta_cb_vpn['cluster'] = meta_cb_vpn.cluster.astype(int)
syn = syn0.merge(
    meta_cb_vpn[['bodyId', 'instance', 'main_groups', 'vision', 'ht', 'cluster']], on='bodyId')

for clu in range(1, cl_r.cluster.max() + 1):
    plot_contribution_layers(
        syn[syn.cluster == clu], 'vision', plot_dir=plot_dir, outlines=outlines_bkgd,
        gp_layers={'nonOL': [2, 3.5, 4, 5]}, vmax='p99', show_ninst=True,
        fname=lambda gp, lo, hi: f'vic_clu_{clu}_ht_{gp}_{lo}_{hi}.pdf')

pf.distribute('si_11')
pf.distribute('si_12')

# %% [markdown]
# <a id="si-13"></a>
# ## SI Fig. 13

# %%
FRAC_CB = 0.17

rf_all      = ol_rf.get_rf_types_combined()
rf_vpn_vcbn = rf_all[(rf_all['main_groups'] == 'VPN') | (rf_all['region'] == 'CB')]
rf_type_cb  = ol_rf.get_rf_type_cb()
ahull_cb    = ol_rf.get_rf_ahull_type_cb()   # per-instance alpha-hull size (VPN + VCBN)
cl_r        = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB)
tb_clu      = cb_data.get_ol_to_cb_weights()
coverage    = cb_data.get_ol_roi_coverage(seed=0)

colored_region, colored_main_groups, _, _ = load_colors()
# SI 10a: VPNs colored by VPN blue (main_groups), CB stays dark gray (region) — mirrors Main 4a.
_rf_colors = colored_region.copy()
_rf_colors.loc['right OL', 'color'] = colored_main_groups.loc['VPN', 'color']

# SI 10b/c/d use the alpha-hull arbor size (cache/data_proc ahull pkls,
# via get_rf_ahull_type_cb) instead of the 2D-Gaussian RF fit `size`.
ahull_vcbn = ahull_cb[ahull_cb['region'] == 'CB']   # VCBNs only (drop VPNs)

# SI 10a (new): VCBN alpha-hull arbor size vs propagated RF size — VCBN analogue
# of the OL examples_ahull_reldiff_vs_size panel. Writes
# cb_examples_size_vs_ahull.pdf + cb_examples_ahull_reldiff_vs_size.pdf.
# NOTE: not yet wired into paper_figs.xlsx (10a still maps to the legacy r2 panel).
_vcbn_cmp = rf_type_cb[['instance', 'size', 'region']].merge(
    ahull_vcbn[['instance', 'ahull_size']], on='instance', how='inner',
)
plot.plot_cb_vcbn_ahull_size_comparison(_vcbn_cmp, colored_region)
# SI 10a (legacy, current manifest source): CB layer vs r2 + r2 cumsum
plot.plot_cb_rf_r2_summary(rf_vpn_vcbn, _rf_colors)
# SI 10b: CB alpha-hull size hist + size vs layer (VCBNs only; highlights = Main 4c example set)
plot.plot_cb_ahull_size_summary(ahull_vcbn, colored_region)
# SI 10c: examples alpha-hull size vs cluster (CB types)
plot.plot_cb_clusters_size_vs_cluster(
    cl_r, ahull_vcbn, colored_region,
    size_col='ahull_size', size_lo=5, size_hi=1000, size_tickvals=(10, 100, 1000),
)
# SI 10d: OL->CB highres subset (tick labels = alpha-hull size; rows=VPN, cols=VCBN — keeps both)
plot.plot_out_cb_highres_sub(tb_clu, ahull_cb.set_index('instance')['ahull_size'])
# SI 10e: relative coverage per ROI summary
plot.plot_ol_roi_coverage_summary(coverage)

pf.distribute('si_13')

# %% [markdown]
# <a id="si-14"></a>
# ## SI Fig. 14 — flywire
#
# Runs last: monkeypatches the shared `core_data` / `polarity` loaders for flywire, so it must follow every malecns panel. Combines malecns + flywire polarity predictions (14a/14b) and the flywire dimorphism panel (14c).

# %%
# --- Flywire polarity predictions (SI 14a/14b) — flywire-fafb-v783b, left side ---
# Flywire adaptation of `figures.ipynb` cells 19 (Main 2d) and 24 (Main 4d),
# rendered in **1-to-1 correspondence with the malecns figures**: the bar set,
# order, and panel split (VIN / VPN / VCBN) are taken from the **malecns**
# reference (`malecns_v1.0`, right side). Each malecns bar gets the flywire
# value for the corresponding cell type, or a **blank** where flywire has none —
# so positions never shift and the two datasets can be compared bar-for-bar.
# Experimental labels come from `TableS2_experimental_data.xlsx` with `*`
# wildcard expansion, ambiguous-polarity removal, and per-entry averaging —
# identical to the malecns pipeline (shared `_build_polarity_labels`), so bar
# names/order match the malecns reference bar-for-bar.
# Writes to `results/exp_comparison/` (bar counts follow the live malecns
# reference, which tracks the spreadsheet):
# - `flywire-fafb-v783b_polarity_predictions_other.pdf` → Main 2d, VINs
# - `flywire-fafb-v783b_polarity_predictions_vpn.pdf`   → Main 2d, VPNs
# - `flywire-fafb-v783b_polarity_predictions_cb.pdf`    → Main 4d, VCBNs
# Flywire-only runtime adaptations (shared modules untouched):
# 1. **Region fix:** `superclass=='optic'` → `'{side} OL'` (no-op on malecns).
# 2. **Alias map:** spreadsheet type → flywire instance (incl. 1→many / many→1);
# wildcards expand against flywire types directly.
# 3. **Display rename:** `TmY9q__perp` → `TmY9qp`.

import pandas as pd

from utils import core_data, polarity
from utils import plotting as plot
from utils.config import DATA_DIR, PARAMS_DIR, FIG_DIR
from utils.palettes import load_colors
from utils.polarity import _build_polarity_labels, _side_cell_types

plot.set_html_output(True)              # write HTML sidecars alongside the PDFs
plot._POLARITY_DISPLAY_RENAMES = {      # (3) display rename
    **plot._POLARITY_DISPLAY_RENAMES, 'TmY9q__perp': 'TmY9qp',
}

DS, SC = 'flywire-fafb-v783b', 'l'
SUFFIX = '_R' if SC == 'r' else '_L'
MAL, MAL_SC = 'malecns_v1.0', 'r'       # malecns reference (matches figures.ipynb)

colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()

# (1) region fix — applied to the shared loader for this session only.
# Idempotent: stash the TRUE original on the module so re-running this cell — or
# the dimorphism cell below, which patches the same loader — never wraps a
# wrapper (stacking wrappers is what caused the RecursionError).
if not hasattr(core_data, '_load_meta_groups_orig'):
    core_data._load_meta_groups_orig = core_data._load_meta_groups
def _patched_lmg(dataset, data_dir):
    meta = core_data._load_meta_groups_orig(dataset, data_dir)
    opt = meta.superclass == 'optic'
    meta.loc[opt & (meta.side == 'left'), 'region'] = 'left OL'
    meta.loc[opt & (meta.side == 'right'), 'region'] = 'right OL'
    return meta
core_data._load_meta_groups = _patched_lmg

# --- malecns reference (defines bar set, order, and VIN/VPN/VCBN split) ---
# Uses the **original** `get_polarity_experiments` (no flywire aliases).

ol_mal = polarity.get_ol_polarity_comparison(dataset=MAL, side_char=MAL_SC)
cb_mal = polarity.get_cb_polarity_comparison(dataset=MAL, side_char=MAL_SC)
ref_other = ol_mal[ol_mal.main_groups != 'VPN'].reset_index(drop=True)   # VINs
ref_vpn = ol_mal[ol_mal.main_groups == 'VPN'].reset_index(drop=True)     # VPNs
ref_cb = cb_mal.reset_index(drop=True)                                   # VCBNs
print(f'malecns bars — 2d_other(VIN): {len(ref_other)}, 2d_vpn(VPN): {len(ref_vpn)}, 4d(VCBN): {len(ref_cb)}')

# --- Flywire adaptations + flywire comparisons ---

# (2) alias map: spreadsheet instance (+_L) -> flywire instance(s) (+_L)
ALIASES = {
    'Dm3a_L': 'Dm3p_L', 'Dm3b_L': 'Dm3q_L', 'Dm3c_L': 'Dm3v_L',
    'Pm1_L': 'Pm01_L', 'Pm3_L': 'Pm09_L', 'Pm5_L': 'Pm02_L',
    'TmY9a_L': 'TmY9q__perp_L', 'Tm29_L': 'Tm5d_L', 'Tm12_L': 'Tm25_L',
    'Y13_L': 'Y1_L', 'TmY18_L': 'Tm27_L',
    # flywire has no PVLP008_a1..a4 arbor subtypes (malecns does), only a single
    # PVLP008; the `PVLP008_a*` wildcard resolves to the synthetic `PVLP008_a`
    # injected into side_types below, which this maps onto flywire's instance.
    'PVLP008_a_L': 'PVLP008_L',
    'TuBu06_L': ['TuBu06a_L', 'TuBu06b_L'],
    'TuBu09_L': 'TuBu09,TuBu10_L', 'TuBu10_L': 'TuBu09,TuBu10_L',
    'PLP219_L': 'CB2102_L',
    'PVLP112_L': ['PVLP112a_L', 'PVLP112b_L'],
    's-LNv_L': 's-LNv_b_L', '5thsLNv_LNd6_L': 's-LNv_a_L',
    # SLP249 -> pSP2/3: not in flywire; left unmapped (renders as a blank bar)
}

def _flywire_labels(*, dataset, data_dir=DATA_DIR, side_char='l',
                    source_file='TableS2_experimental_data.xlsx',
                    sheet='contrast selectivity', force=False, **kw):
    # Same pipeline as the malecns `get_polarity_experiments` (wildcard expand,
    # ambiguous-polarity drop, `*`-stripped group_labels) — but wildcards expand
    # against flywire types and `type + suffix` is remapped to flywire instances
    # via ALIASES (1->many remaps explode). group_labels therefore match the
    # malecns reference bar names, so `_align` can fill them.
    suffix = '_R' if side_char == 'r' else '_L'
    df = pd.read_excel(PARAMS_DIR / source_file, sheet_name=sheet)
    side_types = _side_cell_types(dataset, data_dir, side_char)
    # flywire lacks the PVLP008_a1..a4 subtypes the `PVLP008_a*` wildcard targets
    # (it has a single PVLP008), so the wildcard would expand to nothing. Add the
    # synthetic prefix so it resolves to one type that ALIASES maps onto flywire's
    # PVLP008 instance, keeping the bar aligned to the malecns `PVLP008_a` ref bar.
    side_types = side_types | {'PVLP008_a'}
    out = _build_polarity_labels(df, side_types, suffix, lambda x: ALIASES.get(x, x))
    ol_instances = set(
        core_data.get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        .instance.dropna().unique()
    )
    out['in_ol'] = out['instance'].isin(ol_instances)
    return out[['type', 'Contrast selectivity', 'instance', 'in_ol', 'group_label']].reset_index(drop=True)
polarity.get_polarity_experiments = _flywire_labels   # flywire only, from here on

# rebuild flywire meta + OL sidecars with corrected region (only if stale)
ol_meta_path = DATA_DIR / f'{DS}_OL_{SC}_meta.csv'
need_rebuild = (
    not ol_meta_path.exists()
    or 'Dm3p_L' not in set(pd.read_csv(ol_meta_path, usecols=['instance']).instance)
)
if need_rebuild:
    for suff in ['meta.csv', 'flow_20step_0.1thre_hit.csv', 'prop.npz',
                 'lat_flow_sum.npz', 'lat_flow_0.npz']:
        (DATA_DIR / f'{DS}_OL_{SC}_{suff}').unlink(missing_ok=True)
    print('rebuilding flywire OL sidecars with corrected region…')
core_data.get_meta(dataset=DS, force=need_rebuild)
core_data.get_ol_meta(dataset=DS, side_char=SC, force=need_rebuild)
core_data.get_ol_flow_type(dataset=DS, side_char=SC, force=need_rebuild)

# Reuse the cached flywire comparisons (computed under the `_flywire_labels`
# override). NB: these caches are keyed by dataset/side only, so after editing
# TableS2 regenerate them once with force=True — otherwise the change won't show.
ol_fly = polarity.get_ol_polarity_comparison(dataset=DS, side_char=SC)
cb_fly = polarity.get_cb_polarity_comparison(dataset=DS, side_char=SC)


# %%
# --- Align flywire onto the malecns bar template (blanks where missing) & plot ---

def _align(ref_df, fly_df, key='group_label'):
    """One row per malecns reference bar, in malecns order. Fill flywire seed
    values for the matching cell type; zero-fill (blank bar) when absent."""
    seed_cols = [s + SUFFIX for s in colored_seed.index if s + SUFFIX in fly_df.columns]
    fly_idx = fly_df.drop_duplicates(key).set_index(key)
    rows = []
    for _, r in ref_df.iterrows():
        k = r[key]
        if k in fly_idx.index:
            f = fly_idx.loc[k]
            row = {c: float(f[c]) for c in seed_cols}
            row.update(instance=f['instance'], _blank=False)
            row['Contrast selectivity'] = f['Contrast selectivity']
        else:
            row = {c: 0.0 for c in seed_cols}
            row.update(instance=f'{r["group_label"]}{SUFFIX}', _blank=True)
            row['Contrast selectivity'] = r['Contrast selectivity']
        rows.append(row)
    return pd.DataFrame(rows)

# Malecns decides each bar's panel (OL VIN/VPN vs CB); fill the flywire value
# from whichever flywire comparison computed it. A type malecns calls VPN may be
# 'nonOL' in flywire (landing in cb_fly, e.g. s-LNv) and vice-versa — combining
# the two lookups keeps the malecns OL/CB classification while still filling it.
fly_all = pd.concat([ol_fly, cb_fly], ignore_index=True)
_seed = [s + SUFFIX for s in colored_seed.index if s + SUFFIX in fly_all.columns]
fly_all[_seed] = fly_all[_seed].fillna(0.0)

panel_other = _align(ref_other, fly_all)
panel_vpn = _align(ref_vpn, fly_all)
panel_cb = _align(ref_cb, fly_all)

for name, p in [('2d_other(VIN)', panel_other), ('2d_vpn(VPN)', panel_vpn), ('4d(VCBN)', panel_cb)]:
    nb = int(p['_blank'].sum())
    blanks = p.loc[p['_blank'], 'instance'].str[:-2].tolist()
    print(f'{name}: {len(p)} bars, {nb} blank {blanks}')

# Widths match the analogous malecns Fig 2d / 4d panels (paper_figures.ipynb);
# the 1:1 bar alignment means identical formulas give identical widths.
plot.plot_polarity_stacked_bars(
    panel_other, colored_seed, dataset=DS, filename_stem='polarity_predictions_other',
    width=int(max(400, 35 * len(panel_other) + 150) * 0.8 * 0.9),
)
plot.plot_polarity_stacked_bars(
    panel_vpn, colored_seed, dataset=DS, filename_stem='polarity_predictions_vpn',
    width=int(max(400, 35 * len(panel_vpn) + 150) * 0.9 * 0.95),
)
plot.plot_polarity_stacked_bars(
    panel_cb, colored_seed, dataset=DS, filename_stem='polarity_predictions_cb',
    width=int(max(400, 35 * len(panel_cb) + 150) * 0.9), font_size=12,
)

# %%
# --- Flywire dimorphic-neuron paper panel (SI 14c, visual fraction) ---
# Flywire (FAFB, female brain) analogue of the male-cns
# `*_dimorphic_neurons_paper_panel.pdf` produced by `dimorphism_analysis.py` §5:
# for each dimorphism category (**female-specific**, **sexually dimorphic**),
# a stacked bar showing the fraction of **CB** neurons that are *visual*
# (`VIC > 5e-4`, lighter) vs non-visual (darker).
# `dimorphism` is read straight from `flywire-fafb-v783b_meta.csv` (the column
# added earlier). Region / VIC use the flywire left-flow pipeline:
# - region: `core_data._load_meta_groups` with the optic-superclass region patch
# from `flywire_main_2d_4d_polarity.py` (optic -> left/right OL by side).
# - VIC: `vic.get_ol_cb_vic_raw(dataset, side_char='l')` (per-body summed
# effective input weight from the left visual-input seeds via `_l_lat_flow_sum`).

# Note: In male CNS vs. FlyWire, 3 CB cell types match by exact name but disagree
# on dimorphism status: CRE062 (dimorphic vs. isomorphic), and DNp68/SMP276 
# (dimorphic vs. "potentially" dimorphic — borderline calls). This results in 
# different counts of dimorphic neurons in the two datasets, but the visual fraction is still comparable.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv

PROJECT_ROOT = Path(find_dotenv()).parent
load_dotenv(PROJECT_ROOT / ".env")

from utils import core_data, vic

DS = "flywire-fafb-v783b"
SC = "l"  # flywire flow is left
VIC_THRE = 5e-4
CATEGORIES = ["female-specific", "sexually dimorphic"]

FIG_DIR = PROJECT_ROOT / "results" / "dimorphism"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# same colors as the male-cns panel (dimorphism_analysis.py CAT_BAR): the
# "-specific" category = pink, sexually dimorphic = teal; visual = lighter,
# non-visual = darker.
CAT_BAR = {
    "female-specific":    {"visual": "#f1b6da", "nonvisual": "#c51b7d"},
    "sexually dimorphic": {"visual": "#80cdc1", "nonvisual": "#01665e"},
}

# --- Region patch + meta (carries the new `dimorphism` column) ---
# Reuses `_patched_lmg`, defined in the SI 14a/14b polarity cell above (this
# cell must run after it — see the section markdown). Idempotent: stash the
# TRUE original on the module so this shares the patch with the polarity cell
# above instead of wrapping its wrapper (stacking wrappers is what caused the
# RecursionError).

if not hasattr(core_data, "_load_meta_groups_orig"):
    core_data._load_meta_groups_orig = core_data._load_meta_groups

core_data._load_meta_groups = _patched_lmg

meta = core_data.get_meta(dataset=DS, force=True)  # force: pick up dimorphism + patch
assert "dimorphism" in meta.columns, "meta has no dimorphism column"

# --- Per-neuron CB table: dimorphism + VIC -> visual flag ---

nrn = meta.loc[
    (meta.region == "CB") & (meta.dimorphism.isin(CATEGORIES)),
    ["bodyId", "idx", "region", "dimorphism"],
].copy()

vic_raw = vic.get_ol_cb_vic_raw(dataset=DS, side_char=SC)  # heavy: builds + caches
nrn = nrn.merge(vic_raw[["bodyId", "VIC"]], on="bodyId", how="left")
nrn["VIC"] = nrn["VIC"].fillna(0.0)          # CB body absent from VIC table -> 0
nrn["visual"] = nrn["VIC"] > VIC_THRE

n_by_cat = nrn["dimorphism"].value_counts()
vis_frac = nrn.groupby("dimorphism")["visual"].mean().reindex(CATEGORIES)
print(f"per-neuron visual fraction (VIC > {VIC_THRE:g}):")
for cat in CATEGORIES:
    n_vis = int(nrn[(nrn.dimorphism == cat) & nrn.visual].shape[0])
    print(f"  {cat}: {vis_frac[cat] * 100:.1f}% visual ({n_vis}/{int(n_by_cat[cat])})")

# --- Paper panel: visual fraction (standalone) ---

xpos = np.arange(len(CATEGORIES))


from utils.figure_helpers import _wrap


panel_xlabels = [f"{_wrap(c)}\n({int(n_by_cat[c])})" for c in CATEGORIES]
FS = 13
PANEL_W = 13 / 3 * 0.8

fig, ax = plt.subplots(figsize=(PANEL_W, 4.2))
for i, cat in enumerate(CATEGORIES):
    vf = vis_frac[cat]
    ax.bar(i, vf, color=CAT_BAR[cat]["visual"], edgecolor="white")
    ax.bar(i, 1 - vf, bottom=vf, color=CAT_BAR[cat]["nonvisual"], edgecolor="white")
    ax.text(i, vf / 2, f"{vf * 100:.0f}%\nvisual", ha="center", va="center",
            color="black", fontsize=FS - 1, fontweight="bold")
    ax.text(i, vf + (1 - vf) / 2, "non-visual", ha="center", va="center",
            color="white", fontsize=FS - 3)
ax.set(xticks=xpos, ylim=(0, 1))
ax.set_ylabel("fraction of neurons", fontsize=FS)
ax.set_xticklabels(panel_xlabels, fontsize=FS)
ax.tick_params(axis="y", labelsize=FS)
fig.tight_layout()
out = FIG_DIR / f"{DS}_dimorphic_neurons_paper_panel.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")

pf.distribute('si_14')

# %% [markdown]
# <a id="parity-check"></a>
# ## Parity check

# %%
pf.parity_check(exclude_folders=['si_5', 'si_6'], expected_missing=['si_12/si_12_25.pdf', 'si_12/si_12_26.pdf'])


# %% [markdown]
# <a id="rebuild-viewer-index"></a>
# ## Rebuild viewer index

# %%
# Regenerate results/html_figures/index.html from the distributed .html sidecars
# so the interactive-figure browser (linked from README) matches the current
# panel set. build_index() only globs html_figures/, so it is independent of the
# topic folders drain_topics() removes below.
pf.build_index()


# %%
# Topic folders are now redundant: every panel was copied into its
# main_<N> / si_<N> folder (+ html_figures/). Drain them so the deliverable
# tree holds each panel once, under its manuscript name. Guarded by the
# parity_check above (it raises on any gap, so nothing is deleted unless
# distribution completed). parity_check is one-shot: run it before, not after.
pf.drain_topics()

# %%

# %%
