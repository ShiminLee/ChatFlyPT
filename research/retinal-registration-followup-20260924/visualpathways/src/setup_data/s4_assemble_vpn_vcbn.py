# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %%
# run in this order: this (assemble vp_cb + effweight + RF caches) -> s5_tbar
# -> paper_figures VIC panels + galleries. The former standalone rf_size step is
# incorporated as the last section of this notebook.

# combine meta with vic and ht

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

# %%
from utils import olc_client
c = olc_client.connect(verbose=False)

# %%
import pandas as pd

from utils.config import DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(DATA_DIR)

# %% [markdown]
# # Some setups

# %% [markdown]
# ## Get meta

# %%
# Load meta the same way as paper_figures: the annotated meta (main_groups /
# region, AN's R7/8) via get_meta, instead of reading the raw {DATASET}_meta.csv.
from utils.core_data import get_flow, get_meta, get_ol_meta

meta = get_meta()
print(meta.shape)

# %% [markdown]
# # Compute effective weights from different visual input channels
#
# Pre-build the per-channel per-body effective-weight table for both sides — the
# heavy `lat_flow_sum` pass (one ~6 GB matrix load + one grouped pass per
# visual-input channel). Built first because everything below derives from it:
# `get_vcbn_types` / `get_ol_cb_vic_raw` (total VIC = sum of the channel columns)
# and the SI 11b per-channel VIC panels (`vic.get_effweight_channel`). Building it
# here means the rest of this notebook and the figure code read a cached table
# instead of re-loading the matrix.

# %%
from utils.vic import get_ol_cb_effweight_by_channel

effwt_ch_r = get_ol_cb_effweight_by_channel(side_char='r', force=True)
effwt_ch_l = get_ol_cb_effweight_by_channel(side_char='l', force=True)
print(effwt_ch_r.shape, effwt_ch_l.shape)

# %% [markdown]
# # vpn and vcbn, ht + vic

# %%
from utils.vic import get_vcbn_types

# VPN instances per side, using the same definition as the VPN clusters
# (get_ol_features): the OL-subset meta filtered to main_groups == "VPN".
ol_meta_r = get_ol_meta(side_char='r')
vpn_inst_r = ol_meta_r[ol_meta_r.main_groups == "VPN"].instance.unique()
ol_meta_l = get_ol_meta(side_char='l')
vpn_inst_l = ol_meta_l[ol_meta_l.main_groups == "VPN"].instance.unique()

# right-side vpn and vcbn
vcbn = get_vcbn_types(side_char='r')
vp_cb_r = meta[
    meta['instance'].isin(vcbn.instance) |
    meta['instance'].isin(vpn_inst_r)
]

# left-side vpn and vcbn
vcbn = get_vcbn_types(side_char='l')
vp_cb_l = meta[
    meta['instance'].isin(vcbn.instance) |
    meta['instance'].isin(vpn_inst_l)
]

vp_cb_r.shape, vp_cb_l.shape

# %%
# add ht and vic

# load hitting time (per-body) via the canonical loader
ht_r = get_flow(side_char='r')
ht_l = get_flow(side_char='l')
print(ht_r.shape, ht_l.shape)

from utils.vic import get_ol_cb_vic_raw

# load vic
vic_r = get_ol_cb_vic_raw(side_char='r')
vic_l = get_ol_cb_vic_raw(side_char='l')
print(vic_r.shape, vic_l.shape)

# combine
vp_cb_r = pd.merge(vp_cb_r[['bodyId','cell_type','idx']], ht_r, on='idx', how='inner')
vp_cb_l = pd.merge(vp_cb_l[['bodyId','cell_type','idx']], ht_l, on='idx', how='inner')
print(vp_cb_r.shape, vp_cb_l.shape)

vp_cb_r = pd.merge(vp_cb_r, vic_r[['bodyId','VIC', 'main_groups', 'region']], on='bodyId', how='inner')
vp_cb_l = pd.merge(vp_cb_l, vic_l[['bodyId','VIC', 'main_groups', 'region']], on='bodyId', how='inner')
print(vp_cb_r.shape, vp_cb_l.shape)

# %%
# save to cache
vp_cb_r.to_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l.to_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

# # load
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

# %% [markdown]
# # Build the combined RF caches (ellipse + alpha hull)
#
# Force-rebuilds the two per-neuron RF parquet caches:
#
# * `get_rf_raw_ol` — OL non-visual-input neurons (VIN + VPN + VCN), via the
#   OL-subset `lat_flow_sum`.
# * `get_rf_raw_cb` — VCBN neurons (median VIC > 5e-4), via the full
#   `lat_flow_sum`.
#
# Each getter fits a 2D Gaussian **and** an alpha hull from a *single*
# propagation pass per neuron (`utils.ol_rf._compute_rf_params`), so the ellipse
# `size` and the `ahull_size` live in the same parquet. This replaces the former
# `rf_fit_thr07.pkl` + `meta_ON_ahull_cumsum60.pkl` +
# `meta_VP_CB_ahull_cumsum60.pkl` pickles; every downstream consumer reads the
# parquet through the `ol_rf.get_rf_*` getters. Runs here (after the effweight /
# VIC caches above) because `get_rf_raw_cb` selects its output instances via
# `get_vcbn_types`. Heavy first run — the alpha hull dominates (~1-2 h, CB side).

# %%
from utils import ol_rf

# %% [markdown]
# ## OL non-visual-input: VIN + VPN + VCN (ellipse + alpha hull), OL-subset lat_flow_sum
#
# Right side only. `side_char='r'` is passed explicitly (rather than relying
# on the `side_char=SIDE_CHAR` default) because that default is bound once,
# at `ol_rf` import time, to whatever `config.SIDE_CHAR` happens to be in this
# process — passing it explicitly avoids depending on that process-wide state.

# %%
rf_ol = ol_rf.get_rf_raw_ol(side_char='r', force=True)
print('rf_raw_ol:', rf_ol.shape, '| hull-valid:', int(rf_ol['ahull_size'].notna().sum()))

# %% [markdown]
# ## VCBN (ellipse + alpha hull), full lat_flow_sum  (~1-2 h)

# %%
rf_cb = ol_rf.get_rf_raw_cb(side_char='r', force=True)
print('rf_raw_cb:', rf_cb.shape, '| hull-valid:', int(rf_cb['ahull_size'].notna().sum()))

# %% [markdown]
# # End
