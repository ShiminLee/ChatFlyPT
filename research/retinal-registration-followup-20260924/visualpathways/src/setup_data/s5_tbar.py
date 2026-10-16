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
# Query and cache the presynaptic T-bar (tbar) synapse locations for the
# VPN + VCBN neurons, plus the per-ROI central-brain tbar counts. These are the
# heavy, run-once neuprint queries that feed the visual-input contribution /
# resolution / coverage projection panels (Main Fig. 5g/5h, 6a, 7c-right,
# SI Fig. 11/12)
# rendered in make_figures/paper_figures.ipynb, and the per-instance galleries.
#
# Extracted from the old quan_propagation/contrib_plot_mip + pokemon_plot
# notebooks (the two cells marked "run once, ~100 min" and "run once, ~9 hours").
# Run AFTER s4_assemble_vpn_vcbn.py (needs vp_cb_hit_vic_{r,l}.p).

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
import neuprint
import json
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from collections import defaultdict

from utils.config import (CACHE_DIR, DATA_DIR, FIG_DIR)
from utils.query_roi import get_primary_rois


# %% [markdown]
# # Central-brain ROIs

# %%
rois_cb = get_primary_rois('CentralBrain')
len(rois_cb)

# %% [markdown]
# # Load vpn + vcbn (from s4_assemble_vpn_vcbn.py)

# %%
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

ids_lr = list(set(vp_cb_r['bodyId']).union(set(vp_cb_l['bodyId'])))
len(ids_lr)

# %% [markdown]
# # Query tbar in cb + vpn
# ~ 100 min

# %%
syn = neuprint.fetch_synapses(NC(bodyId=ids_lr), SC(rois=rois_cb, primary_only=True, type='pre'))

# save
syn.to_pickle(Path(DATA_DIR, 'tbar_vp_cb.pkl'))

# %%
# load vp_cb
lr = 'r'
meta_cb_vpn = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))
# print(len(meta_cb_vpn))
# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)
syn = pd.merge(syn, 
               meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'main_groups']], 
               on='bodyId', how='inner')
# print(syn.shape)

# save
syn[['bodyId','roi','vision']].to_pickle(Path(DATA_DIR, f'tbar_VIC_{lr}.pkl'))

# %% [markdown]
# # Sample tbar points for the R alpha-shape outline step

# %%
# 50k-point sample written to csv -> feeds the (offline, R) alpha-shape that
# produces params/outline_cb_{1,2}.csv used as the projection backgrounds.
xy = syn[['x', 'y']].sample(n=50000, random_state=42)
xy.to_csv(Path(DATA_DIR, 'tbar_cb_50k.csv'), index=False)

# %% [markdown]
# # CB ROIs tbar counts

# %%
# get syn count for each roi and save into a df
rois_list = json.dumps(rois_cb)

q = f"""
WITH {rois_list} AS rois
MATCH (n:Neuron)
WHERE any(r IN rois WHERE n[r])
RETURN n.roiInfo AS roiInfo
"""

roi_strings = neuprint.fetch_custom(q)

counts = defaultdict(int)
roiset = set(rois_cb)
for s in roi_strings['roiInfo']:
    ri = json.loads(s)
    for roi in roiset & ri.keys():
        counts[roi] += ri[roi].get('pre', 0)

tbar_count_roi_cb = pd.DataFrame(
    [(roi, counts.get(roi, 0)) for roi in rois_cb],
    columns=['roi', 'tbar_count'],
)

# save tbar_count_roi_cb
tbar_count_roi_cb.to_csv(Path(DATA_DIR, 'tbar_count_roi_cb.csv'), index=False)

# %%
