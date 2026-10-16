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
# # Compute flow hitting times
#
# Works for any dataset. Defaults to the config dataset (malecns, right side);
# to compute the **flywire** flow (left side), uncomment the override in the
# "Dataset selection" cell below — same pattern as `s3_effconn`.
#
# Inputs:
# - `{dataset}_meta.csv`
# - `{dataset}_inprop.npz`
#
# Outputs:
# - `{dataset}_{side_char}_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv` (per-idx)
# - `{dataset}_{side_char}_flow_{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv` (per-instance median)
#
# Runtime: < 2 hours.

# %%
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'setup_data'))

import pandas as pd
import scipy.sparse as sp

from utils.config import (
    DATA_DIR, DATASET, FLYWIRE_DATASET, FLYWIRE_SIDE_CHAR, HIT_THRE,
    MODEL_COMPARISON_DIR, N_FLOW, SIDE_CHAR,
)
import flow as sd

load_dotenv(PROJECT_ROOT / '.env')


# %% [markdown]
# ## Dataset selection
#
# Defaults to the config dataset. Uncomment the override to compute flywire.

# %%
# Defaults to the config dataset (malecns, right). To compute the FlyWire flow
# instead, uncomment the next line (tag + side come from config.py):
# DATASET, SIDE_CHAR = FLYWIRE_DATASET, FLYWIRE_SIDE_CHAR
print(f'dataset: {DATASET}; side: {SIDE_CHAR}; data: {DATA_DIR}')


# %%
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
inprop = sp.load_npz(DATA_DIR / f'{DATASET}_inprop.npz')
print(f'meta rows: {len(meta):,}; inprop shape: {inprop.shape} nnz={inprop.nnz:,}')


# %% [markdown]
# ## Run flow

# %%
idx_to_instance = dict(zip(meta.idx, meta.instance))
# (FlyWire: L_FLOW_SEEDS matches its left instances as-is; the seedless
# R7d_L/R8d_L are silently skipped by find_instance_flow.)
seed_groups = sd.R_FLOW_SEEDS if SIDE_CHAR == 'r' else sd.L_FLOW_SEEDS
save_prefix = f'{DATASET}_{SIDE_CHAR}_flow_'

flow_df = sd.fetch_instance_flow(
    inprop, idx_to_instance, seed_groups, DATA_DIR, save_prefix,
    steps=N_FLOW, thre=HIT_THRE,
)
print(f'flow rows: {len(flow_df):,}')
print(f'wrote {DATA_DIR}/{save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit.csv'
      f' and {save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv')


# %% [markdown]
# ## Model-comparison flow variants (paper SI Fig. 5 & SI Fig. 6)
#
# **These cells are COMMENTED OUT by default** — they are opt-in and heavy (each
# is a full `find_instance_flow` run, < 2 h). Uncomment to (re)build the four
# alternative-flow hitting-time tables that feed the paper's model comparison:
# SI Fig. 5d/5e read these `_hit.csv` tables directly, and SI Fig. 6 reads the
# propagated `_sum` matrices that `s3_effconn` builds *from* these same flow
# files. Outputs go to `cache/model_comparison/` (`MODEL_COMPARISON_DIR`), not
# the main `cache/data/` tree, each with a per-variant suffix:
#
# | variant | change vs. standard | output (in `model_comparison/`) |
# |---|---|---|
# | 0.05 thre | flow threshold 0.1 → 0.05 | `{prefix}20step_0.05thre_hit.csv` |
# | 0.2 thre  | flow threshold 0.1 → 0.2   | `{prefix}20step_0.2thre_hit.csv` |
# | 2syn      | drop `prop` edges with ≤ 2 synapses, recompute `inprop`, then flow | `{prefix}20step_0.1thre_hit_2syn.csv` |
# | r16       | append an idealized R1-6 source (`inprop` 1.0 → every L1/L2/L3, seeded in place of L1/L2/L3) | `{prefix}20step_0.1thre_hit_r16.csv` |
#
# `fetch_instance_flow` names its output `{prefix}{steps}step_{thre}thre_hit.csv`,
# so the 2syn/r16 runs (same 0.1 threshold as the standard) are written under a
# scratch prefix and renamed to the canonical `_2syn` / `_r16` suffix afterwards.
# All inputs (`inprop`/`prop`/`meta`) still come from `cache/data`.

# %%
# --- 0.05 / 0.2 threshold variants: standard inprop, different flow threshold ---
# MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
# for _thre in (0.05, 0.2):
#     sd.fetch_instance_flow(
#         inprop, idx_to_instance, seed_groups, MODEL_COMPARISON_DIR, save_prefix,
#         steps=N_FLOW, thre=_thre,
#     )
#     print(f'wrote model_comparison/{save_prefix}{N_FLOW}step_{_thre}thre_hit.csv')

# %%
# --- 2syn variant: keep only prop edges with > 2 synapses, recompute inprop ---
# `inprop` is `prop` column-normalised per post neuron (setup_data/malecns.py
# build_sparse_matrices); rebuild it from the > 2-synapse-thresholded `prop`.
# import numpy as np
# from scipy.sparse import diags
#
# MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
# _prop = sp.load_npz(DATA_DIR / f'{DATASET}_prop.npz').tocsc().astype(np.float32)
# _prop.data[_prop.data <= 2] = 0
# _prop.eliminate_zeros()
# _colsum = np.asarray(_prop.sum(axis=0)).ravel()
# _inv = np.zeros_like(_colsum)
# _inv[_colsum > 0] = 1.0 / _colsum[_colsum > 0]
# inprop_2syn = (_prop @ diags(_inv)).tocsc().astype(np.float32)
#
# _tmp = f'{DATASET}_{SIDE_CHAR}_flow2syn_'
# sd.fetch_instance_flow(inprop_2syn, idx_to_instance, seed_groups, MODEL_COMPARISON_DIR, _tmp,
#                        steps=N_FLOW, thre=HIT_THRE)
# (MODEL_COMPARISON_DIR / f'{_tmp}{N_FLOW}step_{HIT_THRE}thre_hit.csv').replace(
#     MODEL_COMPARISON_DIR / f'{save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit_2syn.csv')
# (MODEL_COMPARISON_DIR / f'{_tmp}{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv').unlink(missing_ok=True)
# print(f'wrote model_comparison/{save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit_2syn.csv')

# %%
# --- r16 variant: idealized R1-6 photoreceptor source ---
# The EM reconstruction lacks proper R1-6 → lamina connectivity, so the standard
# flow seeds L1/L2/L3 directly at layer 0. Append one idealized R1-6 node that
# feeds every L1/L2/L3 neuron with inprop 1.0 and seed the flow from it instead
# of L1/L2/L3, so L1/L2/L3 shift to layer 1. (Distinct from the real, weakly
# connected `R1-R6_{side}` population already in the meta.)
# import numpy as np
# from scipy.sparse import coo_matrix
#
# MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
# _S = SIDE_CHAR.upper()
# _L = np.concatenate([
#     meta.loc[meta['instance'] == f'{_lm}_{_S}', 'idx'].values
#     for _lm in ('L1', 'L2', 'L3')
# ])
# _N = inprop.shape[0]
# _coo = inprop.tocoo()
# inprop_r16 = coo_matrix(
#     (np.concatenate([_coo.data, np.ones(len(_L), np.float32)]),
#      (np.concatenate([_coo.row, np.full(len(_L), _N, dtype=_coo.row.dtype)]),
#       np.concatenate([_coo.col, _L.astype(_coo.col.dtype)]))),
#     shape=(_N + 1, _N + 1),
# ).tocsc().astype(np.float32)
#
# _R16 = f'R1-6ideal_{_S}'
# idx_to_instance_r16 = dict(idx_to_instance)
# idx_to_instance_r16[_N] = _R16
# seed_groups_r16 = [_R16] + [
#     s for s in seed_groups if s not in (f'L1_{_S}', f'L2_{_S}', f'L3_{_S}')
# ]
#
# _tmp = f'{DATASET}_{SIDE_CHAR}_flowr16_'
# sd.fetch_instance_flow(inprop_r16, idx_to_instance_r16, seed_groups_r16, MODEL_COMPARISON_DIR, _tmp,
#                        steps=N_FLOW, thre=HIT_THRE)
# (MODEL_COMPARISON_DIR / f'{_tmp}{N_FLOW}step_{HIT_THRE}thre_hit.csv').replace(
#     MODEL_COMPARISON_DIR / f'{save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit_r16.csv')
# (MODEL_COMPARISON_DIR / f'{_tmp}{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv').unlink(missing_ok=True)
# print(f'wrote model_comparison/{save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit_r16.csv')
