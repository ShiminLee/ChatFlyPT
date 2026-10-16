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
# # Compute lateral flow (forward-directed connectivity)
#
# Trims `inprop` to forward-directed (lateral) flow using the per-instance
# hitting times, then saves the trimmed matrix and its running sum.
#
# Works for any dataset. Defaults to the config dataset (malecns, right side).
# To build the **flywire** lateral flow instead, override `DATASET` and
# `SIDE_CHAR` in the "Dataset selection" cell below. flywire ships no
# `_per_group` sidecar, so it is derived from the per-neuron `_hit.csv` by the
# per-instance median (matching the malecns `instance`, `hitting_time` schema).
#
# Inputs:
# - `{dataset}_meta.csv`
# - `{dataset}_inprop.npz`
# - `{dataset}_{side}_flow_{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv`
#   (or, if absent, `..._hit.csv` collapsed to a per-instance median)
#
# Outputs:
# - `{dataset}_{side}_lat_flow_0.npz`
# - `{dataset}_{side}_lat_flow_sum.npz`
#
# Runtime: 8× A100 ≈ 15 min; CPU-only (~24 cores) ≈ 2–3 hours.
#

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
# Defaults to the config dataset. Uncomment the overrides to build flywire.

# %%
# Defaults to the config dataset (malecns, right). To build the FlyWire lateral
# flow instead, uncomment the next line (tag + side come from config.py):
# DATASET, SIDE_CHAR = FLYWIRE_DATASET, FLYWIRE_SIDE_CHAR
print(f'dataset: {DATASET}; side: {SIDE_CHAR}; data: {DATA_DIR}')


# %% [markdown]
# ## Load prerequisites
#
# Load meta + inprop and the per-instance hitting times. If the `_per_group`
# sidecar is missing (e.g. flywire), derive it from the per-neuron `_hit.csv`
# by taking the median hitting time over neurons sharing an `instance`.

# %%
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
inprop = sp.load_npz(DATA_DIR / f'{DATASET}_inprop.npz')

per_group_path = DATA_DIR / f'{DATASET}_{SIDE_CHAR}_flow_{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv'
if per_group_path.exists():
    flow_df = pd.read_csv(per_group_path)
else:
    hit_path = DATA_DIR / f'{DATASET}_{SIDE_CHAR}_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv'
    flow_hit = pd.read_csv(hit_path)
    flow_df = flow_hit.groupby('instance', as_index=False)['hitting_time'].median()
    flow_df.to_csv(per_group_path, index=False)
    print(f'derived per-group sidecar from {hit_path.name} -> {per_group_path.name}')

idx_to_instance = dict(zip(meta.idx, meta.instance))
print(f'meta rows: {len(meta):,}; inprop shape: {inprop.shape} nnz={inprop.nnz:,}')
print(f'flow rows: {len(flow_df):,}')


# %% [markdown]
# ## Trim & save

# %%
trimmed_inprop = sd.trim_inprop_by_flow(
    inprop, idx_to_instance, flow_df, flow_diff_min=0.0,
)
print(f'trimmed_inprop nnz: {trimmed_inprop.nnz:,}')

lat_prefix = f'{DATASET}_{SIDE_CHAR}_lat_flow_'
sd.save_lat_flow_0(trimmed_inprop, DATA_DIR, lat_prefix)

summed = sd.compute_lat_flow_sum(trimmed_inprop)
sd.save_lat_flow_sum(summed, DATA_DIR, lat_prefix)
print(f'summed nnz: {summed.nnz:,}')
print(f'wrote {DATA_DIR}/{lat_prefix}0.npz and {DATA_DIR}/{lat_prefix}sum.npz')


# %% [markdown]
# ## Model-comparison variant `_sum` matrices (paper SI Fig. 6) — COMMENTED OUT
#
# **Opt-in and heavy (~15 min GPU per variant).** For each flow variant that
# `s2_layers` writes into `cache/model_comparison/`, build the propagated
# (`_sum`) matrix the same way as the standard model above — trim `inprop` by the
# variant's hitting times, then `compress_paths` + sum — and save it back into
# `cache/model_comparison/` with the variant suffix. SI Fig. 6 reads these `_sum`
# matrices; SI Fig. 5 needs only the `_hit.csv` flow tables (no `_sum`).
#
# The threshold variants (0.05 / 0.2) reuse the **standard** `inprop` (only the
# hitting times differ); `2syn` and `r16` need their own `inprop`, rebuilt below
# exactly as in `s2_layers` (kept in sync by hand — cheap matrix ops, not a flow
# run). The `r16` matrix carries one extra index (the idealized R1-6 node); it is
# a pure source, so downstream readers can drop its row/col from `_sum`.

# %%
# import numpy as np
# from scipy.sparse import coo_matrix, diags
#
# MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
# _pre = f'{DATASET}_{SIDE_CHAR}_flow_{N_FLOW}step'
#
#
# def _build_variant_sum(inprop_v, i2i_v, hit_file, suffix):
#     """Trim `inprop_v` by the variant's per-instance hitting times and save its
#     lateral-flow `_0`/`_sum` matrices into cache/model_comparison/."""
#     hit = pd.read_csv(MODEL_COMPARISON_DIR / hit_file)
#     fdf = hit.groupby('instance', as_index=False)['hitting_time'].median()
#     trimmed = sd.trim_inprop_by_flow(inprop_v, i2i_v, fdf, flow_diff_min=0.0)
#     out_prefix = f'{DATASET}_{SIDE_CHAR}_lat_flow_{suffix}_'
#     sd.save_lat_flow_0(trimmed, MODEL_COMPARISON_DIR, out_prefix)
#     sd.save_lat_flow_sum(sd.compute_lat_flow_sum(trimmed), MODEL_COMPARISON_DIR, out_prefix)
#     print(f'wrote model_comparison/{out_prefix}0.npz and {out_prefix}sum.npz')
#
#
# # threshold variants: standard inprop, variant hitting times
# _build_variant_sum(inprop, idx_to_instance, f'{_pre}_0.05thre_hit.csv', '0.05thre')
# _build_variant_sum(inprop, idx_to_instance, f'{_pre}_0.2thre_hit.csv',  '0.2thre')
#
# # 2syn: rebuild the >2-synapse inprop (mirrors s2_layers)
# _prop = sp.load_npz(DATA_DIR / f'{DATASET}_prop.npz').tocsc().astype(np.float32)
# _prop.data[_prop.data <= 2] = 0
# _prop.eliminate_zeros()
# _cs = np.asarray(_prop.sum(axis=0)).ravel()
# _inv = np.zeros_like(_cs)
# _inv[_cs > 0] = 1.0 / _cs[_cs > 0]
# inprop_2syn = (_prop @ diags(_inv)).tocsc().astype(np.float32)
# _build_variant_sum(inprop_2syn, idx_to_instance, f'{_pre}_{HIT_THRE}thre_hit_2syn.csv', '2syn')
#
# # r16: append the idealized R1-6 node (mirrors s2_layers)
# _S = SIDE_CHAR.upper()
# _Lidx = np.concatenate([
#     meta.loc[meta['instance'] == f'{lm}_{_S}', 'idx'].values for lm in ('L1', 'L2', 'L3')
# ])
# _Nn = inprop.shape[0]
# _co = inprop.tocoo()
# inprop_r16 = coo_matrix(
#     (np.concatenate([_co.data, np.ones(len(_Lidx), np.float32)]),
#      (np.concatenate([_co.row, np.full(len(_Lidx), _Nn, dtype=_co.row.dtype)]),
#       np.concatenate([_co.col, _Lidx.astype(_co.col.dtype)]))),
#     shape=(_Nn + 1, _Nn + 1),
# ).tocsc().astype(np.float32)
# _i2i_r16 = dict(idx_to_instance)
# _i2i_r16[_Nn] = f'R1-6ideal_{_S}'
# _build_variant_sum(inprop_r16, _i2i_r16, f'{_pre}_{HIT_THRE}thre_hit_r16.csv', 'r16')
