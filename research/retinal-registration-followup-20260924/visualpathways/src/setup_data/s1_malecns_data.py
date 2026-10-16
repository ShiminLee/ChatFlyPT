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
# # Setup malecns meta + sparse matrices
#
# Downloads adjacencies and neuron metadata from neuprint, merges the R7/R8 fill-in edge lists under `params/`, assembles hex coordinates, and writes `malecns_{tag}_meta.csv`, `prop.npz`, `inprop.npz` into `cache/data/`. Dataset tag is derived from `config.DATASET` (e.g. `malecns_v1.0` → `v1.0`).
#
# Uses tot_weight normalization and instance override from `cell_type` + side.

# %%
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'setup_data'))

from utils import olc_client
from utils.config import DATA_DIR, DATASET, SIDE_CHAR
import malecns as sd

load_dotenv(PROJECT_ROOT / '.env')
TAG = DATASET.split('_', 1)[-1]  # e.g. 'v1.0'
c = olc_client.connect(verbose=True)
print(f'dataset: {DATASET}; data: {DATA_DIR}')

# %% [markdown]
# ## 1. Connectivity

# %%
conn_df = sd.fetch_connectivity(client=c)
color_df_R, color_df_L = sd.load_r78_fill(DATA_DIR)
conn_df = sd.merge_r78_fill(conn_df, color_df_R, color_df_L)
sorted_nodes = sd.get_sorted_nodes(conn_df)
print(f'edges: {len(conn_df):,}; neurons: {len(sorted_nodes):,}')

# %% [markdown]
# ## 2. Neuron metadata

# %%
meta = sd.fetch_neuron_metadata(sorted_nodes, client=c)
meta = sd.override_r78_metadata(meta, color_df_R, color_df_L)
for inst in ('R7_R', 'R8_R', 'R7_L', 'R8_L'):
    print(f'{inst}: {meta[meta.instance == inst].bodyId.nunique()}')

# %% [markdown]
# ## 3. Add hex coordinates

# %%
meta = sd.add_column_coords(meta, DATA_DIR)
meta = sd.add_r78_dorsal_l3_coords(meta, client=c, data_dir=DATA_DIR)
print(meta[~meta.coords.isna()].instance.value_counts().head(20))

# %% [markdown]
# ## 4. Finalize meta

# %%
meta = sd.finalize_meta(meta, sorted_nodes)
print(meta.shape)
print(meta.side.value_counts())
meta.head()

# %% [markdown]
# ## 5. Build and save sparse matrices

# %%
prop, inprop = sd.build_sparse_matrices(conn_df, sorted_nodes)
sd.save_outputs(meta, prop, inprop, DATA_DIR, TAG)
print(f'prop: {prop.shape} nnz={prop.nnz:,}')
print(f'inprop: {inprop.shape} nnz={inprop.nnz:,}')

# %% [markdown]
# ## 6. Input RF raw data

# %%
from utils import ol_rf

stem = f'malecns_{TAG}_OL_{SIDE_CHAR}'

instances = sorted(ol_rf._ol_non_visual_input_instances(f'malecns_{TAG}', DATA_DIR, SIDE_CHAR))
print(f'fetching input syn-per-col for {len(instances)} instances on {SIDE_CHAR!r}...')

syn_df = sd.fetch_input_syn_per_col(instances, client=c, rel_input_weight=0.4)
path = sd.save_input_syn_per_col(syn_df, DATA_DIR, stem)
print(f'  rows: {len(syn_df):,} -> {path.name}')

# %% [markdown]
# ## 7. ROI synapse counts

# %%
ol_instances = sorted(ol_rf._ol_instances(f'malecns_{TAG}', DATA_DIR, SIDE_CHAR))
roi_counts = sd.fetch_roi_counts(ol_instances, client=c)
print(f'OL instances: {len(ol_instances)}; roi_counts rows: {len(roi_counts):,}')

# %% [markdown]
# ## 8. ROI adjacency

# %%
roi_adj = sd.fetch_roi_adj(ol_instances, SIDE_CHAR, client=c)
sd.save_roi_pickles(roi_counts, roi_adj, DATA_DIR, stem)
print(f'roi_adj rows: {len(roi_adj):,}')

# %%
