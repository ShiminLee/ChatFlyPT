# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: ol-analysis-np2 (3.12.4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup flywire-fafb meta + sparse matrices
#
# FlyWire (FAFB v783) analog of `s1_malecns_data`: downloads adjacencies and
# neuron metadata from neuprint and writes `flywire-fafb-v783b_meta.csv`,
# `_prop.npz`, `_inprop.npz` into `cache/data/` — the external inputs that
# `docs/run_order.md` §2.3 flags as "nothing in this repo currently generates"
# for the FlyWire lateral-flow build (s2/s3 FlyWire mode) and the SI Fig. 14
# flywire cells in `paper_figures.ipynb`.
#
# **Temporary connection override (kept in-script, not in `.env`):** the
# flywire dataset lives on `neuprint-cns.janelia.org` as `flywire-fafb:v783b`;
# the repo `.env` points at `neuprint.janelia.org`, which does not host it.
# The same `NEUPRINT_APPLICATION_CREDENTIALS` token works on both servers.
#
# Scope vs `s1_malecns_data`: sections 1–5 only (connectivity → meta →
# sparse matrices). No R7/R8 fill-in (that patches a malecns-specific EM gap),
# no hex coords (no flywire eyemap under `params/`), and no input-RF / ROI
# sections (the flywire neuprint dataset has no `olHex1/olHex2` synapse fields
# and no OL layer ROIs).
#
# Dataset differences handled here (probed 2026-08-15):
# - `instance` is null server-side → synthesized as `cell_type + '_' + side letter`
# - NT prediction lives in `predictedNt` (long-form names; no histamine class)
# - `side` is a native property: 'left'/'right'/'center'/'na'
# - lamina ROIs are `LA_L`/`LA_R` (malecns: `LA(R)`/`LA(L)`)
# - edge weights are unthresholded (min weight 1, like malecns — not the
#   Codex ≥5-synapse export)

# %%
import sys
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

import numpy as np
import pandas as pd
import scipy as sp

PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'setup_data'))

from neuprint import NeuronCriteria as NC, NotNull, fetch_adjacencies

from utils import olc_client
from utils.config import DATA_DIR, FLYWIRE_DATASET
from malecns import NT_SIGN, SIDE_WORD, get_sorted_nodes, build_sparse_matrices

load_dotenv(PROJECT_ROOT / '.env')

# --- temporary connection override (see header) ---
FLYWIRE_NEUPRINT_SERVER = 'neuprint-cns.janelia.org'
FLYWIRE_NEUPRINT_DATASET = 'flywire-fafb:v783b'

TAG = FLYWIRE_DATASET  # 'flywire-fafb-v783b' — the cache-file prefix from config.py
c = olc_client.connect(instance=FLYWIRE_NEUPRINT_SERVER, dataset=FLYWIRE_NEUPRINT_DATASET, verbose=True)
print(f'dataset tag: {TAG}; data: {DATA_DIR}')

# %% [markdown]
# ## 1. Connectivity

# %%
def fetch_connectivity_flywire(client) -> pd.DataFrame:
    """Download all Neuron→Neuron adjacencies (type IS NOT NULL on both ends),
    drop lamina rows (`LA_L`/`LA_R`) — mirroring `malecns.fetch_connectivity`
     — and collapse per bodyId pair into a flat edge list."""
    neu, adj = fetch_adjacencies(NC(type=NotNull), NC(type=NotNull), client=client)
    adj = adj.loc[~adj["roi"].isin(["LA_L", "LA_R"]), ["bodyId_pre", "bodyId_post", "weight"]]
    adj["weight"] = adj["weight"].astype(np.int32)
    return adj.groupby(["bodyId_pre", "bodyId_post"], as_index=False, sort=False)["weight"].sum()


t0 = time.time()
conn_df = fetch_connectivity_flywire(c)
sorted_nodes = get_sorted_nodes(conn_df)
print(f'edges: {len(conn_df):,}; neurons: {len(sorted_nodes):,} ({time.time() - t0:.0f}s)')

# %% [markdown]
# ## 2. Neuron metadata

# %%
def fetch_neuron_metadata_flywire(bodyIds, client, batch_size: int = 50_000) -> pd.DataFrame:
    """Fetch bodyId, cell_type, nt (`predictedNt`), superclass, subclass,
    class, dimorphism, side. Batched: the full id list in one query risks the
    server's gateway timeout."""
    frames = []
    ids = list(bodyIds)
    for i in range(0, len(ids), batch_size):
        cql = f"""
            WITH {ids[i:i + batch_size]} as bodyIds
            MATCH(n:Neuron)
            WHERE n.bodyId in bodyIds
            RETURN n.bodyId as bodyId, n.type as cell_type,
                   n.predictedNt as nt, n.superclass as superclass,
                   n.subclass as subclass, n.class as class,
                   n.dimorphism as dimorphism, n.side as side
        """
        frames.append(client.fetch_custom(cql))
    neu_df = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame({"bodyId": ids})
    return meta.merge(neu_df, on="bodyId", how="left")


meta = fetch_neuron_metadata_flywire(sorted_nodes, client=c)
print(meta.shape)
print(meta.superclass.value_counts(dropna=False).head(12))

# %% [markdown]
# ## 3. Finalize meta

# %%
def finalize_meta_flywire(meta: pd.DataFrame, sorted_nodes: list[int]) -> pd.DataFrame:
    """Add `sign`, `coords` (empty — no flywire eyemap), `idx`; normalize
    `side` ('center'/'na' → 'noside'); synthesize
    `instance = cell_type + '_' + side letter`; strip `(...)` and
    `_indeterminate` like `malecns.finalize_meta`; sort by `idx`."""
    out = meta.copy()

    out["sign"] = out["nt"].map(NT_SIGN)
    out["side"] = out["side"].where(out["side"].isin(["left", "right"]), "noside")
    out["instance"] = out["cell_type"] + "_" + out["side"].map(SIDE_WORD)

    for col in ("instance", "cell_type"):
        out[col] = (
            out[col]
            .str.replace(r"\s*\(.*?\)", "", regex=True)
            .str.replace(r"_indeterminate", "", regex=True)
        )

    out["coords"] = pd.NA

    nodes_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
    out["idx"] = out["bodyId"].map(nodes_to_idx)
    out = out.sort_values("idx").reset_index(drop=True)
    return out[[
        "bodyId", "instance", "cell_type", "nt", "superclass", "subclass",
        "class", "dimorphism", "coords", "sign", "side", "idx",
    ]]


meta = finalize_meta_flywire(meta, sorted_nodes)
print(meta.shape)
print(meta.side.value_counts())
meta.head()

# %% [markdown]
# ## 4. Build and save sparse matrices

# %%
prop, inprop = build_sparse_matrices(conn_df, sorted_nodes)

DATA_DIR.mkdir(parents=True, exist_ok=True)
meta.to_csv(DATA_DIR / f"{TAG}_meta.csv", index=False)
sp.sparse.save_npz(DATA_DIR / f"{TAG}_prop.npz", prop)
sp.sparse.save_npz(DATA_DIR / f"{TAG}_inprop.npz", inprop)
print(f'prop: {prop.shape} nnz={prop.nnz:,}')
print(f'inprop: {inprop.shape} nnz={inprop.nnz:,}')
print(f'wrote {TAG}_meta.csv / _prop.npz / _inprop.npz to {DATA_DIR}')

# %% [markdown]
# ## 5. Sanity checks
#
# Spot-check against what `core_data._load_meta_groups` and the SI Fig. 14
# cell expect: `Dm3p_L` / `TmY9q__perp_L` instances present, `superclass ==
# 'optic'` populated on both sides, meta row order == matrix index order.

# %%
assert (meta["bodyId"].values == np.array(sorted_nodes)).all(), "meta not in matrix order"
assert meta["idx"].tolist() == list(range(len(meta))), "idx not contiguous"

instances = set(meta.instance.dropna())
for inst in ("Dm3p_L", "Dm3q_L", "Dm3v_L", "TmY9q__perp_L", "Mi1_L", "Tm1_L"):
    print(f'{inst}: {"OK" if inst in instances else "MISSING"}')

opt = meta.superclass == "optic"
print(f"optic left: {(opt & (meta.side == 'left')).sum():,}; "
      f"optic right: {(opt & (meta.side == 'right')).sum():,}")
print(meta.nt.value_counts(dropna=False))
