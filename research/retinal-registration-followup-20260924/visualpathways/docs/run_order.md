# Repository Run Order & Data-Dependency Guide

**Scope:** this describes how to run the pipeline on branch `main`
as of 2026-08-15. It is a factual "what depends on what, in what order"
reference, not a tutorial. Where the current code has a known problem that
affects the order below, it is called out inline with **⚠**.

See [`CLAUDE.md`](../CLAUDE.md) for what each module directory is for; this doc is about
sequencing and inputs/outputs, which CLAUDE.md doesn't cover.

Total run time vary depending on GPU setup. Tested on 4 A100, ~ 6 hours for the 5 scripts in 'setup_data\' which generated a ~15 GB 'cache/data' folder. 

## 0. Environment setup

1. Copy [`.env_sample`](../.env_sample) → `.env`, fill in `NEUPRINT_SERVER_URL`,
   `NEUPRINT_DATASET_NAME`, `NEUPRINT_APPLICATION_CREDENTIALS`.
2. Install the package: `pip install -e .` (this is what [`requirements.txt`](../requirements.txt)
   does under the hood — [`pyproject.toml`](../pyproject.toml) is the real dependency manifest,
   [`requirements-lock.txt`](../requirements-lock.txt) has pinned versions if you want reproducibility).
3. Activate the corresponding environment.

## 1. Config switches you must set before running anything

[`src/utils/config.py`](../src/utils/config.py) drives dataset/side selection through **module-level
constants**, not CLI flags or env vars:

- `DATASET = "malecns_v1.0"` (aliased as `FIG_DATASET`) — the only *global*
  dataset switch. FlyWire has named constants (`FLYWIRE_DATASET` /
  `FLYWIRE_SIDE_CHAR`) in [`config.py`](../src/utils/config.py), but they are deliberately *not* a global
  switch; the FlyWire build applies them locally by uncommenting the override in
  the dataset-selection cells of [`s2_layers`](../src/setup_data/s2_layers.py) / [`s3_effconn`](../src/setup_data/s3_effconn.py), and
  [`s1_flywirefafb_data`](../src/setup_data/s1_flywirefafb_data.py) uses the same tag as its file prefix (see §2b).
- `SIDE = "right"` → `SIDE_CHAR = "r"`.

**⚠ Important:** [`s1_malecns_data`](../src/setup_data/s1_malecns_data.py), [`s2_layers`](../src/setup_data/s2_layers.py), and
[`s3_effconn`](../src/setup_data/s3_effconn.py) all read `SIDE_CHAR` as a
module-level import from [`config.py`](../src/utils/config.py) — it is *not* a parameter you pass in.
To generate both-sides raw cache, you must run each of those scripts once
with `SIDE = "right"`, then edit `config.py` to `SIDE = "left"` and run them
again. Nothing will warn you if you only ever run the "right" pass — you'll
just silently be missing every `_l_` file that later steps need.

**⚠ Switch `SIDE` back to `"right"` as soon as the left-side pass finishes.**
The `SIDE_CHAR` footgun is not confined to `s1`–`s3`: many functions across
`src/utils/` (e.g. most of [`ol_rf.py`](../src/utils/ol_rf.py), including
`get_rf_raw_ol`/`get_rf_raw_cb`) declare a `side_char: str = SIDE_CHAR`
default parameter. That default is bound once, at each module's *import*
time, to whatever `config.SIDE_CHAR` happens to be in the current process —
not re-read per call. Any call site that omits an explicit `side_char=`
(several exist in `paper_figures.py` and the gallery notebooks) silently
follows whatever side `config.py` was left on. Leaving `SIDE = "left"` set
after you're done with the left-side pass is a standing hazard for every
such call for the rest of that session/notebook kernel — restore
`SIDE = "right"` (the side the figure code assumes by default) immediately
after the last left-side script finishes, before running anything else.

(By contrast, the scripts — [`s4_assemble_vpn_vcbn`](../src/setup_data/s4_assemble_vpn_vcbn.py) and the
[`utils.vic`](../src/utils/vic.py) helpers it calls — take `side_char` as an explicit function
argument and loop over both sides internally for most of the notebook, so
they mostly don't have this footgun. The exception is its own two RF-cache
cells at the end (`get_rf_raw_ol`/`get_rf_raw_cb`), which currently build the
right side only via an explicit `side_char='r'` — a deliberate, currently
single-sided choice, not a both-sides loop.)

## 2. Build the raw NeuPrint cache — [`src/setup_data/`](../src/setup_data/) (stages s1–s3)

Run in this order (each script's own header comment/docstring states its
inputs; this is the resolved transitive order):

1. **[`s1_malecns_data`](../src/setup_data/s1_malecns_data.py)** — pure NeuPrint fetch. No prerequisites besides
   the eyemap files in [`params/`](../params/). Writes `{DATASET}_meta.csv`, `_prop.npz`,
   `_inprop.npz`, `_OL_{side}_input_syn_per_col.parquet`,
   `_OL_{side}_roi_counts.pkl`, `_OL_{side}_roi_adj.pkl` to [`cache/data/`](../cache/data/).
2. **[`s2_layers`](../src/setup_data/s2_layers.py)** — despite the module name, its actual job (per its
   own docstring) is computing flow/hitting-time, which [`utils.ol_data`](../src/utils/ol_data.py)
   later combines with `roi_counts.pkl` to build the anatomical-layer table.
   Needs `meta.csv` + `_inprop.npz` from step 1. Writes
   `{DATASET}_{side}_flow_{N}step_{thre}thre_hit[.csv|_per_group.csv]`.
   Dataset-generic: like `s3_effconn`, it has a "Dataset selection" cell whose
   commented override switches it to FlyWire (see §2b).
   (It also holds the item-5 model-comparison flow-variant cells that write the
   `_hit.csv` variants under `cache/model_comparison/` for SI 5/6.)
3. **[`s3_effconn`](../src/setup_data/s3_effconn.py)** — needs `meta.csv` + `_inprop.npz` + the
   `_hit_per_group.csv` from step 2. Writes `{DATASET}_{side}_lat_flow_0.npz`
   and `_lat_flow_sum.npz`. Must run after step 2. Dataset-generic: to build
   the FlyWire lateral flow, uncomment the
   `DATASET, SIDE_CHAR = FLYWIRE_DATASET, FLYWIRE_SIDE_CHAR` line in its
   "Dataset selection" cell (values from [`config.py`](../src/utils/config.py)); when the
   `_per_group` sidecar is absent it is derived from the per-neuron `_hit.csv`
   by per-instance median. FlyWire mode expects a pre-existing FlyWire
   `meta.csv` / `inprop.npz` / per-neuron `_hit.csv` — see §2b below for how
   to build them.
4. **Input-RF derived caches — no separate stage anymore.** The four
   [`utils.ol_rf`](../src/utils/ol_rf.py) input-RF caches are built *lazily* on the first figure call,
   so there is nothing to run here for a normal render. To force a rebuild
   (after re-running [`s1_malecns_data`](../src/setup_data/s1_malecns_data.py) §6 "Input RF raw data", or
   after changing the RF-fitting code), call
   `ol_rf.rebuild_input_rf_caches(force=True)`. 

## 2b. FlyWire (FAFB v783) branch — optional

Only needed for the SI Fig. 14 flywire cells in
[`paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb) (malecns-vs-FlyWire polarity comparison) and
the FlyWire lateral flow. Everything is left-side
(`config.FLYWIRE_SIDE_CHAR = 'l'`); the cache-file prefix is
`config.FLYWIRE_DATASET = "flywire-fafb-v783b"` (distinct from the neuprint
server dataset name `flywire-fafb:v783b`).

1. **[`s1_flywirefafb_data`](../src/setup_data/s1_flywirefafb_data.py)** — pure NeuPrint fetch (~15 min).
   Writes `flywire-fafb-v783b_meta.csv`, `_prop.npz`, `_inprop.npz` to
   [`cache/data/`](../cache/data/). The connection is overridden **in-script** (`.env` is left
   untouched): the dataset lives on `neuprint-cns.janelia.org` as
   `flywire-fafb:v783b` — the repo `.env` points at `neuprint.janelia.org`,
   which does not host it; the same token works on both servers. Covers both
   sides in one pass (no `SIDE` footgun). Scope: no R7/R8 fill-in (that
   patches a malecns-specific EM gap), no hex coords, no input-RF/ROI
   sidecars (the flywire neuprint dataset lacks `olHex*` synapse fields and
   OL-layer ROIs). Lamina (`LA_L`/`LA_R`) edges are dropped, mirroring the
   malecns build.
2. **[`s2_layers`](../src/setup_data/s2_layers.py) FlyWire mode** — uncomment the
   `DATASET, SIDE_CHAR = FLYWIRE_DATASET, FLYWIRE_SIDE_CHAR` override in its
   "Dataset selection" cell (same pattern as `s3_effconn`) and run; writes the
   per-neuron `_l_..._hit.csv` (+ `_per_group.csv`). The seed list needs no
   change: `flow.L_FLOW_SEEDS` matches FlyWire's `L1/L2/L3/R7/R8/HBeyelet`
   left instances as-is, and the two seeds with no FlyWire counterpart
   (`R7d_L`/`R8d_L`) are silently skipped by `find_instance_flow`.
3. **[`s3_effconn`](../src/setup_data/s3_effconn.py) FlyWire mode** — uncomment the documented
   `DATASET, SIDE_CHAR = FLYWIRE_DATASET, FLYWIRE_SIDE_CHAR` override (see
   step 2.3); it derives the `_per_group` sidecar from the per-neuron
   `_hit.csv` if absent and writes `_l_lat_flow_0.npz` / `_l_lat_flow_sum.npz`.
4. **Nothing else to run:** the SI Fig. 14 cell in
   [`paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb) builds the `_OL_l_*` sidecars itself on
   first run (via `core_data._ensure_ol_sidecars`, under its region
   monkeypatch, which relabels `superclass == 'optic'` neurons to
   `left OL`/`right OL`).

## 3. Build the VCBN cache (contribution / resolution / coverage)

This is the chain behind the sections corresponding to paper Figures 5g/5h/6a/6b/7c 
of [`paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb) and the two gallery notebooks. The
order is stated explicitly in [`s4_assemble_vpn_vcbn`](../src/setup_data/s4_assemble_vpn_vcbn.py)'s own header comment:

1. **[`s4_assemble_vpn_vcbn`](../src/setup_data/s4_assemble_vpn_vcbn.py)** — needs `meta.csv` and the `_hit.csv` from
   step 2.3 (both sides), plus `utils.vic.get_ol_cb_vic_raw()`'s output, which
   it obtains by *calling* that function directly (both sides) rather than
   reading a parquet by path. Writes `vp_cb_hit_vic_{r,l}.p` to [`cache/data/`](../cache/data/).
   The **same stage** then force-rebuilds the two combined per-neuron RF parquet
   caches `ol_rf.get_rf_raw_ol` (OL-subset) + `ol_rf.get_rf_raw_cb` (full matrix)
   — the former standalone `rf_size` step, now folded in. Each fits a 2D Gaussian
   **and** an alpha hull from one propagation pass per neuron, so `size` and
   `ahull_size` live in the same parquet (replacing the retired `rf_fit_thr07.pkl`
   / `meta_ON_ahull_cumsum60.pkl` / `meta_VP_CB_ahull_cumsum60.pkl` pickles).
2. **[`s5_tbar`](../src/setup_data/s5_tbar.py)** — heavy NeuPrint queries. Its own comment says explicitly: run after
   [`s4_assemble_vpn_vcbn`](../src/setup_data/s4_assemble_vpn_vcbn.py) (needs `vp_cb_hit_vic_{r,l}.p`).
3. **Manual input, not produced by any script here:** [`params/outline_cb_1.csv`](../params/outline_cb_1.csv)
   / [`outline_cb_2.csv`](../params/outline_cb_2.csv) — offline/R-generated CB outline geometry, consumed
   directly by [`paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb).

## 4. Render figures — [`src/make_figures/`](../src/make_figures/)

Three notebooks currently exist here: [`paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb) (main),
[`gallery_vpn.ipynb`](../src/make_figures/gallery_vpn.ipynb), [`gallery_vcbn.ipynb`](../src/make_figures/gallery_vcbn.ipynb).

- **[`paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb)** — the main figure notebook. Reads the full step-2
  raw+derived cache via [`utils.core_data`](../src/utils/core_data.py) / [`ol_data`](../src/utils/ol_data.py) / [`cb_data`](../src/utils/cb_data.py) / [`ol_rf`](../src/utils/ol_rf.py) /
  [`vic`](../src/utils/vic.py) / [`polarity`](../src/utils/polarity.py) (these derive further "processed" cache in
  [`cache/data_proc/`](../cache/data_proc/) lazily, on first call — no separate build step needed once
  the step-2 raw cache exists). Its VCBN section additionally needs
  step 3's outputs: `tbar_vp_cb.pkl`, `tbar_count_roi_cb.csv`,
  `vp_cb_hit_vic_{r,l}.p`, the combined RF parquet (via `ol_rf.get_rf_raw_ol` /
  `get_rf_raw_cb`, built by step 3's [`s4_assemble_vpn_vcbn`](../src/setup_data/s4_assemble_vpn_vcbn.py)),
  `params/outline_cb_{1,2}.csv`.
- **[`gallery_vpn.ipynb`](../src/make_figures/gallery_vpn.ipynb)** / **[`gallery_vcbn.ipynb`](../src/make_figures/gallery_vcbn.ipynb)** — standalone per-instance
  galleries; need the same step-3 cache chain plus the combined RF parquet (per-body
  ellipse + alpha hull via `ol_rf.get_rf_raw_ol` / `get_rf_raw_cb`; VIC/hitting
  time joined from `vic.load_vp_cb`). The per-neuron RF background is computed on
  demand via `utils.core_data.grouped_effweight`.

The `paper_figures`, `gallery_vpn`, and `gallery_vcbn` notebooks are each
jupytext-paired with a `.py` file ([`paper_figures.py`](../src/make_figures/paper_figures.py), etc.).

Run [`paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb) (base cells) any time after step 2. Only run the
visual-input contribution panel cells / the two gallery notebooks after step 3 is complete.

## 5. Distribute panels into the html viewer

Run **after** all wanted notebook cells have executed (safe to do
incrementally per-folder too):

```python
from make_figures import figures as pf
pf.distribute()   # copy/rename topic-folder sidecars into results/html_figures/
pf.build_index()  # rescan results/html_figures/*.html and rebuild index.html
```
