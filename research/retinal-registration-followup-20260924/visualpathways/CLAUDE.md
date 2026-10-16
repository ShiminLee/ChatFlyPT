# CLAUDE.md — Optic Lobe Connectome Analysis Repository

This file helps Claude Code understand the context, structure, and conventions of this repository.

**See also:**
- [`docs/run_order.md`](docs/run_order.md) — the authoritative setup→figures run order (what depends on what, in what order). This CLAUDE.md covers *what each module is for*; `run_order.md` covers *sequencing*.

## Project Overview

This is a neuroscience research repository for analyzing the **optic lobe (OL) connectome** in the *Drosophila melanogaster* male brain. The codebase processes and visualizes electron microscopy (EM) connectome data from Janelia FlyEM's Male Brain Dataset (MaleCNS), with Princeton's FlyWire (FAFB) comparisons, accessed via the NeuPrint application programming interface (API).

**Maintainer:** Reiser Lab at Janelia Research Campus
**License:** GNU General Public License v3
**Intended Users:** neuroscience researchers
**Project Type:** Research analysis codebase that renders the figures for the OL connectome paper ([bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2025.12.22.696097v2.abstract))

### Main Use Cases
- Query and analyze neural connectivity patterns in the optic lobe
- Compute signal propagation through neural circuits
- Generate receptive field (RF) maps for visual system neurons
- Cluster neurons by connectivity and analyze pathways
- Compare male vs. female (sexual dimorphism) connectivity
- Render the paper's figure panels and assemble them into a browsable HTML viewer

## Repository Structure

```
├── src/                      # Main source code — 3 module dirs: utils/, make_figures/, setup_data/
│   ├── utils/                # Core utilities: cached-data loaders + the plotting stack (36 modules)
│   ├── make_figures/         # Paper-figure orchestrator notebooks + the panel-manifest helper
│   ├── setup_data/           # MaleCNS/FlyWire data download + preprocessing pipeline (s1–s5)
│   └── sync-nb.py            # jupytext helper: keep .ipynb/.py notebook pairs in sync
├── docs/                     # Docs: run_order.md, python-getting-started.md, coordinate-systems.md
├── params/                   # Parameter/input files (git-tracked research inputs, edited live) —
│                             #   paper_figs.xlsx (figure→panel manifest), TableS2_experimental_data.xlsx,
│                             #   eyemap/ + hex-grid pins + column-location xlsx, CB outline CSVs
├── cache/                    # Gitignored: data/ (raw, setup_data-written) + data_proc/ (derived, lazy)
│                             #   + model_comparison/ (SI 5/6 flow variants, built by s2_layers)
├── results/                  # Analysis outputs. Only html_figures/ + tables/ are git-tracked;
│                             #   the numbered topic folders (main_1, si_1, …) are gitignored
├── pyproject.toml            # The real dependency manifest (requirements.txt is a thin `-e .` shim)
├── requirements-lock.txt     # Pinned versions for an exactly-reproducible env
├── jupytext.toml             # jupytext pairing config for the notebooks
├── .env_sample               # Configuration template (copy to .env)
├── README.md                 # Brief project description + install
└── LICENSE                   # GNU GPL v3
```

**Where the data comes from:** connectome data is fetched via the **`connectome_interpreter`** package
(wrapped inside [`utils/core_data.py`](src/utils/core_data.py)), *not* a local query layer. The repo has
**no** hand-rolled NeuPrint query module — an earlier `src/queries/` wrapper was figure-unused and removed.

> **History (context, not action items):** the tree was aggressively pruned to only figure-generating
> code. Removed and **not to be reintroduced**: `network/`, `test/`, `rf/`, `pathways/`, `queries/`,
> `input_distribution/`, `plot_neuron/`, `eyemap/`, `ng_export/`, `judith/`, `quan_propagation/`, plus
> ~18 `utils/` modules. `src/quan_propagation/` was **dissolved** (2026-07) — its pieces were
> redistributed into `setup_data/`, `utils/`, and `make_figures/` (details in the module sections below);
> a stale empty `src/quan_propagation/` folder may linger locally but is not git-tracked.
> `src/dimorphism/` was deleted in commit `2352732` ("aggressive clean up") and stays absent — its two
> paper panels are generated inline by `paper_figures.py` (see the sexual-dimorphism section below);
> the retired exploratory code is recoverable via `git show 2352732^:src/dimorphism/`.

## Core Concepts

### 1. OL Neuron Types and Instances
- **Types:** classes of neurons (e.g., "T4a", "Mi1", "Tm3")
- **Instances:** individual neurons of a type (e.g., "T4a_001")
- Loaded from cached metadata via [`utils/core_data.py`](src/utils/core_data.py) and [`utils/ol_data.py`](src/utils/ol_data.py).

### 2. Region of Interest (ROI) Analysis
- **ROIs:** anatomical brain regions (e.g., "ME", "LO", "LOP")
- **Layers:** subdivisions within ROIs (e.g., "ME(R7)", "ME(R8)")
- Geometry/hex-grid/layer helpers: [`utils/geometry.py`](src/utils/geometry.py), [`utils/hex_hex.py`](src/utils/hex_hex.py); layer assignment is precomputed by [`setup_data/s2_layers.py`](src/setup_data/s2_layers.py).

### 3. Signal Propagation
- Track how signals flow through multi-layer neural circuits; compute effective weights, contributions, influence.
- The paper panels (visual-input contribution (VIC) / resolution / coverage projections) are rendered by the [`make_figures/paper_figures.ipynb`](src/make_figures/paper_figures.ipynb); their heavy cache is built by [`setup_data/s4_assemble_vpn_vcbn.py`](src/setup_data/s4_assemble_vpn_vcbn.py) + [`s5_tbar.py`](src/setup_data/s5_tbar.py).

### 4. Receptive Fields (RF)
- Map visual-space locations that drive neuronal responses (Gaussian fitting, retinotopic mapping).
- Code: [`utils/ol_rf.py`](src/utils/ol_rf.py), [`utils/external_rf.py`](src/utils/external_rf.py). The input syn-per-column data is precomputed by [`setup_data/s1_malecns_data.ipynb`](src/setup_data/s1_malecns_data.ipynb) (§6); RF fits are computed on demand by `ol_rf` (the four input-RF caches build lazily on first figure call).

### 5. Clustering & Pathways
- Cluster neurons by connectivity (Leiden, via `igraph`'s **built-in** implementation — not the separate `leidenalg` package) and analyze pathways / cluster graphs.
- Code: [`utils/ol_data.py`](src/utils/ol_data.py), [`utils/cb_data.py`](src/utils/cb_data.py) (clustering), [`utils/path_filtering.py`](src/utils/path_filtering.py), and the cluster-graph plotters in [`utils/plotting.py`](src/utils/plotting.py).

## Code Organization by Module

### `src/utils/` — Core Utilities (Most Important!)
**Before implementing new functionality, check this directory first** — extensive reusable functions exist here. Two layers live in `utils/`: cached-data loaders and the plotting stack. [`plotting.py`](src/utils/plotting.py) is a thin re-exporting facade over per-section `plotting_<section>.py` modules; the actual save helper (`_save`, re-exported as `plot._save`) lives in [`plotting_shared.py`](src/utils/plotting_shared.py) and writes every per-topic `results/` folder.

Cached-data / analysis loaders:
- **[`config.py`](src/utils/config.py)** — central config: paths (`DATA_DIR`, `CACHE_PROC_DIR`, `MODEL_COMPARISON_DIR`, `FIG_DIR`), dataset/side selection (`DATASET`/`FIG_DATASET = "malecns_v1.0"`, `SIDE`/`SIDE_CHAR`, `FLYWIRE_DATASET`), shared color palettes. **These are module-level constants, not CLI flags or env vars** (see run-order footgun below).
- **[`core_data.py`](src/utils/core_data.py)** — foundational cached-data loaders shared across every downstream module (wraps `connectome_interpreter`)
- **[`ol_data.py`](src/utils/ol_data.py)** — OL-side connectivity, clustering, pathways, and related cached data
- **[`cb_data.py`](src/utils/cb_data.py)** — CB-side connectivity, clustering, and OL↔CB joint analyses
- **[`vic.py`](src/utils/vic.py)** — OL↔CB and binocular visual-input contribution (VIC) cached data; also holds the per-channel effective-weight table (`get_ol_cb_effweight_by_channel`, backing `get_effweight_channel`) that was formerly the `effwt_matrix` step
- **[`cache.py`](src/utils/cache.py)** — cache helpers for preprocessing outputs
- **[`ol_rf.py`](src/utils/ol_rf.py)** — RF utilities for OL neurons (incl. `count_col_ahull`, the hex-grid alpha-hull column-extent metric; `get_rf_raw_ol`/`get_rf_raw_cb` combined RF parquet loaders; `rebuild_input_rf_caches`)
- **[`external_rf.py`](src/utils/external_rf.py)** — external/grid RF fitting helpers
- **[`polarity.py`](src/utils/polarity.py)** — ON/OFF polarity cross-checks against the fly-literature spreadsheet
- **[`geometry.py`](src/utils/geometry.py)**, **[`hex_hex.py`](src/utils/hex_hex.py)** — geometry / hex-grid / depth utilities
- **[`alpha_shape.py`](src/utils/alpha_shape.py)**, **[`ng_view.py`](src/utils/ng_view.py)** — alpha-hull geometry + misc helpers
- **[`query_roi.py`](src/utils/query_roi.py)** — primary-ROI lookup helper (`get_primary_rois`), used by the contribution-panel code
- **[`olc_client.py`](src/utils/olc_client.py)** — NeuPrint/OLC client setup (project/auth bootstrap, with `config.py`)

Plotting stack:
- **[`plotting.py`](src/utils/plotting.py)** — thin re-exporting facade (split 2026-08-20) over one `utils/plotting_<section>.py` module per figure-group section 0..10 (`plotting_shared`, `plotting_inventory`, `plotting_flow`, `plotting_clusters`, `plotting_lr_clustering`, `plotting_polarity`, `plotting_rf`, `plotting_rf_experimental`, `plotting_cb_vic`, `plotting_cb_lr_clustering`, `plotting_cb_rf`; section 11 "Model comparisons" has no functions/module). Existing `from utils import plotting as plot; plot.plot_*(...)` call sites are unaffected. `plotting_shared.py` centralizes saving (`_save`, writing a PDF + interactive HTML sidecar per topic folder).
- **[`figure_helpers.py`](src/utils/figure_helpers.py)** — reusable helpers extracted from `paper_figures.ipynb` (signed polarity-bar family, per-instance effective-#-inputs, label wrapping, `plot_gaussian_params` — moved here from the retired `plotting_functions.py`)
- **[`projection_plots.py`](src/utils/projection_plots.py)** — 2D extreme-value projection / neuron-with-outline plots (visual-input contribution panels)
- **[`pdf_report.py`](src/utils/pdf_report.py)**, **[`plotter.py`](src/utils/plotter.py)**, **[`gallery_index.py`](src/utils/gallery_index.py)**, **[`gallery_common.py`](src/utils/gallery_common.py)** — gallery/PDF-report figure builders (used by the `make_figures/gallery_*.ipynb` notebooks; import `make_bar_fig` from `gallery_common`)
- **[`palettes.py`](src/utils/palettes.py)**, **[`ol_color.py`](src/utils/ol_color.py)** — hardcoded palettes + colour-space helpers for paper-figure renderers
- **[`path_filtering.py`](src/utils/path_filtering.py)** — path-level filters for pathway example diagrams

### `src/make_figures/` — Paper-Figure Rendering
- **Plotting entry points (notebooks):**
  - [`paper_figures.ipynb`](src/make_figures/paper_figures.ipynb) — the **main** figure notebook; includes the visual-input contribution (VIC) / resolution section (the former `quan_propagation` `contrib_plot_mip` and `pokemon_plot` analyses: Main 5g/5h/6a/7c-right, SI 11/12), whose panels still save under `results/quan_propagation/`
  - [`gallery_vpn.ipynb`](src/make_figures/gallery_vpn.ipynb), [`gallery_vcbn.ipynb`](src/make_figures/gallery_vcbn.ipynb) — standalone per-instance galleries
- **[`figures.py`](src/make_figures/figures.py)** — panel-manifest distributor. Maps result PDFs/sidecars produced by the plotters onto per-panel filenames in `results/<topic>/`, driven by the manifest [`params/paper_figs.xlsx`](params/paper_figs.xlsx). `distribute()` + `build_index()` rebuild [`results/html_figures/`](results/html_figures/). It derives its `results/` path from `utils.config.FIG_DIR`. Rows whose `results file` cell is a literal tag from the `EXTERNAL` set — `Blender`/`Neuroglancer`/`custom` (5/2/6 rows as of 2026-08-24) — are excluded from `distribute()` (no auto-distributable file). The VIC panels formerly carried a fourth tag, `AZ`; they now point at real `quan_propagation/*.pdf` paths (85 rows) and *are* distributed normally, so that tag was dropped from `EXTERNAL` on 2026-08-24 (it already matched zero rows). Their `html_figures/` viewer entries are still added by hand, since those panels are PDF-only with no `.html` sidecar. Import as `from make_figures import figures as pf`.
  **Manifest integrity (audited 2026-08-24):** all 313 distributable rows resolve — every
  `results file` basename matches a save-name the current plotting code actually writes, and
  291 of the 313 target panels are present under `results/main_*` / `si_*`. The 22 absent ones
  are all *intentional*, and the notebook's closing
  `pf.parity_check(exclude_folders=['si_5','si_6'], expected_missing=['si_12/si_12_25.pdf','si_12/si_12_26.pdf'])`
  declares exactly those: the 20 `model_comparison/` rows (SI 5d/5e + SI 6b–6e) whose generating
  cells are deliberately commented out as opt-in, and 2 SI-12 cluster panels. So a `distribute()`
  run reporting those 22 as "missing" is expected, not a regression. One row —
  `CB_clustering/layer_hist_left.pdf` (SI 8a) — legitimately carries no `{dataset}` prefix,
  because `plotting_cb_lr_clustering.plot_cb_layer_hist_left` hardcodes that filename.
- **`.py` files:** jupytext pairs of the three notebooks ([`paper_figures.py`](src/make_figures/paper_figures.py), [`gallery_vpn.py`](src/make_figures/gallery_vpn.py), [`gallery_vcbn.py`](src/make_figures/gallery_vcbn.py)) plus `figures.py`. There is **no** `figures.ipynb` and no `make_figures/lib/` subfolder.

### `src/setup_data/` — Data Prep Pipeline
Precomputes the cached data + input RFs the figure code reads. **Kept even though figure code never imports it** — it's a separate entry point you run to (re)build the cache. This hits NeuPrint and is slow. Flat layout: two library modules + five jupytext-paired numbered stages (`s1`–`s5`) + the optional FlyWire stage `s1_flywirefafb_data.py` (`.py`-only, no `.ipynb` pair yet). There is **no** `lib/`, `base_cache/`, or `cb_cache/` subfolder.
- **[`malecns.py`](src/setup_data/malecns.py)** — MaleCNS meta + sparse-matrix assembly functions (library; imported as `sd` by `s1`)
- **[`flow.py`](src/setup_data/flow.py)** — dataset-agnostic lateral-flow matrix builders (library; imported as `sd` by `s3`)
- **[`s1_malecns_data`](src/setup_data/s1_malecns_data.py)**, **[`s2_layers`](src/setup_data/s2_layers.py)**, **[`s3_effconn`](src/setup_data/s3_effconn.py)** — dataset / layer-flow / effective-connectivity builders, run in that order. `s2_layers` also holds the model-comparison flow-variant cells that write `cache/model_comparison/` (SI 5/6). `s2_layers` and `s3_effconn` are dataset-generic: each builds its FlyWire counterpart (flow / lateral flow) by uncommenting the override in its "Dataset selection" cell.
- **[`s1_flywirefafb_data`](src/setup_data/s1_flywirefafb_data.py)** — optional FlyWire (FAFB v783) analog of `s1` (added 2026-08): builds `flywire-fafb-v783b_meta.csv` / `_prop.npz` / `_inprop.npz` for the SI Fig. 14 flywire cells + the FlyWire lateral flow. Connects with an **in-script override** to `neuprint-cns.janelia.org`, dataset `flywire-fafb:v783b` (`.env` untouched — it points at `neuprint.janelia.org`, which doesn't host flywire). The FlyWire flow + lateral flow then come from `s2_layers` and `s3_effconn`, each via the commented override in its "Dataset selection" cell; see `docs/run_order.md` §2b for the full FlyWire sequence.
- **[`s4_assemble_vpn_vcbn`](src/setup_data/s4_assemble_vpn_vcbn.py)**, **[`s5_tbar`](src/setup_data/s5_tbar.py)** — build the contribution-panel cache, run in that order. `s4` assembles the VPN/VCBN tables and force-rebuilds the two combined per-neuron RF parquet caches (2D-Gaussian `size` + alpha-hull `ahull_size` in one parquet — the former standalone `rf_size` step, now folded in). `s5` runs the heavy tbar queries (~100 min tbar-location query + ~9 h per-ROI tbar counts).
- **Input-RF caches** — no separate stage; the four [`utils.ol_rf`](src/utils/ol_rf.py) input-RF caches build lazily on first figure call. Force a rebuild with `ol_rf.rebuild_input_rf_caches(force=True)` after re-running `s1` §6 "Input RF raw data".

> **Run-order footgun:** `s1`/`s2`/`s3` read `SIDE_CHAR` as a module-level import from [`config.py`](src/utils/config.py) — it is *not* a function parameter. To build both-sides raw cache you must run each once with `SIDE = "right"`, then edit `config.py` to `SIDE = "left"` and run them again; nothing warns you if you only ran the "right" pass. (`s4`/`s5` take `side_char` explicitly and loop over both sides, so they don't have this issue.) Full details in [`docs/run_order.md`](docs/run_order.md).

### Sexual dimorphism — no module; the two paper panels are generated inline
There is no `src/dimorphism/` directory (deleted in commit `2352732`), and none is needed:
both dimorphic-neuron panels are now produced directly by [`paper_figures.py`](src/make_figures/paper_figures.py),
which writes them into `results/dimorphism/` where the manifest expects them.
- **Main 7c-left** (`{DATASET}_dimorphic_neurons_paper_panel.pdf`) — the malecns CB panel:
  per dimorphism category (male-specific, sexually dimorphic), the fraction of CB neurons
  that are visual (VIC > 5e-4). The `dimorphism` flag comes from `core_data.get_meta()`
  (fetched by [`setup_data/malecns.py`](src/setup_data/malecns.py)).
- **SI 14c** (`flywire-fafb-v783b_dimorphic_neurons_paper_panel.pdf`) — the FlyWire analog
  (categories female-specific / sexually dimorphic), in the SI-14 flywire section that
  monkeypatches the shared loaders and so must run after every malecns panel.

The exploratory outputs of the retired `dimorphism_analysis` notebook
(`results/neurons_dimorphism.csv`, `results/tables/{DATASET}_dimorphism.csv`) are gone and
are not regenerated — only the two paper panels above survive.

## Key Technologies & Dependencies

The real manifest is [`pyproject.toml`](pyproject.toml); dependencies carry **version ceilings** (not exact pins) to avoid bleeding-edge breakage, with exact versions in [`requirements-lock.txt`](requirements-lock.txt).

### Neuroscience Libraries
- **`connectome_interpreter`** (`@main` fork): connectivity/effective-weight analysis — the **primary data layer**
- **`neuprint-python`** (`artxz` fork `@master`): NeuPrint API — the fork adds `omit_rois` support navis expects
- **`navis`** (`artxz` fork `@master`): neuroanatomy visualization
- **`caveclient`/`fafbseg`, `cloud-volume`**: FlyWire/CAVE tooling + 3D volumetric data access

### Scientific Stack
- **numpy** (2.x — `>=2,<2.3`, migrated 2026-08), **scipy** (`<1.14`), **pandas** (`<3`), **scikit-learn**: data processing
- **networkx**, **python-igraph**: graph analysis & clustering (Leiden uses igraph's built-in impl.)
- **alphashape**, **shapely** (`trimesh` via navis): geometric computations
- **pyarrow** (`>=16`), **openpyxl**: parquet/feather + xlsx I/O

### Visualization
- **matplotlib** (`<3.11`), **plotly**, **pyvis**: 2D/3D visualization
- **kaleido** (`>=1.0`): plotly static (PDF/PNG) export — see the CLI-env caveat under "DON'T"
- **cmap**, **colormath**: color maps / color-space conversions
- **pypdf**, **pymupdf**, **reportlab**: PDF assembly for galleries/reports

## Development Guidelines

### Environment Setup
1. **Copy [`.env_sample`](.env_sample) to `.env`** and fill in credentials:
   ```bash
   NEUPRINT_SERVER_URL=neuprint-cns.janelia.org
   NEUPRINT_DATASET_NAME=male-cns:v1.0   # (or flywire-fafb:v783b)
   NEUPRINT_APPLICATION_CREDENTIALS=<your_token>
   ```
   Get your token from https://neuprint-cns.janelia.org/account. (Note: the NeuPrint server
   dataset name `male-cns:v1.0` is distinct from the internal cache-file prefix
   `config.DATASET = "malecns_v1.0"`.)

2. **Install dependencies:**
   ```bash
   pip install -e .                        # pyproject.toml is the real manifest
   # or, for an exactly-reproducible pinned env:
   pip install -r requirements-lock.txt
   ```
   [`requirements.txt`](requirements.txt) still works but is just a thin `-e .` + dev-tools shim over
   [`pyproject.toml`](pyproject.toml) — don't edit dependencies there. The project's conda env is
   `ol-connectome` (Jupyter kernel `ol-c-kernel`); use that interpreter to run repo code.

3. **Project setup boilerplate** (used in most scripts — the project is not yet a proper package):
   ```python
   from pathlib import Path
   import sys
   from dotenv import load_dotenv, find_dotenv
   PROJECT_ROOT = Path(find_dotenv()).parent
   sys.path.append(str(PROJECT_ROOT.joinpath('src')))
   ```

### Import Conventions
`src/` lacks `__init__.py` files, so imports rely on the `sys.path.append('src')` boilerplate above:
```python
from utils.core_data import get_meta, get_ol_meta
from utils import plotting as plot
from make_figures import figures as pf
```

### Jupyter Notebook Workflow
- Most analysis is done in the `make_figures/*.ipynb` and `setup_data/*.ipynb` notebooks.
- Each notebook is **jupytext-paired** with a `.py` file (config in [`jupytext.toml`](jupytext.toml)); keep the pair in sync (`src/sync-nb.py`, or the `jupytext-sync` skill).
- **Check for pair drift before syncing.** A naive mtime-directed sync can silently clobber
  whichever side is newer with stale content from the other — check `git log -S<marker>` for
  both files first if they might have diverged (branch merges are a common cause: a `git merge`
  can quietly revert a fix already applied on one side, as happened to `s5_tbar.py`'s
  `cache_dir`→`DATA_DIR` fix across a `main`-into-`reorg-aug` merge on 2026-08-24).
- Use `NAVIS_JUPYTER_PLOT3D_BACKEND=plotly` for 3D plotting in notebooks.

### Git Workflow
- PR-based workflow. Branch naming: descriptive names (e.g., `feature/vpn-clustering`); the current working branch is `reorg-aug`, main branch is `main`.
- `results/` topic folders and `cache/` are gitignored; only `results/html_figures/` and `results/tables/` are tracked.

## Common Tasks & Patterns

### Render paper figures
Open [`make_figures/paper_figures.ipynb`](src/make_figures/paper_figures.ipynb), which pulls cached data through
[`utils/core_data.py`](src/utils/core_data.py) / [`utils/ol_data.py`](src/utils/ol_data.py) and plots via [`utils/plotting.py`](src/utils/plotting.py) (a facade over `utils/plotting_<section>.py`). Saving is
centralized in `plotting_shared.py`, which writes a PDF + interactive HTML sidecar into `results/<topic>/`.

### Distribute panels into the viewer
After plots are generated, map result files onto per-panel filenames using the manifest [`params/paper_figs.xlsx`](params/paper_figs.xlsx):
```python
from make_figures import figures as pf
pf.distribute()      # copy/rename topic-folder sidecars into results/html_figures/
pf.build_index()     # regenerate results/html_figures/index.html
```

### Render the visual-input contribution panels
The "Visual-input contribution (VIC) / resolution" section near the end of [`paper_figures.ipynb`](src/make_figures/paper_figures.ipynb)
renders these panels (contribution, resolution, coverage) into `results/quan_propagation/`.
It reads cache built by [`s4_assemble_vpn_vcbn`](src/setup_data/s4_assemble_vpn_vcbn.py)
+ [`s5_tbar`](src/setup_data/s5_tbar.py). The per-instance galleries are the separate standalone notebooks
[`gallery_vpn.ipynb`](src/make_figures/gallery_vpn.ipynb) / [`gallery_vcbn.ipynb`](src/make_figures/gallery_vcbn.ipynb).

### (Re)build cached data
Run the [`setup_data/`](src/setup_data/) numbered stages `s1`→`s5` in order (see the run-order footgun above and [`docs/run_order.md`](docs/run_order.md) for the full dependency chain).

## Important Notes for Claude Code

### DO:
✅ **Check [`src/utils/`](src/utils/) first** before implementing new utilities — extensive reusable functions exist
✅ **Fetch data through [`utils/core_data.py`](src/utils/core_data.py)** (`connectome_interpreter`), not a hand-rolled NeuPrint query
✅ **Respect research context** — this is scientific analysis code; prioritize correctness
✅ **Keep [`src/setup_data/`](src/setup_data/)** — it's the data-prep pipeline even though figure code never imports it
✅ **Follow the PR workflow** for changes
✅ **Be careful with data files** — `cache/data/`, `cache/data_proc/`, and the `params/*.xlsx` manifests are research inputs; the user edits the xlsx files live
✅ **Use the project setup boilerplate** at the top of standalone scripts (see "Environment Setup")

### DON'T:
❌ **Don't reintroduce the removed modules** — `queries/`, `network/`, `rf/`, `pathways/`, `input_distribution/`, `quan_propagation/`, `dimorphism/` (its two paper panels live inline in `paper_figures.py`), etc. were deleted as figure-unused
❌ **Don't modify data files** without explicit permission
❌ **Don't implement functionality that likely exists** in [`src/utils/`](src/utils/)
❌ **Don't break NeuPrint authentication** — the `.env` file is critical
❌ **Don't make breaking changes** to widely-imported modules ([`core_data`](src/utils/core_data.py), [`plotting`](src/utils/plotting.py), [`config`](src/utils/config.py))
❌ **Don't assume package imports work** — the project lacks `__init__.py` files (see "Known Issues")
❌ **Don't rely on `write_image` (PDF/PNG export) in the headless CLI env** — kaleido may not render there; HTML sidecars via `write_html` work. Figures are rendered in the user's own notebook kernel.

### Code Quality Considerations
- **Large files exist:** [`utils/plotter.py`](src/utils/plotter.py), [`utils/plotting_clusters.py`](src/utils/plotting_clusters.py), [`utils/cb_data.py`](src/utils/cb_data.py), [`utils/ol_rf.py`](src/utils/ol_rf.py), and [`make_figures/paper_figures.ipynb`](src/make_figures/paper_figures.ipynb) are large (mixed concerns).
- **Debug code:** DEBUG comments remain (especially in [`utils/projection_plots.py`](src/utils/projection_plots.py) / the contribution-panel cells).
- **Stale docs vs. code:** much historical code has been removed; **verify a file/symbol exists before relying on it.**

## Known Issues & Technical Debt

1. **Not a proper Python package:** no `__init__.py` files, requires the `sys.path.append('src')` boilerplate
2. **Git dependencies unpinned:** `connectome_interpreter` (`@main`), `navis`, and `neuprint-python` (`@master` forks) track moving branches — not fully reproducible (mitigated by `requirements-lock.txt`)
3. **CI is lint-only:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `ruff check` on push/PR, but with a narrow rule set (`E9, F63, F7, F82` — real errors only, no style/formatting); a `[tool.mypy]` config exists in `pyproject.toml` but isn't enforced automatically
4. **No test framework:** there is no test suite (the old `src/test/` scripts were removed)
5. **Large monolithic files:** some modules mix many concerns and are long
6. **No Git Large File Storage (LFS):** large CSV/data files are tracked directly in git (`.gitattributes` has no LFS filters)
7. **`.DS_Store` is committed** despite being in `.gitignore` (the ignore has no effect on an already-tracked file) — a `git rm --cached .DS_Store` would clean it up

## Additional Resources

- **Documentation:** [`docs/`](docs/) — [`run_order.md`](docs/run_order.md) (sequencing).
- **Original codebase:** some code carried over from [male-drosophila-visual-system-connectome-code](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code)
- **Interactive figures:** [GitHub Pages figure browser](https://reiserlab.github.io/visualpathways/results/html_figures/)
- **NeuPrint API:** https://neuprint-cns.janelia.org/ · **Janelia FlyEM:** https://www.janelia.org/project-team/flyem · **Reiser Lab:** https://www.janelia.org/lab/reiser-lab/

---

**Last Updated:** 2026-08-24
**Repository Stats:** 3 module directories under `src/` (`utils/` — 36 modules, `make_figures/`, `setup_data/`). GNU GPL v3.
