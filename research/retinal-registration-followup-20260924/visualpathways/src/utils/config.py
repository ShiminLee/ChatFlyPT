"""Central config: paths and dataset selection.

Colour palettes live in `utils/palettes.py` — `load_colors` / `load_roi_colors`
used to be duplicated here as well; the copies were removed (2026-08-24) because
the `load_roi_colors` pair had silently diverged (this module's copy was missing
`SCL` and every left-side ROI). Import them from `utils.palettes`.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "cache" / "data"
PARAMS_DIR = PROJECT_ROOT / "params"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_PROC_DIR = PROJECT_ROOT / "cache" / "data_proc"
# Model-comparison cache: the alt-model flow (`_hit.csv`) + propagated
# (`_lat_flow_*_sum.npz`) files for the SI Fig. 5/6 comparison, kept out of
# the main `data/` tree because they are optional, heavy to (re)build, and
# named with a per-variant suffix rather than the standard dataset scheme.
MODEL_COMPARISON_DIR = PROJECT_ROOT / "cache" / "model_comparison"
FIG_DIR = PROJECT_ROOT / "results"
HTML_FIG_DIR = FIG_DIR / "html_figures"

DATASET = "malecns_v1.0"
# Figures are named after the data version; keep these in sync.
FIG_DATASET = DATASET
SIDE = "right"
N_FLOW = 20
HIT_THRE = 0.1

SIDE_CHAR = SIDE[0]
DATASET_DIR = PROJECT_ROOT / "cache" / "data" / DATASET

# FlyWire lateral-flow build (used only by setup_effconn.py's FlyWire mode).
# This is NOT a global dataset switch: the figure pipeline + utils always run on
# DATASET above. These are applied only when setup_effconn.py's "Dataset
# selection" override is uncommented, so the global dataset never gets left in a
# FlyWire state.
FLYWIRE_DATASET = "flywire-fafb-v783b"
FLYWIRE_SIDE_CHAR = "l"


def load_nt_colors():
    """Neurotransmitter → hex color mapping."""
    return {
        "acetylcholine": "#EE672D",
        "glutamate":     "#09A64D",
        "gaba":          "#1F4695",
        "dopamine":      "#979DA5",
        "histamine":     "#979DA5",
        "octopamine":    "#979DA5",
        "serotonin":     "#979DA5",
        "other":         "#979DA5",
        "unclear":       "#979DA5",
    }
