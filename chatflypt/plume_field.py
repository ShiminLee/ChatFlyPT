"""Plume field borrowed from the vendored upstream odor arena.

The field model and the grid-indexing math are ported from
`vendor/flygym-gymnasium/flygym_gymnasium/examples/olfaction/plume_tracking_arena.py`
(Apache-2.0, pinned in `research/manifests/sources.lock.json`), and the plume
grid itself is the upstream vendored data file
`vendor/flygym-gymnasium/flygym_gymnasium/data/test_data/plume_tracking/plume_short.hdf5`
(60 frames x 320 x 480, float16, generated upstream by a 2-D Navier-Stokes solver
driven by a Brownian inflow; see `examples/olfaction/simulate_plume_dataset.py`).

What is NOT borrowed: nothing here is calibrated to cVA, to a concentration, or
to this model fly's scale. The upstream arena is a 160 x 240 mm arena at 0.5 mm
per cell; this project's arena is a few millimetres across, so the grid is
sampled with an explicit offset that places the fly near the inflow region. The
upstream code returns NaN outside the grid; that is preserved and treated as
"no measured input" rather than as zero.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import ROOT

PLUME_HDF5 = (ROOT / 'vendor/flygym-gymnasium/flygym_gymnasium/data/test_data'
              / 'plume_tracking/plume_short.hdf5')
UPSTREAM_LICENSE = 'Apache-2.0 (flygym-gymnasium)'
# Upstream defaults (plume_tracking_arena.py: OdorPlumeArena signature).
CELL_MM = .5
FPS = 200.
# Engineering placement: put the fly's arena origin at the plume's filled core
# (upstream grid rows 81-312, cols 16-479; peak near row 160, col 23) so the
# receptor samples a region the upstream solver actually filled.
ORIGIN_CELL = (160, 23)


@lru_cache(maxsize=2)
def _grid(path, modified_ns, size):
    import h5py
    with h5py.File(path, 'r') as handle:
        grid = np.asarray(handle['plume'], dtype=np.float32)
        inflow = {key: np.asarray(handle[key]).tolist()
                  for key in ('inflow_pos', 'inflow_radius', 'inflow_scaler')
                  if key in handle}
    return grid, inflow


def plume_data():
    """Load the vendored plume grid once per file version."""
    if not PLUME_HDF5.is_file():
        raise FileNotFoundError('Vendored plume data missing: ' + str(PLUME_HDF5))
    stat = PLUME_HDF5.stat()
    grid, inflow = _grid(str(PLUME_HDF5), stat.st_mtime_ns, stat.st_size)
    return grid, inflow


def provenance():
    """Record exactly which upstream artifact the field came from."""
    if not PLUME_HDF5.is_file():
        return {'available': False, 'path': str(PLUME_HDF5.relative_to(ROOT))}
    grid, inflow = plume_data()
    return {'available': True, 'path': str(PLUME_HDF5.relative_to(ROOT)),
            'sha256': hashlib.sha256(PLUME_HDF5.read_bytes()).hexdigest(),
            'shape': list(grid.shape), 'dtype': 'float32 (file stores float16)',
            'value_range': [float(grid.min()), float(grid.max())],
            'cell_mm': CELL_MM, 'fps': FPS, 'origin_cell': list(ORIGIN_CELL),
            'inflow': inflow, 'license': UPSTREAM_LICENSE,
            'upstream': 'flygym-gymnasium/examples/olfaction/plume_tracking_arena.py',
            'model': 'Precomputed 2-D Navier-Stokes plume with Brownian inflow (upstream generator)',
            'calibrated_to_cva': False, 'concentration_units': False}


def sample(point_mm, time_ms, strength=1.):
    """Sample the plume at a measured receptor position, upstream indexing math.

    The upstream record is 0.3 s long; a longer observation window wraps around
    it (documented as a repeated record, not as a longer simulation). Returns
    None when the sample falls outside the grid (upstream returns NaN), which the
    caller must treat as "no measured input", never as zero.
    """
    grid, _ = plume_data()
    frames, height, width = grid.shape
    frame = int((float(time_ms) / 1000.) * FPS) % frames
    column = int(round(float(point_mm[0]) / CELL_MM)) + ORIGIN_CELL[1]
    row = int(round(float(point_mm[1]) / CELL_MM)) + ORIGIN_CELL[0]
    if not (0 <= row < height and 0 <= column < width):
        return None
    value = float(grid[frame, row, column])
    if not np.isfinite(value):
        return None
    return float(strength) * value


if __name__ == '__main__':
    print(json.dumps(provenance(), ensure_ascii=False, indent=2))
