# Eyemap Archive

A public archive of pre-computed "eyemaps" for the *Drosophila* optic lobe:
the mapping between each ommatidium/column's hexagonal lattice coordinate
(`p, q`) and its 3D viewing direction (`x, y, z` unit vector / `theta, phi`
spherical angle), optionally linked to a connectome neuron ID (`rootid`).

Live site: `https://artxz.github.io/eyemap-archive/` (once GitHub Pages is
enabled — see below). It lets you browse available datasets, inspect one
interactively (rotatable 3D scatter, Mollweide, or Mercator projection), and
download it by hand or by script.

This repo holds **results only**. The pipeline that generates these files
(lens/eye scans → hex-lattice registration → connectome alignment) lives in
the private `R_eyemap-DIY` project, not here.

## Repo layout

```
maps/                # canonical source data — one folder per dataset
  eyemap_<id>_<date>/
    lens_med.rda          # lens/seed-hex QC data (R), kept as-is
    pqxyztp[id]_{left,right}.xlsx   # p,q,x,y,z,theta,phi[,rootid] per column
scripts/
  build_manifest.py   # generates docs/manifest.json + docs/data/*.{csv,json}
docs/                 # GitHub Pages site (served from /docs on main)
```

## Adding a new dataset

1. Drop a new folder under `maps/` named `eyemap_[DRA_]<source>_[f]<YYYYMMDD>/`
   containing `lens_med.rda` and `pqxyztp[id]_{left,right}.xlsx`.
2. Regenerate the site data:
   ```
   python -m venv .venv   # first time only
   .venv/Scripts/python -m pip install -r scripts/requirements.txt   # first time only
   .venv/Scripts/python scripts/build_manifest.py
   ```
3. Commit the new `maps/` folder and the regenerated `docs/manifest.json` +
   `docs/data/<id>/` files, then push.

## Local preview

```
cd docs
python -m http.server 8000
```
Open `http://localhost:8000/`.

## Programmatic access

Fetch `manifest.json` first — it's the machine-readable index of every
dataset and its download URLs (raw `.xlsx`/`.rda` on GitHub, plus generated
CSV/JSON). See the Home page for Python/R snippets.

## Citation

- [Zhao et al. 2025](https://www.nature.com/articles/s41586-025-09276-5)
- [Nern et al. 2025](https://doi.org/10.1038/s41586-025-08746-0)

## License

Data: CC BY-SA 4.0. Code: MIT. See [LICENSE](LICENSE).
