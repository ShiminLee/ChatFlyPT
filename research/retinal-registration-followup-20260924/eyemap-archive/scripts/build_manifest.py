"""Scan maps/ for eyemap datasets and generate docs/manifest.json plus
per-dataset CSV/JSON exports under docs/data/ for the web site to consume.

Usage:
    .venv/Scripts/python.exe scripts/build_manifest.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO_ROOT / "maps"
DOCS_DIR = REPO_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/artxz/eyemap-archive/main"

SOURCE_LABELS = {
    "783": "FAFB / FlyWire (materialization 783)",
    "mcns": "male CNS (maleCNS)",
}

DATASET_DIR_RE = re.compile(r"^eyemap_(?:(DRA)_)?(.+?)_f?(\d{8})$")

FLOAT_COLS = ["x", "y", "z"]
ANGLE_COLS = ["theta", "phi"]


def parse_dataset_dir(name: str) -> dict:
    m = DATASET_DIR_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized eyemap folder name: {name!r}")
    region, source, date = m.groups()
    date_fmt = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
    source_label = SOURCE_LABELS.get(source, source)
    title = source_label + (" — Dorsal Rim Area (DRA)" if region else "")
    return {
        "id": name,
        "title": title,
        "source": source,
        "region": region,
        "date": date_fmt,
    }


def export_side(xlsx_path: Path, dataset_id: str, side: str) -> dict:
    df = pd.read_excel(xlsx_path)
    df[FLOAT_COLS] = df[FLOAT_COLS].round(6)
    df[ANGLE_COLS] = df[ANGLE_COLS].round(4)

    out_dir = DATA_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{side}.csv"
    json_path = out_dir / f"{side}.json"
    df.to_csv(csv_path, index=False)

    # rootid (18-19 digit connectome IDs) exceeds JS Number.MAX_SAFE_INTEGER
    # (2^53); a bare JSON number would silently round to the wrong ID in the
    # browser, so serialize it as a string instead.
    json_df = df.copy()
    if "rootid" in json_df.columns:
        json_df["rootid"] = json_df["rootid"].astype(str)
    json_path.write_text(json_df.to_json(orient="records"), encoding="utf-8")

    return {
        "n_points": len(df),
        "has_rootid": "rootid" in df.columns,
        "files": {
            "xlsx": f"{GITHUB_RAW_BASE}/maps/{dataset_id}/{xlsx_path.name}",
            "csv": f"data/{dataset_id}/{side}.csv",
            "json": f"data/{dataset_id}/{side}.json",
        },
    }


def build_dataset_entry(dataset_dir: Path) -> dict:
    entry = parse_dataset_dir(dataset_dir.name)

    hemispheres = {}
    has_rootid = False
    for xlsx_path in sorted(dataset_dir.glob("pqxyztp*_*.xlsx")):
        side = xlsx_path.stem.rsplit("_", 1)[-1]
        if side not in ("left", "right"):
            continue
        side_info = export_side(xlsx_path, dataset_dir.name, side)
        rootid_present = side_info.pop("has_rootid")
        has_rootid = has_rootid or rootid_present
        hemispheres[side] = side_info

    entry["has_rootid"] = has_rootid
    entry["hemispheres"] = hemispheres

    rda_path = dataset_dir / "lens_med.rda"
    if rda_path.exists():
        entry["rda_file"] = f"{GITHUB_RAW_BASE}/maps/{dataset_dir.name}/lens_med.rda"

    return entry


def main() -> None:
    dataset_dirs = sorted(p for p in MAPS_DIR.iterdir() if p.is_dir())
    datasets = [build_dataset_entry(d) for d in dataset_dirs]

    manifest = {"datasets": datasets}
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DOCS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path} with {len(datasets)} datasets")
    for d in datasets:
        sides = ", ".join(
            f"{side}={info['n_points']}" for side, info in d["hemispheres"].items()
        )
        print(f"  - {d['id']}: {sides}")


if __name__ == "__main__":
    main()
