"""Fetch three official MaleCNS v1.0 tables, pin generations and verify MD5."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from acquire_sources import fetch

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connections": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
PREFIX = "v1.0/connectome-data/flat-connectome/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-connections", action="store_true",
                        help="Also download the approximately 1.05 GB full connection table.")
    args = parser.parse_args()
    lock = ROOT / "research/manifests/malecns.lock.json"
    records = json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else []
    by_id = {r["id"]: r for r in records}
    for name, filename in FILES.items():
        if name == "connections" and not args.include_connections:
            continue
        record = by_id.get(name)
        if record is None:
            metadata = ROOT / "research/snapshots" / (filename + ".gcs.json")
            fetch("https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o/" +
                  quote(PREFIX + filename, safe=""), metadata)
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            record = {"id": name, "dataset": "MaleCNS", "version": "v1.0",
                      "source_page": "https://male-cns.janelia.org/download/",
                      "license": "CC-BY-4.0", "generation": meta["generation"],
                      "size_bytes": int(meta["size"]),
                      "upstream_md5": base64.b64decode(meta["md5Hash"]).hex(),
                      "url": "https://storage.googleapis.com/flyem-male-cns/" + PREFIX + filename +
                             "?generation=" + meta["generation"],
                      "local_path": "data/raw/malecns-v1.0/" + filename}
        target = ROOT / record["local_path"]
        if not target.exists():
            print(f"Downloading {name}: {record['size_bytes']:,} bytes", flush=True)
            fetch(record["url"], target)
        with target.open("rb") as stream:
            md5 = hashlib.file_digest(stream, "md5").hexdigest()
        if target.stat().st_size != record["size_bytes"] or md5 != record["upstream_md5"]:
            raise ValueError(f"Size/MD5 mismatch: {target}; retained for inspection")
        with target.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        record.update({"sha256": sha256, "verified_at": datetime.now(timezone.utc).isoformat()})
        by_id[name] = record
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(list(by_id.values()), indent=2) + "\n", encoding="utf-8")
        print(f"Verified {name}: MD5 matches official GCS metadata", flush=True)


if __name__ == "__main__":
    main()
