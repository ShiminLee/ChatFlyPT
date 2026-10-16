from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data/derived/reconstruction"
REPORTS = ROOT / "reports/reconstruction"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
