"""Download immutable upstream archives without running upstream code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "shiu-brain": "philshiu/Drosophila_brain_model",
    "eon-fly-brain": "eonsystemspbc/fly-brain",
    "flm": "nftechie/flm",
    "fly-brain-chat": "lixiang1076/fly-brain",
    "flyvis": "TuragaLab/flyvis",
    "flygym": "NeLy-EPFL/flygym",
    "flygym-gymnasium": "NeLy-EPFL/flygym-gymnasium",
    "flywire-annotations": "flyconnectome/flywire_annotations",
    # Pinned body/evidence sources: behavior_body.py / behavior_readout.py verify
    # their file sha256 at load time. No root license upstream — fetched as an
    # immutable archive at the pinned commit, never redistributed from here.
    "badapple-fly": "kevinlinxc/badapple-fly",
}


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    subprocess.run([
        "curl.exe", "--fail", "--location", "--retry", "3",
        "--connect-timeout", "15", "--max-time", "600", "--silent",
        "--show-error", "-H", "User-Agent: ChatFlyPT-research", url,
        "--output", str(partial),
    ], check=True)
    partial.replace(target)


def main() -> None:
    lock_path = ROOT / "research/manifests/sources.lock.json"
    records = json.loads(lock_path.read_text()) if lock_path.exists() else []
    by_id = {r["id"]: r for r in records}
    for name, repo in REPOS.items():
        record = by_id.get(name)
        if record is None:
            metadata = ROOT / "research/snapshots" / (repo.replace("/", "__") + "__commit.json")
            fetch(f"https://api.github.com/repos/{repo}/commits/HEAD", metadata)
            commit = json.loads(metadata.read_text(encoding="utf-8"))["sha"]
            record = {"id": name, "repository": f"https://github.com/{repo}", "commit": commit}
        commit = record["commit"]
        archive = ROOT / "research/archives" / f"{name}-{commit}.tar.gz"
        if not archive.exists():
            print(f"Downloading {repo}@{commit[:12]}", flush=True)
            fetch(f"https://codeload.github.com/{repo}/tar.gz/{commit}", archive)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if record.get("archive_sha256") and digest != record["archive_sha256"]:
            raise ValueError(f"Archive hash mismatch: {name}")
        destination = ROOT / "vendor" / name
        if not destination.exists():
            destination.mkdir(parents=True)
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    parts = Path(member.name).parts
                    if len(parts) < 2:
                        continue
                    member.name = str(Path(*parts[1:]))
                    # Python's data filter rejects traversal, escaping symlinks and devices.
                    tar.extract(member, path=destination, filter="data")
        record.update({"archive_sha256": digest, "local_path": f"vendor/{name}",
                       "retrieved_at": datetime.now(timezone.utc).isoformat(),
                       "archive_url": f"https://codeload.github.com/{repo}/tar.gz/{commit}"})
        by_id[name] = record
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(list(by_id.values()), indent=2) + "\n", encoding="utf-8")
        print(f"Ready {name}: {archive.stat().st_size:,} archive bytes", flush=True)


if __name__ == "__main__":
    main()
