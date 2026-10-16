"""Same-record old/new body comparison. Never reruns or trains a neural model.

Phases: --prepare freezes four source records; --old replays the existing six-leg
adapter; --new replays the Bad Apple adapter after its implementation is ready.
Run using the main .venv Python. Body workers use the existing isolated runtime.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    ("right_foreleg_touch", "7972d97a4ddb4d06a404f9781d6aa09f", "7972d97a4ddb4d06a404f9781d6aa09f"),
    ("left_cold", "301732e8b7334dd9b267ce880cee77b5", "301732e8b7334dd9b267ce880cee77b5"),
    ("left_sugar", "9c5281bc5fda4816b89af0812a26958e", "9c5281bc5fda4816b89af0812a26958e"),
    ("blank", "26b566cee24d4c77b4945fdb91e4c739", "26b566cee24d4c77b4945fdb91e4c739"),
]
ANNOTATIONS = {
    "MaleCNS v1.0": ROOT / "data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "FlyWire v783": ROOT / "vendor/flywire-annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def original_observation(case):
    response = json.loads((ROOT / case["response_file"]).read_text(encoding="utf-8"))
    observations = response.get("observations") or [response["observation"]]
    return next(o for o in observations if o["id"] == case["observation_id"])


def prepare():
    cases = []
    for name, observation_id, response_id in CASES:
        case = {"name": name, "observation_id": observation_id,
                "response_file": f"reports/general-sensory/interactive/{response_id}.json"}
        obs = original_observation(case)
        raw = ROOT / obs["raw_record"]["file"]
        assert sha(raw) == obs["raw_record"]["sha256"]
        with np.load(raw, allow_pickle=False) as z:
            assert int(z["seed"]) == obs["seed"]
            assert float(z["duration_ms"]) == obs["duration_ms"]
            assert int(z["counts"].sum()) == obs["spikes"]
            np.testing.assert_array_equal(np.bincount(z["spike_indices"], minlength=len(z["counts"])), z["counts"])
            if name == "blank":
                assert not z["counts"].any() and not len(z["spike_indices"])
        case.update(dataset=obs["dataset"], seed=obs["seed"], duration_ms=obs["duration_ms"],
                    spikes=obs["spikes"], input_items=obs["reception"]["items"],
                    response_sha256=sha(ROOT / case["response_file"]),
                    raw_record=obs["raw_record"],
                    existing_body_status=obs["scene"]["body"]["status"],
                    existing_body_motor_spikes=obs["scene"]["body"].get("motor_spikes"))
        cases.append(case)
    protocol = {
        "selection": "Four records fixed before the new adapter is run: required right foreleg, different-modality cold and sugar, and a saved all-zero blank. Not selected using new behavior outputs.",
        "cases": cases, "max_duration_ms": 500, "sample_ms": 10, "timeout_seconds": 90,
        "same_neural_source": "Existing NPZ only; counts, spike indices, spike times, IDs, seed and directly driven cells preserved. No neuronal rerun, added spikes or parameter tuning.",
        "old_adapter_sha256": sha(ROOT / "chatflypt/body_simulation.py"),
        "annotation_sha256": {name: sha(path) for name, path in ANNOTATIONS.items()},
        "upstream_manifest": json.loads((ROOT / "research/manifests/badapple-fly-source.json").read_text(encoding="utf-8")),
        "comparison_limit": "Different controllers, mappings and initialization are compared; displacement differences are not biological calibration or a one-factor ablation.",
    }
    path = HERE / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == protocol, "Frozen protocol changed"
    else:
        dump(path, protocol)
    print(json.dumps({"frozen_cases": [{k: c[k] for k in ["name", "dataset", "seed", "spikes", "existing_body_status"]} for c in cases]}), flush=True)


def load_trial(case):
    path = ROOT / case["raw_record"]["file"]
    assert sha(path) == case["raw_record"]["sha256"]
    assert sha(ROOT / case["response_file"]) == case["response_sha256"]
    with np.load(path, allow_pickle=False) as z:
        ids = z["neuron_ids"].copy()
        driven = z["driven_indices"].copy()
        trial = SimpleNamespace(counts=z["counts"].copy(), spike_ids=z["spike_indices"].copy(),
                                spike_steps=z["spike_steps"].copy(), seed=int(z["seed"]),
                                duration_ms=float(z["duration_ms"]))
    path = ANNOTATIONS[case["dataset"]]
    if case["dataset"] == "MaleCNS v1.0":
        rows = pd.read_feather(path, columns=["bodyId", "type", "superclass", "subclass", "somaSide"])
        rows = rows.set_index("bodyId").reindex(ids).fillna("")
    else:
        rows = pd.read_csv(path, sep="\t", usecols=["root_id", "cell_type", "side"],
                           dtype={"root_id": str}, low_memory=False).fillna("")
        # Keep exact 64-bit IDs as the alignment index. Reindexing a duplicated
        # numeric ID column would coerce it to float when a cell has no annotation.
        rows = rows.set_index("root_id").reindex(ids.astype(str)).fillna("")
    return trial, ids, rows, driven


def summary(body):
    frames = body.get("trajectory", [])
    result = {k: body.get(k) for k in [
        "status", "physics_run", "reason", "error", "dataset", "source_spikes_sha256", "source_mapping",
        "motor_spikes", "active_motor_neurons", "motor_activity_in_window", "duration_ms", "source_duration_ms",
        "truncated", "controller", "engine", "controls", "summary", "initialization", "context", "geometry", "geometry_asset",
        "body_source", "source", "mapping_source", "model_variant", "controller_id", "command_summary",
        "biologically_calibrated", "limitations", "elapsed_seconds"] if k in body}
    result["trajectory_frames"] = len(frames)
    if body.get("physics_run"):
        assert frames and body["status"] == "ok", body
        times = np.array([f["time_ms"] for f in frames])
        positions = np.array([f["position_mm"] for f in frames])
        feet = np.array([f["feet_mm"] for f in frames])
        assert positions.shape == (len(frames), 3) and feet.shape == (len(frames), 6, 3)
        assert np.isfinite(positions).all() and np.isfinite(feet).all()
        assert times[0] == 0 and abs(times[-1] - 500) < 1e-6 and np.all(np.diff(times) > 0)
        yaw = np.unwrap([f["yaw_rad"] for f in frames])
        assert np.isfinite(yaw).all()
        result["measured"] = {"displacement_mm": (positions[-1] - positions[0]).tolist(),
                              "horizontal_displacement_mm": float(np.linalg.norm(positions[-1, :2] - positions[0, :2])),
                              "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
                              "min_thorax_height_mm": float(positions[:, 2].min()),
                              "first_frame_ms": float(times[0]), "last_frame_ms": float(times[-1]),
                              "feet_finite": True}
        result["trajectory_sha256"] = hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()
        if all("quaternion_wxyz" in frame for frame in frames):
            rotations = np.asarray([frame["quaternion_wxyz"] for frame in frames])
            assert rotations.shape == (len(frames), 4) and np.isfinite(rotations).all()
            upright_z = 1 - 2 * (rotations[:, 1] ** 2 + rotations[:, 2] ** 2)
            tilt = np.degrees(np.arccos(np.clip(upright_z, -1, 1)))
            result["measured"].update(max_tilt_deg=float(tilt.max()),
                                       overturned_sampled=bool((tilt >= 90).any()),
                                       overturn_metric="Sampled thorax local vertical at least 90 degrees from world vertical; not a complete fall detector")
        if body.get("geometry_asset"):
            asset_path = ROOT / "chatflypt/web" / body["geometry_asset"].lstrip("/")
            asset = json.loads(asset_path.read_text(encoding="utf-8"))
            result["actual_geometry"] = {
                "path": str(asset_path.relative_to(ROOT)), "sha256": sha(asset_path),
                "asset_top_level_keys": list(asset),
                "frame_geom_count": len(frames[0].get("geom_positions_mm", [])),
                "geom_indices": body.get("geom_indices"), "geom_names": body.get("geom_names"),
            }
    else:
        assert not frames, "Failed/zero run must not invent movement frames"
    return result


def verify_completed_bin_timing(body, drive):
    """Independently verify the command applied to each sampled integration end."""
    if not body.get("physics_run"):
        return {"checked": False, "reason": "No physics trajectory was generated"}
    available = np.asarray(drive["available_at_ms"])
    dt = float(body["engine"]["timestep_ms"])
    zero = {key: 0.0 for key in drive["commands"][0]}
    for frame in body["trajectory"]:
        integration_start = float(frame["time_ms"]) - dt
        row = int(np.searchsorted(available, integration_start + 1e-7, side="right")) - 1
        expected = drive["commands"][row] if row >= 0 else zero
        assert frame["commands"] == expected
        assert frame["source_command_bin"] == row
        if row >= 0:
            assert available[row] <= integration_start + 1e-7
            assert frame["command_available_at_ms"] == available[row]
        else:
            assert frame["command_available_at_ms"] is None
    return {"checked": True, "frames_checked": len(body["trajectory"]),
            "timestep_ms": dt, "no_future_bin_commands": True,
            "timing_rule": "Sampled state is at integration end; its recorded command was available at or before that step's start. Initial frame has no neural command."}


def command_summary(drive):
    times = np.asarray(drive["available_at_ms"])
    commands = drive["commands"]
    window = [(float(t), c) for t, c in zip(times, commands) if t < 500]
    keys = list(commands[0]) if commands else []
    rows = [{"available_at_ms": float(t), **c} for t, c in zip(times, commands)]
    activity = [(t, c) for t, c in window if any(abs(float(v)) > 0 for v in c.values())]
    return {
        "groups": drive["groups"], "commands_count": len(commands),
        "first_nonzero_command_available_ms": activity[0][0] if activity else None,
        "nonzero_command_rows_in_window": len(activity),
        "command_ranges_in_window": {k: {"min": min(float(c[k]) for _, c in window),
                                         "max": max(float(c[k]) for _, c in window)} for k in keys},
        "command_rows": rows,
        "readouts_hz": drive["readouts_hz"],
    }


def replay(variant):
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    assert sha(ROOT / "chatflypt/body_simulation.py") == protocol["old_adapter_sha256"]
    for dataset, digest in protocol["annotation_sha256"].items():
        assert sha(ANNOTATIONS[dataset]) == digest
    if variant == "old":
        from chatflypt.body_simulation import build_motor_drive as build, run_body as run
        sources = ["chatflypt/body_simulation.py"]
    else:
        from chatflypt.behavior_readout import build_behavior_drive as build
        from chatflypt.behavior_body import run_behavior_body as run
        sources = ["chatflypt/behavior_readout.py", "chatflypt/behavior_body.py"]
    source_hashes = {s: sha(ROOT / s) for s in sources}
    results = []
    for case in protocol["cases"]:
        print(f"{variant}: {case['name']} / {case['observation_id']}", flush=True)
        start = time.perf_counter()
        trial, ids, rows, driven = load_trial(case)
        drive = build(trial, ids, rows, dataset=case["dataset"], driven_indices=driven)
        directory = HERE / case["name"]
        dump(directory / f"{variant}-drive.json", drive)
        body = run(drive, max_duration_ms=500, sample_ms=10, timeout=90,
                   output_dir=directory / variant, render=False)
        dump(directory / f"{variant}-result.json", body)
        row = {"name": case["name"], "observation_id": case["observation_id"],
               "dataset": case["dataset"], "seed": trial.seed, "raw_record": case["raw_record"],
               "source_sha256_unchanged": sha(ROOT / case["raw_record"]["file"]) == case["raw_record"]["sha256"],
               "body": summary(body), "wall_seconds": time.perf_counter() - start}
        assert row["source_sha256_unchanged"]
        if variant == "old":
            saved = original_observation(case)["scene"]["body"]
            row["historical_body_status_matches"] = saved["status"] == body["status"]
            row["historical_trajectory_exact_match"] = saved.get("trajectory", []) == body.get("trajectory", [])
        else:
            row["behavior"] = command_summary(drive)
            row["completed_bin_timing"] = verify_completed_bin_timing(body, drive)
            old_drive = json.loads((directory / "old-drive.json").read_text(encoding="utf-8"))
            row["old_new_source_spikes_sha256_equal"] = old_drive["source_spikes_sha256"] == drive["source_spikes_sha256"]
            assert row["old_new_source_spikes_sha256_equal"]
            if not row["behavior"]["nonzero_command_rows_in_window"]:
                assert body["status"] == "no_motor_activity" and not body["physics_run"]
            keys = list(drive["commands"][0])
            with (directory / "new-commands.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["available_at_ms"] + keys)
                writer.writeheader()
                writer.writerows(row["behavior"]["command_rows"])
        if case["name"] == "blank":
            assert body["status"] == "no_motor_activity" and not body["physics_run"]
        results.append(row)
        dump(HERE / f"{variant}-summary.json", {"variant": variant, "implementation_sha256": source_hashes,
                                               "cases": results, "complete": len(results) == len(protocol["cases"])})
        print(json.dumps({"variant": variant, "name": case["name"], "status": body["status"],
                          "frames": len(body.get("trajectory", [])), "seconds": row["wall_seconds"]}), flush=True)
    assert all(sha(ROOT / s) == digest for s, digest in source_hashes.items()), "Implementation changed during run"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--new", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.old:
        replay("old")
    if args.new:
        replay("new")
