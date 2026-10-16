r"""Replay a saved neural trial into the physical model, plus explicit controls.

Run from repository root using the MAIN .venv (pandas/pyarrow are there):
  .venv\Scripts\python.exe reports\body-simulation\reproduce.py
No language, target direction, stimulus name, remote model or API key is read.
"""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chatflypt.body_simulation import build_motor_drive, run_body


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/derived/pathway-trials/902c68d5bf3e4fe9a01ef36623886b87.npz"
    output = ROOT / "data/derived/body-simulation"
    annotations_path = ROOT / "data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    with np.load(source, allow_pickle=False) as z:
        ids = z["neuron_ids"]
        t = SimpleNamespace(counts=z["counts"], spike_ids=z["spike_indices"],
                            spike_steps=z["spike_steps"], seed=int(z["seed"]),
                            duration_ms=float(z["duration_ms"]))
        driven = z["driven_indices"]
    rows = pd.read_feather(annotations_path, columns=["bodyId", "superclass", "subclass", "somaSide"]).set_index("bodyId").reindex(ids).fillna("")
    drive = build_motor_drive(t, ids, rows, dataset="MaleCNS v1.0", driven_indices=driven)
    output.mkdir(parents=True, exist_ok=True)
    (output / "measured-drive.json").write_text(json.dumps(drive, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Actual recorded motor spikes:", drive["motor_spikes"], flush=True)
    actual = run_body(drive, output_dir=output / "neural", render=True)
    if actual["status"] != "ok":
        raise RuntimeError(actual)
    # Same physical model, seed, initial condition and time; no actuator gains.
    passive = run_body(drive, output_dir=output / "passive", passive_control=True)
    if passive["status"] != "passive_control":
        raise RuntimeError(passive)
    zero_trial = SimpleNamespace(counts=np.zeros_like(t.counts), spike_ids=np.array([], dtype=int),
                                 spike_steps=np.array([], dtype=int), seed=t.seed, duration_ms=t.duration_ms)
    zero_drive = build_motor_drive(zero_trial, ids, rows, dataset="MaleCNS v1.0", driven_indices=driven)
    zero = run_body(zero_drive)
    assert zero["status"] == "no_motor_activity" and zero["trajectory"] == []
    assert not zero["physics_run"]
    assert actual["controls"]["max_force_on_zero_drive_actuators"] == 0
    assert passive["controls"]["max_actuator_force_model_units"] == 0
    assert actual["controls"]["max_actuator_force_model_units"] > 0
    assert actual["summary"]["finite_state"] and actual["summary"]["physics_warnings"] == 0
    assert abs(actual["duration_ms"] - min(500, t.duration_ms)) < 1e-6
    first_active = next(f["time_ms"] for f in actual["trajectory"] if any(f["drive"]))
    q_actual = np.array([f["joint_angles_rad"] for f in actual["trajectory"]])
    q_passive = np.array([f["joint_angles_rad"] for f in passive["trajectory"]])
    pos_actual = np.array([f["position_mm"] for f in actual["trajectory"]])
    pos_passive = np.array([f["position_mm"] for f in passive["trajectory"]])
    assert np.max(np.abs(q_actual - q_passive)) > 0
    summary = {"source_trial": str(source.relative_to(ROOT)),
               "source_trial_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "annotation_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
               "motor_spikes": drive["motor_spikes"], "active_motor_neurons": drive["active_motor_neurons"],
               "groups": [{k: g[k] for k in ("leg", "neuron_count", "spikes")} for g in drive["groups"]],
               "engine": actual["engine"], "controls": actual["controls"],
               "neural_physics": actual["summary"], "passive_physics": passive["summary"],
               "first_active_frame_ms": first_active,
               "max_neural_minus_passive_joint_angle_rad": float(np.max(np.abs(q_actual - q_passive))),
               "final_neural_minus_passive_position_mm": (pos_actual[-1] - pos_passive[-1]).tolist(),
               "zero_control": zero,
               "trajectory_frames": len(actual["trajectory"]),
               "elapsed_seconds": {"neural": actual["elapsed_seconds"], "passive": passive["elapsed_seconds"]},
               "artifacts": actual.get("render_artifacts", {}),
               "render_error": actual.get("render_error"),
               "checks": "passed: finite physical state; no numerical warnings; no force on zero-drive actuators; passive forces zero; actual neural forces positive; neural/passive state difference; zero trial no animation; duration preserved"}
    report = ROOT / "reports/body-simulation/evidence.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
