"""Neural provenance/causality checks; real physics evidence has a separate runner."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from chatflypt.body_simulation import build_motor_drive, run_body, LEGS


def trial(indices=(), steps=(), duration=100, n=8):
    indices = np.asarray(indices, dtype=int)
    return SimpleNamespace(counts=np.bincount(indices, minlength=n), spike_ids=indices,
                           spike_steps=np.asarray(steps, dtype=int), seed=2, duration_ms=duration)


def male_rows():
    return [dict(superclass="vnc_motor", subclass=leg + "l", somaSide=side)
            for side in ("L", "R") for leg in "fmh"] + [
        dict(superclass="sensory", subclass="fl", somaSide="L"),
        dict(superclass="vnc_motor", subclass="wm", somaSide="R")]


def build(t, **kwargs):
    return build_motor_drive(t, np.arange(8), male_rows(), dataset="MaleCNS v1.0", **kwargs)


def test_selected_population_and_causal_spike_timing():
    drive = build(trial([0, 5, 6, 7], [200, 450, 120, 120]))
    assert drive["legs"] == list(LEGS)
    assert drive["motor_spikes"] == 2
    assert [g["spikes"] for g in drive["groups"]] == [1, 0, 0, 0, 0, 1]
    amplitudes = np.asarray(drive["amplitudes"])
    # A spike stamped 20 ms becomes available at 30 ms, never at 20 ms.
    assert not amplitudes[:2].any()
    assert amplitudes[2, 0] > 0
    assert not amplitudes[:4, 5].any()
    assert amplitudes[4, 5] > 0
    assert not amplitudes[:, 1:5].any()


def test_sensory_and_nonleg_activity_do_not_create_body_actions():
    drive = build(trial([6, 7], [200, 210]))
    with patch("chatflypt.body_simulation.subprocess.run") as worker:
        result = run_body(drive)
    worker.assert_not_called()
    assert result["status"] == "no_motor_activity"
    assert not result["physics_run"] and result["trajectory"] == []


def test_completely_zero_trial_cannot_launch_an_autonomous_gait():
    with patch("chatflypt.body_simulation.subprocess.run") as worker:
        result = run_body(build(trial()))
    worker.assert_not_called()
    assert result["status"] == "no_motor_activity"
    assert result["motor_spikes"] == 0 and result["trajectory"] == []


def test_directly_driven_motor_cells_are_not_counted_as_downstream_activity():
    drive = build(trial([0, 5], [200, 250]), driven_indices=[0, 5])
    assert drive["motor_spikes"] == 0
    assert not np.any(drive["amplitudes"])


def test_missing_spike_record_is_rejected_instead_of_inventing_timing():
    t = trial()
    t.counts[0] = 5
    with pytest.raises(ValueError, match="invent spike timing"):
        build(t)


def test_unknown_laterality_does_not_gain_a_side():
    rows = male_rows()
    rows[0]["somaSide"] = ""
    drive = build_motor_drive(trial([0], [200]), np.arange(8), rows, dataset="MaleCNS v1.0")
    assert drive["motor_spikes"] == 0


def test_flywire_engineering_fanout_is_explicit_and_counts_unique_spikes():
    rows = [dict(cell_type="DNa01", side="left"), dict(cell_type="DNa02", side="right"),
            dict(cell_type="MN9", side="left")]
    drive = build_motor_drive(trial([0, 1, 2], [200, 300, 200], n=3),
                              np.arange(3), rows, dataset="FlyWire 783")
    assert drive["motor_spikes"] == 2
    assert [g["spikes"] for g in drive["groups"]] == [1, 1, 1, 1, 1, 1]
    assert "shared by 3 legs" in drive["source_mapping"]


def test_activity_outside_truncated_window_cannot_create_passive_animation():
    drive = build(trial([0], [700], duration=100))
    with patch("chatflypt.body_simulation.subprocess.run") as worker:
        result = run_body(drive, max_duration_ms=50)
    worker.assert_not_called()
    assert result["status"] == "no_motor_activity" and result["truncated"]
    assert result["duration_ms"] == 50 and result["source_duration_ms"] == 100


def test_nonzero_amplitude_without_motor_spikes_is_rejected():
    drive = build(trial())
    drive["amplitudes"][0][0] = 1
    assert run_body(drive)["status"] == "error"


def test_invalid_spike_and_duration_are_rejected():
    t = trial([0], [1000], duration=100)
    with pytest.raises(ValueError, match="outside"):
        build(t)
    assert run_body(build(trial()), max_duration_ms=float("nan"))["status"] == "error"
