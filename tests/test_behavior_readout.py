"""Recorded-spike provenance, causal timing, and original behavioral formula."""
from copy import deepcopy
import hashlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chatflypt.behavior_readout import (
    SOURCE_PATH, COMMAND_KEYS, GROUP_KEYS, build_behavior_drive, _validate_behavior_drive,
)


def trial(indices=(), steps=(), *, duration=100., n=4):
    indices = np.asarray(indices, dtype=np.int32)
    return SimpleNamespace(counts=np.bincount(indices, minlength=n), spike_ids=indices,
                           spike_steps=np.asarray(steps, dtype=np.int32), seed=7341,
                           duration_ms=duration)


def rows():
    return [{"type": "DNg100", "somaSide": "L"}, {"type": "DNg100", "somaSide": "R"},
            {"type": "MDN", "somaSide": "L"}, {"type": "DNa02", "somaSide": "R"}]


def build(t=None, **kwargs):
    return build_behavior_drive(t or trial(), np.arange(4, dtype=np.int64), rows(),
                                dataset="MaleCNS v1.0", **kwargs)


def group(drive, name):
    return next(g for g in drive["groups"] if g["id"] == name)


def test_prepared_anatomy_matches_complete_original_drive_and_rejects_other_ids():
    from chatflypt.behavior_readout import PreparedMotorAnnotations
    ids=np.arange(4,dtype=np.int64);annotation=rows()
    prepared=PreparedMotorAnnotations(ids,annotation,'MaleCNS v1.0')
    for driven in ([],[0],[1,2]):
        recorded=trial([0,1,2,3,0],[1,190,210,410,900])
        expected=build_behavior_drive(recorded,ids,annotation,dataset='MaleCNS v1.0',driven_indices=driven)
        actual=build_behavior_drive(recorded,ids,prepared,dataset='MaleCNS v1.0',driven_indices=driven)
        assert actual==expected
    annotation[0]['type']='unmapped'
    assert prepared.rows[0]['type']=='DNg100'
    with pytest.raises(ValueError,match='connectome'):
        build_behavior_drive(recorded,ids[::-1],prepared,dataset='MaleCNS v1.0')


def test_every_command_matches_original_formula_for_all_groups_and_contexts():
    spec = importlib.util.spec_from_file_location("original_behavior_reference", SOURCE_PATH)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    annotation = [{"type": m["types"][0] + ("_a" if m.get("family") else ""), "somaSide": side}
                  for m in original.MAPPINGS for side in ("L", "R")]
    indices = [i for i in range(26) for _ in range(i % 4 + 1)]
    steps = [25 * j + 1 for i in range(26) for j in range(i % 4 + 1)]
    t = trial(indices, steps, n=26)
    for context in ("ground", "tethered_flight"):
        drive = build_behavior_drive(t, np.arange(26), annotation, dataset="MaleCNS v1.0", context=context)
        assert [g["id"] for g in drive["groups"]] == list(GROUP_KEYS)
        for readout, command in zip(drive["readouts_hz"], drive["commands"]):
            assert command == original.motor_commands(readout, context, 100.)
        assert drive["motor_spikes"] == len(indices)
        if context == "tethered_flight":
            assert all(c["left"] == c["right"] == c["escape"] == 0 for c in drive["commands"])
            assert drive["commands"][0]["flight"] > 0
        else:
            assert all(c["flight"] == c["landing"] == 0 for c in drive["commands"])


def test_bin_end_availability_and_boundary_spikes_are_causal():
    drive = build(trial([0, 1], [199, 200]))
    assert drive["available_at_ms"] == [20., 40., 60., 80., 100.]
    assert drive["spike_counts"][0]["forward"] == {"L": 1, "R": 0}
    assert drive["spike_counts"][1]["forward"] == {"L": 0, "R": 1}
    assert drive["readouts_hz"][0]["forward"]["R"] == 0
    early = build(trial([0], [199]))
    assert drive["commands"][0] == early["commands"][0]
    assert drive["commands"][1] != early["commands"][1]
    assert not any(t < 20 for t in drive["available_at_ms"])


def test_filter_is_applied_to_hz_before_clipping_and_partial_bin_uses_actual_width():
    t = trial([0] * 4 + [1], [0, 30, 60, 90, 400], duration=45.)
    drive = build(t)
    alpha = 1 - np.exp(-20 / 40)
    assert drive["population_rate_hz"][0]["forward"]["L"] == 200.
    assert drive["readouts_hz"][0]["forward"]["L"] == pytest.approx(200 * alpha)
    assert drive["readouts_hz"][1]["forward"]["L"] == pytest.approx(200 * alpha * (1-alpha))
    assert drive["available_at_ms"] == [20., 40., 45.]
    assert drive["population_rate_hz"][-1]["forward"]["R"] == 200.
    assert drive["readouts_hz"][-1]["forward"]["R"] == pytest.approx(200 * (1-np.exp(-5/40)))


def test_directly_driven_cells_are_excluded_from_numerator_and_population():
    annotation = [{"type": "DNg100", "somaSide": "L"}] * 3
    drive = build_behavior_drive(trial([0, 0, 1], [0, 50, 100], n=3), [1, 2, 3], annotation,
                                dataset="MaleCNS v1.0", driven_indices=[0, 0])
    left = group(drive, "forward")["sides"]["L"]
    assert left["neuron_ids"] == ["2", "3"] and left["neuron_count"] == 2
    assert left["excluded_neuron_ids"] == ["1"] and left["excluded_spikes"] == 2
    assert drive["directly_driven_cells_excluded"] == 1
    assert drive["motor_spikes"] == 1 and drive["active_motor_neurons"] == 1
    assert drive["population_rate_hz"][0]["forward"]["L"] == 25.


def test_zero_unmapped_and_final_bin_only_do_not_claim_usable_commands():
    zero = build()
    assert zero["motor_spikes"] == zero["active_motor_neurons"] == 0
    assert not zero["has_command_in_window"]
    assert all(set(c) == set(COMMAND_KEYS) and not any(c.values()) for c in zero["commands"])
    final = build(trial([0], [999]))
    assert final["commands"][-1]["left"] > 0
    assert not final["has_command_in_window"]  # Last command only available after this trial.
    excluded = build(trial([0], [0]), driven_indices=[0])
    assert not excluded["has_command_in_window"] and excluded["motor_spikes"] == 0
    assert group(excluded, "forward")["sides"]["L"]["status"] == "directly_driven_only"


def test_left_right_steering_and_backward_come_from_recorded_cell_types():
    backward = build(trial([2], [0]), smoothing_ms=0)
    assert backward["commands"][0]["left"] < 0 and backward["commands"][0]["right"] < 0
    steering = build(trial([0, 3], [0, 0]), smoothing_ms=0)
    assert steering["commands"][0]["left"] > steering["commands"][0]["right"] > 0
    annotation = [{"type": "DNg13", "somaSide": "L"}]
    outside = build_behavior_drive(trial([0], [0], n=1), [10], annotation,
                                  dataset="MaleCNS v1.0", smoothing_ms=0)
    assert outside["commands"][0]["right"] > 0 and outside["commands"][0]["left"] == 0


def test_flywire_long_ids_exact_types_and_missing_sides_are_explicit():
    ids = ["720575940600000001", "720575940600000002", "720575940600000003",
           "720575940600000004", "720575940600000005", "720575940600000006"]
    annotation = [dict(root_id=x, cell_type=t, side=s) for x, t, s in zip(ids,
                  ["DNa02", "DNa02", "DNg12_a", "BDN2", "DNa020", "DNg12ab"],
                  ["left", "center", "right", "left", "right", "right"])]
    drive = build_behavior_drive(trial(range(6), [0]*6, n=6), ids, annotation, dataset="FlyWire v783")
    assert drive["motor_spikes"] == 2
    assert group(drive, "turn_inside")["sides"]["L"]["neuron_ids"] == [ids[0]]
    assert group(drive, "turn_inside")["unknown_side_ids"] == [ids[1]]
    assert group(drive, "turn_inside")["sides"]["R"]["status"] == "empty"
    assert group(drive, "anterior")["sides"]["R"]["neuron_ids"] == [ids[2]]
    assert group(drive, "forward")["sides"]["L"]["missing_types"] == ["DNg100", "DNg97"]
    repeated = build_behavior_drive(trial(range(6), [0]*6, n=6), np.asarray(ids, dtype=object),
                                    annotation, dataset="FlyWire v783")
    assert repeated["source_spikes_sha256"] == drive["source_spikes_sha256"]


def test_input_labels_and_extra_trial_fields_cannot_change_commands():
    original = trial([0, 3], [100, 300])
    injected = dict(vars(original), message="任意意图", targets=["other"], decoded={"direction": "right"})
    a = build(original)
    b = build_behavior_drive(injected, np.arange(4),
                            [dict(row, stimulus="unrelated", requested_direction="left") for row in rows()],
                            dataset="MaleCNS v1.0")
    assert a == b


def test_source_digest_remains_compatible_with_existing_numeric_records():
    t = trial([0, 3], [100, 200])
    digest = hashlib.sha256()
    for value in (np.arange(4, dtype=np.int64), t.spike_ids.astype(np.int64), t.spike_steps):
        digest.update(np.ascontiguousarray(value).tobytes())
    assert build(t)["source_spikes_sha256"] == digest.hexdigest()


def test_annotation_alignment_is_checked_for_both_named_index_and_id_column():
    frame = pd.DataFrame(rows(), index=pd.Index([1, 0, 2, 3], name="bodyId"))
    with pytest.raises(ValueError, match="Annotation index"):
        build_behavior_drive(trial(), np.arange(4), frame, dataset="MaleCNS v1.0")
    annotation = [dict(row, bodyId=i) for i, row in enumerate(rows())]
    annotation[0]["bodyId"] = 9
    with pytest.raises(ValueError, match="Annotation IDs"):
        build_behavior_drive(trial(), np.arange(4), annotation, dataset="MaleCNS v1.0")


@pytest.mark.parametrize("field,value", [
    ("duration_ms", float("nan")), ("duration_ms", 20.05), ("seed", 1.5),
    ("spike_ids", np.array([.5])), ("spike_steps", np.array([1.5])),
    ("spike_steps", np.array([1000])), ("counts", np.array([5, 0, 0, 0])),
])
def test_invalid_trial_cannot_invent_activity(field, value):
    t = trial([0], [0])
    setattr(t, field, value)
    with pytest.raises(ValueError):
        build(t)


@pytest.mark.parametrize("kwargs", [
    {"bin_ms": float("nan")}, {"bin_ms": .15}, {"reference_rate_hz": 0},
    {"smoothing_ms": -1}, {"context": "arbitrary"}, {"driven_indices": [-1]},
    {"driven_indices": [1.5]},
])
def test_invalid_controller_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        build(**kwargs)


def test_floating_point_ids_are_rejected_before_precision_loss_is_hidden():
    with pytest.raises(ValueError, match="floats lose precision"):
        build_behavior_drive(trial(n=1), [float(720575940600000001)],
                             [{"cell_type": "DNa02", "side": "left"}], dataset="FlyWire v783")


@pytest.mark.parametrize("mutate", [
    lambda d: d["available_at_ms"].__setitem__(0, 0.),
    lambda d: d["commands"][0].__setitem__("left", float("nan")),
    lambda d: d["commands"][0].__setitem__("left", .2),
    lambda d: d["readouts_hz"][0]["forward"].__setitem__("L", -1.),
    lambda d: d["readouts_hz"][0]["forward"].__setitem__("L", 100.),
    lambda d: d.__setitem__("motor_spikes", 0),
    lambda d: d.__setitem__("has_command_in_window", False),
    lambda d: d.__setitem__("directly_driven_neuron_ids", ["0"]),
])
def test_serialized_drive_validator_rejects_forged_times_rates_and_commands(mutate):
    drive = deepcopy(build(trial([0], [0])))
    mutate(drive)
    with pytest.raises(ValueError):
        _validate_behavior_drive(drive)


def test_validator_accepts_signed_original_backward_and_json_roundtrip():
    import json
    drive = json.loads(json.dumps(build(trial([2], [0])), allow_nan=False))
    _validate_behavior_drive(drive)
    assert drive["commands"][0]["left"] < 0
