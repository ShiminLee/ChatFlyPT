"""Recorded spikes -> Bad Apple Fly's thirteen output groups and motor formula.

No message, stimulus label, decoded sensory label, or requested action is read.
The upstream formula is loaded unchanged from a hash-pinned local source file.
Directly stimulated cells are removed from both rate numerators and populations.
Each half-open spike bin becomes available only at its end, unlike the upstream
same-window body replay. This adapter does not train or change the neural model.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "chatflypt.behavior-drive.v1"
SOURCE_COMMIT = "6726892e7ebd93e6c5ffab056ab8db09433c28a6"
SOURCE_SHA256 = "17c7a2aec96a027e0bc8a1b4691ca7ec9048abba4b81fda58fe4642dc9ad891e"
SOURCE_PATH = ROOT / "vendor/badapple-fly/backend/evidence.py"
SOURCE_URL = f"https://github.com/kevinlinxc/badapple-fly/blob/{SOURCE_COMMIT}/backend/evidence.py"
COMMAND_KEYS = ("left", "right", "antenna", "rub", "anterior", "feeding",
                "song", "escape", "flight", "landing")
GROUP_KEYS = ("forward", "backward", "turn_inside", "turn_outside", "context_walk",
              "antenna", "rub", "anterior", "feeding", "song", "flight", "landing", "escape")
SIDES = ("L", "R")
LIMITATIONS = [
    "采用 Bad Apple Fly 的定性细胞群映射及原控制公式，增益和动作模板未经行为生理校准。",
    "只读真实记录的 spike；直接刺激细胞已排除，不能把刺激输出细胞当作下游反应。",
    "MaleCNS 按 type/somaSide，FlyWire 按 cell_type/side；只匹配原类型名，缺失类型或侧别不猜测补齐。",
    "DNg12/DNg02 采用原作者明确的下划线子类型家族规则；个别子类型的动作作用未验证。",
    "FlyWire 不含完整 VNC；同名细胞到同一动作程序的跨数据集应用仍是工程映射。",
    "动作适配器只读取已完成的神经时间窗；感觉反馈由外层具身会话提供，独立身体回放没有该反馈。",
    "行为命令不等于已发生的动作；是否运动须由实际身体仿真轨迹确定。",
]


@lru_cache(maxsize=1)
def _source():
    data = SOURCE_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Bad Apple Fly mapping source differs from the pinned commit")
    spec = importlib.util.spec_from_file_location("_chatflypt_badapple_evidence", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(m["id"] for m in module.MAPPINGS) != GROUP_KEYS:
        raise ValueError("Unexpected upstream behavior group order")
    return module


def _field(trial: Any, name: str):
    try:
        return trial[name] if isinstance(trial, dict) else getattr(trial, name)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Missing recorded trial field: {name}") from exc


def _number(value, name, *, minimum=None, maximum=None, strict_min=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if (not math.isfinite(result)
            or (minimum is not None and (result <= minimum if strict_min else result < minimum))
            or (maximum is not None and result > maximum)):
        raise ValueError(f"Invalid {name}")
    return result


def _integer(value, name, *, maximum=None):
    result = _number(value, name, minimum=0, maximum=maximum)
    if result != math.floor(result):
        raise ValueError(f"{name} must be an integer")
    return int(result)


def _integer_array(value, name):
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in "iuf"
            or not np.isfinite(array).all() or (array < 0).any()
            or (array >= 2**63).any() or not np.equal(array, np.floor(array)).all()):
        raise ValueError(f"{name} must be a nonnegative integer array")
    return array, array.astype(np.int64)


def _string_ids(value):
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("Neuron IDs must be a one-dimensional array")
    ids = []
    for item in array:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError("Neuron IDs must be exact integer strings or integers")
        if isinstance(item, (int, np.integer)):
            text = str(int(item))
        elif isinstance(item, (str, np.str_)):
            text = str(item)
        else:
            raise ValueError("Neuron IDs must be exact integer strings or integers; floats lose precision")
        if not text.isascii() or not text.isdigit() or str(int(text)) != text:
            raise ValueError("Neuron IDs must be canonical nonnegative integer strings")
        ids.append(text)
    if len(set(ids)) != len(ids):
        raise ValueError("Neuron IDs must be unique")
    return array, ids


def _dataset_fields(dataset):
    if not isinstance(dataset, str):
        raise ValueError("dataset must identify MaleCNS or FlyWire")
    if dataset.lower().startswith("malecns"):
        return "type", "somaSide", {"L": "L", "R": "R"}, "bodyId"
    if dataset.lower().startswith("flywire"):
        return "cell_type", "side", {"left": "L", "right": "R"}, "root_id"
    raise ValueError("Behavior adapter supports only annotated MaleCNS or FlyWire trials")


def _annotation_rows(annotations, ids, type_key, side_key, id_key):
    if hasattr(annotations, "to_dict"):
        if not {type_key, side_key}.issubset(annotations.columns):
            raise ValueError("Annotation type/side columns are missing")
        if getattr(annotations.index, "name", None) in ("bodyId", "root_id"):
            if _string_ids(annotations.index)[1] != ids:
                raise ValueError("Annotation index must align with neuron_ids")
        keys = [type_key, side_key] + ([id_key] if id_key in annotations.columns else [])
        keys += [k for k in ('superclass', 'super_class', 'somaNeuromere', 'exitNerve')
                 if k in annotations.columns and k not in keys]
        rows = annotations[keys].to_dict("records")
    else:
        rows = list(annotations)
    if len(rows) != len(ids) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Annotation rows must align with neuron_ids")
    for i, row in enumerate(rows):
        if id_key in row and _string_ids([row[id_key]])[1] != [ids[i]]:
            raise ValueError("Annotation IDs must align with neuron_ids")
    return rows


class PreparedMotorAnnotations:
    """Validated immutable session snapshot of static anatomical membership."""
    def __init__(self, neuron_ids, annotations, dataset):
        original,ids=_string_ids(neuron_ids)
        self.neuron_ids=original.copy();self.neuron_ids.setflags(write=False)
        self.ids=ids;self.dataset=dataset
        type_key,side_key,side_map,id_key=_dataset_fields(dataset)
        self.rows=[dict(row) for row in _annotation_rows(annotations,ids,type_key,side_key,id_key)]
        self.type_indices={};self.normalized_sides=[]
        for i,row in enumerate(self.rows):
            cell_type=row.get(type_key)
            if isinstance(cell_type,str) and cell_type:self.type_indices.setdefault(cell_type,[]).append(i)
            side=row.get(side_key)
            self.normalized_sides.append(side_map.get(side) if isinstance(side,str) else None)
        self.mapping_digest=hashlib.sha256(json.dumps(
            [[ids[i],row.get(type_key) if isinstance(row.get(type_key),str) else '',self.normalized_sides[i]]
             for i,row in enumerate(self.rows)],ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def _course_dict(array):
    return [{group: {side: float(row[2 * j + k]) for k, side in enumerate(SIDES)}
             for j, group in enumerate(GROUP_KEYS)} for row in array]


def _source_digest(original_ids, string_ids, spike_indices, original_steps):
    digest = hashlib.sha256()
    # Preserve the existing provenance digest for real integer-array records.
    # Object/string IDs use deterministic text, never NumPy object pointer bytes.
    if original_ids.dtype.kind in "iu":
        encoded_ids = np.ascontiguousarray(original_ids).tobytes()
        id_encoding = f"numpy:{original_ids.dtype.str}"
    else:
        encoded_ids = json.dumps(string_ids, ensure_ascii=True, separators=(",", ":")).encode()
        id_encoding = "json-canonical-string-ids:utf8"
    for data in (encoded_ids, np.ascontiguousarray(spike_indices).tobytes(),
                 np.ascontiguousarray(original_steps).tobytes()):
        digest.update(data)
    return digest.hexdigest(), {"neuron_ids": id_encoding, "spike_ids": "numpy:<i8",
                                "spike_steps": f"numpy:{original_steps.dtype.str}"}


def build_behavior_drive(trial: Any, neuron_ids, annotations, *, dataset: str,
                         driven_indices=(), context: str = "ground", bin_ms: float = 20.,
                         smoothing_ms: float = 40., reference_rate_hz: float = 100.) -> dict:
    """Use aligned exact annotations and recorded 0.1-ms spike ticks only.

    Commands for [start, end) spikes become available at end. The final bin's
    command is recorded, but cannot act inside the original trial duration.
    No left/right aliases or cross-dataset type-name guesses are introduced.
    """
    source = _source()
    prepared=annotations if isinstance(annotations,PreparedMotorAnnotations) else None
    if prepared is not None:
        original_ids=np.asarray(neuron_ids)
        if prepared.dataset!=dataset or not np.array_equal(original_ids,prepared.neuron_ids):
            raise ValueError('Prepared anatomy does not match this connectome')
        ids=prepared.ids
    else:original_ids, ids = _string_ids(neuron_ids)
    _, counts = _integer_array(_field(trial, "counts"), "counts")
    _, spike_indices = _integer_array(_field(trial, "spike_ids"), "spike_ids")
    original_steps, steps = _integer_array(_field(trial, "spike_steps"), "spike_steps")
    duration = _number(_field(trial, "duration_ms"), "duration_ms", minimum=0, maximum=60000, strict_min=True)
    bin_ms = _number(bin_ms, "bin_ms", minimum=.1, maximum=100)
    smoothing_ms = _number(smoothing_ms, "smoothing_ms", minimum=0)
    reference_rate_hz = _number(reference_rate_hz, "reference_rate_hz", minimum=0, strict_min=True)
    seed = _integer(_field(trial, "seed"), "seed", maximum=2**32 - 1)
    if context not in ("ground", "tethered_flight"):
        raise ValueError("context must be ground or tethered_flight")
    for name, value in (("duration_ms", duration), ("bin_ms", bin_ms)):
        if abs(value * 10 - round(value * 10)) > 1e-8:
            raise ValueError(f"{name} must use 0.1 ms timing")
    if counts.shape != (len(ids),) or steps.shape != spike_indices.shape:
        raise ValueError("Counts and spike arrays must align with recorded neuron IDs")
    if ((spike_indices >= len(ids)).any() or (steps >= round(duration * 10)).any()):
        raise ValueError("Recorded spikes outside the neural trial")
    if not np.array_equal(np.bincount(spike_indices, minlength=len(ids)), counts):
        raise ValueError("Counts do not match recorded spikes; cannot invent spike timing")
    type_key, side_key, side_map, id_key = _dataset_fields(dataset)
    rows = prepared.rows if prepared is not None else _annotation_rows(annotations, ids, type_key, side_key, id_key)
    _, driven = _integer_array(list(driven_indices), "driven_indices")
    if (driven >= len(ids)).any():
        raise ValueError("Directly driven cell indices out of range")
    excluded = set(driven.tolist())
    if prepared is not None:
        type_indices,normalized_sides=prepared.type_indices,prepared.normalized_sides
    else:
        type_indices = {}
        normalized_sides = []
        for i, row in enumerate(rows):
            cell_type = row.get(type_key)
            if isinstance(cell_type, str) and cell_type:
                type_indices.setdefault(cell_type, []).append(i)
            side = row.get(side_key)
            normalized_sides.append(side_map.get(side) if isinstance(side, str) else None)
    groups, selected_groups = [], []
    for mapping in source.MAPPINGS:
        matching = {}
        for wanted in mapping["types"]:
            matching[wanted] = [i for actual, members in type_indices.items()
                                if actual == wanted or (mapping.get("family", False)
                                                       and actual.startswith(wanted + "_"))
                                for i in members]
        matched = sorted({i for members in matching.values() for i in members})
        sides = {}
        for side in SIDES:
            annotated = [i for i in matched if normalized_sides[i] == side]
            selected = [i for i in annotated if i not in excluded]
            removed = [i for i in annotated if i in excluded]
            selected_groups.append(selected)
            sides[side] = {
                "neuron_ids": [ids[i] for i in selected], "neuron_count": len(selected),
                "annotated_neuron_count": len(annotated),
                "spikes": int(counts[selected].sum()),
                "active_neurons": int(np.count_nonzero(counts[selected])),
                "excluded_neuron_ids": [ids[i] for i in removed],
                "excluded_spikes": int(counts[removed].sum()),
                "missing_types": [t for t, members in matching.items()
                                  if not any(normalized_sides[i] == side for i in members)],
                "status": "available" if selected else "directly_driven_only" if removed else "empty",
            }
        groups.append({"id": mapping["id"], "label": mapping["label"],
                       "types": list(mapping["types"]), "family": mapping.get("family", False),
                       "sides": sides, "unknown_side_ids": [ids[i] for i in matched
                                                               if normalized_sides[i] is None]})
    ticks_per_bin = round(bin_ms * 10)
    duration_ticks = round(duration * 10)
    end_ticks = np.minimum(np.arange(ticks_per_bin, duration_ticks + ticks_per_bin, ticks_per_bin), duration_ticks)
    ends = end_ticks.astype(float) / 10
    widths = np.diff(np.r_[0., ends])
    spike_bins = steps // ticks_per_bin
    binned = np.zeros((len(ends), 26), dtype=np.int64)
    for j, group in enumerate(selected_groups):
        selected = np.isin(spike_indices, group)
        binned[:, j] = np.bincount(spike_bins[selected], minlength=len(ends))
    sizes = np.asarray([len(group) for group in selected_groups])
    rates = binned / np.maximum(sizes, 1)[None, :] / (widths[:, None] / 1000.)
    filtered = np.empty_like(rates)
    state = np.zeros(26)
    for i, rate in enumerate(rates):
        # Filter Hz before clipping/normalization, exactly as in upstream Brain.
        alpha = 1 - math.exp(-widths[i] / smoothing_ms) if smoothing_ms else 1.
        state += alpha * (rate - state)
        filtered[i] = state
    readouts = _course_dict(filtered)
    commands = [source.motor_commands(row, context, reference_rate_hz) for row in readouts]
    unique = sorted({i for group in selected_groups for i in group})
    digest, encoding = _source_digest(original_ids, ids, spike_indices, original_steps)
    mapping_digest = prepared.mapping_digest if prepared is not None else hashlib.sha256(json.dumps(
        [[ids[i], rows[i].get(type_key) if isinstance(rows[i].get(type_key), str) else "",
          normalized_sides[i]] for i in range(len(ids))],
        ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    drive = {
        "schema": SCHEMA, "dataset": dataset, "context": context, "seed": seed,
        "duration_ms": duration, "bin_ms": bin_ms, "smoothing_ms": smoothing_ms,
        "reference_rate_hz": reference_rate_hz, "available_at_ms": ends.tolist(),
        "commands": commands, "readouts_hz": readouts,
        "population_rate_hz": _course_dict(rates),
        "spike_counts": [{group: {side: int(row[2*j+k]) for k, side in enumerate(SIDES)}
                          for j, group in enumerate(GROUP_KEYS)} for row in binned],
        "motor_spikes": int(counts[unique].sum()),
        "active_motor_neurons": int(np.count_nonzero(counts[unique])),
        "source_spikes_sha256": digest, "source_spikes_encoding": encoding,
        "annotation_mapping_sha256": mapping_digest,
        "directly_driven_cells_excluded": len(excluded),
        "directly_driven_neuron_ids": [ids[i] for i in sorted(excluded)],
        "directly_driven_spikes_excluded": int(counts[sorted(excluded)].sum()),
        "groups": groups,
        "source_mapping": "Bad Apple Fly 13 original groups and unchanged motor_commands; "
                          + type_key + "+" + side_key + "; directly driven cells excluded",
        "mapping_source": {"repository": "https://github.com/kevinlinxc/badapple-fly",
                           "commit": SOURCE_COMMIT, "file": "backend/evidence.py",
                           "sha256": SOURCE_SHA256, "url": SOURCE_URL},
        "causality": "Half-open bins; commands available at completed-bin ends only; no future spikes or time stretching",
        "has_command_in_window": any(end < duration and any(value != 0 for value in command.values())
                                     for end, command in zip(ends, commands)),
        "limitations": list(LIMITATIONS),
    }
    from .joint_motor import build_joint_motor, LIMITATION
    drive['joint_motor'] = build_joint_motor(rows, ids, counts, spike_indices, steps, excluded,
        ends, ticks_per_bin, smoothing_ms, reference_rate_hz, dataset, unique)
    drive['motor_coverage'] = drive['joint_motor']['coverage']
    drive['limitations'].append(LIMITATION)
    drive['has_joint_command_in_window'] = any(t < duration and any(r.values())
        for t, r in zip(ends, drive['joint_motor']['offsets_rad']))
    _validate_behavior_drive(drive)
    return drive


def _rate_array(course, n_bins, name):
    if not isinstance(course, list) or len(course) != n_bins:
        raise ValueError(f"Invalid {name} time course")
    rows = []
    for row in course:
        if not isinstance(row, dict) or set(row) != set(GROUP_KEYS):
            raise ValueError(f"Invalid {name} groups")
        values = []
        for group in GROUP_KEYS:
            if not isinstance(row[group], dict) or set(row[group]) != set(SIDES):
                raise ValueError(f"Invalid {name} sides")
            values.extend(_number(row[group][side], name, minimum=0) for side in SIDES)
        rows.append(values)
    return np.asarray(rows)


def _validate_behavior_drive(drive):
    """Validate serialized drives before a body worker accepts signed commands.

    Rechecks completed-bin times, spike/rate/filter consistency and the original
    motor formula. This proves internal consistency, not external authenticity;
    the raw recorded trial must still be retained with its provenance digest.
    """
    if not isinstance(drive, dict) or drive.get("schema") != SCHEMA:
        raise ValueError("Invalid behavior-drive schema")
    _dataset_fields(drive.get("dataset"))
    if drive.get("context") not in ("ground", "tethered_flight"):
        raise ValueError("Invalid behavior context")
    duration = _number(drive.get("duration_ms"), "duration_ms", minimum=0, maximum=60000, strict_min=True)
    bin_ms = _number(drive.get("bin_ms"), "bin_ms", minimum=.1, maximum=100)
    smoothing = _number(drive.get("smoothing_ms"), "smoothing_ms", minimum=0)
    reference = _number(drive.get("reference_rate_hz"), "reference_rate_hz", minimum=0, strict_min=True)
    _integer(drive.get("seed"), "seed", maximum=2**32-1)
    for value in (duration, bin_ms):
        if abs(value * 10 - round(value * 10)) > 1e-8:
            raise ValueError("Drive must use 0.1 ms timing")
    ticks = round(bin_ms * 10)
    expected = np.minimum(np.arange(ticks, round(duration * 10) + ticks, ticks), round(duration * 10)) / 10.
    times = np.asarray(drive.get("available_at_ms"))
    if times.dtype.kind not in "iuf" or times.shape != expected.shape or not np.array_equal(times, expected):
        raise ValueError("Invalid completed-bin availability times")
    n_bins = len(times)
    readouts = _rate_array(drive.get("readouts_hz"), n_bins, "readouts_hz")
    rates = _rate_array(drive.get("population_rate_hz"), n_bins, "population_rate_hz")
    counts = _rate_array(drive.get("spike_counts"), n_bins, "spike_counts")
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Spike counts must be integers")
    groups = drive.get("groups")
    if not isinstance(groups, list) or [g.get("id") for g in groups] != list(GROUP_KEYS):
        raise ValueError("Invalid behavior group metadata")
    sizes, selected_ids = [], set()
    for j, group in enumerate(groups):
        for k, side in enumerate(SIDES):
            entry = group["sides"][side]
            neuron_ids = _string_ids(entry["neuron_ids"])[1]
            size = _integer(entry["neuron_count"], "neuron_count")
            if size != len(neuron_ids) or not all(isinstance(item, str) for item in entry["neuron_ids"]):
                raise ValueError("Group neuron IDs must be exact strings")
            if any(item in selected_ids for item in neuron_ids):
                raise ValueError("A neuron cannot belong to multiple original behavior groups/sides")
            selected_ids.update(neuron_ids)
            sizes.append(size)
            if _integer(entry["spikes"], "group spikes") != int(counts[:, 2*j+k].sum()):
                raise ValueError("Group spikes do not match recorded bins")
            if not size and np.any(counts[:, 2*j+k]):
                raise ValueError("An empty group cannot contain spikes")
    motor_spikes = _integer(drive.get("motor_spikes"), "motor_spikes")
    active = _integer(drive.get("active_motor_neurons"), "active_motor_neurons")
    if motor_spikes != int(counts.sum()) or active > min(motor_spikes, len(selected_ids)):
        raise ValueError("Invalid motor spike totals")
    excluded = _string_ids(drive.get("directly_driven_neuron_ids"))[1]
    if selected_ids.intersection(excluded) or len(excluded) != _integer(
            drive.get("directly_driven_cells_excluded"), "directly_driven_cells_excluded"):
        raise ValueError("Directly driven cells must be excluded from behavior groups")
    widths = np.diff(np.r_[0., times])
    expected_rates = counts / np.maximum(sizes, 1)[None, :] / (widths[:, None] / 1000.)
    if not np.allclose(rates, expected_rates, atol=1e-12, rtol=1e-12):
        raise ValueError("Rates do not match recorded spike populations")
    state = np.zeros(26)
    for i, rate in enumerate(rates):
        alpha = 1 - math.exp(-widths[i] / smoothing) if smoothing else 1.
        state += alpha * (rate - state)
        if not np.allclose(readouts[i], state, atol=1e-12, rtol=1e-12):
            raise ValueError("Readouts violate causal rate filtering")
    commands = drive.get("commands")
    if not isinstance(commands, list) or len(commands) != n_bins:
        raise ValueError("Invalid command time course")
    for i, command in enumerate(commands):
        if not isinstance(command, dict) or set(command) != set(COMMAND_KEYS):
            raise ValueError("Invalid command keys")
        for key in COMMAND_KEYS:
            _number(command[key], key, minimum=-2.4 if key in ("left", "right") else 0,
                    maximum=2.4 if key in ("left", "right") else 1.5)
        expected_command = _source().motor_commands(drive["readouts_hz"][i], drive["context"], reference)
        if any(not math.isclose(command[k], expected_command[k], rel_tol=1e-12, abs_tol=1e-12)
               for k in COMMAND_KEYS):
            raise ValueError("Commands differ from original motor formula")
    has_command = any(t < duration and any(v != 0 for v in row.values()) for t, row in zip(times, commands))
    if drive.get("has_command_in_window") is not has_command:
        raise ValueError("Invalid command availability flag")
    if not motor_spikes and any(any(v != 0 for v in row.values()) for row in commands):
        raise ValueError("Nonzero actuation without motor spikes is forbidden")
    from .joint_motor import validate_joint_motor
    validate_joint_motor(drive, selected_ids)
    if 'joint_motor' in drive:
        expected_joint_flag = any(t < duration and any(r.values())
            for t, r in zip(times, drive['joint_motor']['offsets_rad']))
        if drive.get('has_joint_command_in_window') is not expected_joint_flag:
            raise ValueError('Invalid joint command availability flag')
    for name in ("source_spikes_sha256", "annotation_mapping_sha256"):
        digest = drive.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid source digest")
