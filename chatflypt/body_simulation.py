"""Measured neural spikes -> explicitly uncalibrated controller -> MuJoCo body.

The main process only needs numpy. FlyGym/MuJoCo run in the isolated .venv-body.
This is a physical engineering adapter, not a decoded/validated fly behaviour.
Neither messages, stimulus names, readout labels nor target directions are inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
SCHEMA = "chatflypt.motor-drive.v1"
LIMITATIONS = [
    "工程控制器，未经过果蝇行为或肌肉生理校准，不能把轨迹称为真实果蝇的预测动作。",
    "MaleCNS 只使用 fl/ml/hl 腿运动神经元；somaSide 到同侧腿的映射是工程假设，未逐一核验肌肉末梢。",
    "FlyWire 只使用 DNa01/DNa02 的记录活动；复制到同侧三腿是工程连接，缺少该数据集的 VNC。",
    "步态相位、12 Hz 振荡器和关节模板来自官方 FlyGym 控制器；神经活动只调制各腿幅值与执行器增益。",
    "神经试次先运行，身体再回放；地面接触尚未反馈进连接组，不是完整闭环。",
    "官方 NeuroMechFly 身体源于雌蝇；MaleCNS 是雄蝇连接组，二者未做个体或性别配准。",
    "位移同时受重力、被动关节和接触影响；位移本身不证明神经系统选择了某种行为。",
]


def _field(trial: Any, name: str) -> Any:
    return trial[name] if isinstance(trial, dict) else getattr(trial, name)


def build_motor_drive(trial: Any, neuron_ids, annotations, *, dataset: str,
                      driven_indices=(), bin_ms: float = 10.0,
                      reference_rate_hz: float = 20.0,
                      smoothing_ms: float = 30.0) -> dict:
    """Build six-leg input exclusively from recorded spikes and annotation rows.

    annotations must already be aligned to neuron_ids (brain.rows for MaleCNS;
    NeuralView.rows for FlyWire). Directly stimulated cells are excluded. A bin
    becomes available at its END: the controller never sees future spikes.
    Rate = spike count / all annotated neurons in that group / bin seconds.
    A causal exponential filter and fixed 20 Hz reference are engineering choices.
    """
    ids = np.asarray(neuron_ids)
    counts = np.asarray(_field(trial, "counts"))
    raw_indices = np.asarray(_field(trial, "spike_ids"))
    raw_steps = np.asarray(_field(trial, "spike_steps"))
    duration = float(_field(trial, "duration_ms"))
    if (ids.ndim != 1 or counts.shape != ids.shape or len(np.unique(ids)) != len(ids)
            or not np.isfinite(counts).all() or (counts < 0).any()
            or not np.equal(counts, np.floor(counts)).all()):
        raise ValueError("Neuron IDs/counts must be aligned, unique, nonnegative integers")
    if not 0 < duration <= 5000 or not 0.1 <= bin_ms <= 100:
        raise ValueError("duration_ms must be in (0,5000], bin_ms in [0.1,100]")
    if (not math.isfinite(reference_rate_hz) or reference_rate_hz <= 0
            or not math.isfinite(smoothing_ms) or smoothing_ms < 0):
        raise ValueError("Invalid rate reference/filter duration")
    if (raw_indices.ndim != 1 or raw_steps.shape != raw_indices.shape
            or not np.isfinite(raw_indices).all() or not np.isfinite(raw_steps).all()
            or not np.equal(raw_indices, np.floor(raw_indices)).all()
            or not np.equal(raw_steps, np.floor(raw_steps)).all()):
        raise ValueError("Recorded spike indices/steps must be aligned integer arrays")
    spike_indices = raw_indices.astype(np.int64)
    spike_ms = raw_steps.astype(np.float64) * 0.1
    if ((spike_indices < 0).any() or (spike_indices >= len(ids)).any()
            or (spike_ms < 0).any() or (spike_ms >= duration).any()):
        raise ValueError("Recorded spikes outside the neural trial")
    recorded_counts = np.bincount(spike_indices, minlength=len(ids))
    if not np.array_equal(recorded_counts, counts):
        raise ValueError("Counts do not match recorded spikes; cannot invent spike timing")
    is_male = "malecns" in str(dataset).lower()
    is_flywire = "flywire" in str(dataset).lower()
    if not (is_male or is_flywire):
        raise ValueError("Body adapter supports only annotated MaleCNS or FlyWire trials")
    if hasattr(annotations, "to_dict"):
        # Reject common accidental misalignment rather than silently swapping IDs.
        index_name = getattr(getattr(annotations, "index", None), "name", None)
        if index_name in ("bodyId", "root_id"):
            if not np.array_equal(np.asarray(annotations.index).astype(str), ids.astype(str)):
                raise ValueError("Annotation index must align with neuron_ids")
        needed = ("superclass", "subclass", "somaSide") if is_male else ("cell_type", "side")
        # The full table includes large arrays of anatomical coordinates. Never
        # copy them into hundreds of thousands of per-neuron Python dicts.
        rows = annotations[[c for c in needed if c in annotations.columns]].to_dict("records")
    else:
        rows = list(annotations)
    if len(rows) != len(ids):
        raise ValueError("Annotation rows must align with neuron_ids")
    excluded = {int(i) for i in driven_indices}
    if any(i < 0 or i >= len(ids) for i in excluded):
        raise ValueError("Directly driven cell indices out of range")
    groups = [[] for _ in LEGS]
    for i, row in enumerate(rows):
        if i in excluded:
            continue
        if is_male:
            if row.get("superclass") != "vnc_motor":
                continue
            leg = {"fl": "f", "ml": "m", "hl": "h"}.get(row.get("subclass"))
            side = {"L": "l", "R": "r"}.get(row.get("somaSide"))
            selected_legs = [side + leg] if side and leg else []
        else:
            if row.get("cell_type") not in ("DNa01", "DNa02"):
                continue
            side = {"left": "l", "right": "r"}.get(row.get("side"))
            selected_legs = [side + leg for leg in "fmh"] if side else []
        for leg in selected_legs:
            groups[LEGS.index(leg)].append(i)
    n_bins = int(math.ceil(duration / bin_ms))
    bins = np.minimum((spike_ms / bin_ms).astype(np.int64), n_bins - 1)
    binned = np.zeros((n_bins, 6), dtype=np.int64)
    group_sizes = np.asarray([len(g) for g in groups])
    for leg_i, group in enumerate(groups):
        selected = np.isin(spike_indices, group)
        binned[:, leg_i] = np.bincount(bins[selected], minlength=n_bins)
    # Partial last bin uses its actual width; no synthetic extension of the trial.
    end_ms = np.minimum(np.arange(1, n_bins + 1) * bin_ms, duration)
    widths = np.diff(np.r_[0.0, end_ms])
    rates = binned / np.maximum(group_sizes, 1)[None, :] / (widths[:, None] / 1000)
    amplitudes = np.zeros_like(rates)
    state = np.zeros(6)
    for i, rate in enumerate(rates):
        decay = math.exp(-widths[i] / smoothing_ms) if smoothing_ms else 0.0
        state = decay * state + (1 - decay) * np.clip(rate / reference_rate_hz, 0, 1)
        amplitudes[i] = state
    selected_unique = sorted({i for group in groups for i in group})
    spike_digest = hashlib.sha256()
    for value in (ids, spike_indices, raw_steps):
        spike_digest.update(np.ascontiguousarray(value).tobytes())
    return {
        "schema": SCHEMA, "dataset": str(dataset), "legs": list(LEGS),
        "duration_ms": duration, "seed": int(_field(trial, "seed")),
        "bin_ms": float(bin_ms), "available_at_ms": end_ms.tolist(),
        "spike_counts": binned.tolist(), "population_rate_hz": rates.tolist(),
        "amplitudes": amplitudes.tolist(), "reference_rate_hz": reference_rate_hz,
        "smoothing_ms": smoothing_ms, "source_spikes_sha256": spike_digest.hexdigest(),
        "motor_spikes": int(counts[selected_unique].sum()),
        "active_motor_neurons": int(np.count_nonzero(counts[selected_unique])),
        "directly_driven_cells_excluded": len(excluded),
        "groups": [{"leg": leg, "neuron_count": len(group),
                    "neuron_ids": [str(ids[i]) for i in group],
                    "spikes": int(counts[group].sum())}
                   for leg, group in zip(LEGS, groups)],
        "source_mapping": ("MaleCNS vnc_motor subclass fl/ml/hl + somaSide"
                           if is_male else "FlyWire DNa01/DNa02 + side; each side shared by 3 legs"),
        "causality": "Completed-bin rates only; up to bin_ms delay, no time stretching",
        "limitations": list(LIMITATIONS),
    }


def _base_result(drive: dict) -> dict:
    return {"schema": "chatflypt.body.v1", "dataset": drive.get("dataset"),
            "physics_run": False, "trajectory": [], "legs": list(LEGS),
            "source_duration_ms": drive.get("duration_ms"),
            "motor_spikes": drive.get("motor_spikes", 0),
            "active_motor_neurons": drive.get("active_motor_neurons", 0),
            "source_spikes_sha256": drive.get("source_spikes_sha256"),
            "source_mapping": drive.get("source_mapping"),
            "units": {"position": "mm", "time": "ms", "angles": "rad",
                      "actuator_force": "Native MuJoCo per-actuator units; force and torque are mixed, no SI conversion"},
            "controller": "Official FlyGym tripod CPG + recorded joint template; per-leg neural amplitude/gain",
            "biologically_calibrated": False, "limitations": list(LIMITATIONS)}


def run_body(drive: dict, *, max_duration_ms: float = 500.0,
             sample_ms: float = 10.0, timeout: float = 90.0,
             output_dir: str | Path | None = None, render: bool = False,
             passive_control: bool = False) -> dict:
    """Run a bounded worker, returning status ok/no_motor_activity/unavailable/error/timeout.

    No-motor trials return no animation. passive_control=True is an explicitly
    labelled validation-only gravity/contact control with every actuator disabled.
    A truncated run starts at neural time zero; a 500 ms trial stays 500 ms.
    Each frame has actual time_ms, position_mm, yaw_rad, feet_mm, joint_angles_rad,
    body_positions_mm and body_quaternions_wxyz. No synthetic interpolation here.
    """
    result = _base_result(drive)
    try:
        _validate_drive(drive)
        if (not math.isfinite(max_duration_ms) or not 0 < max_duration_ms <= 5000
                or not math.isfinite(sample_ms) or not 1 <= sample_ms <= 100
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Invalid body duration, sample interval or timeout")
        duration = min(float(drive["duration_ms"]), max_duration_ms)
        result.update(duration_ms=duration, truncated=duration < float(drive["duration_ms"]))
        available = np.asarray(drive["available_at_ms"]) < duration
        in_window = bool(np.any(np.asarray(drive["amplitudes"])[available] > 0))
        result["motor_activity_in_window"] = in_window
        if not in_window and not passive_control:
            return dict(result, status="no_motor_activity", reason="当前物理时间窗内没有可用于此腿控制器的运动神经驱动，未生成主动动作。")
        python = ROOT / ".venv-body" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            return dict(result, status="unavailable", reason="未安装隔离的 FlyGym 物理环境 .venv-body。")
        request = {"drive": drive, "duration_ms": duration, "sample_ms": sample_ms,
                   "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
                   "render": bool(render), "passive_control": bool(passive_control)}
        done = subprocess.run([str(python), str(Path(__file__).resolve()), "--worker"],
                              input=json.dumps(request), capture_output=True, text=True,
                              encoding="utf-8", timeout=timeout, cwd=ROOT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if done.returncode:
            return dict(result, status="error", reason="MuJoCo worker failed", detail=done.stderr[-1500:])
        output = json.loads(done.stdout)
        return {**result, **output}
    except subprocess.TimeoutExpired:
        return dict(result, status="timeout", reason=f"物理仿真超过 {timeout:g} 秒，未提供未完成轨迹。")
    except Exception as exc:
        return dict(result, status="error", reason=str(exc), error_type=type(exc).__name__)


def _validate_drive(drive):
    if drive.get("schema") != SCHEMA or drive.get("legs") != list(LEGS):
        raise ValueError("Invalid motor-drive schema/order")
    times = np.asarray(drive["available_at_ms"], dtype=float)
    values = np.asarray(drive["amplitudes"], dtype=float)
    duration = float(drive["duration_ms"])
    if (not 0 < duration <= 5000 or times.ndim != 1 or not len(times)
            or values.shape != (len(times), 6) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
            or times[0] <= 0 or times[-1] > duration):
        raise ValueError("Invalid neural drive time course")
    if int(drive["motor_spikes"]) < 0 or (not drive["motor_spikes"] and np.any(values)):
        raise ValueError("Nonzero actuation without motor spikes is forbidden")


def _simulate_physics(request: dict) -> dict:
    """Worker only: use compiled MuJoCo states, never manufactured trajectories."""
    import importlib.metadata
    import mujoco
    from flygym.compose import FlatGroundWorld, ActuatorType
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain.common import make_locomotion_fly, apply_locomotion_action
    from flygym_demo.complex_terrain.cpg_controller import CPGController, make_tripod_cpg_network
    from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

    started = time.perf_counter()
    drive = request["drive"]
    _validate_drive(drive)
    passive = request["passive_control"]
    fly = make_locomotion_fly()
    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position=[0, 0, 1.0],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    from chatflypt.body_measurement import install_body_pose_accessors
    install_body_pose_accessors(sim)
    sim.reset()
    model, data = sim.mj_model, sim.mj_data
    timestep = float(model.opt.timestep)
    cpg = make_tripod_cpg_network(timestep, seed=int(drive["seed"]) % (2 ** 32))
    # Fixed tripod phase, zero initial magnitude: no hidden warm-up locomotion.
    cpg.reset(init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]),
              init_magnitudes=np.zeros(6))
    controller = CPGController(cpg, PreprogrammedSteps(), fly.get_actuated_jointdofs_order(ActuatorType.POSITION))
    actuator_ids, actuator_legs = [], []
    for dof, actuator in fly.jointdof_to_mjcfactuator_by_type[ActuatorType.POSITION].items():
        actuator_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name))
        actuator_legs.append(LEGS.index(dof.child.pos))
    for leg, actuator in fly.leg_to_adhesionactuator.items():
        actuator_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name))
        actuator_legs.append(LEGS.index(leg))
    actuator_ids = np.array(actuator_ids)
    actuator_legs = np.array(actuator_legs)
    if len(actuator_ids) != model.nu or (actuator_ids < 0).any():
        raise RuntimeError("Every model actuator must be assigned to a measured leg channel")
    gains = model.actuator_gainprm.copy()
    biases = model.actuator_biasprm.copy()
    model.actuator_gainprm[:] = 0
    model.actuator_biasprm[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    body_names = [b.name for b in fly.get_bodysegs_order()]
    thorax_i = body_names.index("c_thorax")
    foot_ids = [body_names.index(f"{leg}_tarsus5") for leg in LEGS]
    joint_names = [j.name for j in fly.get_jointdofs_order()]
    times = np.asarray(drive["available_at_ms"])
    amplitudes = np.asarray(drive["amplitudes"])
    duration = float(request["duration_ms"])
    steps = int(math.floor((duration / 1000) / timestep + 1e-8))
    sample_every = max(1, int(round(request["sample_ms"] / (1000 * timestep))))
    trajectory = []
    current = np.zeros(6)
    max_force = 0.0
    max_zero_drive_force = 0.0
    powered_steps = 0

    def frame():
        # mj_step integrates qpos after computing xpos. Refresh derived kinematics
        # so every exported coordinate is at the reported post-step time.
        mujoco.mj_forward(model, data)
        positions = sim.get_body_positions(fly.name)
        rotations = sim.get_body_rotations(fly.name)
        quat = rotations[thorax_i]
        w, x, y, z = quat
        return {"time_ms": round(float(data.time * 1000), 7),
                "position_mm": positions[thorax_i].tolist(),
                "quaternion_wxyz": quat.tolist(),
                "yaw_rad": float(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))),
                "feet_mm": positions[foot_ids].tolist(),
                "joint_angles_rad": sim.get_joint_angles(fly.name).tolist(),
                "body_positions_mm": positions.tolist(),
                "body_quaternions_wxyz": rotations.tolist(),
                "contact_count": int(data.ncon),
                "drive": current.tolist(),
                "max_actuator_force_model_units": float(np.max(np.abs(data.actuator_force), initial=0))}

    trajectory.append(frame())
    render_frames = []
    renderer = None
    render_error = None
    if request.get("render") and request.get("output_dir"):
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as exc:
            render_error = f"{type(exc).__name__}: {exc}"
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -25, 9

    def render_frame():
        nonlocal renderer, render_error
        if renderer is not None:
            try:
                camera.lookat[:] = trajectory[-1]["position_mm"]
                renderer.update_scene(data, camera=camera)
                render_frames.append(renderer.render().copy())
            except Exception as exc:
                render_error = f"{type(exc).__name__}: {exc}"
                renderer.close()
                renderer = None

    render_frame()
    for step in range(steps):
        # The row stamped 60 ms may first affect integration starting at 60 ms.
        row = int(np.searchsorted(times, step * timestep * 1000 + 1e-8, side="right")) - 1
        current = amplitudes[row] if row >= 0 and not passive else np.zeros(6)
        cpg.intrinsic_amps[:] = current
        action = controller.step()
        scale = current[actuator_legs, None]
        model.actuator_gainprm[actuator_ids] = gains[actuator_ids] * scale
        model.actuator_biasprm[actuator_ids] = biases[actuator_ids] * scale
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        force = np.abs(data.actuator_force)
        max_force = max(max_force, float(np.max(force, initial=0)))
        inactive_ids = actuator_ids[current[actuator_legs] == 0]
        max_zero_drive_force = max(max_zero_drive_force, float(np.max(force[inactive_ids], initial=0)))
        powered_steps += int(np.any(current > 0))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("MuJoCo produced non-finite physical state")
        if (step + 1) % sample_every == 0 or step + 1 == steps:
            trajectory.append(frame())
            render_frame()
    if renderer is not None:
        renderer.close()
    warning_count = int(sum(int(w.number) for w in data.warning))
    if warning_count:
        raise RuntimeError(f"MuJoCo reported {warning_count} numerical/physics warnings")
    position = np.array([f["position_mm"] for f in trajectory])
    result = _base_result(drive)
    result.update(status="passive_control" if passive else "ok", physics_run=True,
                  duration_ms=float(data.time * 1000), requested_duration_ms=duration,
                  truncated=duration < drive["duration_ms"], trajectory=trajectory,
                  body_names=body_names, joint_names=joint_names,
                  engine={"flygym": importlib.metadata.version("flygym"),
                          "mujoco": mujoco.__version__, "nq": model.nq, "nv": model.nv,
                          "nu": model.nu, "timestep_ms": timestep * 1000,
                          "model": "NeuroMechFly simplified_max2000faces / FlatGroundWorld"},
                  controls={"reference_rate_hz": drive["reference_rate_hz"],
                            "smoothing_ms": drive["smoothing_ms"], "bin_ms": drive["bin_ms"],
                            "frequency_hz": 12, "passive_control": passive,
                            "powered_steps": powered_steps,
                            "max_drive": float(np.max([f["drive"] for f in trajectory])),
                            "max_actuator_force_model_units": max_force,
                            "max_force_on_zero_drive_actuators": max_zero_drive_force},
                  summary={"displacement_mm": (position[-1] - position[0]).tolist(),
                           "path_length_mm": float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
                           "max_contacts": max(f["contact_count"] for f in trajectory),
                           "finite_state": True, "physics_warnings": warning_count},
                  elapsed_seconds=time.perf_counter() - started)
    if render_error:
        result["render_error"] = render_error
    if request.get("output_dir"):
        output = Path(request["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if render_frames:
            import imageio.v3 as iio
            iio.imwrite(output / "body-final.png", render_frames[-1])
            # Playback is labelled, export physical timestamps in JSON alongside.
            iio.imwrite(output / "body.gif", np.stack(render_frames),
                        duration=float(request["sample_ms"]), loop=0)
            result["render_artifacts"] = {"image": str(output / "body-final.png"),
                                           "animation": str(output / "body.gif")}
        (output / "body-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module runs through run_body(); --worker is an internal JSON protocol")
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        payload = json.load(sys.stdin)
        response = _simulate_physics(payload)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"status": "error", "physics_run": False, "trajectory": [],
                          "reason": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False))
