"""Replay recorded behavior commands through the unchanged Bad Apple Fly body.

FlyGym is loaded only in the isolated physics worker. The upstream qualitative
programs and Hybrid controller are reused, not inferred from user messages.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "vendor/badapple-fly/backend/body.py"
SOURCE_SHA256 = "1702dc665e4fce9196fa3e18bb19454aae1ad743b81a6834cab6118337af27cc"
SOURCE_COMMIT = "6726892e7ebd93e6c5ffab056ab8db09433c28a6"
GEOMETRY_ASSET = "/assets/neuromechfly-behavior-scene.json"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
COMMAND_KEYS = ("left", "right", "antenna", "rub", "anterior", "feeding",
                "song", "escape", "flight", "landing")
CONTROLLER = "Bad Apple Fly Body + official FlyGym HybridTurningController + recorded tibia MN adapter; qualitative, uncalibrated"
LIMITATIONS = [
    "复用 Bad Apple Fly 原动作程序与增益；它们是定性工程映射，未经行为或肌肉生理校准。",
    "原版零命令仍有主动姿态维持及 Hybrid 局部反馈；这种力和位移不是神经指令证据。",
    "预热使用原版姿态执行器，预热结束另设神经回放时间零点，不计作神经响应。",
    "身体接触和关节状态只反馈到 Hybrid 局部控制器，没有反馈到连接组神经网络。",
    "飞行场景为显式系留：固定胸部位置姿态并挥翅；没有气动力或自由飞行。",
    "NeuroMechFly 雌蝇几何与 MaleCNS 雄蝇连接组未做个体或性别配准。",
    "此官方模型有 rostrum/haustellum 口器关节，没有独立 labellum 网格或关节；原程序中该名称的分支不会创建新结构。",
    "行为命令不等于实现的动作；实际位姿来自 MuJoCo，重力和被动关节同样影响运动。",
]


def command_execution(commands, joint_offsets=None, *, passive=False):
    """Report branch eligibility in the pinned body; never change its inputs.

    Eligibility is not a claim of movement, new neural spikes, or actuator work.
    Small escape/landing/grooming tails are ignored by the upstream gesture.
    """
    joint_offsets = joint_offsets or {}
    requested = bool(any(commands.values()) or any(joint_offsets.values()))
    grooming = max(min(1., max(commands['antenna'], commands['anterior'])),
                   min(1., commands['rub']))
    programs = {
        'locomotion': bool(commands['left'] or commands['right']),
        'grooming': grooming > .01,
        'escape': min(1., commands['escape']) > .01,
        'landing': min(1., commands['landing']) > .01,
        'feeding': min(1., commands['feeding']) != 0.,
        'song': min(1., commands['song']) != 0.,
        'flight': min(1.5, commands['flight']) != 0.,
        'joint_targets': bool(any(joint_offsets.values())),
    }
    if passive:
        programs = dict.fromkeys(programs, False)
    return {'requested_command_nonzero': requested,
            'neural_command_active': any(programs.values()),
            'effective_programs': programs}


class BodyControlStatistics:
    """Window-local statistics of inputs actually used by integration steps."""
    def __init__(self, start_ms=0.):
        self.start_ms = float(start_ms)
        self.steps = self.requested = self.effective = self.zero = 0
        self.max_force = self.max_zero_force = None
        self.max_inactive_force = None
        self.commands = {k: {'min': None, 'max': None} for k in COMMAND_KEYS}
        self.joints = {k: {'min': None, 'max': None} for k in LEGS}
        self.program_steps = {}

    def record(self, commands, joints, force, *, passive=False):
        status = command_execution(commands, joints, passive=passive)
        raw, effective = status['requested_command_nonzero'], status['neural_command_active']
        self.steps += 1
        self.requested += int(raw)
        self.effective += int(effective)
        self.zero += int(not raw)
        self.max_force = max(self.max_force or 0., force)
        if not raw:
            self.max_zero_force = max(self.max_zero_force or 0., force)
        if not effective:
            self.max_inactive_force = max(self.max_inactive_force or 0., force)
        for key, active in status['effective_programs'].items():
            self.program_steps[key] = self.program_steps.get(key, 0) + int(active)
        for source, summary in ((commands, self.commands), (joints, self.joints)):
            for key, bounds in summary.items():
                value = float(source.get(key, 0.))
                bounds['min'] = value if bounds['min'] is None else min(bounds['min'], value)
                bounds['max'] = value if bounds['max'] is None else max(bounds['max'], value)

    def report(self):
        return {
            'command_summary': self.commands,
            'joint_command_summary_rad': self.joints,
            'requested_command_in_window': self.requested > 0,
            'motor_activity_in_window': self.effective > 0,
            'controls': {
                'statistics_scope': 'Actual integration steps in this observation window; no cumulative maxima',
                'activity_definition': 'Pinned controller branch eligibility, not measured action or new neural spikes',
                'integration_steps': self.steps, 'requested_command_steps': self.requested,
                'powered_steps': self.effective, 'effective_program_steps': self.program_steps,
                'zero_command_steps': self.zero,
                'no_effective_command_steps': self.steps - self.effective,
                'max_actuator_force_model_units': self.max_force,
                'max_force_during_zero_commands': self.max_zero_force,
                'max_force_without_effective_commands': self.max_inactive_force,
            },
        }


@lru_cache(maxsize=1)
def load_body_class():
    """Load precisely the recorded upstream implementation, without editing it."""
    if hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Bad Apple Fly body source differs from the pinned commit")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
    os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(ROOT / "data/cache/flygym"))
    spec = importlib.util.spec_from_file_location("_chatflypt_badapple_body", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Body


def initialize_body(seed=1, context="ground", environment=None, *, inspection_contacts=False):
    """Preserve the original constructor and per-trial reset/warmup sequence."""
    cls = load_body_class()
    if environment is not None:
        from chatflypt.environment import add_environment_to_world
        # Isolated worker only: replace the world factory during construction.
        scope = cls.__init__.__globals__
        original_world = scope['FlatGroundWorld']
        original_fly = scope['NeuroMechFly']
        def seeing_fly(*args, **kwargs):
            fly = original_fly(*args, **kwargs)
            fly.add_vision()
            return fly
        scope['FlatGroundWorld'] = add_environment_to_world(original_world, environment,full_body_contacts=inspection_contacts)
        scope['NeuroMechFly'] = seeing_fly
        try:
            body = cls()
        finally:
            scope['FlatGroundWorld'] = original_world
            scope['NeuroMechFly'] = original_fly
    else:
        body = cls()
    discarded_ms = float(body.sim.mj_data.time * 1000)
    body.reset(seed=int(seed), context=context)
    from chatflypt.body_measurement import install_body_pose_accessors
    install_body_pose_accessors(body.sim)
    from chatflypt.joint_motor import install_joint_adapter
    install_joint_adapter(body)
    initialization = {
        "method": "Unchanged upstream Body() then Body.reset(seed, context)",
        "constructor_warmup_ms": discarded_ms,
        "constructor_state_discarded": True,
        "warmup_ms": float(body.sim.mj_data.time * 1000),
        "warmup_active_posture_control": True,
        "warmup_neural_commands": False,
        "neural_time_origin_mujoco_ms": float(body.sim.mj_data.time * 1000),
        "neural_time_origin": "After the selected reset's original warmup; body.elapsed = 0",
        "context": context,
        "seed": int(seed),
    }
    return body, initialization


def body_source() -> dict:
    from chatflypt.behavior_readout import SOURCE_SHA256 as evidence_sha
    return {
        "repository": "https://github.com/kevinlinxc/badapple-fly",
        "commit": SOURCE_COMMIT,
        "body_path": "vendor/badapple-fly/backend/body.py",
        "body_sha256": SOURCE_SHA256,
        "evidence_path": "vendor/badapple-fly/backend/evidence.py",
        "evidence_sha256": evidence_sha,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reuse": "Pinned Body construction/reset/advance, original gestures and Hybrid controller retained; external tibia target decorator added after warmup",
        "joint_adapter_sha256": hashlib.sha256((ROOT / 'chatflypt/joint_motor.py').read_bytes()).hexdigest(),
        "adaptations": [
            "Replay one original physics step at a time using only completed neural bins.",
            "Export full-precision body/geom poses and separate warmup and neural timestamps.",
            "Recorded, anatomically selected tibia MN rates add bounded antagonist joint targets; no stimulus labels or LLM action advice.",
            "No invented motor commands in zero-command windows; environmental physics and posture control may still move the body.",
            "Validation passive replay disables all actuator gains/biases and bypasses the motor controller after the same warmup.",
        ],
        "license": "No root license file found in the recorded Bad Apple Fly tree. Official FlyGym model/assets: Apache-2.0.",
    }


def _base_result(drive):
    return {
        "schema": "chatflypt.body.v1", "dataset": drive.get("dataset"),
        "context": drive.get("context", "ground"),
        "physics_run": False, "trajectory": [], "legs": list(LEGS),
        "geometry_asset": GEOMETRY_ASSET,
        "model_variant": "badapple-fly-full-behavior",
        "source_duration_ms": drive.get("duration_ms"),
        "motor_spikes": drive.get("motor_spikes", 0),
        "active_motor_neurons": drive.get("active_motor_neurons", 0),
        "motor_coverage": drive.get('motor_coverage', {}),
        "joint_motor_spikes": drive.get('joint_motor', {}).get('motor_spikes', 0),
        "source_spikes_sha256": drive.get("source_spikes_sha256"),
        "source_mapping": drive.get("source_mapping"),
        "mapping_source": drive.get("mapping_source"),
        "controller": CONTROLLER, "controller_id": "badapple_hybrid_v1", "source": body_source(),
        "controller_extensions": ['tibia_motor_v1'] if 'joint_motor' in drive else [],
        "sourcehash": SOURCE_SHA256,
        "biologically_calibrated": False,
        "units": {"position": "mm", "time": "ms", "angles": "rad",
                  "actuator_force": "Native MuJoCo per-actuator units; mixed force/torque, not SI-converted"},
        "initialization": {"performed": False},
        "controls": {"active_posture_hold_when_commands_zero": True,
                     "controller_local_feedback": "Official Hybrid foot height/contact and heading feedback; no feedback into the neural connectome"},
        "summary": {"finite_state": None, "physics_warnings": 0},
        "limitations": list(dict.fromkeys([*drive.get("limitations", []), *LIMITATIONS])),
    }


def run_behavior_body(drive: dict, *, max_duration_ms: float = 500.0,
                      sample_ms: float = 10.0, timeout: float = 90.0,
                      output_dir: str | Path | None = None, render: bool = False,
                      passive_control: bool = False, environment: dict | None = None) -> dict:
    """Run a bounded .venv-body worker; never synthesize trajectories on failure.

    Passive control disables every actuator during the reported neural window;
    the shared upstream active warmup is explicitly retained and disclosed.
    """
    result = _base_result(drive)
    if environment:
        result['environment'] = environment
    try:
        from chatflypt.behavior_readout import _validate_behavior_drive
        _validate_behavior_drive(drive)
        if (not math.isfinite(max_duration_ms) or not 0 < max_duration_ms <= 5000
                or not math.isfinite(sample_ms) or not 1 <= sample_ms <= 100
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Invalid body duration, sample interval or timeout")
        duration = min(float(drive["duration_ms"]), max_duration_ms)
        from chatflypt.joint_motor import joint_summary, held_rows_in_window
        joint_commands = joint_summary(drive, end_ms=duration)
        held = held_rows_in_window(drive['available_at_ms'], drive['commands'],
                                   0., duration, dict.fromkeys(COMMAND_KEYS, 0.))
        in_window = any(any(row.values()) for row in held) or any(any(r.values()) for r in joint_commands.values())
        effective = any(command_execution(row)['neural_command_active'] for row in held)
        effective |= any(any(r.values()) for r in joint_commands.values())
        command_summary = {key: {"min": min(row[key] for row in held),
                                 "max": max(row[key] for row in held)}
                           for i, key in enumerate(COMMAND_KEYS)}
        result.update(duration_ms=duration, requested_duration_ms=duration,
                      truncated=duration < float(drive["duration_ms"]),
                      requested_command_in_window=in_window,
                      motor_activity_in_window=bool(effective and not passive_control), command_summary=command_summary,
                      joint_command_summary_rad=joint_commands)
        result["controls"]["passive_control"] = bool(passive_control)
        if not in_window and not passive_control and not (environment and environment.get('objects')):
            result.update(status="no_motor_activity", reason="当前时间窗没有可用的非零行为指令，未生成主动动作记录。")
            if output_dir:
                _write_result(result, output_dir, drive)
            return result
        python = ROOT / ".venv-body" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            return dict(result, status="unavailable", reason="缺少隔离的 .venv-body 物理环境。")
        request = {"drive": drive, "duration_ms": duration, "sample_ms": sample_ms,
                   "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
                   "render": bool(render), "passive_control": bool(passive_control), 'environment': environment}
        done = subprocess.run([str(python), str(Path(__file__).resolve()), "--worker"],
                              input=json.dumps(request), capture_output=True, text=True,
                              encoding="utf-8", timeout=timeout, cwd=ROOT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if done.returncode:
            return dict(result, status="error", reason="Bad Apple Fly worker failed", detail=done.stderr[-2000:])
        result.update(json.loads(done.stdout))
        if output_dir:
            _write_result(result, output_dir, drive)
        return result
    except subprocess.TimeoutExpired:
        return dict(result, status="timeout", reason=f"物理仿真超过 {timeout:g} 秒，未提供未完成轨迹。")
    except Exception as exc:
        return dict(result, status="error", reason=str(exc), error_type=type(exc).__name__)


def _write_result(result, directory, drive):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "behavior-drive.json"
    source_path.write_text(json.dumps(drive, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    result["source_drive_artifact"] = str(source_path.resolve())
    (directory / "body-result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def _simulate_physics(request):
    import importlib.metadata
    import mujoco
    from chatflypt.behavior_readout import _validate_behavior_drive, GROUP_KEYS

    started = time.perf_counter()
    if request.get('probe_only'):
        from chatflypt.environment import measure_environment_contacts
        body, initialization = initialize_body(request['seed'], request['context'], request['environment'],inspection_contacts=True)
        sim = body.sim
        names = [segment.name for segment in body.fly.get_bodysegs_order()]
        if not np.isfinite(sim.mj_data.qpos).all() or sum(int(w.number) for w in sim.mj_data.warning):
            raise ValueError('物品布局导致物理初始化不稳定')
        positions = sim.get_body_positions(body.fly.name)
        rotations = sim.get_body_rotations(body.fly.name)
        frame = {'time_ms': 0., 'position_mm': positions[names.index('c_thorax')].tolist(),
                 'body_positions_mm': positions.tolist(), 'body_quaternions_wxyz': rotations.tolist()}
        from chatflypt.environment import measure_vision
        vision = measure_vision(sim, body.fly.name, request['environment'], names, positions)
        return {'status': 'ok', 'frame': frame, 'body_names': names, 'initialization': initialization, 'vision': vision,
                'environment_contacts': measure_environment_contacts(sim.mj_model, sim.mj_data),
                'environment_geom_names': [mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
                    for i in range(sim.mj_model.ngeom) if (mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('env_')]}
    drive = request["drive"]
    _validate_behavior_drive(drive)
    body, initialization = initialize_body(drive["seed"], drive["context"], request.get('environment'),inspection_contacts=True)
    initialization["performed"] = True
    sim, fly = body.sim, body.fly
    model, data = sim.mj_model, sim.mj_data
    passive = bool(request.get("passive_control"))
    if passive:
        model.actuator_gainprm[:] = 0
        model.actuator_biasprm[:] = 0
        data.ctrl[:] = 0
        initialization["passive_control_scope"] = "All actuators disabled after the same upstream active warmup; reported neural window is passive"
    timestep = float(sim.timestep)
    timestep_ms = timestep * 1000
    duration = float(request["duration_ms"])
    steps = int(math.floor(duration / timestep_ms + 1e-8))
    sample_every = max(1, int(round(float(request["sample_ms"]) / timestep_ms)))
    names = [segment.name for segment in fly.get_bodysegs_order()]
    thorax = names.index("c_thorax")
    foot_ids = [names.index(f"{leg}_tarsus5") for leg in LEGS]
    geom_ids = body.geometry_indices
    times = np.asarray(drive["available_at_ms"])
    zero = {key: 0. for key in COMMAND_KEYS}
    current = dict(zero)
    readouts = {group: {"L": 0., "R": 0.} for group in GROUP_KEYS}
    current_row = -1
    statistics = BodyControlStatistics()
    initial_warning_count = int(sum(int(w.number) for w in data.warning))

    def frame():
        from chatflypt.environment import measure_environment_contacts
        mujoco.mj_forward(model, data)
        positions = sim.get_body_positions(fly.name)
        rotations = sim.get_body_rotations(fly.name)
        w, x, y, z = rotations[thorax]
        return {"time_ms": round(float(body.elapsed * 1000), 7),
                "position_mm": positions[thorax].tolist(),
                "quaternion_wxyz": rotations[thorax].tolist(),
                "yaw_rad": float(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                "feet_mm": positions[foot_ids].tolist(),
                "joint_angles_rad": sim.get_joint_angles(fly.name).tolist(),
                "body_positions_mm": positions.tolist(),
                "body_quaternions_wxyz": rotations.tolist(),
                "geom_positions_mm": data.geom_xpos[geom_ids].tolist(),
                "geom_rotation_matrices": data.geom_xmat[geom_ids].tolist(),
                "contact_count": int(data.ncon), "environment_contacts": measure_environment_contacts(model, data), "commands": dict(current),
                "drive": dict(current), "readouts_hz": readouts,
                "source_command_bin": current_row,
                "command_available_at_ms": float(times[current_row]) if current_row >= 0 else None,
                "joint_motor_offsets_rad": dict(body.joint_motor_offsets),
                **command_execution(current, body.joint_motor_offsets, passive=passive),
                "max_actuator_force_model_units": float(np.max(np.abs(data.actuator_force), initial=0))}

    trajectory = [frame()]
    render_frames, renderer, render_error = [], None, None
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
    try:
        for step in range(steps):
            # Commands stamped t may first affect the integration starting at t.
            row = int(np.searchsorted(times, step*timestep_ms + 1e-8, side="right")) - 1
            current_row = row
            current = dict(drive["commands"][row]) if row >= 0 and not passive else dict(zero)
            readouts = drive["readouts_hz"][row] if row >= 0 else readouts
            from chatflypt.joint_motor import offsets_at
            body.joint_motor_offsets = offsets_at(drive, row if not passive else -1)
            if passive:
                data.ctrl[:] = 0
                sim.step()
                if body.context == "tethered_flight":
                    # Same explicit kinematic tether as upstream, not an actuator.
                    data.qpos[:7] = [0, 0, 2, 1, 0, 0, 0]
                    data.qvel[:6] = 0
                body.elapsed += timestep
            else:
                body.advance(timestep_ms, current)
            force = float(np.max(np.abs(data.actuator_force), initial=0))
            statistics.record(current, body.joint_motor_offsets, force, passive=passive)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("MuJoCo produced non-finite physical state")
            if (step+1) % sample_every == 0 or step+1 == steps:
                trajectory.append(frame())
                render_frame()
    finally:
        if renderer is not None:
            renderer.close()
    warnings = int(sum(int(w.number) for w in data.warning))
    if warnings:
        raise RuntimeError(f"MuJoCo reported {warnings} numerical/physics warnings, including {initial_warning_count} during warmup")
    if passive and statistics.max_force != 0:
        raise RuntimeError("Passive control produced nonzero actuator forces")
    positions = np.asarray([item["position_mm"] for item in trajectory])
    result = _base_result(drive)
    if request.get('environment'):
        result['environment'] = request['environment']
    result.update(status="passive_control" if passive else "ok", physics_run=True,
                  trajectory=trajectory, duration_ms=round(float(body.elapsed*1000), 7),
                  requested_duration_ms=duration, truncated=duration < drive["duration_ms"],
                  body_names=names, joint_names=[j.name for j in fly.get_jointdofs_order()],
                  geom_indices=geom_ids,
                  geom_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in geom_ids],
                  initialization=initialization,
                  engine={"flygym": importlib.metadata.version("flygym"), "mujoco": mujoco.__version__,
                          "nq": model.nq, "nv": model.nv, "nu": model.nu,
                          "timestep_ms": timestep_ms,
                          "model": "Bad Apple Fly NeuroMechFly full behavior body / FlatGroundWorld"},
                  summary={"displacement_mm": (positions[-1]-positions[0]).tolist(),
                           "path_length_mm": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                           "max_contacts": max(item["contact_count"] for item in trajectory),
                           "finite_state": True, "physics_warnings": warnings},
                  elapsed_seconds=time.perf_counter()-started)
    result["controls"].update(
        passive_control=passive, all_actuators_disabled_during_replay=passive,
        active_posture_hold_when_commands_zero=not passive,
        reference_rate_hz=drive["reference_rate_hz"], smoothing_ms=drive["smoothing_ms"],
        bin_ms=drive["bin_ms"],
        neural_feedback=False, aerodynamics=False,
        kinematic_tether=drive["context"] == "tethered_flight")
    measured = statistics.report()
    result['controls'].update(measured.pop('controls'))
    result.update(measured)
    if render_error:
        result["render_error"] = render_error
    if render_frames:
        import imageio.v3 as iio
        output = Path(request["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output / "body-final.png", render_frames[-1])
        iio.imwrite(output / "body.gif", np.stack(render_frames), duration=float(request["sample_ms"]), loop=0)
        result["render_artifacts"] = {"image": str(output / "body-final.png"), "animation": str(output / "body.gif")}
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use run_behavior_body(); --worker is an internal JSON protocol")
    sys.path.insert(0, str(ROOT))
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        print(json.dumps(_simulate_physics(json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"status": "error", "physics_run": False, "trajectory": [],
                          "reason": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False))
