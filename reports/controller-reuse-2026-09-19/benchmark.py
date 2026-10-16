"""Four fixed external-command baselines for the unmodified official controller.

Run with .venv-body/Scripts/python.exe from the repository root. This benchmark
does not import ChatFlyPT brain/server modules or read any neural trial or key.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridTurningController, HybridControllerObservation, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
# Freeze these before running. Do not tune them using the resulting trajectories.
CASES = (
    ("forward", (1.0, 1.0)),
    ("left_command", (0.4, 1.2)),
    ("right_command", (1.2, 0.4)),
    ("zero_descending", (0.0, 0.0)),
)
DURATION_S = 1.0
WARMUP_S = 0.05
SEED = 0


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    manifest = json.loads((ROOT / "research/manifests/sources.lock.json").read_text(encoding="utf-8"))
    source = next(row for row in manifest if row["id"] == "flygym")
    selected = [
        "tutorials/4d_turning_controller.ipynb",
        "src/flygym_demo/complex_terrain/common.py",
        "src/flygym_demo/complex_terrain/cpg_controller.py",
        "src/flygym_demo/complex_terrain/hybrid_controller.py",
        "src/flygym_demo/complex_terrain/turning_controller.py",
        "src/flygym_demo/complex_terrain/preprogrammed.py",
        "src/flygym/assets/model/neuromechfly/rigging.yaml",
        "src/flygym/assets/model/neuromechfly/mujoco_globals.yaml",
    ]
    return {
        "scope": "Unmodified official controller; four external commands, not neural outputs",
        "source": source,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "flygym": importlib.metadata.version("flygym"), "mujoco": mujoco.__version__,
                    "controller_module": str(Path(inspect.getfile(HybridTurningController)).resolve())},
        "source_sha256": {name: file_hash(ROOT / "vendor/flygym" / name) for name in selected},
        "benchmark_sha256": file_hash(__file__),
        "cases": [{"name": name, "external_descending_signal_left_right": list(command)} for name, command in CASES],
        "seed": SEED, "duration_s_per_case": DURATION_S, "warmup_s": WARMUP_S,
        "tutorial_configuration": {
            "spawn_position_mm": [0, 0, 0.8], "spawn_quaternion_wxyz": [1, 0, 0, 0],
            "ground_contacts": "ContactBodiesPreset.TIBIA_TARSUS_ONLY",
            "fly": "make_locomotion_fly(add_adhesion=True, colorize=True); other defaults unchanged",
            "joint_stiffness": .05, "joint_damping": .06,
            "passive_tarsus_stiffness": 7.5, "passive_tarsus_damping": .01,
            "actuator_gain": 45, "actuator_forcerange": [-65, 65], "adhesion_gain": 40,
            "warmup": "Official template default pose plus all-leg adhesion, then sim.warmup(0.05); not included in measured 1 s",
            "controller": "HybridTurningController with official PreprogrammedSteps and output DOF order",
            "cpg": {"frequency_hz": 12, "coupling_strength": 10, "convergence_coefficient": 20,
                    "initial_phases": "official reset(seed=0) random phases", "initial_magnitudes": "zero"},
            "hybrid": {"stumbling_force_threshold": -1, "retraction_height_threshold_mm": .05,
                       "retraction_rates": [800, 700], "stumbling_rates": [2200, 1800],
                       "max_correction": 80, "swing_extension_rad": math.pi / 4,
                       "retraction_persistence_steps": 20},
        },
        "fall_criterion": "Engineering diagnostic: thorax tilt >60 degrees OR thorax center height <0.5 mm continuously for >=100 ms. Not biological validation.",
        "zero_command_note": "Zero descending amplitude is NOT passive: position actuators, adhesion and hybrid feedback remain enabled, as in the official API.",
        "deviations_from_tutorial": [
            "The tutorial alternates [1.2,.4] and [.4,1.2] for 1 s each in a single 2 s run; here each is an independent 1 s reset for comparability.",
            "Adds symmetric [1,1] and zero [0,0] controls; no controller equations, gains or feedback rules changed.",
            "No profiling or continuous video loop; records sampled state and one final render.",
        ],
    }


def run_case(name, command):
    wall_start = time.perf_counter()
    fly = make_locomotion_fly(name="turning_demo", add_adhesion=True, colorize=True)
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, .8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    steps_template = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    controller = HybridTurningController(timestep=sim.timestep,
        preprogrammed_steps=steps_template, output_dof_order=dof_order)
    sim.reset()
    controller.reset(seed=SEED)
    initial_action = LocomotionAction(
        joint_angles=steps_template.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool))
    apply_locomotion_action(sim, fly.name, initial_action)
    sim.warmup(WARMUP_S)
    model, data = sim.mj_model, sim.mj_data
    timestep = float(sim.timestep)
    body_order = fly.get_bodysegs_order()
    thorax_i = body_order.index(BodySegment("c_thorax"))
    body_ids = sim._internal_bodyids_by_fly[fly.name]
    thorax_body_id = int(body_ids[thorax_i])
    foot_ids = [int(body_ids[body_order.index(BodySegment(f"{leg}_tarsus5"))]) for leg in LEGS]
    # Sample derived post-integration kinematics into a separate MjData. This does
    # not call mj_forward or change the official controller's live physics state.
    sampled = mujoco.MjData(model)
    start_time = float(data.time)
    frames = []
    warning_before = [int(w.number) for w in data.warning]
    max_tilt = 0.
    min_height = math.inf
    longest_fall_steps = fall_steps = 0
    inversion_steps = 0
    correction_steps = {"retraction": 0, "stumbling": 0, "net": 0}
    max_correction = 0.
    max_contact = 0
    max_actuator = 0.
    finite = True
    failure = None

    def snapshot():
        sampled.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, sampled)
        q = sampled.xquat[thorax_body_id]
        w, x, y, z = q
        up = float(sampled.xmat[thorax_body_id].reshape(3, 3)[2, 2])
        return {"time_ms": round((float(data.time) - start_time) * 1000, 7),
                "absolute_simulation_time_ms": round(float(data.time) * 1000, 7),
                "position_mm": sampled.xpos[thorax_body_id].tolist(),
                "quaternion_wxyz": q.tolist(),
                "yaw_rad": float(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(up, -1, 1)))),
                "feet_mm": sampled.xpos[foot_ids].tolist(),
                "contact_count": int(data.ncon),
                "cpg_magnitudes": controller.cpg_network.curr_magnitudes.tolist(),
                "net_corrections": controller.last_info.get("net_corrections", np.zeros(6)).tolist()}

    frames.append(snapshot())
    nsteps = int(round(DURATION_S / timestep))
    sample_every = max(1, round(.01 / timestep))
    descending = np.array(command)
    for i in range(nsteps):
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        action = controller.step(descending, obs)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            finite = False
            failure = "Non-finite MuJoCo state"
            break
        # Per-step diagnostics from the official simulation's derived state have
        # at most one dt (0.1 ms) lag; output snapshots use current qpos instead.
        up = float(data.xmat[thorax_body_id].reshape(3, 3)[2, 2])
        tilt = float(np.degrees(np.arccos(np.clip(up, -1, 1))))
        height = float(data.xpos[thorax_body_id, 2])
        max_tilt = max(max_tilt, tilt)
        min_height = min(min_height, height)
        fall_steps = fall_steps + 1 if tilt > 60 or height < .5 else 0
        longest_fall_steps = max(longest_fall_steps, fall_steps)
        inversion_steps += int(up < 0)
        max_contact = max(max_contact, int(data.ncon))
        max_actuator = max(max_actuator, float(np.max(np.abs(data.actuator_force))))
        for key, field in [("retraction", "retraction_correction"), ("stumbling", "stumbling_correction"), ("net", "net_corrections")]:
            correction_steps[key] += int(np.any(np.abs(controller.last_info[field]) > 1e-12))
        max_correction = max(max_correction, float(np.max(np.abs(controller.last_info["net_corrections"]))))
        if (i + 1) % sample_every == 0 or i + 1 == nsteps:
            frames.append(snapshot())
    warnings = [int(w.number) for w in data.warning]
    coordinates = np.array([f["position_mm"] for f in frames])
    yaw = np.unwrap([f["yaw_rad"] for f in frames])
    delta = coordinates[-1] - coordinates[0]
    initial_heading = np.array([math.cos(yaw[0]), math.sin(yaw[0])])
    yaw_delta = float(np.degrees(yaw[-1] - yaw[0]))
    result = {
        "name": name, "source": "External fixed command; not neural activity",
        "external_descending_signal_left_right": list(command),
        "status": "completed" if finite and not failure else "failed", "failure": failure,
        "physics": {"flygym": importlib.metadata.version("flygym"), "mujoco": mujoco.__version__,
                    "nq": model.nq, "nv": model.nv, "nu": model.nu, "dt_ms": timestep * 1000,
                    "measured_duration_ms": frames[-1]["time_ms"], "warmup_ms": start_time * 1000,
                    "finite_state": finite, "warnings_total": sum(warnings),
                    "warnings_during_warmup": sum(warning_before), "warning_counts": warnings},
        "motion": {"displacement_mm": delta.tolist(),
                   "horizontal_displacement_mm": float(np.linalg.norm(delta[:2])),
                   "forward_projection_mm": float(delta[:2] @ initial_heading),
                   "xy_path_length_mm": float(np.linalg.norm(np.diff(coordinates[:, :2], axis=0), axis=1).sum()),
                   "yaw_change_deg_unwrapped": yaw_delta,
                   "observed_yaw_sign": "left" if yaw_delta > 2 else "right" if yaw_delta < -2 else "near_zero"},
        "stability": {"max_thorax_tilt_deg": max_tilt, "min_thorax_height_mm": min_height,
                      "longest_fall_criterion_ms": longest_fall_steps * timestep * 1000,
                      "fell_by_engineering_criterion": longest_fall_steps * timestep >= .1,
                      "inverted_ms": inversion_steps * timestep * 1000, "max_contacts": max_contact},
        "feedback": {"correction_steps": correction_steps, "max_abs_joint_correction_multiplier": max_correction,
                     "max_actuator_force_native_mixed_units": max_actuator},
        "units": {"position": "mm", "time": "ms", "orientation": "radians in frames; degrees in summary",
                  "actuator": "native MuJoCo per-actuator units; mixed force/torque, no SI conversion"},
        "trajectory": frames,
        "elapsed_seconds": time.perf_counter() - wall_start,
    }
    if name == "forward" and finite:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.azimuth, camera.elevation, camera.distance = 135, -25, 9
            camera.lookat[:] = coordinates[-1]
            renderer.update_scene(data, camera=camera)
            import imageio.v3 as iio
            iio.imwrite(HERE / "official-forward-final.png", renderer.render())
            renderer.close()
            result["render"] = "official-forward-final.png"
        except Exception as exc:
            result["render_error"] = f"{type(exc).__name__}: {exc}"
    dump(HERE / f"{name}.json", result)
    return {k: v for k, v in result.items() if k != "trajectory"}


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (path_ax, yaw_ax) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for name, _ in CASES:
        result = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
        frames = result["trajectory"]
        xyz = np.array([f["position_mm"] for f in frames])
        xy = xyz[:, :2] - xyz[0, :2]
        yaw = np.degrees(np.unwrap([f["yaw_rad"] for f in frames]))
        t = np.array([f["time_ms"] for f in frames]) / 1000
        path_ax.plot(xy[:, 0], xy[:, 1], label=name)
        yaw_ax.plot(t, yaw - yaw[0], label=name)
    path_ax.set(xlabel="World x displacement (mm)", ylabel="World y displacement (mm)", title="Measured thorax paths")
    path_ax.set_aspect("equal", adjustable="datalim")
    yaw_ax.set(xlabel="Physical time after warmup (s)", ylabel="Unwrapped yaw change (degrees)", title="Measured orientation change")
    for ax in (path_ax, yaw_ax):
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Official HybridTurningController: external commands, not connectome outputs", fontsize=11)
    fig.savefig(HERE / "baseline-trajectories.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    frozen = protocol()
    dump(HERE / "protocol.json", frozen)
    results = []
    for name, command in CASES:
        print(f"Starting {name}: external command {command}", flush=True)
        result = run_case(name, command)
        results.append(result)
        dump(HERE / "summary.json", {"protocol": "protocol.json", "results": results})
        print(json.dumps(result, ensure_ascii=True), flush=True)
    assert len(results) == 4
    assert all(file_hash(ROOT / "vendor/flygym" / name) == sha for name, sha in frozen["source_sha256"].items())
    plot()
