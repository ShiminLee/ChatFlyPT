"""Read actual body frames without indexing MuJoCo with missing IDs.

Static-body fusion may remove an anatomical body. Recover only its retained
MjSpec rigid transform from a surviving parent; never invent a new joint or
use the final body in the array. Physics and controller inputs are unchanged.
"""
from types import MethodType

import numpy as np


def body_poses(sim, fly_name):
    import mujoco as mj
    fly = sim.world.fly_lookup[fly_name]
    model, data = sim.mj_model, sim.mj_data
    memo = {}

    def resolve(body):
        if body.name in memo:
            return memo[body.name]
        index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body.name)
        if index >= 0:
            result = data.xpos[index].copy(), data.xquat[index].copy()
        else:
            if list(body.joints):
                raise ValueError(f'Missing movable anatomical body: {body.name}')
            if body.parent is None or body.parent.name == body.name:
                raise ValueError(f'Cannot recover missing anatomical body frame: {body.name}')
            parent_position, parent_quaternion = resolve(body.parent)
            local_position = np.asarray(body.pos, float)
            local_quaternion = np.asarray(body.quat, float)
            if (not np.isfinite(local_position).all() or not np.isfinite(local_quaternion).all()
                    or not np.isclose(np.linalg.norm(local_quaternion), 1., atol=1e-10)):
                raise ValueError(f'Unsupported static-body orientation: {body.name}')
            matrix = np.empty(9); mj.mju_quat2Mat(matrix, parent_quaternion)
            quaternion = np.empty(4); mj.mju_mulQuat(quaternion, parent_quaternion, local_quaternion)
            result = parent_position + matrix.reshape(3, 3) @ local_position, quaternion
        memo[body.name] = result
        return result

    frames = [resolve(fly.bodyseg_to_mjcfbody[segment]) for segment in fly.get_bodysegs_order()]
    return np.asarray([p for p, q in frames]), np.asarray([q for p, q in frames])


def install_body_pose_accessors(sim):
    """Override read access on this instance only, including after recompile."""
    if hasattr(sim, '_chatflypt_raw_body_positions'):
        return
    sim._chatflypt_raw_body_positions = sim.get_body_positions
    sim._chatflypt_raw_body_rotations = sim.get_body_rotations
    sim.get_body_positions = MethodType(lambda instance, name: body_poses(instance, name)[0], sim)
    sim.get_body_rotations = MethodType(lambda instance, name: body_poses(instance, name)[1], sim)


def eye_camera_positions(sim, fly_name):
    """Optical sensing locations are the actual camera frames, not eye origins."""
    import mujoco as mj
    fly = sim.world.fly_lookup[fly_name]
    cameras = list(fly.eyecameraname_to_mjcfcamera.values())
    if len(cameras) != 2:
        raise ValueError('Expected two official eye cameras')
    positions = {}
    for side, camera in zip(('left', 'right'), cameras):
        index = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, camera.name)
        if index < 0:
            raise ValueError(f'Missing compiled eye camera: {camera.name}')
        positions[side] = sim.mj_data.cam_xpos[index].copy()
    return positions
