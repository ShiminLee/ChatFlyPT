"""Edit scene geometry without resetting the fly, controller or simulation clock.

Only static environment geoms/lights/contact pairs are replaced. The MuJoCo
integration state is copied explicitly: recompile alone does not guarantee all
integration fields survive on every supported MuJoCo release.
"""
import hashlib
import math


def update_body_environment(body, previous, environment):
    import mujoco as mj
    import numpy as np
    from .environment import TYPES, validate_objects

    objects = validate_objects(environment['objects'])
    sim = body.sim
    old_model, old_data = sim.mj_model, sim.mj_data
    state_type = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(old_model, state_type))
    mj.mj_getState(old_model, old_data, state, state_type)
    before_hash = hashlib.sha256(state.tobytes()).hexdigest()
    spec = sim.world.mjcf_root.copy()
    removed = {'env_' + obj['id'] for obj in previous['objects']}
    removed_lights = {'light_' + obj['id'] for obj in previous['objects']}
    ground_names = [g.name for g in sim.world.ground_geoms if g.name not in removed]

    # Preserve the original contact parameters for each body geometry. These
    # are already present for the ground; new objects use the same mechanics.
    templates = {}
    for pair in spec.pairs:
        if pair.geomname1.startswith('fly/') and pair.geomname2 in ground_names:
            templates.setdefault(pair.geomname1, {
                key: np.asarray(getattr(pair, key)).copy() if key in ('friction', 'solref', 'solimp')
                else getattr(pair, key)
                for key in ('friction', 'solref', 'solimp', 'margin', 'gap', 'condim')})
    if not templates:
        raise ValueError('No original body contact parameters available')
    for pair in list(spec.pairs):
        if pair.geomname1 in removed or pair.geomname2 in removed:
            spec.delete(pair)
    for geom in list(spec.geoms):
        if geom.name in removed:
            spec.delete(geom)
    for light in list(spec.lights):
        if light.name in removed_lights:
            spec.delete(light)
    environment_names = [g.name for g in sim.world.environment_geoms if g.name not in removed]
    for obj in objects:
        if obj['type'] == 'light':
            spec.worldbody.add_light(name='light_' + obj['id'], type=mj.mjtLightType.mjLIGHT_POINT,
                pos=obj['position_mm'], diffuse=[obj['strength']] * 3,
                attenuation=[1., 0., 1. / obj['range_mm']**2], castshadow=True)
        if not TYPES[obj['type']]['solid']:
            continue
        name = 'env_' + obj['id']
        color = TYPES[obj['type']]['color'].lstrip('#')
        angle = math.radians(obj['yaw_deg']) / 2
        spec.worldbody.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX,
            pos=obj['position_mm'], size=np.array(obj['size_mm']) / 2,
            quat=[math.cos(angle), 0, 0, math.sin(angle)],
            rgba=[int(color[i:i+2], 16) / 255 for i in (0, 2, 4)] + [1.],
            contype=0, conaffinity=0)
        for fly_geom, parameters in templates.items():
            spec.add_pair(name=fly_geom + '-' + name + '-object',
                geomname1=fly_geom, geomname2=name, **parameters)
        ground_names.append(name)
        environment_names.append(name)

    # Static objects may change contact arrays and geometry indices, never the
    # fly's degrees of freedom, body names, actuator order or mocap identities.
    # FlyGym enables static-body fusion. Compilation mutates its spec, so keep
    # the editable tree intact exactly as FlyGym's own compile() does.
    new_model, new_data = spec.copy().recompile(old_model, old_data)
    for count, kind in [('njnt', mj.mjtObj.mjOBJ_JOINT), ('nu', mj.mjtObj.mjOBJ_ACTUATOR),
                        ('nbody', mj.mjtObj.mjOBJ_BODY)]:
        old_names = [mj.mj_id2name(old_model, kind, i) for i in range(getattr(old_model, count))]
        new_names = [mj.mj_id2name(new_model, kind, i) for i in range(getattr(new_model, count))]
        if old_names != new_names:
            differing=[(a,b) for a,b in zip(old_names,new_names) if a!=b]
            raise ValueError(f'Scene update changed {count} ordering: {differing[:3]} ({len(old_names)} -> {len(new_names)})')
    if mj.mj_stateSize(new_model, state_type) != len(state):
        raise ValueError('Scene update changed integration state dimensions')
    mj.mj_setState(new_model, new_data, state, state_type)
    verified = np.empty_like(state)
    mj.mj_getState(new_model, new_data, verified, state_type)
    if not np.array_equal(state, verified):
        raise ValueError('Scene update did not preserve integration state')

    for renderer_name in ('renderer', 'eye_renderer'):
        renderer = getattr(sim, renderer_name, None)
        if renderer is not None:
            renderer.close()
            setattr(sim, renderer_name, None)
    sim.mj_model, sim.mj_data = new_model, new_data
    sim.world._mjcf_root = spec
    sim.world._neutral_keyframe = spec.key('neutral')
    sim.world.ground_geoms = [spec.geom(name) for name in ground_names]
    sim.world.environment_geoms = [spec.geom(name) for name in environment_names]
    for mapper in ('bodyids', 'geom_ids', 'ground_geom_ids', 'qposqveladrs', 'actuator_ids',
                   'adhesionactuator_ids', 'tendonactuator_ids', 'jointids',
                   'groundcontactsensor_ids', 'site_ids', 'eye_camera_ids'):
        getattr(sim, '_map_internal_' + mapper)()
    body.geometry_indices = [i for i in range(new_model.ngeom)
                             if new_model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH]
    mj.mj_forward(new_model, new_data)
    # mj_forward refreshes derived forces; restore the solver warm start that
    # belongs to the preserved integration state before the next step.
    mj.mj_setState(new_model, new_data, state, state_type)
    return {'status': 'ok', 'state_preserved': True, 'controller_reset': False,
        'body_reset': False, 'time_ms': body.elapsed * 1000,
        'integration_state_before_sha256': before_hash,
        'integration_state_after_sha256': hashlib.sha256(verified.tobytes()).hexdigest(),
        'method': 'MjSpec.recompile plus explicit mjSTATE_INTEGRATION preservation',
        'environment_revision': environment['revision']}
