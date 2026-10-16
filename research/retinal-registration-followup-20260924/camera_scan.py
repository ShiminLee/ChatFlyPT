"""Read-only geometry extraction and physical renderer scan; no neural runs.

.venv-body/Scripts/python.exe research/retinal-registration-followup-20260924/camera_scan.py --capture
.venv-body/Scripts/python.exe research/retinal-registration-followup-20260924/camera_scan.py
The controlled optical scene uses actual compiled eye cameras but no fly mesh
occluders: it validates projection handedness / resampling, not animal behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def dump(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')


def capture():
    from chatflypt.behavior_body import initialize_body
    body, init = initialize_body(environment={'objects': [], 'revision': 0})
    sim = body.sim
    m, d = sim.mj_model, sim.mj_data
    mj.mj_forward(m, d)
    head = m.body('fly/c_head').id
    H = d.xmat[head].reshape(3, 3); origin = d.xpos[head]
    report = {'mujoco_version': mj.__version__, 'head_position_world': origin.tolist(),
        'head_rotation_world': H.tolist(), 'initialization': init, 'cameras': {}}
    source_names = [
        'vendor/flygym/src/flygym/assets/model/neuromechfly/'+name
        for name in ['compound_eye.npz','vision.yaml','rigging.yaml','mujoco_globals.yaml']]
    source_names += ['vendor/flygym/src/flygym/'+name for name in
        ['compose/fly/base_fly.py','compose/fly/neuromechfly.py','vision/retina.py','simulation.py']]
    source_names += ['vendor/badapple-fly/backend/body.py','chatflypt/behavior_body.py','chatflypt/environment.py','chatflypt/body_measurement.py']
    report['geometry_source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    for side, cid in zip(['left', 'right'], sim._intern_eye_camera_ids_by_fly[body.fly.name]):
        C = d.cam_xmat[cid].reshape(3, 3); P = d.cam_xpos[cid]
        report['cameras'][side] = {'id': int(cid), 'name': m.camera(cid).name,
            'world_rotation': C.tolist(), 'head_rotation': (H.T @ C).tolist(),
            'world_position': P.tolist(), 'head_position': (H.T @ (P - origin)).tolist(),
            'fovy_deg': float(m.cam_fovy[cid])}
    dump('camera-geometry.json', report)
    print('Captured actual compiled cameras after unchanged initialization; no brain or neural commands.')


def unit(x):
    a = np.asarray(x, float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b)), -1, 1))))


def scan():
    from flygym.vision.retina import Retina
    from PIL import Image
    geometry = json.loads((HERE / 'camera-geometry.json').read_text())
    retina = Retina()
    with np.load(HERE / 'candidate-receptor-map.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    candidate = json.loads((HERE / 'candidate-receptor-map.json').read_text())['mapped']
    H = np.asarray(geometry['head_rotation_world'])
    h, w = retina.nrows, retina.ncols
    camera_xml = []
    for side, camera in geometry['cameras'].items():
        C = np.asarray(camera['world_rotation'])
        axes = np.concatenate((C[:, 0], C[:, 1]))
        camera_xml.append(f'<camera name="{side}" pos="{" ".join(map(str,camera["world_position"]))}" xyaxes="{" ".join(map(str,axes))}" fovy="{camera["fovy_deg"]}"/>')
    xml = f'''<mujoco><visual><global offwidth="{w}" offheight="{h}"/><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/><quality offsamples="4"/></visual>
    <asset><material name="white" rgba="1 1 1 1" emission="1"/></asset>
    <worldbody>{''.join(camera_xml)}<body name="spot" mocap="true"><geom name="spot" type="sphere" size="0.6" material="white" contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
    model = mj.MjModel.from_xml_string(xml); data = mj.MjData(model)
    renderer = mj.Renderer(model, height=h, width=w)
    records = []
    try:
        for side in ['left', 'right']:
            camera = geometry['cameras'][side]
            rays = arrays[f'{side}_ommatidium_head_rays']
            mapped = [r for r in candidate if r['side'] == side]
            # Test a broad world-direction grid independently of author columns.
            cases = []
            for az in range(-150, 151, 30):
                for el in [-60, -30, 0, 30, 60]:
                    a, e = np.radians([az, el])
                    cases.append({'kind': 'head_direction_grid', 'azimuth_left_deg': az, 'elevation_deg': el,
                        'direction': [np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]})
            # Also target a stratified set of actual author column axes, including
            # peripheral ones. No brain outputs or expected animal actions used.
            cols = {tuple(r['author_hex']): r for r in mapped}
            keys = sorted(cols)
            for i in np.linspace(0, len(keys)-1, 20).round().astype(int):
                r = cols[keys[i]]
                cases.append({'kind': 'author_column_axis', 'author_hex': r['author_hex'],
                    'mapped_ommatidium_index': r['ommatidium_index'], 'direction': r['author_head_ray']})
            for case in cases:
                direction = unit(case['direction'])
                target = np.asarray(camera['world_position']) + H @ direction * 20
                data.mocap_pos[0] = target
                mj.mj_forward(model, data)
                renderer.update_scene(data, camera=side)
                raw = renderer.render()
                fish = retina.correct_fisheye(raw)
                readings = retina.raw_image_to_hex_pxls(fish).sum(axis=-1)
                nearest = int(np.argmax(rays @ direction)); peak = int(np.argmax(readings))
                visible = bool(readings.max() > .005)
                active = np.flatnonzero(readings > .005)
                record = dict(case, side=side, target_world_mm=target.tolist(), distance_from_eye_mm=20.,
                    sphere_radius_mm=.6, angular_radius_deg=float(np.degrees(np.arcsin(.6/20))),
                    renderer_visible=visible, peak_intensity=float(readings[peak]),
                    peak_ommatidium_index=peak if visible else None,
                    nearest_ray_ommatidium_index=nearest,
                    nearest_ray_angular_error_deg=angle(direction, rays[nearest]),
                    observed_peak_ray_error_deg=angle(direction, rays[peak]) if visible else None,
                    nearest_ray_ommatidium_intensity=float(readings[nearest]),
                    illuminated_ommatidia=active.tolist())
                if 'mapped_ommatidium_index' in case:
                    record['mapped_sample_received_light'] = bool(readings[case['mapped_ommatidium_index']] > .005)
                    record['mapped_sample_intensity'] = float(readings[case['mapped_ommatidium_index']])
                records.append(record)
                if case['kind'] == 'head_direction_grid' and case['elevation_deg'] == 0 and case['azimuth_left_deg'] in [-90, 0, 90]:
                    Image.fromarray(raw).save(HERE/f'scan-{side}-az{case["azimuth_left_deg"]}-raw.png')
                    Image.fromarray(fish).save(HERE/f'scan-{side}-az{case["azimuth_left_deg"]}-retina.png')
    finally:
        renderer.close()
    visible = [r for r in records if r['renderer_visible']]
    cols = [r for r in records if r['kind'] == 'author_column_axis']
    report = {'schema': 'offline.retinal_camera_scan.v1', 'mujoco_version': mj.__version__,
        'scope': 'Real MuJoCo sphere rendering with actual compiled eye-camera extrinsics and official Retina; controlled scene excludes body/ground occlusion; no neural runs.',
        'cases': len(records), 'visible_cases': len(visible), 'author_axis_cases': len(cols),
        'author_axis_received_light': sum(r['mapped_sample_received_light'] for r in cols),
        'peak_ray_error_deg': dict(zip(['min','median','p95','max'], np.quantile([r['observed_peak_ray_error_deg'] for r in visible],[0,.5,.95,1]).tolist())),
        'records': records}
    dump('renderer-scan.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--capture', action='store_true')
    if parser.parse_args().capture: capture()
    else: scan()
