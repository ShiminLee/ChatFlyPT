"""Author MaleCNS optical axes registered to the actual official eye sampler.

Geometry is reproducible, but cross-specimen anatomy and population physiology
remain uncalibrated. No object identity, requested action, or eye-mean fallback.
The source atlas is CC BY-SA 4.0, Arthur Zhao / Reiser Lab; derived maps retain it.
"""
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .paths import ROOT

EVIDENCE = ROOT / 'research/retinal-registration-followup-20260924'
ATLAS = EVIDENCE / 'eyemap-archive/maps/eyemap_mcns_f20240701'
AUTHOR_COLUMNS = EVIDENCE / 'visualpathways/params'
CAMERA_GEOMETRY = EVIDENCE / 'camera-geometry.json'
FLYGYM = ROOT / 'vendor/flygym/src/flygym'
EYE_ASSETS = FLYGYM / 'assets/model/neuromechfly'
ATLAS_COMMIT = '503c7f055d5491a48b60b49ade8c71798d24d8f1'
COLUMNS_COMMIT = '23f6ac131529b5f56894c6eeb9b88b17894fc00d'
CAMERA_SOURCE_PATHS = (
    EYE_ASSETS / 'compound_eye.npz', EYE_ASSETS / 'vision.yaml',
    EYE_ASSETS / 'rigging.yaml', EYE_ASSETS / 'mujoco_globals.yaml',
    FLYGYM / 'compose/fly/base_fly.py', FLYGYM / 'compose/fly/neuromechfly.py',
    FLYGYM / 'vision/retina.py', FLYGYM / 'simulation.py',
    ROOT / 'vendor/badapple-fly/backend/body.py',
    ROOT / 'chatflypt/behavior_body.py', ROOT / 'chatflypt/environment.py',
    ROOT / 'chatflypt/body_measurement.py',
)

# Both the model contract and map loader bind current files. A static camera
# snapshot alone must never allow a modified rig/Retina to use obsolete rays.
SOURCE_PATHS = (
    Path(__file__).resolve(), ROOT / 'chatflypt/spatial_vision.py',
    CAMERA_GEOMETRY, EVIDENCE / 'camera_scan.py',
    EVIDENCE / 'eyemap-archive-manifest.json', EVIDENCE / 'visualpathways-manifest.json',
    ATLAS / 'pqxyztp_left.xlsx', ATLAS / 'pqxyztp_right.xlsx',
    AUTHOR_COLUMNS / 'ME_L_columnar-cells_location.xlsx',
    AUTHOR_COLUMNS / 'ME_columnar-cells_location.xlsx',
    *CAMERA_SOURCE_PATHS,
)


def source_paths():
    return SOURCE_PATHS


def read_numeric_xlsx(path):
    """Read the author's first ordinary value table at original precision.

    Formulas are rejected and no downloaded code or pickle is executed.
    """
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        rows = []
        for row in ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('.//m:row', ns):
            values = {}
            for cell in row.findall('m:c', ns):
                if cell.find('m:f', ns) is not None:
                    raise ValueError('Unexpected formula in author retinal data')
                ix = 0
                for char in re.match('[A-Z]+', cell.attrib['r'])[0]: ix = 26*ix + ord(char)-64
                value = cell.find('m:v', ns)
                if value is None: value = ''.join(cell.itertext())
                elif cell.attrib.get('t') == 's': value = strings[int(value.text)]
                else: value = float(value.text)
                values[ix-1] = value
            rows.append(values)
    return pd.DataFrame([{rows[0][k]: v for k,v in row.items()} for row in rows[1:]])


def unit(values):
    values = np.asarray(values, float)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def angle_deg(first, second):
    return np.degrees(np.arccos(np.clip(np.sum(unit(first)*unit(second), axis=-1), -1, 1)))


def numeric_vision_config():
    return {k: float(v) for k,v in re.findall(r'^([a-z_]+): ([0-9.]+)$',
        (EYE_ASSETS/'vision.yaml').read_text(), flags=re.M)}


def retina_mean_rays(camera, config, id_map):
    """Exact pinned fisheye source pixels -> ommatidial mean direction in head.

    Camera -Z is forward, +X right, +Y up. Do not mirror the left eye again.
    """
    h,w = id_map.shape
    row,col = np.indices(id_map.shape)
    dr=(2*row-h)/h/config['fisheye_zoom'];dc=(2*col-w)/w/config['fisheye_zoom']
    denominator=1-config['fisheye_distortion_coefficient']*(dr**2+dc**2)+1e-6
    sr=np.trunc((dr/denominator+1)*h/2).astype(int)
    sc=np.trunc((dc/denominator+1)*w/2).astype(int)
    valid=(sr>=0)&(sr<h)&(sc>=0)&(sc<w)&(id_map>0)
    focal=h/(2*np.tan(np.radians(camera['fovy_deg'])/2))
    rays=unit(np.stack(((sc+.5-w/2)/focal,-(sr+.5-h/2)/focal,-np.ones_like(sr)),axis=-1))
    rays=rays@np.asarray(camera['head_rotation']).T
    mean,spread=[],[]
    for label in range(1,int(id_map.max())+1):
        selected=rays[valid&(id_map==label)]
        if not len(selected): raise ValueError('An official ommatidium has no valid camera samples')
        center=unit(selected.mean(axis=0));mean.append(center)
        spread.append(float(angle_deg(selected,center).max()))
    return np.asarray(mean),np.asarray(spread)


def retinal_support(ray, camera, config, id_map):
    """Reject directions outside actual camera / compound-eye coverage."""
    ray=np.asarray(ray)@np.asarray(camera['head_rotation'])
    if ray[2]>=0: return 'behind_eye_camera'
    h,w=id_map.shape;focal=h/(2*np.tan(np.radians(camera['fovy_deg'])/2))
    src=np.array([focal*ray[0]/-ray[2]*2/w,-focal*ray[1]/-ray[2]*2/h])
    if np.any(np.abs(src)>=1): return 'outside_rectilinear_camera'
    radius=np.linalg.norm(src);k=config['fisheye_distortion_coefficient']
    dst_radius=2*radius*(1+1e-6)/(1+np.sqrt(1+4*k*radius**2*(1+1e-6)))
    dst=src*(dst_radius/radius if radius else 1.)*config['fisheye_zoom']
    col,row=np.rint((dst+1)*np.array([w,h])/2).astype(int)
    if not(0<=col<w and 0<=row<h) or id_map[row,col]==0: return 'outside_compound_eye_mask'
    return None


def column_crosswalk(pins, annotations, side):
    """Same body IDs, >=2 distinct types, unanimous and bijective columns."""
    witnesses={};checks={'same_column':0,'changed_column':0,'missing_or_invalid_identity':0}
    for _,row in pins.iterrows():
        author=tuple(map(int,[row.hex1_id,row.hex2_id]))
        for kind in ['L1','L2','L3','Mi1']:
            body_id=row.get(kind,np.nan)
            if not np.isfinite(body_id): continue
            if int(body_id) not in annotations.index:
                checks['missing_or_invalid_identity']+=1;continue
            current=annotations.loc[int(body_id)]
            coords=[current.assignedOlHex1,current.assignedOlHex2]
            if not np.isfinite(coords).all() or current.somaSide!=side or current.type!=kind:
                checks['missing_or_invalid_identity']+=1;continue
            key=tuple(map(int,coords));checks['same_column' if key==author else 'changed_column']+=1
            witnesses.setdefault(key,[]).append({'body_id':int(body_id),'type':kind,'author_hex':list(author)})
    accepted={};rejected={}
    for current,proof in witnesses.items():
        choices={tuple(w['author_hex']) for w in proof}
        if len(choices)==1 and len({w['type'] for w in proof})>=2: accepted[current]=next(iter(choices))
        else: rejected[current]='conflicting_or_insufficient_body_id_witnesses'
    inverse={}
    for current,author in accepted.items(): inverse.setdefault(author,[]).append(current)
    for currents in inverse.values():
        if len(currents)>1:
            for current in currents:
                accepted.pop(current);rejected[current]='non_bijective_body_id_crosswalk'
    evidence={'method':'Same body IDs; >=2 distinct types; all agree; bijective.',
        'identity_checks':checks,
        'accepted':[{'current_hex':list(k),'author_hex':list(v),'witnesses':witnesses[k]} for k,v in sorted(accepted.items())],
        'rejected':[{'current_hex':list(k),'reason':v,'witnesses':witnesses[k]} for k,v in sorted(rejected.items())]}
    return accepted,evidence


def prepare_registration(annotations, id_map):
    """Return geometry and complete column provenance; no graph / body running."""
    from .paths import sha256
    for package,commit,paths in [
        ('eyemap-archive',ATLAS_COMMIT,[ATLAS/f'pqxyztp_{side}.xlsx' for side in ('left','right')]),
        ('visualpathways',COLUMNS_COMMIT,[AUTHOR_COLUMNS/'ME_L_columnar-cells_location.xlsx',AUTHOR_COLUMNS/'ME_columnar-cells_location.xlsx'])]:
        manifest=json.loads((EVIDENCE/f'{package}-manifest.json').read_text())
        if manifest['commit']!=commit: raise ValueError('Author retinal source commit mismatch')
        expected={r['path']:r['sha256'] for r in manifest['files']}
        for path in paths:
            name=str(path.relative_to(EVIDENCE/package)).replace('\\','/')
            if expected.get(name)!=sha256(path): raise ValueError(f'Author retinal source checksum mismatch: {name}')
    cameras=json.loads(CAMERA_GEOMETRY.read_text())
    expected={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in CAMERA_SOURCE_PATHS}
    if cameras.get('geometry_source_sha256') != expected:
        raise ValueError('Captured eye-camera geometry sources changed; rerun camera_scan.py --capture before rebuilding')
    config=numeric_vision_config();sides={}
    for side,letter in [('left','L'),('right','R')]:
        atlas=read_numeric_xlsx(ATLAS/f'pqxyztp_{side}.xlsx')
        if atlas.duplicated(['hex1','hex2']).any(): raise ValueError('Duplicate author retinal column')
        if not(np.all(atlas.p==atlas.hex2-19) and np.all(atlas.q==atlas.hex1-18)):
            raise ValueError('Unexpected author p/q and hex convention')
        if np.max(np.abs(np.linalg.norm(atlas[['x','y','z']],axis=1)-1))>1e-10:
            raise ValueError('Author optical axes must be unit vectors')
        pins=read_numeric_xlsx(AUTHOR_COLUMNS/('ME_L_columnar-cells_location.xlsx' if side=='left' else 'ME_columnar-cells_location.xlsx'))
        crosswalk,proof=column_crosswalk(pins,annotations,letter)
        camera=cameras['cameras'][side];rays,spread=retina_mean_rays(camera,config,id_map)
        tree=cKDTree(rays);columns={}
        for _,row in atlas.iterrows():
            ray=row[['x','y','z']].to_numpy(float)
            reason=retinal_support(ray,camera,config,id_map);_,closest=tree.query(ray)
            key=tuple(map(int,[row.hex1,row.hex2]))
            columns[key]={'author_hex':list(key),'author_head_ray':ray.tolist(),'rejected_reason':reason,
                'ommatidium_index':int(closest) if not reason else None,
                'angular_error_deg':float(angle_deg(ray,rays[closest]))}
        sides[side]={'crosswalk':crosswalk,'crosswalk_evidence':proof,'columns':columns,
            'ommatidium_head_rays':rays,'ommatidium_spread_deg':spread,'camera':camera}
    return sides


def registered_column(registration, current_hex):
    author=registration['crosswalk'].get(tuple(current_hex))
    if author is None: return None,'no_unambiguous_body_id_column_crosswalk'
    column=registration['columns'].get(author)
    if column is None: return None,'no_author_atlas_column'
    if column['rejected_reason']: return column,column['rejected_reason']
    return column,None
