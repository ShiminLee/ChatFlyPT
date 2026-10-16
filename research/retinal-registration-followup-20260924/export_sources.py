"""Decode immutable author RData files as data, export joins / schema / QC.

Uses rdata, numpy, pandas and scipy in .venv. Does not execute author R code.
"""
import json
import warnings
import numpy as np
import pandas as pd
import rdata
from scipy.spatial import cKDTree
from registration import HERE, ANN, ATLAS, sha, dump, xlsx_numeric, angle, quantiles

warnings.filterwarnings('ignore', category=UserWarning)


def arr(a):
    return np.asarray(a)


def main():
    schemas = {}
    for relative in ['data/eyemap.RData', 'data/med_ixy.RData', 'data/lens_ixy.RData',
                     'data/Mi1_ind.RData', 'data/microCT/20240701.RData', 'data/microCT/20240701_normals.RData']:
        path = HERE / 'upstream' / relative
        data = rdata.read_rda(path)
        fields = []
        for k, v in data.items():
            f = {'name': k, 'python_type': type(v).__name__, 'shape': list(v.shape) if hasattr(v, 'shape') else None}
            if hasattr(v, 'coords'):
                f['dimensions'] = {dim: [str(x) for x in v.coords[dim].values[:5]] for dim in v.dims}
            if isinstance(v, pd.DataFrame): f['columns'] = list(v.columns)
            if isinstance(v, dict): f['keys'] = list(v)
            fields.append(f)
        schemas[relative] = {'sha256': sha(path), 'fields': fields}
    dump(HERE / 'rdata-schema.json', schemas)
    eye = rdata.read_rda(HERE / 'upstream/data/eyemap.RData')
    med = arr(rdata.read_rda(HERE / 'upstream/data/med_ixy.RData')['med_ixy'])
    pair = arr(eye['eyemap']).astype(int)
    out = pd.DataFrame({'fafb_Mi1_row_index': pair[:, 0], 'microct_right_lens_local_index': pair[:, 1]})
    out[['axis_anterior', 'axis_left', 'axis_dorsal']] = arr(eye['ucl_rot_sm'])
    out[['fafb_medulla_native_x', 'fafb_medulla_native_y', 'fafb_medulla_native_z']] = arr(eye['med_xyz'])
    out = out.merge(pd.DataFrame(med, columns=['fafb_Mi1_row_index', 'grid_x', 'grid_y']), on='fafb_Mi1_row_index', validate='one_to_one')
    out.to_csv(HERE / 'fafb-right-778-lens-mi1-axes.csv', index=False)
    pd.DataFrame(arr(eye['lens_ixy']), columns=['microct_right_lens_local_index', 'grid_x', 'grid_y']).to_csv(HERE / 'eyemap-embedded-852-lens-grid.csv', index=False)
    source = rdata.read_rda(HERE / 'upstream/data/microCT/20240701.RData')
    # R i_match is one-based lens ID for each cone row; argsort restores lens order.
    axes = arr(source['ucl_rot_sm'])[np.argsort(arr(source['i_match']))]
    isleft = arr(source['ind_left_lens']).astype(bool)
    micro = pd.DataFrame(axes, columns=['axis_anterior', 'axis_left', 'axis_dorsal'])
    micro.insert(0, 'lens_bilateral_index', np.arange(1, len(micro)+1))
    micro['side'] = np.where(isleft, 'left', 'right')
    micro[['lens_aligned_x','lens_aligned_y','lens_aligned_z']] = arr(source['lens'])
    micro.to_csv(HERE / 'microct-1709-lens-axes.csv', index=False)
    checks = {'microct_sample': '20240701; separate eye specimen, not the MaleCNS EM specimen',
        'source_axes_basis': {'vf': arr(source['vf']).tolist(), 'vleft': arr(source['vleft']).tolist(), 'vn': arr(source['vn']).tolist()},
        'source_code_basis_lines': 'upstream/proc_uCT.R:37-81 and 141-182',
        'fafb_pairs': len(out), 'source_eye_lens_grid_count': len(eye['lens_ixy']),
        'standalone_lens_grid_count': len(rdata.read_rda(HERE/'upstream/data/lens_ixy.RData')['lens_ixy']),
        'auxiliary_extrapolated_count': len(eye['eyemap_aux'])-len(eye['eyemap']), 'sides': {}}
    annotations = pd.read_feather(ANN).set_index('bodyId')
    for side, mask, sheet in [('left', isleft, 2), ('right', ~isleft, 1)]:
        a = xlsx_numeric(ATLAS/f'pqxyztp_{side}.xlsx');v=a[['x','y','z']].to_numpy()
        d, idx = cKDTree(axes[mask]).query(v)
        nearest_angles = angle(v, axes[mask][idx])
        dra = xlsx_numeric(HERE/'eyemap-archive/maps/eyemap_DRA_mcns_f20240701'/f'pqxyztp_{side}.xlsx')
        joined = a.merge(dra, on=['hex1','hex2'], suffixes=['_regular','_DRA'])
        latest = xlsx_numeric(HERE/'optic-column-type-assignments-v1.0.xlsx', sheet=sheet)
        matches = 0; mismatches = []; checked = []
        for _, row in latest.iterrows():
            bid = row['L1']
            if not np.isfinite(bid) or bid < 0 or int(bid) not in annotations.index: continue
            pieces = str(row['column']).split('_');expected=list(map(int,pieces[-2:]));ann=annotations.loc[int(bid)]
            got=[int(x) if np.isfinite(x) else None for x in [ann.assignedOlHex1,ann.assignedOlHex2]]
            checked.append({'body_id':int(bid),'column':str(row['column']),'official_hex':expected,'annotation_hex':got})
            if got==expected: matches+=1
            else: mismatches.append(checked[-1])
        dump(HERE/f'v1-official-L1-check-{side}.json',{'matched':matches,'mismatches':mismatches,'checked':checked})
        checks['sides'][side]={'microct_lenses':int(mask.sum()),'atlas_axes':len(a),
            'nearest_source_microct_axis_angle_deg':quantiles(nearest_angles),
            'source_axis_agreement_status':'Supports unchanged anatomical axis convention; nearest-neighbor similarity is not a column identity proof.',
            'theta_matches_acos_z_max_abs_deg':float(np.max(np.abs(a.theta-np.degrees(np.arccos(a.z))))),
            'phi_matches_negative_atan2_y_x_wrapped_max_abs_deg':float(np.max(np.abs((a.phi+np.degrees(np.arctan2(a.y,a.x))+180)%360-180))),
            'DRA_variant_common_columns':len(joined),
            'DRA_variant_axis_difference_deg':quantiles(angle(joined[['x_regular','y_regular','z_regular']],joined[['x_DRA','y_DRA','z_DRA']])),
            'v1_official_L1_column_table_matches':matches,'v1_official_L1_column_table_mismatches':len(mismatches)}
    dump(HERE/'source-geometry-checks.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
