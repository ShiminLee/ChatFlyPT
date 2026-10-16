"""Inspect actual input tables; report versions, rows, filters and file hashes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest}


def main():
    report = {'flywire': [], 'malecns': {}}
    vendor = ROOT / 'vendor/shiu-brain'
    for version, comp, con in [
        ('630', '2023_03_23_completeness_630_final.csv', '2023_03_23_connectivity_630_final.parquet'),
        ('783', 'Completeness_783.csv', 'Connectivity_783.parquet'),
    ]:
        neurons = pd.read_csv(vendor / comp, index_col=0)
        edges = pd.read_parquet(vendor / con)
        pre = edges['Presynaptic_Index'].to_numpy()
        post = edges['Postsynaptic_Index'].to_numpy()
        weights = edges['Excitatory x Connectivity'].to_numpy()
        assert neurons.index.is_unique
        assert min(pre.min(), post.min()) >= 0
        assert max(pre.max(), post.max()) < len(neurons)
        assert np.isfinite(weights).all()
        report['flywire'].append({
            'version': version, 'origin': 'philshiu/Drosophila_brain_model',
            'neurons': len(neurons), 'connection_rows': len(edges),
            'columns': list(edges.columns), 'neuron_columns': list(neurons.columns),
            'neuron_id_dtype': str(neurons.index.dtype),
            'positive_rows': int((weights > 0).sum()), 'negative_rows': int((weights < 0).sum()),
            'zero_rows': int((weights == 0).sum()),
            'sum_abs_model_weights': float(np.abs(weights).sum()),
            'index_bounds_valid': True, 'files': [fingerprint(vendor / comp), fingerprint(vendor / con)]})
        del neurons, edges, pre, post, weights
    raw = ROOT / 'data/raw/malecns-v1.0'
    lock = json.loads((ROOT / 'research/manifests/malecns.lock.json').read_text())
    paths = {r['id']: ROOT / r['local_path'] for r in lock}
    annotations = feather.read_table(paths['annotations'])
    nt = feather.read_table(paths['neurotransmitters'])
    ids = np.asarray(annotations['bodyId'], dtype=np.int64)
    keep = np.array([bool(x) for x in annotations['superclass'].to_pylist()])
    keep &= np.array([x != 'Glia' for x in annotations['status'].to_pylist()])
    retained = np.sort(ids[keep])
    assert len(np.unique(retained)) == len(retained)
    raw_rows = kept_rows = raw_contacts = kept_contacts = 0
    with pa.memory_map(str(paths['connections']), 'r') as mapped:
        reader = pa.ipc.open_file(mapped)
        schema = str(reader.schema)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre, post, count = [np.asarray(batch.column(batch.schema.get_field_index(c)))
                                for c in ['body_pre', 'body_post', 'weight']]
            assert (count >= 0).all()
            valid = np.isin(pre, retained) & np.isin(post, retained)
            raw_rows += len(pre)
            raw_contacts += int(count.sum(dtype=np.int64))
            kept_rows += int(valid.sum())
            kept_contacts += int(count[valid].sum(dtype=np.int64))
    report['malecns'] = {
        'version': 'v1.0', 'annotation_rows': len(ids), 'retained_neurons': len(retained),
        'filter': 'nonempty superclass and status != Glia; retain both edge endpoints',
        'annotation_schema': str(annotations.schema), 'nt_rows': nt.num_rows, 'nt_schema': str(nt.schema),
        'raw_connection_rows': raw_rows, 'raw_contact_sum': raw_contacts,
        'retained_connection_rows': kept_rows, 'retained_contact_sum': kept_contacts,
        'connection_schema': schema,
        'weight_interpretation': 'unsigned anatomical contact count, not fitted physiological weight',
        'files': [fingerprint(p) for p in paths.values()]}
    out = ROOT / 'reports/data-audit.json'
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'flywire': [{k:v for k,v in r.items() if k in ['version','neurons','connection_rows','index_bounds_valid']}
                    for r in report['flywire']],
        'malecns': {k:v for k,v in report['malecns'].items() if k.endswith('rows') or k.endswith('neurons') or k.endswith('sum')}
    }, indent=2))


if __name__ == '__main__':
    main()
