"""Read-only provenance diagnosis for the frozen zero-response cases.

Run from the repository root using its .venv Python. No graphs are changed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chatflypt.malecns import MALE, EDGES, TRANSMITTERS, SIGN
from chatflypt.paths import sha256
from chatflypt.stimuli import ANNOTATIONS

REPORT = ROOT / 'reports/gap-closure'
cases = json.loads((REPORT / 'zero-response-diagnosis.json').read_text(encoding='utf-8'))
for case in cases:
    with np.load(ROOT / case['raw_record'], allow_pickle=False) as data:
        case['input_neuron_ids'] = data['input_neuron_ids'].astype(np.int64).tolist()

male = pd.read_feather(MALE).fillna('')
male.index = male.bodyId
valid_male = pd.Index(male.loc[male.superclass.ne(''), 'bodyId'].astype(np.int64))
nt = pd.read_feather(TRANSMITTERS).fillna('').set_index('body')
fly = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
fly.index = fly.root_id

records = {}
source_stats = {}
for dataset in ['MaleCNS v1.0', 'FlyWire v783']:
    ids = sorted({i for case in cases if case['dataset'] == dataset for i in case['input_neuron_ids']})
    index = pd.Index(np.array(ids, np.int64))
    fields = ['raw_outgoing_rows', 'raw_outgoing_weight', 'raw_incoming_rows', 'raw_incoming_weight',
              'positive_outgoing_rows', 'annotated_target_rows', 'annotated_target_weight',
              'outside_graph_target_rows', 'outside_graph_target_weight', 'modeled_sign_rows',
              'modeled_sign_weight']
    counters = {key: np.zeros(len(ids), np.int64) for key in fields}
    if dataset.startswith('Male'):
        path = EDGES
        handle = pa.memory_map(str(path), 'r')
        reader = pa.ipc.open_file(handle)
        batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
        pre_col, post_col, weight_col = 'body_pre', 'body_post', 'weight'
    else:
        path = ROOT / 'vendor/shiu-brain/Connectivity_783.parquet'
        reader = pq.ParquetFile(path)
        batches = reader.iter_batches(batch_size=1000000,
            columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity', 'Excitatory x Connectivity'])
        pre_col, post_col, weight_col = 'Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity'
        valid_fly = pd.Index(pd.read_csv(ROOT / 'vendor/shiu-brain/Completeness_783.csv', index_col=0).index.astype(np.int64))
    scanned = 0
    for batch in batches:
        pre = batch.column(pre_col).to_numpy()
        post = batch.column(post_col).to_numpy()
        weight = batch.column(weight_col).to_numpy()
        scanned += len(pre)
        out = index.get_indexer(pre)
        inc = index.get_indexer(post)
        out_mask, in_mask = out >= 0, inc >= 0
        for field, positions, values in [
            ('raw_outgoing_rows', out[out_mask], None),
            ('raw_outgoing_weight', out[out_mask], weight[out_mask]),
            ('raw_incoming_rows', inc[in_mask], None),
            ('raw_incoming_weight', inc[in_mask], weight[in_mask]),
        ]:
            counters[field] += np.bincount(positions, weights=values, minlength=len(ids)).astype(np.int64)
        selected = np.flatnonzero(out_mask)
        if not len(selected):
            continue
        positive = selected[weight[selected] > 0]
        counters['positive_outgoing_rows'] += np.bincount(out[positive], minlength=len(ids))
        valid_index = valid_male if dataset.startswith('Male') else valid_fly
        in_graph = valid_index.get_indexer(post[positive]) >= 0
        for label, subset in [('annotated_target', positive[in_graph]), ('outside_graph_target', positive[~in_graph])]:
            counters[label + '_rows'] += np.bincount(out[subset], minlength=len(ids))
            counters[label + '_weight'] += np.bincount(out[subset], weights=weight[subset], minlength=len(ids)).astype(np.int64)
        selected = positive[in_graph]
        if dataset.startswith('Male'):
            signs = nt.reindex(pre[selected]).consensus_nt.map(SIGN).fillna(0).to_numpy()
            modeled = selected[signs != 0]
        else:
            modeled = selected[batch.column('Excitatory x Connectivity').to_numpy()[selected] != 0]
        counters['modeled_sign_rows'] += np.bincount(out[modeled], minlength=len(ids))
        counters['modeled_sign_weight'] += np.bincount(out[modeled], weights=weight[modeled], minlength=len(ids)).astype(np.int64)
    if dataset.startswith('Male'):
        handle.close()
    source_stats[dataset] = {'source': str(path.relative_to(ROOT)), 'scanned_rows': scanned,
                            'source_sha256': sha256(path)}
    records[dataset] = {}
    for j, body in enumerate(ids):
        row = {'neuron_id': str(body), **{key: int(val[j]) for key, val in counters.items()}}
        if dataset.startswith('Male'):
            row['annotation'] = male.loc[body, ['type', 'superclass', 'class', 'subclass', 'rootSide', 'entryNerve', 'statusLabel']].to_dict()
            row['neurotransmitter_record'] = nt.loc[body].to_dict() if body in nt.index else None
        else:
            row['annotation'] = fly.loc[body, ['cell_type', 'super_class', 'cell_class', 'cell_sub_class', 'side', 'nerve', 'known_nt']].to_dict()
        records[dataset][str(body)] = row

result = {'scope': 'Read-only diagnosis of original zero-response input groups; no neurotransmitter sign or graph changes.',
          'selection_source': 'reports/gap-closure/zero-response-diagnosis.json',
          'sources': source_stats, 'modeled_male_signs': SIGN, 'channels': []}
for case in cases:
    rows = [records[case['dataset']][str(i)] for i in case['input_neuron_ids']]
    summary = {key: sum(row[key] for row in rows) for key in fields}
    row = {'channel': case['channel'], 'dataset': case['dataset'], 'input_cells': len(rows),
           **summary, 'cells': rows}
    if not summary['raw_outgoing_rows']:
        row['cause'] = 'No outgoing rows in the available source connectome table; not a sign-filter loss.'
    elif not summary['annotated_target_rows']:
        row['cause'] = 'Source outgoing rows exist, but none have positive weight and a target in the selected modeled graph.'
    elif not summary['modeled_sign_rows']:
        row['cause'] = 'Anatomical outgoing edges exist; all retained-target edges lack an implemented nonzero fast transmitter sign.'
    else:
        row['cause'] = 'Source rows predict modeled edges; investigate graph cache mismatch.'
    result['channels'].append(row)
    print(json.dumps({key: value for key, value in row.items() if key != 'cells'}, ensure_ascii=False), flush=True)
target = REPORT / 'zero-outgoing-provenance.json'
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target, flush=True)
