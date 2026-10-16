"""Exercise the live HTTP pathway after freezing the neural-state probes.

These are integration checks, not a new classifier training/selection set.
The pathway is neural-only: no model request is made or accepted here.
"""
import argparse
import hashlib
import json
import time
import urllib.request

import numpy as np

from .paths import ROOT, dump, sha256
from .sensory_readout import targets
from .state_readout import OUT, StateDecoder, DATASETS


CASES = [
    ('visual_voltage', '/api/stimulate', {'items': [{'channel': 'vision_r7', 'side': 'left'}]}),
    ('ordinary_sugar', '/api/communicate', {'message': '模拟左侧甜味'}),
    ('ordinary_water', '/api/communicate', {'message': '模拟右侧水'}),
    ('high_salt', '/api/stimulate', {'items': [{'channel': 'high_salt', 'side': 'left'}]}),
    ('hexanoic_candidate', '/api/stimulate', {'items': [{'channel': 'fatty_acid_taste', 'side': 'right'}]}),
    ('cross_dataset_and_body', '/api/stimulate', {'items': [
        {'channel': 'cold', 'side': 'left'}, {'channel': 'foreleg_touch', 'side': 'right'}]}),
    ('zero_intensity', '/api/stimulate', {'items': [{'channel': 'bitter', 'side': 'left', 'intensity': 0}]}),
    ('blank', '/api/communicate', {'message': '停止刺激'}),
    ('unmapped', '/api/communicate', {'message': '施加饥饿刺激'}),
]


def verify_saved_body_sources(report):
    """Tie each returned body to the exact saved neural spike sequence."""
    checked = 0
    for row in report['cases']:
        if not row['observations']: continue
        primary_id = row['observations'][0]['id']
        saved = json.loads((ROOT / 'reports/general-sensory/interactive' /
                            f'{primary_id}.json').read_text(encoding='utf8'))
        saved_by_id = {obs['id']: obs for obs in saved['observations']}
        for obs in row['observations']:
            body = saved_by_id[obs['id']]['scene']['body']
            digest = hashlib.sha256()
            with np.load(ROOT / obs['raw_record']['file'], allow_pickle=False) as raw:
                for values in (raw['neuron_ids'], raw['spike_indices'].astype(np.int64), raw['spike_steps']):
                    digest.update(np.ascontiguousarray(values).tobytes())
            assert body['source_spikes_sha256'] == digest.hexdigest()
            assert body['dataset'] == obs['dataset']
            obs['body'].update(neural_source_verified=True, source_spikes_sha256=digest.hexdigest())
            checked += 1
    report['summary']['body_neural_sources_verified'] = checked


def run(port=8766):
    directory = ROOT / 'reports/gap-closure'
    catalogue = json.loads((ROOT / 'data/derived/sensory/catalogue.json').read_text(encoding='utf8'))
    models = {dataset: StateDecoder(dataset) for dataset in DATASETS.values()}
    hashes = {key: sha256(OUT / f'{key}-decoder.npz') for key in DATASETS}
    dump(directory / 'acceptance-protocol.json', {'cases': CASES, 'seed_base': 5900000,
        'model_sha256': hashes,
        'purpose': 'HTTP integration and provenance; classifier not tuned using these results.'})

    def post(route, payload):
        req = urllib.request.Request(f'http://127.0.0.1:{port}{route}',
            json.dumps(payload, ensure_ascii=False).encode('utf8'),
            {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            assert response.status == 200
            return json.load(response)

    rows = []; body_runs = 0; voltage_only = 0; last_id = ''
    for index, (name, route, payload) in enumerate(CASES):
        started = time.perf_counter()
        result = post(route, dict(payload, seed=5900000 + index))
        row = {'name': name, 'http_status': 200, 'handled': result['handled'],
               'wall_seconds': time.perf_counter() - started, 'reply': result['reply'], 'observations': []}
        if name == 'unmapped':
            assert not result['handled'] and not result['neural_simulation_run']
            assert result['reply_source'] == 'input_status'
            row['rejected_without_execution'] = True
        else:
            assert result['handled'] and result['pathway']['completed']
            assert result['reply_source'] == 'neural_readout'
            if name == 'cross_dataset_and_body':
                assert result['pathway']['independent_datasets'] and len(result['observations']) == 2
            for obs in result['observations']:
                assert obs['readout_model'] == 'general_sensory_state'
                model = models[obs['dataset']]
                raw_path = ROOT / obs['raw_record']['file']
                assert sha256(raw_path) == obs['raw_record']['sha256']
                with np.load(raw_path, allow_pickle=False) as raw:
                    counts, state = raw['counts'], raw['state_record']
                    np.testing.assert_array_equal(np.bincount(raw['spike_indices'], minlength=len(counts)), counts)
                    decoded = model.decode(counts, state)
                    assert decoded == obs['decoded']
                    assert not np.intersect1d(model.indices, raw['driven_indices']).size
                    if name in ['zero_intensity', 'blank']:
                        assert not counts.any() and not state.any() and not decoded['words']
                if 'event_record' in obs:
                    path = ROOT / obs['event_record']['file']
                    assert sha256(path) == obs['event_record']['sha256']
                    with np.load(path, allow_pickle=False) as events:
                        assert str(events['dataset']) == obs['dataset']
                        assert set(events['event_neuron_ids']) <= set(events['input_neuron_ids'])
                body = obs['scene']['body']
                assert body['status'] in ['ok', 'no_motor_activity'], body.get('reason', body)
                assert bool(body.get('physics_run')) == (body['status'] == 'ok')
                if body.get('physics_run'):
                    assert body['status'] == 'ok' and body['trajectory']
                    assert body['summary']['finite_state'] and not body['summary']['physics_warnings']
                    saved = ROOT / 'data/derived/body-trials' / obs['id'] / 'body-result.json'
                    # The body worker records the exact frames exposed by HTTP.
                    stored = json.loads(saved.read_text(encoding='utf8'))
                    assert stored['trajectory'] == body['trajectory']
                    body_runs += 1
                if name in ['zero_intensity', 'blank']:
                    assert body['status'] == 'no_motor_activity' and not body.get('physics_run')
                    assert not body['trajectory']
                membrane = obs['reaction'].get('membrane_response', {})
                if not obs['reaction']['non_sensory_spikes'] and membrane.get('internal_cells', 0):
                    voltage_only += 1
                expected = [label for label, yes in zip(model.labels,
                    targets({'items': [item for item in obs['reception']['items'] if item['intensity'] > 0]},
                            catalogue, model.labels)) if yes]
                row['observations'].append({'id': obs['id'], 'dataset': obs['dataset'],
                    'input_items': obs['reception']['items'], 'readout_model': obs['readout_model'],
                    'expected_intervention_labels': expected, 'decoded_labels': decoded['words'],
                    'exact_label_match': set(expected) == set(decoded['words']),
                    'raw_replay_matches': True, 'input_cells_excluded': True,
                    'non_sensory_spikes': obs['reaction']['non_sensory_spikes'],
                    'membrane_response': membrane, 'raw_record': obs['raw_record'],
                    'body': {k: body.get(k) for k in ['status', 'physics_run', 'duration_ms', 'summary']}})
                if obs['reaction']['non_sensory_spikes'] or membrane.get('internal_cells', 0):
                    last_id = obs['id']
        rows.append(row)
        dump(directory / 'acceptance.json', {'in_progress': True, 'cases': rows})
        print(name, 'passed', round(row['wall_seconds'], 2), 'seconds', flush=True)
    assert body_runs > 0 and voltage_only > 0
    assert all(sha256(OUT / f'{key}-decoder.npz') == value for key, value in hashes.items())
    report = {'in_progress': False, 'summary': {'http_cases_passed': len(rows),
        'observations_replayed': sum(len(row['observations']) for row in rows),
        'physical_runs': body_runs, 'voltage_only_observations': voltage_only,
        'models_unchanged': True},
        'model_sha256': hashes, 'cases': rows}
    verify_saved_body_sources(report)
    dump(directory / 'acceptance.json', report)
    print(json.dumps(report['summary']), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    run(args.port)
