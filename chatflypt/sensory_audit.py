"""Run every executable registry input against its actual full connectome.

Delivery is distinct from perception/behavior: the audit measures whether a
specified external tape evokes nonsensory model activity. Zero downstream
responses and exceptions are retained rather than tuning the stimulus.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .brain import Brain, simulate
from .malecns import MaleCNS
from .paths import ROOT, dump, sha256
from .sensory_catalogue import OUT, FW, MC, build_catalogue
from .sensory_events import encode_inputs
from .social import to_engine_tape
from .stimuli import ANNOTATIONS

REPORT = ROOT / 'reports/general-sensory'


def _atomic_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    dump(temporary, value)
    os.replace(temporary, path)


def _atomic_save(path, **values):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def audit(workers=2):
    if not 1 <= workers <= 6:
        raise ValueError('Audit workers must be between one and six')
    catalogue_path = OUT / 'catalogue.json'
    catalogue = json.loads(catalogue_path.read_text(encoding='utf8')) if catalogue_path.exists() else build_catalogue()
    sources = {name: sha256(ROOT / 'chatflypt' / name) for name in
               ['sensory_catalogue.py', 'sensory_events.py', 'brain.py', 'malecns.py']}
    digest = hashlib.sha256((json.dumps(catalogue, sort_keys=True) + json.dumps(sources, sort_keys=True)).encode()).hexdigest()
    cache = OUT / 'channel-audit' / digest[:16]
    cache.mkdir(parents=True, exist_ok=True)
    specs = sorted((s for s in catalogue['channels'].values() if s['status'] in ['ready', 'candidate']),
                   key=lambda s: (s['dataset'], s['id']))
    protocol = {'name': 'all-executable-sensory-delivery-v1', 'duration_ms': 500.,
                'side': 'bilateral', 'intensity': 1., 'onset_ms': 50., 'input_duration_ms': 300.,
                'seed_rule': '250000 plus index in sorted(dataset, channel_id)',
                'fingerprint': digest, 'catalogue_sha256': sha256(catalogue_path),
                'code_sha256': sources,
                'cases': [{'channel': spec['id'], 'dataset': spec['dataset'], 'seed': 250000 + i}
                          for i, spec in enumerate(specs)],
                'criterion': 'At least one external event, at least one driven-cell spike, and at least one nonsensory spike.',
                'scope': 'Model input delivery only; no identification accuracy, behavioral validation, or physical calibration.'}
    _atomic_dump(REPORT / 'channel-delivery-protocol.json', protocol)
    results = []
    start = time.perf_counter()
    # Each graph is loaded once. Process datasets sequentially to keep memory
    # bounded while a separate readout training job may use the same machine.
    for dataset in [FW, MC]:
        selected_cases = [c for c in protocol['cases'] if c['dataset'] == dataset]
        if not selected_cases:
            continue
        brain = Brain() if dataset == FW else MaleCNS()
        if dataset == FW:
            table = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
            rows = table.set_index(table.root_id.astype(str)).reindex(brain.ids.astype(str)).fillna('')
            nonsensory = ~rows.super_class.str.contains('sensory').to_numpy()
        else:
            nonsensory = brain.nonsensory
        print(f'Audit {dataset}: {len(selected_cases)} channels, {len(brain.ids)} neurons, {len(brain.indices)} edges', flush=True)

        def run(case):
            cid, seed = case['channel'], case['seed']
            spec = catalogue['channels'][cid]
            item = {'channel': cid, 'side': 'bilateral', 'intensity': 1.,
                    'onset_ms': 50., 'duration_ms': 300.}
            stamp = time.perf_counter()
            row = {**case, 'status': spec['status'], 'advanced': spec['advanced'],
                   'input_events': 0, 'driven_count': 0, 'driven_spikes': 0,
                   'non_sensory_spikes': 0, 'active_internal_neurons': 0,
                   'input_delivery_passed': False, 'failure_reason': None}
            try:
                events = encode_inputs(catalogue, [item], seed, duration_ms=500.)[dataset]
                tape = to_engine_tape(brain, events)
                path = cache / (hashlib.sha256(cid.encode()).hexdigest()[:16] + '.npz')
                if path.exists():
                    with np.load(path, allow_pickle=False) as f:
                        counts = f['counts']
                else:
                    counts = simulate(brain.indptr, brain.indices, brain.weights, *tape, False)[0]
                    _atomic_save(path, counts=counts, seed=seed,
                                 input_neuron_ids=events['input_neuron_ids'],
                                 event_neuron_ids=events['event_neuron_ids'], event_steps=events['event_steps'])
                mask = nonsensory.copy()
                mask[tape[0]] = False
                row.update(input_events=len(events['event_steps']), driven_count=len(tape[0]),
                           driven_spikes=int(counts[tape[0]].sum()),
                           non_sensory_spikes=int(counts[mask].sum()),
                           active_internal_neurons=int(np.count_nonzero(counts[mask])),
                           total_spikes=int(counts.sum()),
                           raw_record=str(path.relative_to(ROOT)))
                if row['input_events'] == 0:
                    row['failure_reason'] = 'The fixed random draw produced no external input events.'
                elif row['driven_spikes'] == 0:
                    row['failure_reason'] = 'External events were generated but driven cells recorded no spikes.'
                elif row['non_sensory_spikes'] == 0:
                    row['failure_reason'] = 'No nonsensory activity at the fixed default stimulus; anatomy alone does not guarantee model propagation.'
                else:
                    row['input_delivery_passed'] = True
            except Exception as exc:
                row['failure_reason'] = f'{type(exc).__name__}: {exc}'
            row['elapsed_seconds'] = time.perf_counter() - stamp
            return row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, c) for c in selected_cases]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 16 == 0 or len(results) == len(specs):
                    completed = {'status': 'running', 'protocol': protocol,
                                 'completed': len(results), 'total': len(specs),
                                 'cases': sorted(results, key=lambda r: (r['dataset'], r['channel']))}
                    _atomic_dump(REPORT / 'channel-delivery.json', completed)
                    print(f'Audit {len(results)}/{len(specs)}; failures {sum(not r["input_delivery_passed"] for r in results)}; {time.perf_counter()-start:.1f}s', flush=True)
        del brain

    results.sort(key=lambda r: (r['dataset'], r['channel']))
    by_dataset = {}
    for dataset in [FW, MC]:
        rows = [r for r in results if r['dataset'] == dataset]
        by_dataset[dataset] = {'channels': len(rows),
                              'delivery_passed': sum(r['input_delivery_passed'] for r in rows),
                              'no_internal_response': sum(r['non_sensory_spikes'] == 0 for r in rows),
                              'failures': [{'channel': r['channel'], 'reason': r['failure_reason']}
                                           for r in rows if not r['input_delivery_passed']]}
    report = {'status': 'completed', 'protocol': protocol, 'total': len(results),
              'delivery_passed': sum(r['input_delivery_passed'] for r in results),
              'by_dataset': by_dataset, 'elapsed_seconds': time.perf_counter() - start,
              'cases': results,
              'unmapped_channels_not_run': [s['id'] for s in catalogue['channels'].values() if s['status'] == 'unmapped']}
    _atomic_dump(REPORT / 'channel-delivery.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'total', 'delivery_passed', 'elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run'])
    parser.add_argument('--workers', type=int, default=2)
    audit(parser.parse_args().workers)
