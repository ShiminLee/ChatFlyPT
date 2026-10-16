"""Rebuild identity/graph evidence and a bounded neural smoke diagnostic.

Run from the project root with .venv/Scripts/python.exe -m
reports.malecns-senses-20260924.audit (or pass the script with PYTHONPATH=. ).
These direct interventions diagnose the new populations, not normal chat.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from chatflypt.brain import recorded_trial
from chatflypt.malecns import MaleCNS
from chatflypt.malecns_senses import CHANNEL_MAP, DATASET
from chatflypt.paths import dump, sha256
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.sensory_events import encode_inputs
from chatflypt.social import MALE, to_engine_tape


def main():
    out = Path(__file__).resolve().parent
    catalogue = build_catalogue(save=False)
    specs = {cid: catalogue['channels'][cid] for cid in CHANNEL_MAP.values()}
    dump(out / 'populations.json', {'dataset': DATASET, 'channel_map': CHANNEL_MAP,
        'annotation_sha256': sha256(MALE), 'channels': specs})
    rows = pd.read_feather(MALE)
    both = rows[['assignedOlHex1', 'assignedOlHex2']].notna().all(axis=1)
    visual = rows[rows.superclass.fillna('').str.contains('sensory') & rows['class'].eq('visual')]
    dump(out / 'retinotopy-inventory.json', {
        'annotation_sha256': sha256(MALE),
        'available_fields': list(rows.columns),
        'cells_with_both_hex_fields': int(both.sum()),
        'types_with_both_hex_fields': rows[both].groupby('type').size().to_dict(),
        'visual_receptors': len(visual),
        'visual_receptors_with_hex_fields': int(visual[['assignedOlHex1', 'assignedOlHex2']].notna().all(axis=1).sum()),
        'status': 'No direct photoreceptor-column assignment in this annotation table; no image-to-neuron registration claimed.',
        'resources': [
            'https://www.nature.com/articles/s41586-025-08746-0',
            'https://github.com/reiserlab/male-drosophila-visual-system-connectome-code',
            'https://raw.githubusercontent.com/reiserlab/male-drosophila-visual-system-connectome-code/main/docs/coordinate-systems.md',
            'https://www.nature.com/articles/s41586-025-09276-5',
            'https://github.com/reiserlab/eyemap_T4',
        ]})
    brain = MaleCNS()
    results = []
    for k, channels in enumerate([['mc_bright'], ['mc_yeast'], ['mc_warm'], ['mc_cold'],
                                  ['mc_bright', 'mc_yeast', 'mc_warm', 'foreleg_touch']]):
        items = [{'channel': c, 'side': 'left', 'onset_ms': 0., 'duration_ms': 100., 'intensity': 1.}
                 for c in channels]
        events = encode_inputs(catalogue, items, 801927 + k, duration_ms=200.)[DATASET]
        trial = recorded_trial(brain, to_engine_tape(brain, events), 801927 + k, 200., True)
        mask = brain.nonsensory.copy()
        mask[[brain.lookup[str(i)] for i in events['input_neuron_ids']]] = False
        row = {'channels': channels, 'seed': 801927 + k,
               'duration_ms': 200., 'input_events': len(events['event_steps']),
               'internal_spikes': int(trial.counts[mask].sum()),
               'internal_active_cells': int(np.count_nonzero(trial.counts[mask])),
               'internal_voltage_response_cells': int(np.count_nonzero(np.max(trial.state_record[2:, mask], axis=0) > 1e-6)),
               'finite_state': bool(np.isfinite(trial.state_record).all()),
               'dataset': DATASET, 'namespace': 'malecns_body_id',
               'normal_chat_acceptance': False}
        results.append(row)
        np.savez_compressed(out / f'neural-smoke-{k}.npz', counts=trial.counts,
            state=trial.state_record, spike_ids=trial.spike_ids, spike_steps=trial.spike_steps,
            input_neuron_ids=events['input_neuron_ids'],
            event_neuron_ids=events['event_neuron_ids'], event_steps=events['event_steps'])
        print(json.dumps(row, ensure_ascii=False), flush=True)
    dump(out / 'neural-smoke.json', {
        'scope': 'Independent labeled direct-neural diagnostics; not environmental or behavioral validation.',
        'graph_sha256': specs['mc_bright']['graph_audit']['graph_sha256'],
        'transmitter_signs_changed': False, 'background_drive_added': False,
        'tests': results})


if __name__ == '__main__':
    main()
