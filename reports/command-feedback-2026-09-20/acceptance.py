"""Four fresh command-codebook neural trials; no training or body claims."""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from chatflypt.brain import Brain
from chatflypt.command_feedback import CommandFeedback, MODEL_PATH
from chatflypt.paths import ARTIFACTS, sha256


def main():
    model_before = sha256(MODEL_PATH)
    manifest = json.loads((ARTIFACTS / 'stimuli.json').read_text(encoding='utf-8'))
    brain = Brain()
    feedback = CommandFeedback(brain, manifest)
    report = {'scope': 'Four new independent 300 ms neural trials; no retraining or physical/body execution.',
              'model_sha256': model_before, 'source': feedback.source, 'cases': []}
    # Not part of the saved 82000-series training or 83000-series held-out seeds.
    for action, seed in [('left', 9200001), ('right', 9200002), ('forward', 9200003), ('stop', 9200004)]:
        started = time.perf_counter()
        recorded = feedback.run(action, seed)
        trial = recorded['trial']
        decoded = recorded['decoder'].decode(trial.counts)
        np.testing.assert_array_equal(np.bincount(trial.spike_ids, minlength=len(brain.ids)), trial.counts)
        raw = HERE / (action + '-neural.npz')
        np.savez_compressed(raw, counts=trial.counts, neuron_ids=brain.ids.astype(str),
                            spike_indices=trial.spike_ids, spike_steps=trial.spike_steps,
                            driven_indices=recorded['driven'], seed=seed, duration_ms=trial.duration_ms)
        event_path = HERE / (action + '-inputs.npz')
        np.savez_compressed(event_path, **recorded['events'])
        row = {'action': action, 'seed': seed, 'duration_ms': trial.duration_ms,
               'decoded': decoded, 'command_matches_codebook': decoded['command'] == action,
               'spikes': int(trial.counts.sum()), 'active_neurons': int((trial.counts > 0).sum()),
               'readout_spikes': int(trial.counts[feedback.decoder.indices].sum()),
               'input_events': recorded['reception']['input_events'],
               'reception': recorded['reception'], 'encoding': recorded['encoding'],
               'raw_record': {'file': str(raw.relative_to(ROOT)), 'sha256': sha256(raw)},
               'input_record': {'file': str(event_path.relative_to(ROOT)), 'sha256': sha256(event_path)},
               'elapsed_seconds': time.perf_counter() - started}
        report['cases'].append(row)
        report['model_sha256_unchanged'] = sha256(MODEL_PATH) == model_before
        assert report['model_sha256_unchanged']
        if action == 'stop':
            assert decoded['status'] == 'silence' and decoded['words'] == []
            assert row['spikes'] == row['input_events'] == 0
        (HERE / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({key: row[key] for key in ['action', 'seed', 'spikes', 'readout_spikes',
                         'input_events', 'command_matches_codebook', 'elapsed_seconds']}), flush=True)
    report['complete'] = True
    (HERE / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
