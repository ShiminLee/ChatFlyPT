"""Index a collected corpus as layer-separated, re-annotatable records.

Following documentary-linguistics practice, every window is stored as an
"annotated communicative event" with four separated layers plus metadata, so the
material can be re-annotated later without re-running the simulation:

  context        the applied scene and its revision (what the world was)
  measured       the receptor items actually delivered and their evidence
  internal       internal-activity summary (the only readout input)
  behaviour      measured body outcome from the recorded trajectory
  metadata       protocol/contract hashes, seeds, session id, record hashes

No layer may be substituted for another: the index never feeds measured receptors
back into a decoder, and never presents the applied scene as if it were received.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump, sha256
from chatflypt.intent_readout import measured_outcome

CORPUS = ROOT / 'data/derived/unified-readout/runs'
OUT = ROOT / 'reports/corpus'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    trials = CORPUS / args.run_id / 'trials'
    if not trials.is_dir():
        raise SystemExit('No corpus at ' + str(trials))

    windows = []
    for episode_dir in sorted(trials.glob('*')):
        complete = episode_dir / 'complete.json'
        if not complete.is_file():
            continue
        saved = json.loads(complete.read_text(encoding='utf8'))
        for row in saved['rows']:
            feedback = json.loads(Path(row['feedback']).read_text(encoding='utf8'))
            body = json.loads(Path(row['body']).read_text(encoding='utf8'))
            with np.load(row['record'], allow_pickle=False) as observation:
                counts = observation['counts']
                states = observation['state_record']
                duration = float(observation['duration_ms'])
            events = np.load(row['events'], allow_pickle=False)
            samples = feedback.get('samples', [])
            delivered = Counter()
            for sample in samples:
                for item in sample.get('items', []):
                    if item.get('intensity', 1.) > 0:
                        delivered[f"{item['channel']}:{item.get('side', 'bilateral')}"
                                  + (f"@{item.get('body_part')}" if item.get('body_part') else '')] += 1
            outcome = None
            if body.get('trajectory'):
                try:
                    outcome = measured_outcome(body['trajectory'])
                except (KeyError, TypeError, ValueError):
                    outcome = None
            windows.append({
                'id': f"{row['episode']}#turn{row['turn']}",
                'metadata': {'episode': row['episode'], 'turn': row['turn'], 'split': row['split'],
                             'session_id': row['session_id'], 'protocol_sha256': row['protocol_sha256'],
                             'contract_sha256': sha256_of(row['contract']),
                             'record': row['record'], 'record_sha256': row['record_sha256'],
                             'events_sha256': row['events_sha256'], 'body_sha256': row['body_sha256'],
                             'feedback_sha256': row['feedback_sha256'],
                             'session_start_ms': feedback.get('session_start_ms'),
                             'interval_ms': feedback.get('interval_ms'),
                             'duration_ms': duration},
                'context': {'environment_revision': feedback.get('environment_revision'),
                            'input_mode': feedback.get('input_mode'),
                            'objects': [obj.get('type') for obj in (feedback.get('environment') or {}).get('objects', [])],
                            'direct_neural_intervention': feedback.get('direct_neural_intervention'),
                            'llm_called': feedback.get('llm_called'),
                            'goal_controller': feedback.get('goal_controller')},
                'measured': {'delivered_items': dict(delivered),
                             'physical_presentation': feedback.get('physical_presentation'),
                             'proprioception_events': sum(p.get('encoded_event_count', 0)
                                                          for s in samples for p in [s.get('proprioception', {})] if p),
                             'input_events': int(len(events['event_steps'])) if 'event_steps' in events else 0,
                             'unmapped': [u.get('reason') for s in samples for u in s.get('unmapped', [])]},
                'internal': {'spikes': int(counts.sum()), 'active_cells': int((counts > 0).sum()),
                             'nonsensory_spikes': int(counts[feedback.get('nonsensory_mask', slice(None))].sum())
                                 if feedback.get('nonsensory_mask') else None,
                             'membrane_internal_cells': feedback.get('membrane_internal_cells'),
                             'state_record_shape': list(np.asarray(states).shape) if states is not None else None},
                'behaviour': {'outcome': outcome,
                              'body_status': body.get('status'), 'physics_run': body.get('physics_run'),
                              'physics_warnings': (body.get('summary') or {}).get('physics_warnings'),
                              'max_contacts': (body.get('summary') or {}).get('max_contacts'),
                              'controller': body.get('controller')}})
    summary = {'run_id': args.run_id, 'windows': len(windows),
               'episodes': len({w['metadata']['episode'] for w in windows}),
               'by_split': dict(Counter(w['metadata']['split'] for w in windows)),
               'conditions': dict(Counter(w['metadata']['episode'].split('-', 1)[1]
                                          for w in windows if '-' in w['metadata']['episode'])),
               'delivered_channels': dict(Counter(key.split('@')[0].split(':')[0]
                                                  for w in windows for key in w['measured']['delivered_items'])),
               'physics_failures': sum(1 for w in windows
                                       if not w['behaviour']['physics_run']
                                       or w['behaviour']['physics_warnings']),
               'interventions': sum(1 for w in windows if w['context']['direct_neural_intervention']),
               'llm_calls': sum(1 for w in windows if w['context']['llm_called']),
               'delivered_by_family': delivered_by_family(windows),
               'turn_amplitude_by_family': turn_amplitude_by_family(windows)}
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / f'{args.run_id}-index.json', {'schema': 'chatflypt.corpus-index.v1', 'summary': summary,
                                             'layers': {'context': 'applied scene and revision',
                                                        'measured': 'actually delivered receptor items',
                                                        'internal': 'internal-activity summary',
                                                        'behaviour': 'measured body outcome'},
                                             'windows': windows})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def condition_family(episode):
    """Coarse family from the condition name, for delivery and turn summaries."""
    name = episode.split('-', 1)[1] if '-' in episode else episode
    for family, markers in (('probe', ('touch_', 'tap', 'nudge', 'unilateral')),
                            ('platform', ('leg_sugar', 'leg_block', 'mouth_')),
                            ('odor', ('yeast', 'pheromone')),
                            ('song', ('song',)),
                            ('thermal', ('heat', 'cold', 'warm')),
                            ('visual', ('bright', 'dark'))):
        if any(marker in name for marker in markers):
            return family
    return 'other'


def delivered_by_family(windows):
    """Which channels actually arrived, per condition family."""
    families = {}
    for window in windows:
        family = condition_family(window['metadata']['episode'])
        entry = families.setdefault(family, {'windows': 0, 'windows_with_signal': 0, 'channels': Counter()})
        entry['windows'] += 1
        keys = list(window['measured']['delivered_items'])
        if keys:
            entry['windows_with_signal'] += 1
        for key in keys:
            entry['channels'][key.split('@')[0]] += 1
    for entry in families.values():
        entry['channels'] = dict(entry['channels'])
    return families


def turn_amplitude_by_family(windows):
    """Measured |yaw| and path per family: are there commanded turns to decode?"""
    families = {}
    for window in windows:
        outcome = window['behaviour']['outcome']
        if not outcome:
            continue
        family = condition_family(window['metadata']['episode'])
        entry = families.setdefault(family, {'windows': 0, 'abs_yaw': [], 'path': []})
        entry['windows'] += 1
        entry['abs_yaw'].append(abs(outcome['yaw_change_deg']))
        entry['path'].append(outcome['path_length_mm'])
    for entry in families.values():
        yaw = np.asarray(entry.pop('abs_yaw')); path = np.asarray(entry.pop('path'))
        entry['abs_yaw_deg'] = {'median': float(np.median(yaw)), 'p85': float(np.quantile(yaw, .85)),
                                'max': float(yaw.max()), 'over_5deg': int((yaw >= 5.).sum())}
        entry['path_mm'] = {'median': float(np.median(path)), 'max': float(path.max())}
    return families


def sha256_of(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    main()
