"""Answer the preregistered rhythm question: are equal-contact-time tap trains distinguishable?

The elicitation battery asks whether three rhythms with the SAME total contact
time (1x180 ms, 3x60 ms, 6x30 ms) can be told apart. Total contact time is held
fixed precisely so that any difference cannot be explained by "more contact".

Two levels are reported, both from measurements:
  * delivery  - how many contact bouts the probe actually produced (physics)
  * internal  - internal-activity statistics and the fitted readout's score for
                the touched leg (the same blind readout used in the chat)

A difference is only claimed when the rhythm groups separate on a measured
quantity; otherwise the honest result is "not distinguishable in this corpus".
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.malecns import MaleCNS
from chatflypt.unified_readout import load_unified_decoder

CORPUS = ROOT / 'data/derived/unified-readout/runs'
RHYTHMS = ('tap1x180', 'tap3x60', 'tap6x30')


def probe_bouts(feedback):
    """Derive the measured contact-bout count and requested tap count from a turn.

    A bout starts in a sampled window where the probe key appears after at least
    one sampled window without it, so a tap train yields several bouts. The
    requested count comes from the recorded probe program.
    """
    previous, bouts = set(), 0
    for sample in feedback.get('samples', []):
        active = {(e.get('object_id'), e.get('side'), e.get('body_part'))
                  for e in sample.get('evidence', [])
                  if e.get('mode') == 'mujoco_leg_contact' and str(e.get('object_id', '')).startswith('message_probe_')}
        bouts += len(active - previous)
        previous = active
    program = (feedback.get('physical_presentation') or {}).get('program') or []
    requested = sum(int(entry.get('taps', 1)) for entry in program)
    return bouts, requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v4')
    args = parser.parse_args()
    trials = CORPUS / args.run_id / 'trials'
    brain, catalogue = MaleCNS(), build_catalogue(save=False)
    decoder, note = load_unified_decoder(brain, catalogue)
    if decoder is None:
        raise SystemExit('No published decoder to score with: ' + json.dumps(note))

    groups = defaultdict(list)
    for episode_dir in sorted(trials.glob('*')):
        name = episode_dir.name
        rhythm = next((r for r in RHYTHMS if r in name), None)
        if rhythm is None:
            continue
        side = 'right' if name.endswith('_right') else 'left'
        split = name.split('-', 1)[0]
        complete = episode_dir / 'complete.json'
        if not complete.is_file():
            continue
        for row in json.loads(complete.read_text(encoding='utf8'))['rows']:
            feedback = json.loads(Path(row['feedback']).read_text(encoding='utf8'))
            bouts, requested = probe_bouts(feedback)
            with np.load(row['record'], allow_pickle=False) as saved:
                counts = saved['counts']
                state = saved['state_record']
                duration = float(saved['duration_ms'])
            decoded = decoder.decode(counts, state, duration)
            label = f'foreleg_touch:{side}'
            groups[rhythm].append({
                'episode': name, 'split': split, 'side': side, 'turn': row['turn'],
                'measured_contact_bouts': bouts,
                'requested_taps': requested,
                'spikes': int(counts.sum()),
                'active_cells': int((counts > 0).sum()),
                'score': float(decoded['scores'].get(label, float('nan'))),
                'spoken': label in decoded['words'],
            })

    summary, verdict = {}, {}
    for rhythm, rows in sorted(groups.items()):
        bouts = [r['measured_contact_bouts'] for r in rows if r['measured_contact_bouts'] is not None]
        scores = [r['score'] for r in rows if np.isfinite(r['score'])]
        spikes = [r['spikes'] for r in rows]
        summary[rhythm] = {
            'windows': len(rows),
            'requested_taps': sorted({r['requested_taps'] for r in rows}),
            'measured_contact_bouts': {'min': min(bouts), 'median': statistics.median(bouts),
                                       'max': max(bouts)} if bouts else None,
            'spikes': {'min': min(spikes), 'median': statistics.median(spikes), 'max': max(spikes)},
            'foreleg_touch_score': {'min': round(min(scores), 3), 'median': round(statistics.median(scores), 3),
                                    'max': round(max(scores), 3)} if scores else None,
            'spoken_windows': sum(r['spoken'] for r in rows),
        }
    # Separation is judged on the score for the touched leg: do the rhythm groups
    # overlap, or does one group sit entirely above/below another?
    scored = {r: [row['score'] for row in rows if np.isfinite(row['score'])] for r, rows in groups.items()}
    pairs = [(a, b) for i, a in enumerate(RHYTHMS) for b in RHYTHMS[i + 1:]]
    for a, b in pairs:
        first, second = scored.get(a, []), scored.get(b, [])
        if not first or not second:
            verdict[f'{a} vs {b}'] = 'insufficient windows'
            continue
        overlap = not (max(first) < min(second) or max(second) < min(first))
        verdict[f'{a} vs {b}'] = ('ranges overlap; not separated in this corpus' if overlap
                                  else 'ranges are disjoint; separated on the readout score')
    report = {'schema': 'chatflypt.tap-rhythm-analysis.v1', 'run_id': args.run_id,
              'question': 'Equal total contact time (180 ms), three rhythms: distinguishable?',
              'criterion': 'Report separation on measured bout counts and on the blind readout score for the touched leg; '
                           'no separation is reported as such.',
              'decoder': decoder.artifact, 'decoder_sha256': decoder.artifact_sha256,
              'per_rhythm': summary, 'comparisons': verdict,
              'limitations': ['Windows within a session are correlated; a few sessions per rhythm.',
                              'Contact bouts are counted from 10 ms sampled contacts and can miss very short taps.',
                              'The readout score is a linear neural-state score, not a probability.']}
    dump(ROOT / 'reports/corpus' / f'{args.run_id}-tap-rhythms.json', report)
    print(json.dumps({'per_rhythm': summary, 'comparisons': verdict}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
