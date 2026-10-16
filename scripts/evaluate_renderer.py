"""Evaluate the compositional renderer over a corpus: variety, grounding, abstention.

For every window of the test split this script
  1. takes the *measured* facts (delivered receptor items, measured internal
     activity band, measured delivery fraction),
  2. decodes the slots from blind internal activity with the published decoders
     (unified channel labels + slot readout),
  3. renders one utterance with `generative_renderer.render_state`, and
  4. audits the utterance against the measurements.

Reported: how many *distinct* sentences appear (the whole point of composing
instead of templating), the invented/unsupported/contradicted slot counts, and how
often the system abstains. A contradiction is a decoded slot whose measured value
disagrees; they are reported, never hidden.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.malecns import MaleCNS
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.unified_readout import blind_mask, load_unified_decoder
from chatflypt.slot_readout import (ACTIVITY_LABELS, SlotDecoder, activity_labels,
                                    load_slot_decoders, pattern_labels, slot_rules)
from chatflypt.generative_renderer import audit_grounding, render_state
from chatflypt.dialogue import sensory_report

CORPUS = ROOT / 'data/derived/unified-readout/runs'
REPORT = ROOT / 'reports/generative-renderer'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v5')
    parser.add_argument('--slot-artifact', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    brain, catalogue = MaleCNS(), build_catalogue(save=False)
    allowed = blind_mask(brain, catalogue)
    unified, note = load_unified_decoder(brain, catalogue)
    if unified is None:
        raise SystemExit('No published decoder: ' + json.dumps(note, ensure_ascii=False))
    slots_decoder, slot_note = load_slot_decoders(brain, catalogue, args.slot_artifact)
    if slots_decoder is None:
        raise SystemExit('No published slot readout: ' + json.dumps(slot_note, ensure_ascii=False))

    trials = CORPUS / args.run_id / 'trials'
    rows = []
    for episode in sorted(trials.glob('*')):
        complete = episode / 'complete.json'
        if complete.is_file():
            rows.extend(json.loads(complete.read_text(encoding='utf8'))['rows'])
    # Slot bands are fixed by the training split of the same corpus.
    train_facts = []
    for row in rows:
        if row['split'] != 'train':
            continue
        with np.load(row['record'], allow_pickle=False) as saved:
            counts = saved['counts']
        feedback = json.loads(Path(row['feedback']).read_text(encoding='utf8'))
        samples = feedback.get('samples', [])
        delivered = sum(1 for sample in samples
                        if any(item.get('intensity', 1.) > 0 for item in sample.get('items', [])))
        train_facts.append({'delivery_fraction': delivered / max(1, len(samples)),
                            'nonsensory_spikes': float(np.asarray(counts)[allowed].sum())})
    rules = slot_rules(train_facts)

    sentences, contradictions, unsupported, invented, abstained, windows = Counter(), [], 0, 0, 0, 0
    slot_histogram = Counter()
    for row in rows:
        if row['split'] != 'test':
            continue
        if args.limit and windows >= args.limit:
            break
        windows += 1
        with np.load(row['record'], allow_pickle=False) as saved:
            counts = saved['counts']; state = saved['state_record']; duration = float(saved['duration_ms'])
        feedback = json.loads(Path(row['feedback']).read_text(encoding='utf8'))
        samples = feedback.get('samples', [])
        delivered_items, delivered_samples = {}, 0
        for sample in samples:
            hit = False
            for item in sample.get('items', []):
                if item.get('intensity', 1.) > 0:
                    key = f"{item['channel']}:{item.get('side', 'bilateral')}"
                    delivered_items[key] = delivered_items.get(key, 0) + 1
                    hit = True
            delivered_samples += hit
        measured = {'delivered_items': delivered_items,
                    'activity_band': activity_labels(float(np.asarray(counts)[allowed].sum()), rules)[0],
                    'pattern_band': pattern_labels(delivered_samples / max(1, len(samples)), rules)[0]}
        decoded = unified.decode(counts, state, duration)
        slot_out = slots_decoder.decode(counts, state, duration)
        report = sensory_report(decoded, {'non_sensory_spikes': int(np.asarray(counts)[allowed].sum()),
                                          'membrane_response': {'internal_cells': 0}})
        feelings = [f for f in report['feelings'] if f['channel'] != 'command']
        activity = next((w for w in slot_out['words'] if w in ACTIVITY_LABELS), None)
        pattern = next((w for w in slot_out['words'] if w.startswith('pattern:')), None)
        rendered = render_state(feelings=feelings, activity=activity, pattern=pattern,
                                seed=int(row.get('seed') or 0) if row.get('seed') else hash(row['episode']) % 2**31)
        audit = audit_grounding(rendered, measured)
        if rendered['text'] is None:
            abstained += 1
            continue
        sentences[rendered['text']] += 1
        slot_histogram.update(rendered['slots'])
        invented += len(audit['invented'])
        unsupported += len(audit['unsupported'])
        contradicted = audit['contradicted']
        if contradicted:
            contradictions.append({'episode': row['episode'], 'turn': row['turn'],
                                   'contradicted': contradicted, 'rendered': rendered['slots']})
    rendered_windows = windows - abstained
    report = {'schema': 'chatflypt.generative-renderer-evaluation.v1', 'run_id': args.run_id,
              'windows': windows, 'rendered': rendered_windows, 'abstained': abstained,
              'distinct_sentences': len(sentences),
              'fixed_template_baseline': 'the per-channel template join produced at most one sentence per label set',
              'top_sentences': sentences.most_common(8),
              'slot_histogram': dict(slot_histogram),
              'invented_slots': invented, 'unsupported_slots': unsupported,
              'contradicted_slots': len(contradictions), 'contradictions': contradictions[:12],
              'rules': rules, 'decoder': unified.artifact, 'slot_artifact': slots_decoder.artifact,
              'scope': 'Every slot is decoded from internal activity; the renderer varies wording only.',
              'limitations': ['Contradictions are decoded misreads, reported with their measured values.',
                              'Slot bands are fixed from the training split of this corpus.',
                              'No language model is involved; the realizer is a fixed grammar.']}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / f'{args.run_id}-evaluation.json', report)
    print(json.dumps({k: v for k, v in report.items()
                      if k in ('windows', 'rendered', 'abstained', 'distinct_sentences',
                               'invented_slots', 'unsupported_slots', 'contradicted_slots')},
                     ensure_ascii=False))
    print('top sentences:')
    for text, count in sentences.most_common(6):
        print(f'  x{count}  {text}')


if __name__ == '__main__':
    main()
