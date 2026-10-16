"""Multi-turn exchange: the next stimulus is chosen from the fly's measured state.

Unlike the normal chat (one human message, one 500 ms window), this runs a closed
round-based exchange inside one MaleCNS session. Each round:

  1. presents something physical (a tap train, a cVA field, a song field),
  2. measures what actually arrived (delivered receptor events, contacts, body),
  3. reads the fly blindly (channel labels, activity/pattern slots, motion), and
  4. lets a **declared policy** pick the next presentation from that *measured*
     state - never from the human text and never from an LLM.

The policy is an engineering convention, printed in the report: replying to song
motor activity with a song field, answering a measured touch with a nudge on the
opposite leg, offering sugar when it is walking, and prompting with a gentle tap
when nothing happened. Rounds where nothing is measured stay in the record: a
non-response is a result, not a failure to paper over.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.malecns import MaleCNS
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.environment import empty_environment, probe_environment, validate_objects
from chatflypt.embodied import EmbodiedSession, run_embodied
from chatflypt.intent_readout import load_intent_decoder, measured_outcome
from chatflypt.slot_readout import load_slot_decoders
from chatflypt.unified_readout import load_unified_decoder
from chatflypt.dialogue import sensory_report
from chatflypt.generative_renderer import audit_grounding, render_state

OUT = ROOT / 'data/derived/turn-exchange'
REPORT = ROOT / 'reports/turn-exchange'
PLACEMENT = ROOT / 'reports/unified-readout/conspecific-v4-placement.json'
POLICY = {
    'song_reply': 'measured song motor activity > 0 -> present a courtship-song field at the pedicel',
    'nudge_back': 'a measured foreleg touch -> tap the opposite foreleg three times',
    'offer_food': 'measured motion active -> place a sugar platform 1 mm ahead',
    'prompt': 'nothing measured -> a single gentle tap on the right foreleg',
}

OPENER_REASON = '\u5f00\u573a\uff1a\u5bf9\u5de6\u524d\u817f\u8f7b\u6572\u4e09\u6b21\uff08\u58f0\u660e\u7684\u5f00\u573a\u523a\u6fc0\uff09'
REASON_SONG = ('\u5b9e\u6d4b\u5230 song \u8fd0\u52a8\u6d3b\u52a8 \u2192 '
               '\u5728\u89e6\u89d2\u6897\u8282\u5904\u4ee5\u9e23\u58f0\u573a\u56de\u5e94\uff08\u540c\u79cd\u56de\u5e94\uff09')
REASON_NUDGE = ('\u5b9e\u6d4b\u5230 {touched} \u7684\u63a5\u89e6 \u2192 '
                '\u6572{side}\u524d\u817f\u4e09\u6b21\uff08\u5bf9\u4fa7\u8f7b\u63a8\uff09')
REASON_WALK = ('\u5b9e\u6d4b\u5230\u5b83\u5728\u8d70\u52a8 \u2192 '
               '\u5728\u524d\u65b9 1 mm \u5904\u653e\u4e00\u5757\u7cd6\u5e73\u53f0')
REASON_PROMPT = ('\u672c\u8f6e\u6ca1\u6709\u5b9e\u6d4b\u5230\u4e0a\u8ff0\u4efb\u4f55\u4e00\u9879 \u2192 '
                 '\u8f7b\u78b0\u53f3\u524d\u817f\u4f5c\u4e3a\u63d0\u793a')


def tap(side, strength=1., taps=1, onset=50.):
    probe = {'kind': 'touch', 'side': side, 'body_part': 'foreleg', 'strength': strength,
             'onset_ms': onset, 'duration_ms': 60. if taps > 1 else 300.}
    if taps > 1:
        probe.update(taps=taps, interval_ms=140.)
    return probe


def song_environment(placement, side='left'):
    spec = placement['local_sources']['song'][side]
    environment = empty_environment()
    environment['objects'] = validate_objects([{'id': 'reply_song', 'type': 'song',
        'position_mm': spec['position_mm'], 'size_mm': [.05, .05, .05],
        'range_mm': spec['range_mm'], 'strength': 1.}])
    return environment


def sugar_environment(neutral):
    """A platform placed 1 mm ahead of the measured thorax pose."""
    index = neutral['body_names'].index('c_thorax')
    position = neutral['frame']['body_positions_mm'][index]
    environment = empty_environment()
    environment['objects'] = validate_objects([{'id': 'reply_sugar', 'type': 'sugar',
        'position_mm': [position[0] + 1., position[1], .1], 'size_mm': [1., 1., .2],
        'range_mm': 2., 'strength': 1.}])
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--seed', type=int, default=980000)
    args = parser.parse_args()
    placement = json.loads(PLACEMENT.read_text(encoding='utf8'))
    brain, catalogue = MaleCNS(), build_catalogue(save=False)
    unified, unified_note = load_unified_decoder(brain, catalogue)
    slots, slot_note = load_slot_decoders(brain, catalogue)
    intent, intent_note = load_intent_decoder(brain, catalogue)
    if unified is None or slots is None:
        raise SystemExit('Readouts must be published first: ' + json.dumps(
            {'unified': unified_note, 'slots': slot_note}, ensure_ascii=False))

    log = {'schema': 'chatflypt.turn-exchange.v1', 'rounds': args.rounds,
           'policy': POLICY, 'llm_called': False, 'human_text_used': False,
           'session': 'one MaleCNS neural/body session; state preserved across rounds',
           'readouts': {'unified': unified.artifact, 'slots': slots.artifact,
                        'intent': getattr(intent, 'artifact', None)},
           'exchange': []}
    environment = empty_environment()
    session = EmbodiedSession(environment, 'ground', 'exchange')
    presentation = [tap('left', 1., 3)]
    reason = OPENER_REASON
    try:
        for round_index in range(args.rounds):
            neutral = probe_environment(environment, args.seed + round_index, 'ground')
            result = run_embodied(brain, brain.rows, catalogue, 'MaleCNS v1.0', [], environment,
                                  args.seed + round_index, 'ground', duration_ms=500.,
                                  session=session, input_mode='environment_only',
                                  presentations=presentation)
            trial, body, feedback = result['trial'], result['body'], result['feedback']
            counts, state, duration = trial.counts, trial.state_record, trial.duration_ms
            delivered, contacts = {}, 0
            for sample in feedback.get('samples', []):
                for item in sample.get('items', []):
                    if item.get('intensity', 1.) > 0:
                        key = f"{item['channel']}:{item.get('side', 'bilateral')}"
                        delivered[key] = delivered.get(key, 0) + 1
                contacts += sum(1 for e in sample.get('evidence', [])
                                if e.get('mode') == 'mujoco_leg_contact'
                                and str(e.get('object_id', '')).startswith('message_probe_'))
            decoded = unified.decode(counts, state, duration)
            slot_out = slots.decode(counts, state, duration)
            intent_out = intent.decode(counts, state, duration) if intent is not None else {'words': []}
            measured = {'delivered_items': delivered,
                        'activity_band': next((w for w in slot_out['words'] if w.startswith('activity:')), None),
                        'pattern_band': next((w for w in slot_out['words'] if w.startswith('pattern:')), None)}
            report = sensory_report(decoded, {'non_sensory_spikes': int(np.asarray(counts).sum()),
                                              'membrane_response': {'internal_cells': 0}})
            feelings = [f for f in report['feelings'] if f['channel'] != 'command']
            rendered = render_state(feelings=feelings, activity=measured['activity_band'],
                                    pattern=measured['pattern_band'], seed=args.seed + round_index)
            audit = audit_grounding(rendered, measured)
            song_activity = float((body.get('command_summary') or {}).get('song', {}).get('max', 0.))
            touched_keys = sorted({key for key in delivered if 'touch' in key})
            outcome = measured_outcome(body['trajectory']) if body.get('trajectory') else None
            log['exchange'].append({
                'round': round_index, 'presented': presentation, 'presented_reason': reason,
                'measured': {'delivered_items': delivered, 'contact_samples': contacts,
                             'song_motor_max': song_activity, 'touched_channels': touched_keys,
                             'outcome': outcome},
                'readout': {'labels': decoded['words'], 'slots': slot_out['words'],
                            'intent': intent_out.get('words', []),
                            'text': rendered['text'], 'render_slots': rendered['slots']},
                'audit': audit})
            # Declared policy: the next presentation comes from the measured state.
            if song_activity > 0:
                environment = song_environment(placement)
                presentation = []
                reason = REASON_SONG
            elif touched_keys:
                # Opposite leg from the one actually measured as touched; the side
                # comes from the measured channel, never from the plan.
                side = 'right' if any(key.endswith(':left') for key in touched_keys) else 'left'
                presentation = [tap(side, 1.2, 3)]
                reason = REASON_NUDGE.format(touched='\u3001'.join(touched_keys),
                                             side='\u53f3' if side == 'right' else '\u5de6')
            elif outcome and outcome['path_length_mm'] >= 1.3:
                environment = sugar_environment(neutral)
                presentation = []
                reason = REASON_WALK
            else:
                environment = empty_environment()
                presentation = [tap('right', .8, 1)]
                reason = REASON_PROMPT
    finally:
        session.close()
    REPORT.mkdir(parents=True, exist_ok=True)
    log['summary'] = {
        'rounds_recorded': len(log['exchange']),
        'rounds_with_sensation': sum(1 for row in log['exchange'] if row['readout']['labels']),
        'rounds_with_motion': sum(1 for row in log['exchange'] if row['readout']['intent']),
        'rounds_song_reply': sum(1 for row in log['exchange'] if row['presented_reason'] == REASON_SONG),
        'rounds_nudge_back': sum(1 for row in log['exchange'] if '\u5bf9\u4fa7\u8f7b\u63a8' in row['presented_reason']),
        'nudged_side_by_round': [(row['round'], next((p['side'] for p in row['presented']
                                                      if p.get('kind') == 'touch'), None))
                                 for row in log['exchange']],
        'touched_side_by_round': [(row['round'], row['measured']['touched_channels'])
                                  for row in log['exchange']],
        'contradictions': sum(len(row['audit']['contradicted']) for row in log['exchange']),
        'invented_slots': sum(len(row['audit']['invented']) for row in log['exchange'])}
    dump(REPORT / 'latest.json', log)
    for row in log['exchange']:
        print(json.dumps({'round': row['round'], 'reason': row['presented_reason'],
                          'text': row['readout']['text'], 'intent': row['readout']['intent'],
                          'delivered': list(row['measured']['delivered_items']),
                          'contacts': row['measured']['contact_samples']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
