"""Two-fly conspecific loop: one fly's recorded motor output becomes the other's stimulus.

Fly A runs its own MaleCNS neural/body session. Whether A "sang" in a window is
read from the pinned controller's recorded song motor group in that window - the
same recorded motor activity that drives A's body, never a human instruction.
When A sang, a song field is placed at fly B's actual pedicel (where Johnston's
organ sits) and B runs its own session; B's measured reception, blind readout and
measured movement are recorded.

Declared limits: the map from A's song motor activity to a field at B's pedicel is
an engineering convention, not a decoded song; nothing here says A "meant" to
communicate or that B understood. If A never sings in the sampled windows, that
no-response is the result.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump, sha256
from chatflypt.malecns import MaleCNS
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.environment import empty_environment, probe_environment, validate_objects
from chatflypt.embodied import EmbodiedSession, run_embodied
from chatflypt.intent_readout import measured_outcome

OUT = ROOT / 'data/derived/conspecific-loop'
REPORT = ROOT / 'reports/conspecific-loop'
SONG_GROUP = 'song'
SOURCE_SIZE_MM = [.05, .05, .05]
SOURCE_RANGE_MM = .20
SOURCE_OFFSET_MM = .08


def song_activity(body):
    summary = body.get('command_summary') or {}
    return float((summary.get(SONG_GROUP) or {}).get('max', 0.))


def run_window(brain, annotations, catalogue, environment, seed, session, presentations=None):
    return run_embodied(brain, annotations, catalogue, 'MaleCNS v1.0', [], environment, seed, 'ground',
                        duration_ms=500., session=session, input_mode='environment_only',
                        presentations=presentations or [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fly-a-windows', type=int, default=6, help='How many windows to sample A before giving up')
    parser.add_argument('--seed', type=int, default=970000)
    args = parser.parse_args()

    brain, catalogue = MaleCNS(), build_catalogue(save=False)
    annotations = brain.rows
    environment = empty_environment()
    log = {'schema': 'chatflypt.conspecific-loop.v1',
           'mapping': {'direction': 'A recorded song motor activity -> song field at B pedicel',
                       'convention': 'engineering, not a decoded song',
                       'song_group': SONG_GROUP,
                       'source_geometry_mm': SOURCE_SIZE_MM, 'source_range_mm': SOURCE_RANGE_MM,
                       'source_offset_mm': SOURCE_OFFSET_MM, 'biologically_calibrated': False},
           'stages': []}

    # Stage 1: does fly A sing in any sampled window?
    a_session = EmbodiedSession(environment, 'ground', 'loop-fly-a')
    try:
        for index in range(args.fly_a_windows):
            result = run_window(brain, annotations, catalogue, environment, args.seed + index, a_session)
            body = result['body']
            song = song_activity(body)
            outcome = measured_outcome(body['trajectory']) if body.get('trajectory') else None
            log['stages'].append({'stage': 'fly_a_window', 'index': index, 'seed': args.seed + index,
                                  'song_motor_max': song,
                                  'spikes': int(result['trial'].counts.sum()),
                                  'measured_outcome': outcome})
            if song > 0:
                log['fly_a_sang'] = True
                log['fly_a_song_window'] = index
                break
        else:
            log['fly_a_sang'] = False
    finally:
        a_session.close()
    if not log.get('fly_a_sang'):
        log['result'] = ('fly A produced no recorded song motor activity in '
                         f'{args.fly_a_windows} sampled windows; no conspecific signal was presented to B')
        REPORT.mkdir(parents=True, exist_ok=True)
        dump(REPORT / 'latest.json', log)
        print(json.dumps({'fly_a_sang': False, 'result': log['result']}, ensure_ascii=False))
        return

    # Stage 2: place the field at B's actual pedicel and run B.
    neutral = probe_environment(environment, args.seed, 'ground')
    names = neutral['body_names']
    positions = neutral['frame']['body_positions_mm']
    site = 'l_pedicel' if 'l_pedicel' in names else None
    if site is None:
        raise SystemExit('The compiled body has no pedicel site; a song field cannot be placed')
    center = (np.array(positions[names.index(site)]) + [0., SOURCE_OFFSET_MM, 0.]).tolist()
    environment_b = empty_environment()
    environment_b['objects'] = validate_objects([{'id': 'conspecific_song', 'type': 'song',
        'position_mm': center, 'size_mm': SOURCE_SIZE_MM, 'range_mm': SOURCE_RANGE_MM, 'strength': 1.}])
    log['presented'] = {'object': 'conspecific_song', 'type': 'song', 'position_mm': center,
                        'receptor_site': site, 'declared_side': 'left'}
    b_session = EmbodiedSession(environment_b, 'ground', 'loop-fly-b')
    try:
        result = run_window(brain, annotations, catalogue, environment_b, args.seed + 100, b_session)
        body = result['body']
        feedback = result['feedback']
        delivered = sorted({f"{item['channel']}:{item.get('side')}"
                            for sample in feedback.get('samples', []) for item in sample.get('items', [])
                            if item.get('intensity', 1.) > 0})
        decoded = None
        from chatflypt.unified_readout import load_unified_decoder
        decoder, note = load_unified_decoder(brain, catalogue)
        if decoder is not None and result['trial'].state_record is not None:
            decoded = decoder.decode(result['trial'].counts, result['trial'].state_record, result['trial'].duration_ms)
        else:
            decoded = {'status': 'uncalibrated', 'note': note, 'words': []}
        log['stages'].append({'stage': 'fly_b_window', 'seed': args.seed + 100,
                              'measured_delivered_items': delivered,
                              'measured_contact_bouts': None,
                              'spikes': int(result['trial'].counts.sum()),
                              'blind_readout': {'status': decoded.get('status'), 'words': decoded.get('words'),
                                                'scope': decoded.get('scope')},
                              'measured_outcome': measured_outcome(body['trajectory']) if body.get('trajectory') else None,
                              'physics_warnings': (body.get('summary') or {}).get('physics_warnings')})
    finally:
        b_session.close()
    log['result'] = ('A recorded song motor activity was mapped to a song field at B\'s left pedicel; '
                     'B\'s measured reception and blind readout are recorded. This is an engineering '
                     'channel, not evidence that A communicated or B understood.')
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / 'latest.json', log)
    print(json.dumps({'fly_a_sang': True, 'fly_b': log['stages'][-1]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
