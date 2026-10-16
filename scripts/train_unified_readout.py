"""Collect and fit a single-MaleCNS, strictly environmental readout benchmark.

Collection never requests direct neural events. Entire physical/neural sessions
are assigned to one split before collection. The test split is opened only after
weights, feature selection and thresholds have been saved. Targets mean actual
delivery to a receptor population, not subjective sensation or behavior.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump, sha256
from chatflypt.unified_readout import (DATASET, OUT, SCHEMA, MIN_VOLTAGE_MV,
    SEMANTIC_GROUPS, source_artifact_paths, UnifiedDecoder, blind_mask, internal_features, model_contract)

REPORT = ROOT / 'reports/unified-readout'
V3_PLACEMENT = REPORT / 'retinal-axes-v3-placement.json'
V4_PLACEMENT = REPORT / 'conspecific-v4-placement.json'
CHANNELS = ('mc_bright', 'mc_yeast', 'mc_warm', 'mc_cold', 'foreleg_touch',
            'midleg_touch', 'hindleg_touch', 'head_touch', 'body_touch', 'wing_touch',
            'sugar', 'sugar_specific', 'high_salt',
            'leg_sweet_local', 'leg_appetitive_ascending', 'mc_cva', 'mc_auditory')
# Conspecific rhythm contrasts: equal total contact time (180 ms), three
# different rhythms. Timing is an engineering parameter set, not a measurement.
TAP_RHYTHMS = (('tap1x180', {'duration_ms': 180.}),
               ('tap3x60', {'duration_ms': 60., 'taps': 3, 'interval_ms': 140.}),
               ('tap6x30', {'duration_ms': 30., 'taps': 6, 'interval_ms': 60.}))
# Side-dominance transcription rule (declared, tested, REJECTED - kept as code
# with its negative result). The antennal chemosensory/thermosensory groups
# receive bilateral delivery by geometry: a source in front of the head reaches
# both funiculi or both aristae. Measured on the v5 corpus, bilateral windows have
# minor/major delivered-event ratios with median 0.82 (yeast), 0.85 (warm),
# 0.77 (cold) - near-symmetric - and only 6/6/5 strictly unilateral windows exist
# in 294 windows. Applying a 2x dominance criterion therefore collapses the four
# side labels from 5/5/5 and 3/3/3 positives to 0-1 per split. The contrast is not
# measurable in this apparatus at window resolution, so no transcription rule can
# recover it and the four labels stay abstentions. See the negative registry.
DOMINANCE_GROUPS = ('yeast', 'warm', 'cold')
SIDE_MIN_EVENTS = 2
SIDE_RATIO = 2.0


def protocol(run_id=None):
    """Bounded preregistered scenes, including physical sham and withdrawal."""
    from chatflypt.environment import validate_objects
    episodes = []
    def obj(kind, pos, size, strength=1., distance=5.):
        return {'id': 'sample', 'type': kind, 'position_mm': pos,
                'size_mm': size, 'strength': strength, 'range_mm': distance}
    for split, index in (('train', 0), ('validation', 1), ('test', 2)):
        strength = (1., .85, 1.15)[index]
        shade = {'id': 'shade', 'type': 'shade', 'position_mm': [0, 0, 10],
                 'size_mm': [20, 20, 20], 'strength': 1.}
        scenes = [
            ('bright', [], [[], []]),
            ('dark', [shade], [[], []]),
            ('yeast', [obj('yeast', [3 + .2 * index, -.2 * index, .4], [.5] * 3, strength, 7.)], [[], []]),
            ('dark_yeast', [shade, obj('yeast', [3, .2 * index, .4], [.5] * 3, strength, 7.)], [[], []]),
            ('heat', [obj('heat', [1, 0, 1], [.2] * 3, strength)], [[], []]),
            ('cold', [obj('cold', [1, 0, 1], [.2] * 3, strength)], [[], []]),
            ('mouth_block', [obj('block', [1.1, 0, .4], [.7, 1., .8])], [[], []]),
            ('mouth_sugar', [obj('sugar', [1.1, 0, .4], [.7, 1., .8])], [[], []]),
            ('mouth_high_salt', [obj('high_salt', [1.1, 0, .4], [.7, 1., .8])], [[], []]),
            ('leg_sugar', [obj('sugar', [0, 0, .1], [5., 4., .2])], [[], []]),
        ]
        for side in ('left', 'right'):
            probe = {'kind': 'touch', 'side': side, 'body_part': 'foreleg',
                     'strength': strength, 'onset_ms': 50., 'duration_ms': 300.}
            opposite = dict(probe, side='right' if side == 'left' else 'left')
            # Two independent ordering episodes provide each side a chance at
            # two positive validation windows, separated by real withdrawals.
            scenes.append(('touch_' + side, [], [[probe], [], [opposite], []]))
        for scene, objects, turns in scenes:
            episodes.append({'id': f'{split}-{scene}', 'split': split, 'condition': scene,
                'seed': 924500 + len(episodes) * 10, 'turns': turns,
                'environment': {'schema': 'chatflypt.environment.v1', 'revision': 1,
                                'objects': validate_objects(objects)}})
    plan = {'schema': 'chatflypt.unified-readout-protocol.v1', 'dataset': DATASET,
            'input_mode': 'environment_only', 'duration_ms': 500.,
            'interval_ms': 'environment-only default; actual interval saved in every physical record',
            'split_unit': 'whole_session', 'episodes': episodes, 'channels': list(CHANNELS),
            'alpha': 10., 'threshold_grid': [.2, .8, .01],
            'threshold_candidates': 'v7 revision: the absolute grid 0.2..0.8 plus per-label quantiles '
                                    '(0.5..0.999) of that label\'s own training scores. Ridge shrinkage is '
                                    'label-dependent, so an absolute grid can sit entirely above one label\'s '
                                    'scores and report it unreadable when the problem is the threshold scale. '
                                    'Quantiles come from the training split; validation still selects the '
                                    'threshold and the per-label qualification.',
            'qualification': {'minimum_positive_windows': 2, 'minimum_negative_windows': 2,
                              'precision': .9, 'recall': .75},
            'target': 'Actually delivered receptor-population events in this observation window',
            'candidate_only': True, 'biologically_calibrated': False,
            'limitations': ['Some conditions occupy only one session per split.',
                'Persistent windows within a session are correlated, not independent replicates.',
                'Lateralized population delivery is not spatial object localization.',
                'No natural-language intent or behavior target is fitted.']}
    if run_id is not None:
        plan['run_id'] = run_id
    if run_id == 'retinal-axes-v3':
        extend_v3_protocol(plan)
    if run_id == 'conspecific-v4':
        extend_v4_protocol(plan)
    if run_id == 'conspecific-v5-probe':
        extend_v5_probe_protocol(plan)
    if run_id == 'conspecific-v5':
        extend_v4_protocol(plan)
        extend_v5_probe_protocol(plan)
    if run_id == 'conspecific-v6':
        extend_v4_protocol(plan)
        extend_v5_probe_protocol(plan)
        extend_v6_protocol(plan)
    if run_id in ('conspecific-v7', 'conspecific-v8'):
        extend_v4_protocol(plan)
        extend_v5_probe_protocol(plan)
        extend_v6_protocol(plan)
        extend_v7_protocol(plan)
    if run_id == 'conspecific-v9':
        # Same episode construction as v7 plus the side-corrected following
        # sources and stronger asymmetric steering stimuli. v9 exists because the
        # anchored field model was fixed, which changes a contract-hashed file and
        # therefore requires a fresh collection of the whole union.
        extend_v4_protocol(plan)
        extend_v5_probe_protocol(plan)
        extend_v6_protocol(plan)
        extend_v7_protocol(plan)
        extend_v9_protocol(plan)
    return plan


def extend_v9_protocol(plan):
    """Side-corrected following sources, and stronger asymmetric steering stimuli.

    Two things this round fixes that the earlier ones could not:

    * ``measured_field`` now honours a declared ``anchor_side``: the source keeps
      its offset from **its own** receptor instead of being re-placed next to
      whichever receptor is sampled. These conditions declare the side, so they are
      reliably delivered *and* strictly one-sided - the property the antennal side
      labels need and the legacy bilateral mode could never provide.
    * Left-turn commands were elicited but too small to clear the training-split
      informative threshold. These conditions combine, in one window, the stimulus
      families whose windows carried a left-turn command (cold on the antennal
      arista, courtship song at the pedicel, a foreleg nudge train), mirrored for
      the right direction, so the steering asymmetry is larger.

    Planned sides and planned commands are never targets; only delivered events and
    recorded commands count.
    """
    from chatflypt.environment import validate_objects
    if not (V3_PLACEMENT.is_file() and V4_PLACEMENT.is_file()):
        raise ValueError('Both placement verifications are required before the v9 round')
    local = json.loads(V3_PLACEMENT.read_text(encoding='utf8'))['local_sources']
    conspecific = json.loads(V4_PLACEMENT.read_text(encoding='utf8'))
    emitters = conspecific['local_sources']
    receptors = conspecific['neutral_receptors_mm']
    extra_index = 0
    for split, index in (('train', 0), ('validation', 1), ('test', 2)):
        strength = (1., .85, 1.15)[index]
        additions = []
        # Following sources that declare which receptor they belong to. The range
        # stays above the source-to-own-receptor distance and well below the
        # separation between the two receptors, so the far side reads exactly zero.
        for kind, site, reach in (('yeast', 'funiculus', .16), ('heat', 'arista', .12),
                                  ('cold', 'arista', .12)):
            for side in ('left', 'right'):
                spec = local[site][side]
                neutral = receptors[side + '_' + site]
                for repeat in range(1, 4):
                    additions.append((f'{kind}_side_{side}_r{repeat}',
                        [{'id': 'local_source', 'type': kind, 'position_mm': spec['position_mm'],
                          'size_mm': [.05, .05, .05], 'range_mm': reach, 'strength': strength,
                          'field': 'anchored', 'anchor': site, 'anchor_origin_mm': neutral,
                          'anchor_side': side}], [[], []]))
        # Composite steering stimuli, mirrored. Left-rich combines everything that
        # produced left-turn commands; right-rich is its mirror image.
        for direction in ('left', 'right'):
            source_side = 'left' if direction == 'left' else 'right'
            song_side = 'right' if direction == 'left' else 'left'
            tap_side = 'left' if direction == 'left' else 'right'
            song_spec = emitters['song'][song_side]
            song_neutral = receptors[song_side + '_pedicel']
            cold_spec = local['arista'][source_side]
            cold_neutral = receptors[source_side + '_arista']
            scene = [
                {'id': 'steering_cold', 'type': 'cold', 'position_mm': cold_spec['position_mm'],
                 'size_mm': [.05, .05, .05], 'range_mm': 0.12, 'strength': strength,
                 'field': 'anchored', 'anchor': 'arista', 'anchor_origin_mm': cold_neutral,
                 'anchor_side': source_side},
                {'id': 'steering_song', 'type': 'song', 'position_mm': song_spec['position_mm'],
                 'size_mm': [.05, .05, .05], 'range_mm': song_spec['range_mm'], 'strength': strength,
                 'field': 'anchored', 'anchor': 'pedicel', 'anchor_origin_mm': song_neutral,
                 'anchor_side': song_side}]
            train = {'kind': 'touch', 'side': tap_side, 'body_part': 'foreleg',
                     'strength': min(2., 1.5 * strength), 'onset_ms': 50.,
                     'duration_ms': 60., 'taps': 3, 'interval_ms': 140.}
            for repeat in range(1, 4):
                additions.append((f'turn_{direction}_combo_r{repeat}', scene, [[train], [train]]))
        for condition, objects, turns in additions:
            if split + '-' + condition in {episode['id'] for episode in plan['episodes']}:
                continue
            plan['episodes'].append({'id': split + '-' + condition, 'split': split,
                'condition': condition, 'seed': 980000 + extra_index * 10,
                'turns': turns, 'environment': {'schema': 'chatflypt.environment.v1',
                    'revision': 1, 'objects': validate_objects(objects)}})
            extra_index += 1
    plan['elicitation_round_3'] = {
        'fills': ['yeast:left', 'yeast:right', 'warm:left', 'warm:right', 'cold:left', 'cold:right'],
        'why_this_round_exists': 'environment.measured_field was fixed to honour anchor_side; that file is '
                                 'contract-hashed, so every previously collected window is unfittable and the '
                                 'whole union is re-collected here.',
        'method': 'following sources with a declared side (reliable and strictly one-sided by construction)',
        'steering': 'mirrored composite stimuli that stack the families whose windows carried left/right '
                    'turn commands, to raise the asymmetry above the informative threshold',
        'legacy_mode': 'scenes without anchor_side keep the old bilateral behaviour and are labelled '
                       'anchor_mode=legacy_bilateral in their records',
        'planned_side_is_not_a_target': True}
    plan['scope_additions'] = [*plan.get('scope_additions', []),
        'Side-corrected following sources on each side (declared anchor side)',
        'Mirrored composite steering stimuli (cold + song + foreleg nudge)']
    return plan


def extend_v7_protocol(plan):
    """Elicitation round 2: enough strictly unilateral windows and more left-turn commands.

    Two measured gaps drove this round, both recorded in the negative registry:

    * The four antennal side labels (yeast:left, warm:left, cold:left, cold:right)
      need windows in which only *one* funiculus or arista is reached. The v3
      ``unilateral_*`` scenes are verified strictly unilateral at the initial pose,
      but the animal walks out of the small source, so only 6/6/5 such windows
      exist in 294 - too few for the per-label gates. The anchored ("following")
      model cannot help: as implemented it re-places the source next to *whichever*
      receptor is being sampled, so every receptor receives the same field by
      construction and a side contrast is impossible. That is a real defect; fixing
      it means editing contract-hashed transduction code and re-collecting the whole
      corpus, so this round repeats the verified fixed-source scenes instead.
    * Left-turn commands are elicited but barely present in the *training* split
      (1 window against 21 right-turn windows), which is why the readout never
      predicts them. This round adds more of the families that produced them.

    Planned sides and planned commands are never targets; only delivered events and
    recorded commands count.
    """
    from chatflypt.environment import validate_objects
    if not (V3_PLACEMENT.is_file() and V4_PLACEMENT.is_file()):
        raise ValueError('Both placement verifications are required before the v7 round')
    local = json.loads(V3_PLACEMENT.read_text(encoding='utf8'))['local_sources']
    conspecific = json.loads(V4_PLACEMENT.read_text(encoding='utf8'))
    emitters = conspecific['local_sources']
    receptors = conspecific['neutral_receptors_mm']
    extra_index = 0
    for split, index in (('train', 0), ('validation', 1), ('test', 2)):
        strength = (1., .85, 1.15)[index]
        additions = []
        for kind, site in (('yeast', 'funiculus'), ('heat', 'arista'), ('cold', 'arista')):
            for side in ('left', 'right'):
                spec = local[site][side]
                for repeat in range(1, 5):
                    additions.append((f'unilateral_{kind}_{side}_u{repeat}',
                        [{'id': 'local_source', 'type': kind, 'position_mm': spec['position_mm'],
                          'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'],
                          'strength': strength}], [[], []]))
        for kind in ('pheromone', 'song'):
            for side in ('left', 'right'):
                spec = emitters[kind][side]
                site = 'funiculus' if kind == 'pheromone' else 'pedicel'
                neutral = receptors[side + '_' + site]
                for repeat in (2, 3):
                    additions.append((f'{kind}_follow_{side}_r{repeat}',
                        [{'id': 'conspecific_source', 'type': kind, 'position_mm': spec['position_mm'],
                          'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'], 'strength': strength,
                          'field': 'anchored', 'anchor': site, 'anchor_origin_mm': neutral}], [[], []]))
        for side in ('left', 'right'):
            opposite = 'right' if side == 'left' else 'left'
            train = {'kind': 'touch', 'side': opposite, 'body_part': 'foreleg',
                     'strength': min(2., 1.5 * strength), 'onset_ms': 50.,
                     'duration_ms': 60., 'taps': 3, 'interval_ms': 140.}
            for repeat in (2, 3):
                additions.append((f'nudge_{side}_r{repeat}', [], [[train], [train]]))
        for condition, objects, turns in additions:
            if split + '-' + condition in {episode['id'] for episode in plan['episodes']}:
                continue
            plan['episodes'].append({'id': split + '-' + condition, 'split': split,
                'condition': condition, 'seed': 970000 + extra_index * 10,
                'turns': turns, 'environment': {'schema': 'chatflypt.environment.v1',
                    'revision': 1, 'objects': validate_objects(objects)}})
            extra_index += 1
    plan['elicitation_round_2'] = {
        'fills': ['yeast:left', 'warm:left', 'cold:left', 'cold:right'],
        'method': 'repeats of the placement-verified strictly-unilateral fixed sources',
        'enriches_left_turn_commands': ['pheromone_follow_*_r2/r3', 'song_follow_*_r2/r3', 'nudge_*_r2/r3'],
        'anchored_model_defect': 'As implemented, the anchored model re-places the source next to whichever '
                                 'receptor is sampled, so every receptor sees the same field and no side contrast '
                                 'is possible; recorded in the negative registry. Fixing it changes '
                                 'contract-hashed transduction code and would force a full re-collection.',
        'planned_side_is_not_a_target': True}
    plan['scope_additions'] = [*plan.get('scope_additions', []),
        'Repeated strictly-unilateral fixed odor/thermal sources on each side',
        'Repeated song-field and foreleg-nudge sessions in every split']
    return plan


def extend_v6_protocol(plan):
    """Targeted elicitation round for the labels the delivery matrix shows as unreached.

    The v5 coverage matrix (`reports/unified-readout/coverage-v5.json`) splits the
    twelve unqualified labels in two:

    * `yeast:left`, `warm:left`, `cold:left`, `cold:right` are not short of windows;
      their delivered positives come mostly from *bilateral* scenes, so the side
      contrast is never actually elicited. This round adds *following* unilateral
      sources - the anchored model that already fixed cVA and song delivery - with
      several sessions per split, so each side has its own positives.
    * `body_touch:*`, `wing_touch:*`, `sugar_specific:*`, `high_salt:*` have no
      elicitation route at all: `environment.py` refuses to register wing-membrane,
      whole-thorax and labellum contacts to those populations. No condition is added
      for them; they stay abstentions by design, not by missing data.

    It also enriches the few families whose recorded windows actually carried a
    left-turn command (song field/partner, foreleg nudges, unilateral cold), because
    the corpus held 43 right-turn against 1 left-turn command above the informative
    threshold. Planned sides and planned commands are never targets: only delivered
    events and recorded commands count.
    """
    from chatflypt.environment import validate_objects
    if not (V3_PLACEMENT.is_file() and V4_PLACEMENT.is_file()):
        raise ValueError('Both placement verifications are required before the v6 elicitation round')
    placement = json.loads(V4_PLACEMENT.read_text(encoding='utf8'))
    receptors = placement['neutral_receptors_mm']
    # Odor/thermal source geometry lives in the v3 placement check; the
    # conspecific emitters live in the v4 one.
    local = json.loads(V3_PLACEMENT.read_text(encoding='utf8'))['local_sources']
    emitters = placement['local_sources']
    declared = {episode['condition'] for episode in plan['episodes']}
    extra_index = 0
    for split, index in (('train', 0), ('validation', 1), ('test', 2)):
        strength = (1., .85, 1.15)[index]
        additions = []
        # Side-contrast follow sources: same declared geometry, source keeps its
        # offset relative to the measured receptor instead of the fixed room position.
        for kind, site in (('yeast', 'funiculus'), ('heat', 'arista'), ('cold', 'arista')):
            for side in ('left', 'right'):
                spec = local[site][side]
                neutral = receptors[side + '_' + site]
                for repeat in range(3):
                    additions.append((f'{kind}_follow_{side}_r{repeat + 1}',
                        [{'id': 'local_source', 'type': kind, 'position_mm': spec['position_mm'],
                          'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'], 'strength': strength,
                          'field': 'anchored', 'anchor': site, 'anchor_origin_mm': neutral}], [[], []]))
        # Left-turn command families measured in the v5 corpus.
        for kind in ('pheromone', 'song'):
            for side in ('left', 'right'):
                spec = emitters[kind][side]
                site = 'funiculus' if kind == 'pheromone' else 'pedicel'
                neutral = receptors[side + '_' + site]
                additions.append((f'{kind}_follow_{side}_r1',
                    [{'id': 'conspecific_source', 'type': kind, 'position_mm': spec['position_mm'],
                      'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'], 'strength': strength,
                      'field': 'anchored', 'anchor': site, 'anchor_origin_mm': neutral}], [[], []]))
        for side in ('left', 'right'):
            opposite = 'right' if side == 'left' else 'left'
            train = {'kind': 'touch', 'side': opposite, 'body_part': 'foreleg',
                     'strength': min(2., 1.5 * strength), 'onset_ms': 50.,
                     'duration_ms': 60., 'taps': 3, 'interval_ms': 140.}
            additions.append((f'nudge_{side}_r1', [], [[train], [train]]))
        for condition, objects, turns in additions:
            if split + '-' + condition in {episode['id'] for episode in plan['episodes']}:
                continue
            plan['episodes'].append({'id': split + '-' + condition, 'split': split,
                'condition': condition, 'seed': 960000 + extra_index * 10,
                'turns': turns, 'environment': {'schema': 'chatflypt.environment.v1',
                    'revision': 1, 'objects': validate_objects(objects)}})
            extra_index += 1
    plan['elicitation_round'] = {
        'fills': ['yeast:left', 'warm:left', 'cold:left', 'cold:right'],
        'enriches_left_turn_commands': ['pheromone_follow_*', 'song_follow_*', 'nudge_*_r*'],
        'no_route_kept_as_abstention': ['body_touch:*', 'wing_touch:*', 'sugar_specific:*', 'high_salt:*'],
        'reason_for_abstention': 'environment.py refuses to register wing-membrane, whole-thorax and '
                                 'labellum contacts to those populations; there is no physical route to elicit them',
        'planned_side_is_not_a_target': True}
    plan['scope_additions'] = [*plan.get('scope_additions', []),
        'Following unilateral odor/thermal sources on each side, several sessions per split',
        'Repeated song-field and foreleg-nudge sessions in every split']
    plan['limitations'].append('A following source can still become bilateral if the two antennae come '
                               'within its declared range; delivered events, never the planned side, decide labels.')
    return plan


def extend_v5_probe_protocol(plan, include_base_reference=True):
    """Focused probe: do delivery-limited modalities and turn-rich contacts change the verdicts?

    Three declared fixes to the v4 abstentions, each with a measured path:
      * `pheromone_plume_*`   - the borrowed upstream plume grid sampled at the
                                funiculus (spatially and temporally structured).
      * `*_partner_*`         - the source keeps its declared offset but follows
                                the measured receptor pose, i.e. a conspecific
                                that stays with the animal.
      * `nudge_*`             - multi-tap trains on the opposite foreleg
                                (Ramdya-style nudging), aimed at producing
                                turn-rich windows for the motor-side readout.

    With ``include_base_reference`` the run also carries the base reference
    conditions, so it can stand alone as a probe; the union run
    (``conspecific-v5``) sets it False because the v4 extension already adds them.
    """
    from chatflypt.environment import validate_objects
    if not V4_PLACEMENT.is_file():
        raise ValueError('Conspecific placement verification is required before the v5 probe')
    placement = json.loads(V4_PLACEMENT.read_text(encoding='utf8'))
    sources = placement['local_sources']
    receptors = placement['neutral_receptors_mm']
    # The union run already declares these two controls in the v4 extension; do
    # not declare a condition twice (a duplicate id would make the corpus's
    # episode list disagree with the number of collected sessions).
    declared = {episode['condition'] for episode in plan['episodes']}
    extra_index = 0
    for split, index in (('train', 0), ('validation', 1), ('test', 2)):
        strength = (1., .85, 1.15)[index]
        additions = []
        for kind, channel, site in (('pheromone', 'funiculus', 'funiculus'), ('song', 'pedicel', 'pedicel')):
            for side in ('left', 'right'):
                spec = sources[kind][side]
                neutral = receptors[side + '_' + site]
                # Borrowed upstream plume grid: same declared geometry, different model.
                additions.append((f'{kind}_plume_{side}',
                    [{'id': 'conspecific_source', 'type': kind, 'position_mm': spec['position_mm'],
                      'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'], 'strength': strength,
                      'field': 'plume'}], [[], []]))
                # Following source: declared offset preserved relative to the neutral receptor.
                additions.append((f'{kind}_partner_{side}',
                    [{'id': 'conspecific_source', 'type': kind, 'position_mm': spec['position_mm'],
                      'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'], 'strength': strength,
                      'field': 'anchored', 'anchor': site, 'anchor_origin_mm': neutral}], [[], []]))
            far = dict(sources[kind]['left'])
            far['position_mm'] = [far['position_mm'][0] + 4., far['position_mm'][1], far['position_mm'][2]]
            additions.append((f'far_{kind}_control',
                [{'id': 'conspecific_source', 'type': kind, 'position_mm': far['position_mm'],
                  'size_mm': [.05, .05, .05], 'range_mm': far['range_mm'], 'strength': strength}], [[], []]))
        for side in ('left', 'right'):
            opposite = 'right' if side == 'left' else 'left'
            train = {'kind': 'touch', 'side': opposite, 'body_part': 'foreleg',
                     'strength': min(2., 1.5 * strength), 'onset_ms': 50.,
                     'duration_ms': 60., 'taps': 3, 'interval_ms': 140.}
            # Two windows per session: the train and a second train after withdrawal.
            additions.append((f'nudge_{side}', [], [[train], [train]]))
        alternating = [
            {'kind': 'touch', 'side': 'right', 'body_part': 'foreleg', 'strength': min(2., 1.5 * strength),
             'onset_ms': 50., 'duration_ms': 60., 'taps': 1},
            {'kind': 'touch', 'side': 'left', 'body_part': 'foreleg', 'strength': min(2., 1.5 * strength),
             'onset_ms': 210., 'duration_ms': 60., 'taps': 1},
            {'kind': 'touch', 'side': 'right', 'body_part': 'foreleg', 'strength': min(2., 1.5 * strength),
             'onset_ms': 370., 'duration_ms': 60., 'taps': 1}]
        additions.append(('nudge_alternating', [], [[*alternating], []]))
        for condition, objects, turns in additions:
            # Never declare a condition the union plan already carries: a duplicate
            # episode id would make the declared list disagree with the number of
            # sessions the collector actually runs.
            if split + '-' + condition in {episode['id'] for episode in plan['episodes']}:
                continue
            plan['episodes'].append({'id': split + '-' + condition, 'split': split,
                'condition': condition, 'seed': 950000 + extra_index * 10,
                'turns': turns, 'environment': {'schema': 'chatflypt.environment.v1',
                    'revision': 1, 'objects': validate_objects(objects)}})
            extra_index += 1
    plan['plume_provenance'] = __import__('chatflypt.plume_field', fromlist=['provenance']).provenance()
    plan['scope_additions'] = ['Borrowed upstream plume grid sampled at the funiculus (spatially and temporally structured)',
        'Following sources that keep their declared offset relative to the measured receptor pose',
        'Multi-tap nudges on the opposite foreleg and an alternating nudge, for turn-rich windows']
    plan['limitations'].append('The plume is upstream data from a different arena scale; it is not a cVA concentration. '
                               'The following source is a different physical claim from a fixed source and is labelled as such. '
                               'Nudge trains are engineering timings, and no direction code is claimed for them.')


def extend_v4_protocol(plan):
    """Add conspecific-signal conditions on top of the v3 confound controls.

    The v3 additions are kept, not replaced: the equal-geometry platform and
    darkness controls are what separate leg taste from merely standing on a
    platform, and the unilateral sources carry the laterality evidence. Rhythm
    contrasts hold total contact time fixed (180 ms) and vary only the number of
    taps. Emitter conditions use the placement-checked funiculus (cVA) and
    pedicel (song) fields, each with an out-of-range far control. Planned side is
    never a target: only actually delivered receptor events count.
    """
    from chatflypt.environment import validate_objects
    extend_v3_protocol(plan)
    if not V4_PLACEMENT.is_file():
        raise ValueError('Conspecific placement verification is required before freezing v4')
    placement = json.loads(V4_PLACEMENT.read_text(encoding='utf8'))
    sources = placement['local_sources']
    extra_index = 0
    for split, index in (('train', 0), ('validation', 1), ('test', 2)):
        strength = (1., .85, 1.15)[index]
        additions = []
        for name, timing in TAP_RHYTHMS:
            for side in ('left', 'right'):
                additions.append((f'{name}_{side}',
                    [], [[{'kind': 'touch', 'side': side, 'body_part': 'foreleg',
                           'strength': strength, 'onset_ms': 50., **timing}], []]))
        for kind in ('pheromone', 'song'):
            for side in ('left', 'right'):
                spec = sources[kind][side]
                additions.append((f'{kind}_{side}',
                    [{'id': 'conspecific_source', 'type': kind, 'position_mm': spec['position_mm'],
                      'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'], 'strength': strength}],
                    [[], []]))
            far = dict(sources[kind]['left'])
            far['position_mm'] = [far['position_mm'][0] + 4., far['position_mm'][1], far['position_mm'][2]]
            additions.append((f'far_{kind}_control',
                [{'id': 'conspecific_source', 'type': kind, 'position_mm': far['position_mm'],
                  'size_mm': [.05, .05, .05], 'range_mm': far['range_mm'], 'strength': strength}],
                [[], []]))
        for condition, objects, turns in additions:
            plan['episodes'].append({'id': split + '-' + condition, 'split': split,
                'condition': condition, 'seed': 940000 + extra_index * 10,
                'turns': turns, 'environment': {'schema': 'chatflypt.environment.v1',
                    'revision': 1, 'objects': validate_objects(objects)}})
            extra_index += 1
    plan['placement_reference_v4'] = {'path': str(V4_PLACEMENT.relative_to(ROOT)),
        'sha256': sha256(V4_PLACEMENT), 'scope': 'Fixed initial placement only; no tracking or tethering'}
    plan['scope_additions'] = ['Equal-total-contact tap rhythms (1x180 ms, 3x60 ms, 6x30 ms) on each foreleg',
        'Funiculus-field cVA source and pedicel-field song source, per side, plus an out-of-range far control for each',
        'Planned side is never a target; only actually delivered receptor events count']
    plan['limitations'].append('Tap timing and emitter field strengths are engineering parameter sets; the sources are not calibrated concentrations or sound pressures.')


def extend_v3_protocol(plan):
    """Predeclare physical confound controls and attempted unilateral scenes.

    The original 84 windows and their seeds remain byte-for-byte equivalent.
    These additions do not guarantee a receptor response or track the animal.
    """
    from chatflypt.environment import validate_objects
    if not V3_PLACEMENT.is_file():
        raise ValueError('Actual camera/receptor placement verification is required before freezing v3')
    placement = json.loads(V3_PLACEMENT.read_text(encoding='utf8'))
    sites = placement['neutral_receptors_mm']
    eyes = placement['neutral_eye_cameras_mm']
    if not placement.get('verified_no_initial_yeast_contact'):
        raise ValueError('Small yeast source must be verified not to collide with antennae')
    eye_separation = abs(eyes['left'][1] - eyes['right'][1])
    if eye_separation < .065:
        raise ValueError('Actual eyes cannot be separated by the supported fixed shade geometry')
    eye_width = min(.25, .8 * eye_separation)
    extra_index = 0
    for split, index in (('train', 0), ('validation', 1), ('test', 2)):
        shade = {'id': 'shade', 'type': 'shade', 'position_mm': [0, 0, 10],
                 'size_mm': [20, 20, 20], 'strength': 1.}
        block = {'id': 'platform', 'type': 'block', 'position_mm': [0, 0, .1], 'size_mm': [5., 4., .2]}
        sugar = dict(block, type='sugar')
        far = {'id': 'sample', 'type': 'sugar', 'position_mm': [4., -2., .5], 'size_mm': [1., 1., 1.]}
        additions = [('leg_block_control', [block]), ('far_sugar_control', [far]),
                     ('dark_leg_sugar_control', [shade, sugar]), ('dark_leg_block_control', [shade, block])]
        for target_side in ('left', 'right'):
            other = 'right' if target_side == 'left' else 'left'
            additions.append(('unilateral_bright_' + target_side,
                [{'id': 'opposite_eye_shade', 'type': 'shade',
                  'position_mm': [0., eyes[other][1], 10.],
                  'size_mm': [20., eye_width, 20.], 'strength': 1.}]))
            for kind, receptor in (('yeast', 'funiculus'), ('heat', 'arista'), ('cold', 'arista')):
                spec = placement['local_sources'][receptor][target_side]
                additions.append((f'unilateral_{kind}_{target_side}',
                    [{'id': 'local_source', 'type': kind, 'position_mm': spec['position_mm'],
                      'size_mm': [.05, .05, .05], 'range_mm': spec['range_mm'],
                      'strength': (1., .85, 1.15)[index]}]))
        for condition, objects in additions:
            plan['episodes'].append({'id': split + '-' + condition, 'split': split,
                'condition': condition, 'seed': 930000 + extra_index * 10,
                'turns': [[], []], 'environment': {'schema': 'chatflypt.environment.v1',
                    'revision': 1, 'objects': validate_objects(objects)}})
            extra_index += 1
    plan['placement_reference'] = {'path': str(V3_PLACEMENT.relative_to(ROOT)),
        'sha256': sha256(V3_PLACEMENT), 'scope': 'Fixed initial placement only; no tracking or tethering'}
    plan['scope_additions'] = ['Same-geometry sugar versus ordinary platform under light and darkness',
        'Distant visible sugar without planned contact',
        'Fixed small-range odor/temperature sources and opposing-eye shade; actual events decide side']
    plan['limitations'].append('A nominal unilateral scene can become bilateral or silent as the free body moves; planned side is never a target.')


def labels_for(catalogue, channels):
    labels, names = [], {}
    for channel in channels:
        spec = catalogue['channels'].get(channel)
        if not spec or spec['dataset'] != DATASET:
            continue
        group = spec.get('readout_group', channel)
        group = SEMANTIC_GROUPS.get(group, group)
        for side, prefix in (('left', '左侧'), ('right', '右侧')):
            if spec.get('side_ids', {}).get(side):
                label = group + ':' + side
                if label not in labels:
                    labels.append(label)
                    names[label] = prefix + spec['label']
    if not labels:
        raise ValueError('No declared MaleCNS environmental labels are available')
    return labels, names


def measured_targets(samples, events, catalogue, channels, dominance=False):
    """Offline targets require matching measured samples AND actual event IDs.

    Neither the named scene nor the intended probe side is sufficient. Matching
    each sample's time window avoids assigning a label from a later unrelated
    use of an overlapping receptor population.

    With ``dominance`` the antennal chemosensory/thermosensory groups use the
    declared side-dominance rule instead of any-delivery (see DOMINANCE_GROUPS);
    every other group keeps any-delivery.
    """
    if str(events['dataset']) != DATASET:
        raise ValueError('Targets cannot mix connectome namespaces')
    event_times = np.asarray(events['event_steps']) * float(events['dt_ms'])
    event_ids = np.asarray(events['event_neuron_ids']).astype(str)
    result = set()
    # The observation window is the transcription unit: per-side delivery is
    # summed over its samples and the dominance rule is applied once at the end,
    # otherwise a window in which each side wins one sample would label both.
    sided = {}
    for sample in samples:
        active = (event_times >= sample['time_ms']) & (event_times < sample['until_ms'])
        delivered = set(event_ids[active])
        for item in sample['items']:
            channel = item['channel']
            if channel not in channels or item.get('intensity', 1.) <= 0:
                continue
            spec = catalogue['channels'][channel]
            if spec['dataset'] != DATASET:
                continue
            side_ids = spec.get('body_part_ids', {}).get(item.get('body_part'), spec['side_ids'])
            group = SEMANTIC_GROUPS.get(spec.get('readout_group', channel), spec.get('readout_group', channel))
            sides = ('left', 'right') if item.get('side', 'bilateral') == 'bilateral' else (item['side'],)
            if dominance and group in DOMINANCE_GROUPS:
                # Items may arrive per side or bilateral; accumulate both ways and
                # decide once per window after all samples are summed.
                for side in sides:
                    counts = sided.setdefault(group, {'left': 0, 'right': 0})
                    counts[side] += len(delivered.intersection(map(str, side_ids.get(side, []))))
                continue
            for side in sides:
                if delivered.intersection(map(str, side_ids.get(side, []))):
                    result.add(group + ':' + side)
    for group, counts in sided.items():
        for side in ('left', 'right'):
            other = 'right' if side == 'left' else 'left'
            if counts[side] >= SIDE_MIN_EVENTS and (counts[other] == 0
                                                    or counts[side] >= SIDE_RATIO * counts[other]):
                result.add(group + ':' + side)
    return result


def write_observation(result, folder, episode, turn, contract, protocol_sha256):
    """Export an actual run_embodied result for the offline protocol.

    This hook lets integration acceptance collect once without rerunning a
    large brain. It rejects intervention trials instead of relabeling them.
    """
    folder = Path(folder)
    feedback, body, trial = result['feedback'], result['body'], result['trial']
    if (feedback.get('input_mode') != 'environment_only' or feedback.get('direct_neural_intervention') is not False or
            feedback.get('llm_called') is not False or feedback.get('goal_controller') is not False):
        raise ValueError('Unified environmental fitting requires a measured environment-only trial')
    if body.get('status') != 'ok' or not body.get('physics_run') or not body['summary']['finite_state'] or body['summary']['physics_warnings']:
        raise ValueError('A completed finite physical record without warnings is required')
    np.testing.assert_array_equal(np.bincount(trial.spike_ids, minlength=len(trial.counts)), trial.counts)
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / 'observation.npz'
    np.savez_compressed(raw, counts=trial.counts, state_record=trial.state_record,
        spike_indices=trial.spike_ids, spike_steps=trial.spike_steps,
        driven_indices=result['driven'], duration_ms=trial.duration_ms)
    event_path = folder / 'window-events.npz'
    np.savez_compressed(event_path, **result['events'])
    # Save the exact source evidence, even when a caller used another output_dir.
    dump(folder / 'measured-feedback.json', feedback)
    dump(folder / 'measured-body.json', body)
    row = {'episode': episode['id'], 'session_id': feedback['session_id'],
           'split': episode['split'], 'turn': turn,
           'contract': contract, 'protocol_sha256': protocol_sha256,
           'record': str(raw.resolve()), 'record_sha256': sha256(raw),
           'events': str(event_path.resolve()), 'events_sha256': sha256(event_path),
           'feedback': str((folder / 'measured-feedback.json').resolve()),
           'feedback_sha256': sha256(folder / 'measured-feedback.json'),
           'body': str((folder / 'measured-body.json').resolve()),
           'body_sha256': sha256(folder / 'measured-body.json')}
    dump(folder / 'observation.json', row)
    return row


@contextmanager
def episode_lock(episode_id, protocol_hash):
    """Fail closed on accidental overlapping workers; never overwrite a trial."""
    folder = OUT / 'locks'; folder.mkdir(parents=True, exist_ok=True)
    path = folder / (episode_id + '.lock')
    with path.open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'protocol_sha256': protocol_hash}, stream)
    try:
        yield
    finally:
        path.unlink()


def snapshot_sources(contract, catalogue):
    """Preserve frozen transduction code/assets, not just unrecoverable hashes."""
    folder = OUT / 'model-snapshot'; folder.mkdir(parents=True, exist_ok=True)
    entries = {}
    entries['code/unified_readout.py'] = (ROOT / 'chatflypt/unified_readout.py', contract['readout_features_sha256'])
    entries['code/train_unified_readout.py'] = (Path(__file__), sha256(__file__))
    for section in ('engine', 'physical_transduction'):
        for name, digest in contract[section].items():
            if digest is not None:
                entries['code/' + name] = (ROOT / 'chatflypt' / name, digest)
    source_paths = source_artifact_paths()
    for name, digest in contract['source_artifacts'].items():
        if digest is not None:
            source = source_paths[name]
            # Preserve directory structure so two modules called __init__.py
            # or two left/right source tables cannot overwrite one another.
            relative = str(source.relative_to(ROOT)).replace('\\', '/') if source.is_relative_to(ROOT) else name.replace(':', '_')
            entries['assets/' + relative] = (source, digest)
    for relative, (source, digest) in entries.items():
        target = folder / relative
        if sha256(source) != digest:
            raise ValueError('Model changed while freezing sources: ' + relative)
        if target.is_file():
            if sha256(target) != digest:
                raise ValueError('Cannot replace a prior model snapshot: ' + relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(target.suffix + f'.{os.getpid()}.pending')
        shutil.copyfile(source, pending)
        if sha256(pending) != digest:
            raise ValueError('Source changed during snapshot: ' + relative)
        pending.replace(target)
    # Both shards write identical complete JSON via atomic replacement.
    for name, value in (('contract.json', contract), ('catalogue.json', catalogue)):
        target = folder / name
        if target.is_file():
            if json.loads(target.read_text(encoding='utf8')) != value:
                raise ValueError('Cannot replace a prior model snapshot: ' + name)
            continue
        pending = folder / (name + f'.{os.getpid()}.pending')
        dump(pending, value); pending.replace(target)


def unique_episodes(plan):
    """One entry per episode id, keeping the first declaration.

    A protocol may declare the same condition twice by mistake (e.g. a union run
    whose additions overlap). The collector runs whichever declaration comes
    first and skips the second as already complete, so the corpus has one session
    per id; report the duplicates explicitly instead of hiding them, and never
    edit a frozen protocol file to remove them (that would invalidate every
    collected session).
    """
    seen, unique, duplicates = set(), [], []
    for episode in plan['episodes']:
        if episode['id'] in seen:
            duplicates.append(episode['id'])
            continue
        seen.add(episode['id'])
        unique.append(episode)
    return unique, duplicates


def extension_chain(plan, protocol_path, base_plan, base_protocol_path):
    """Verify that an extension protocol only appends episodes to a frozen one.

    Re-collecting a whole corpus costs hours, so an elicitation round may instead
    extend a frozen corpus: every base episode must appear in the new protocol with
    the same id, split, condition, turns and environment. Only then may a fit read
    the union of the two runs, and each row must still carry the protocol hash of
    the run it was actually collected under.

    Seeds are compared and *reported* rather than required: they only steer the
    neural random stream of a re-run, and the base corpus is reused exactly as
    recorded, never regenerated. The record states how many seeds differ and why,
    so a reader can see that the reused rows keep their own seeds.
    """
    def canonical(episode):
        return json.dumps({key: episode[key] for key in
                           ('id', 'split', 'condition', 'turns', 'environment')},
                          sort_keys=True, ensure_ascii=False)
    index = {episode['id']: episode for episode in plan['episodes']}
    changed = [{'id': episode['id'],
                'problem': 'missing from the extension protocol' if episode['id'] not in index
                           else 'episode differs from the base protocol'}
               for episode in base_plan['episodes']
               if episode['id'] not in index or canonical(index[episode['id']]) != canonical(episode)]
    if changed:
        raise ValueError('Extension protocol is not a superset of the base protocol: '
                         + json.dumps(changed[:5], ensure_ascii=False))
    seed_mismatches = [episode['id'] for episode in base_plan['episodes']
                       if index[episode['id']]['seed'] != episode['seed']]
    base_ids = {episode['id'] for episode in base_plan['episodes']}
    appended = [episode['id'] for episode in plan['episodes'] if episode['id'] not in base_ids]
    return {'base_protocol_sha256': sha256(base_protocol_path), 'protocol_sha256': sha256(protocol_path),
            'base_episodes': len(base_plan['episodes']), 'appended_episodes': len(appended),
            'appended_examples': appended[:12], 'verified': True,
            'seed_mismatches': len(seed_mismatches),
            'seed_mismatch_note': 'Differs only in the RNG seed of the base episode; the reused rows keep '
                                  'the seeds they were collected with, and no base episode is regenerated.',
            'seed_mismatch_examples': seed_mismatches[:6],
            'scope': 'Every base episode keeps its id, split, condition, turns and environment in the '
                     'extension protocol; a union fit may mix exactly these two protocol revisions, and '
                     'each row keeps the hash of its own run.'}


def collect(plan, protocol_path, split='all', limit=None, shard_index=0, shard_count=1, base=None):
    from chatflypt.malecns import MaleCNS
    from chatflypt.sensory_catalogue import build_catalogue
    from chatflypt.embodied import EmbodiedSession, run_embodied
    brain = MaleCNS()
    catalogue = build_catalogue(save=False)
    contract = model_contract(brain, catalogue)
    snapshot_sources(contract, catalogue)
    protocol_hash = sha256(protocol_path)
    episodes, duplicates = unique_episodes(plan)
    if duplicates:
        print(json.dumps({'note': 'protocol declares duplicate episode ids; one session per id',
                          'duplicates': sorted(set(duplicates))}, ensure_ascii=False), flush=True)
    missing = [c for c in plan['channels'] if c.startswith('mc_') and c not in catalogue['channels']]
    if missing:
        raise ValueError('Unified environmental catalogue is not ready: ' + ', '.join(missing))
    completed = 0
    if not 0 <= shard_index < shard_count:
        raise ValueError('Shard index must be in [0, shard_count)')
    chains = []
    bases = []
    if base is not None:
        for base_run, base_protocol_path in base:
            chain = extension_chain(plan, protocol_path,
                                   json.loads(Path(base_protocol_path).read_text(encoding='utf8')),
                                   base_protocol_path)
            chains.append(chain)
            bases.append((base_run, OUT.parent / base_run / 'trials', chain['base_protocol_sha256']))
        print(json.dumps({'extension_chains': chains}, ensure_ascii=False), flush=True)
    for episode_index, episode in enumerate(episodes):
        if episode_index % shard_count != shard_index:
            continue
        if split != 'all' and episode['split'] != split:
            continue
        folder = OUT / 'trials' / episode['id']
        complete = folder / 'complete.json'
        if complete.is_file():
            saved = json.loads(complete.read_text(encoding='utf8'))
            if saved['contract'] != contract or saved['protocol_sha256'] != protocol_hash:
                raise ValueError('Existing corpus is stale; choose another output directory, do not overwrite evidence')
            continue
        reused = None
        for base_run, base_trials, base_hash in bases:
            if (base_trials / episode['id'] / 'complete.json').is_file():
                saved = json.loads((base_trials / episode['id'] / 'complete.json').read_text(encoding='utf8'))
                if saved['contract'] != contract or saved['protocol_sha256'] != base_hash:
                    raise ValueError(f'Base corpus {base_run} is stale; it cannot serve as an extension base')
                reused = base_run
                break
        if reused is not None:
            continue
        if limit is not None and completed >= limit:
            break
        with episode_lock(episode['id'], protocol_hash):
            session = EmbodiedSession(episode['environment'], 'ground', 'unified-' + episode['id'])
            rows = []
            try:
                for turn, presentations in enumerate(episode['turns']):
                    dest = folder / f'turn-{turn}'
                    result = run_embodied(brain, brain.rows, catalogue, DATASET, [], episode['environment'],
                        episode['seed'] + turn, 'ground', duration_ms=plan['duration_ms'],
                        output_dir=dest, session=session, input_mode='environment_only', presentations=presentations)
                    row = write_observation(result, dest, episode, turn, contract, protocol_hash)
                    rows.append(row)
                    print(json.dumps({'episode': episode['id'], 'turn': turn,
                        'recorded_spikes': int(result['trial'].counts.sum()),
                        'channels': sorted({i['channel'] for s in result['feedback']['samples'] for i in s['items']})}), flush=True)
            finally:
                session.close()
            dump(complete, {'contract': contract, 'protocol_sha256': protocol_hash, 'rows': rows})
        completed += 1


def validate_splits(rows):
    groups = {split: {r['session_id'] for r in rows if r['split'] == split}
              for split in ('train', 'validation', 'test')}
    if any(not value for value in groups.values()):
        raise ValueError('Independent train, validation and test sessions are required')
    if any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('A neural/body session cannot occur in more than one split')
    if len({(r['episode'], r['turn']) for r in rows}) != len(rows):
        raise ValueError('Duplicate observation in corpus')
    if len({str(Path(r['record']).resolve()) for r in rows}) != len(rows):
        raise ValueError('A recorded observation cannot be reused under another session name')
    return groups


def metrics(truth, predicted, labels):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp = (truth & predicted).sum(0)
    fp = (~truth & predicted).sum(0)
    fn = (truth & ~predicted).sum(0)
    return {'windows': len(truth), 'exact_set_accuracy': float(np.all(truth == predicted, axis=1).mean()),
            'micro_f1': float(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())),
            'per_label': {w: {'tp': int(t), 'fp': int(f), 'fn': int(n),
                'positives': int(y.sum()), 'negatives': int((~y).sum()),
                'precision': float(t / max(1, t + f)), 'recall': float(t / max(1, t + n))}
                for w, t, f, n, y in zip(labels, tp, fp, fn, truth.T)}}


def choose_thresholds(scores, truth, signal, training_truth, rules, training_scores=None):
    """Independent validation only, per label; one failure does not disable all.

    Candidate thresholds are absolute (0.2..0.8) *and* per-label quantiles of the
    label's own **training** scores. The second family exists because Ridge
    shrinkage is label-dependent: under the v7 corpus the yeast scores sat below
    the whole absolute grid while other labels sat at 0.3-0.5, so no absolute
    threshold could ever fire and the label was reported as unreadable when the
    problem was the threshold scale. Quantiles are computed from the training split
    only; validation still selects the final threshold and the qualification.
    """
    thresholds, qualified = [], []
    for index in range(truth.shape[1]):
        target = truth[:, index]
        supported = (target.sum() >= rules['minimum_positive_windows'] and
                     (~target).sum() >= rules['minimum_negative_windows'] and
                     training_truth[:, index].any() and (~training_truth[:, index]).any())
        candidates = list(np.linspace(.2, .8, 61))
        if training_scores is not None:
            column = training_scores[:, index]
            if column.size:
                candidates.extend(np.quantile(column, np.linspace(.5, .999, 60)).tolist())
        choices = []
        for threshold in sorted(set(round(float(t), 6) for t in candidates)):
            predicted = (scores[:, index] >= threshold) & signal
            tp = int(np.sum(predicted & target)); fp = int(np.sum(predicted & ~target)); fn = int(np.sum(~predicted & target))
            passes = bool(supported and tp / max(1, tp + fp) >= rules['precision'] and tp / max(1, tp + fn) >= rules['recall'])
            choices.append((passes, 2 * tp / max(1, 2 * tp + fp + fn), -abs(threshold - .5), float(threshold)))
        best = max(choices)
        thresholds.append(best[3]); qualified.append(best[0])
    return np.asarray(thresholds), np.asarray(qualified, bool)


def fit(plan, protocol_path, manifest=None, base=None):
    from chatflypt.malecns import MaleCNS
    from chatflypt.sensory_catalogue import build_catalogue
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits
    brain = MaleCNS(); catalogue = build_catalogue(save=False)
    contract = model_contract(brain, catalogue); allowed = blind_mask(brain, catalogue)
    protocol_hash = sha256(protocol_path)
    rows = []
    episodes, duplicates = unique_episodes(plan)
    if duplicates:
        print(json.dumps({'note': 'protocol declares duplicate episode ids; one session per id',
                          'duplicates': sorted(set(duplicates))}, ensure_ascii=False), flush=True)
    chains = []
    tagged = []
    if base is not None:
        for base_run, base_protocol_path in base:
            chain = extension_chain(plan, protocol_path,
                                    json.loads(Path(base_protocol_path).read_text(encoding='utf8')),
                                    base_protocol_path)
            chains.append(chain)
            tagged.append((base_run, OUT.parent / base_run / 'trials', chain['base_protocol_sha256']))
        print(json.dumps({'extension_chains': chains}, ensure_ascii=False), flush=True)
    if manifest:
        rows = [(row, protocol_hash) for row in json.loads(Path(manifest).read_text(encoding='utf8'))['rows']]
    else:
        for episode in episodes:
            complete = OUT / 'trials' / episode['id'] / 'complete.json'
            if complete.is_file():
                rows.extend((row, protocol_hash)
                            for row in json.loads(complete.read_text(encoding='utf8'))['rows'])
                continue
            for base_run, base_trials, base_hash in tagged:
                base_complete = base_trials / episode['id'] / 'complete.json'
                if base_complete.is_file():
                    rows.extend((row, base_hash)
                                for row in json.loads(base_complete.read_text(encoding='utf8'))['rows'])
                    break
    groups = validate_splits([row for row, _ in rows])
    labels, names = labels_for(catalogue, plan['channels'])
    dominance_rule = bool((plan.get('target_dominance') or {}).get('applied', False))
    for row, expected_hash in rows:
        if row['contract'] != contract:
            raise ValueError('Corpus model fingerprint mismatch')
        if row['protocol_sha256'] != expected_hash:
            raise ValueError('A row does not match the protocol of the run it came from')
    rows = [row for row, _ in rows]

    def load(split, selected=None):
        vectors, truth, signals, source_rows, durations = [], [], [], [], []
        checkpoints = {}
        for row in rows:
            if row['split'] != split:
                continue
            for key in ('record', 'events', 'feedback', 'body'):
                if sha256(row[key]) != row[key + '_sha256']:
                    raise ValueError('Measured source checksum mismatch: ' + key)
            feedback = json.loads(Path(row['feedback']).read_text(encoding='utf8'))
            body = json.loads(Path(row['body']).read_text(encoding='utf8'))
            if (feedback.get('session_id') != row['session_id'] or
                    feedback.get('input_mode') != 'environment_only' or feedback.get('direct_neural_intervention') is not False or
                    feedback.get('llm_called') is not False or feedback.get('goal_controller') is not False or
                    body.get('status') != 'ok' or not body.get('physics_run') or not body['summary']['finite_state'] or body['summary']['physics_warnings']):
                raise ValueError('Corpus contains a non-environmental or failed physical trial')
            previous = checkpoints.get(row['session_id'])
            if previous is None:
                if feedback.get('session_start_ms') != 0:
                    raise ValueError('Corpus omitted the start of a persistent session')
            elif (feedback.get('session_start_ms') != previous['session_end_ms'] or
                    feedback.get('initial_neural_state_sha256') != previous['neural_state_sha256'] or
                    feedback.get('neural_reset_count') != 0 or feedback.get('body_reset_count') != 0):
                raise ValueError('Recorded session continuity failed')
            checkpoints[row['session_id']] = feedback
            with np.load(row['events'], allow_pickle=False) as events:
                expected = measured_targets(feedback['samples'], events, catalogue, plan['channels'],
                                            dominance=dominance_rule)
            with np.load(row['record'], allow_pickle=False) as raw:
                counts, state = raw['counts'], raw['state_record']
                duration = float(raw['duration_ms'])
                if len(counts) != len(brain.ids) or np.any(allowed[raw['driven_indices']]):
                    raise ValueError('Actual direct inputs escaped the blind feature mask')
                np.testing.assert_array_equal(np.bincount(raw['spike_indices'], minlength=len(counts)), counts)
                vector = internal_features(counts, state, duration)
                vectors.append(vector if selected is None else vector[selected])
                indices = np.flatnonzero(allowed) if selected is None else np.unique(selected % len(brain.ids))
                signals.append(bool(np.any(counts[indices]) or np.any(state[2:, indices] > MIN_VOLTAGE_MV)))
                durations.append(duration)
            truth.append([w in expected for w in labels]); source_rows.append(row)
        return np.asarray(vectors), np.asarray(truth, bool), np.asarray(signals, bool), source_rows, durations

    xt, yt, _, train_rows, durations = load('train')
    # Training alone chooses features. Validation/test never add active cells.
    selected = np.flatnonzero(np.tile(allowed, 3) & np.any(xt != 0., axis=0))
    if not len(selected):
        raise ValueError('Training has no blind internal neural features')
    xt = xt[:, selected]
    xv, yv, sv, validation_rows, vd = load('validation', selected)
    with threadpool_limits(limits=2):
        model = Ridge(alpha=plan['alpha'], solver='cholesky').fit(xt, yt.astype(float))
        validation_scores = model.predict(xv)
        training_scores = model.predict(xt)
    thresholds, qualified = choose_thresholds(validation_scores, yv, sv, yt, plan['qualification'],
                                              training_scores)
    metadata = {'schema': SCHEMA, 'contract': contract, 'input_mode': 'environment_only',
        'split_unit': 'whole_session', 'protocol_sha256': protocol_hash,
        'tested_durations_ms': sorted(set(durations) & set(vd)), 'candidate_only': True,
        'qualification': plan['qualification'], 'dataset': DATASET,
        'source_groups': {k: sorted(v) for k, v in groups.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep unfinished evaluation out of the live loader's default path.
    artifact = OUT / 'malecns-decoder.candidate.npz'
    temporary = artifact.with_suffix('.pending.npz')
    np.savez_compressed(temporary, feature_indices=selected,
        indices=np.unique(selected % len(brain.ids)), weights=model.coef_, intercept=model.intercept_,
        thresholds=thresholds, qualified=qualified, labels=labels, names=[names[w] for w in labels],
        metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(artifact)
    locked_hash = sha256(artifact)
    # The test split remains unopened until the artifact is fixed.
    reader = UnifiedDecoder(brain, catalogue, artifact)
    xe, ye, se, test_rows, td = load('test', selected)
    predicted = (model.predict(xe) >= thresholds) & se[:, None] & qualified
    served = []
    for row in test_rows:
        with np.load(row['record'], allow_pickle=False) as raw:
            output = reader.decode(raw['counts'], raw['state_record'], float(raw['duration_ms']))
            served.append([w in output['words'] for w in labels])
    np.testing.assert_array_equal(served, predicted)
    zero = reader.decode(np.zeros(len(brain.ids)), np.zeros((4, len(brain.ids))), 500.)
    assert not zero['words'] and not zero['unqualified_words']
    report = {'dataset': DATASET, 'artifact': str(artifact.relative_to(ROOT)),
        'artifact_sha256': locked_hash, 'protocol_sha256': protocol_hash, 'contract': contract,
        'features': len(selected), 'all_sensory_and_direct_cells_excluded': bool(np.all(allowed[reader.indices])),
        'train_windows': len(yt), 'validation_windows': len(yv), 'test_windows': len(ye),
        'groups': {k: sorted(v) for k, v in groups.items()}, 'qualified_labels': [w for w, q in zip(labels, qualified) if q],
        'unqualified_labels': [w for w, q in zip(labels, qualified) if not q],
        'validation': metrics(yv, (validation_scores >= thresholds) & sv[:, None] & qualified, labels),
        'test': metrics(ye, predicted, labels),
        'shuffled_control': metrics(ye, predicted[np.random.default_rng(92741).permutation(len(predicted))], labels),
        'zero_false_positives': len(zero['words']), 'serving_matches_test_matrix': True,
        'selection': 'Train selects features/weights; validation selects thresholds and per-label qualification; saved before opening test.',
        'extension': chains or None,
        'candidate_only': True, 'biologically_calibrated': False,
        'test_predictions': [{'episode': r['episode'], 'turn': r['turn'],
            'expected': [w for w, y in zip(labels, truth) if y],
            'predicted': [w for w, p in zip(labels, prediction) if p]}
            for r, truth, prediction in zip(test_rows, ye, predicted)],
        'limitations': plan['limitations'] + ['No held-out test failure is retuned away in this run.',
            'Whole-brain dynamics, receptor transduction and motor behavior are not calibrated by this classifier.']}
    assert sha256(artifact) == locked_hash
    dump(REPORT / 'evaluation.json', report)
    print(json.dumps({k: report[k] for k in ('qualified_labels', 'unqualified_labels', 'test', 'zero_false_positives')}, ensure_ascii=False), flush=True)
    return report


def main():
    global OUT, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--split', choices=('all', 'train', 'validation', 'test'), default='all')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--run-id', help='New immutable corpus namespace after a model/distribution change')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--base-run', help='Frozen run id(s) this protocol extends; that corpus is reused, not recollected')
    parser.add_argument('--base-protocol', type=Path, help='Protocol file of a frozen base run (single base only)')
    args = parser.parse_args()
    base = None
    if args.base_run:
        names = [name for name in args.base_run.split(',') if name]
        base = []
        for index, name in enumerate(names):
            protocol_file = args.base_protocol if (args.base_protocol and len(names) == 1) else \
                ROOT / 'reports/unified-readout' / 'runs' / name / 'protocol.json'
            base.append((name, protocol_file))
    if args.run_id:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', args.run_id):
            raise ValueError('Run ID must be a short lowercase filesystem-safe identifier')
        OUT = OUT / 'runs' / args.run_id
        REPORT = REPORT / 'runs' / args.run_id
    protocol_path = args.protocol or REPORT / 'protocol.json'
    if protocol_path.is_file():
        plan = json.loads(protocol_path.read_text(encoding='utf8'))
    else:
        plan = protocol(args.run_id); dump(protocol_path, plan)
    if args.collect:
        collect(plan, protocol_path, args.split, args.limit, args.shard_index, args.shard_count, base)
    if args.fit:
        fit(plan, protocol_path, args.manifest, base)
    if not args.collect and not args.fit:
        print(str(protocol_path))


if __name__ == '__main__':
    main()
