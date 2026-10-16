"""Semantic map: signal x context x reaction, with holes left explicit.

Borrowed from how underdescribed languages are handled: a form is not given one
gloss, it is placed on a map of the contexts it occurs in and the reactions that
follow it, and *absent* combinations stay visible as holes instead of being filled
in. Everything here is measured: signals are actually delivered receptor items,
contexts are the applied scene and the presented probe, reactions are the
measured body outcome (and, when available, the spoken readout words).

No human message, no intended stimulus and no scene *name* is used as a meaning.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.intent_readout import outcome_rules, TURN_QUANTILE, ACTIVITY_QUANTILE

CORPUS = ROOT / 'reports/corpus'
MIN_WINDOWS = 3


def context_family(episode):
    """Context label from the *condition* name only, never from a scene description."""
    name = episode.split('-', 1)[1] if '-' in episode else episode
    for family, markers in (('probe', ('touch_', 'tap', 'nudge', 'unilateral')),
                            ('platform', ('leg_sugar', 'leg_block', 'mouth_')),
                            ('odor', ('yeast', 'pheromone', 'cares')),
                            ('song', ('song',)),
                            ('thermal', ('heat', 'cold', 'warm')),
                            ('visual', ('bright', 'dark'))):
        if any(marker in name for marker in markers):
            return family
    return 'other'


def reaction_of(outcome, rules):
    """Discretise the measured outcome into reportable reactions."""
    reactions = []
    if outcome:
        if outcome['yaw_change_deg'] >= rules['turn_deg']:
            reactions.append('turn_left')
        elif outcome['yaw_change_deg'] <= -rules['turn_deg']:
            reactions.append('turn_right')
        reactions.append('active' if outcome['path_length_mm'] >= rules['move_mm'] else 'still')
    return reactions or ['no_measurement']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v4')
    args = parser.parse_args()
    index_path = CORPUS / f'{args.run_id}-index.json'
    if not index_path.is_file():
        raise SystemExit('No corpus index at ' + str(index_path))
    index = json.loads(index_path.read_text(encoding='utf8'))
    windows = [w for w in index['windows'] if w['metadata']['split'] in ('train', 'validation', 'test')]

    # Rules come from the training split's measured outcomes, as elsewhere.
    training = [w['behaviour']['outcome'] for w in windows
                if w['metadata']['split'] == 'train' and w['behaviour']['outcome']]
    rules = outcome_rules(training) if training else {'turn_deg': 5., 'move_mm': .5}

    signal_context = defaultdict(Counter)
    signal_reaction = defaultdict(Counter)
    signal_sessions = defaultdict(set)
    context_windows = Counter()
    signal_windows = Counter()
    for window in windows:
        context = context_family(window['metadata']['episode'])
        context_windows[context] += 1
        reactions = reaction_of(window['behaviour']['outcome'], rules)
        signals = sorted(window['measured']['delivered_items'])
        if not signals:
            signal_context['(no delivered signal)'][context] += 1
            signal_reaction['(no delivered signal)'].update(reactions)
        for signal in signals:
            signal_windows[signal] += 1
            signal_sessions[signal].add(window['metadata']['session_id'])
            signal_context[signal][context] += 1
            signal_reaction[signal].update(reactions)

    signals = sorted(signal_windows)
    contexts = sorted(context_windows)
    matrix = [[signal_context[signal][context] or None for context in contexts] for signal in signals]
    rows = []
    for signal in signals:
        total = signal_windows[signal]
        distribution = {reaction: round(count / total, 3) for reaction, count in signal_reaction[signal].most_common()}
        rows.append({'signal': signal, 'windows': total, 'sessions': len(signal_sessions[signal]),
                     'contexts': {context: (signal_context[signal][context] or None) for context in contexts},
                     'reactions': distribution,
                     'holes': [context for context in contexts if not signal_context[signal][context]]})
    # Cluster signals by their context profile (the "semantic field" analogue).
    fields = []
    if len(signals) >= 2:
        profiles = np.asarray([[signal_context[signal][context] for context in contexts] for signal in signals], dtype=float)
        rowsum = profiles.sum(axis=1, keepdims=True)
        profiles = np.divide(profiles, rowsum, out=np.zeros_like(profiles), where=rowsum > 0)
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=min(4, len(signals)), metric='euclidean',
                                         linkage='average').fit_predict(profiles)
        groups = defaultdict(list)
        for signal, label in zip(signals, labels):
            groups[int(label)].append(signal)
        fields = [{'field': index, 'signals': members} for index, members in sorted(groups.items())]
    report = {'schema': 'chatflypt.semantic-map.v1', 'run_id': args.run_id,
              'source_index': str(index_path.relative_to(ROOT)), 'windows': len(windows),
              'rules': rules,
              'scope': 'Measured signals x measured contexts x measured reactions; holes are explicit.',
              'contexts': contexts, 'context_windows': dict(context_windows),
              'signals': rows, 'cooccurrence': {'signals': signals, 'contexts': contexts, 'counts': matrix},
              'fields_by_context_profile': fields,
              'limitations': ['A signal appearing in several contexts is polysemy, not several words.',
                              'One session per condition per split; counts are not independent replicates.',
                              'No human message or intended stimulus enters any cell of this map.']}
    dump(CORPUS / f'{args.run_id}-semantic-map.json', report)
    print(json.dumps({'signals': len(signals), 'contexts': contexts, 'fields': fields,
                      'holes_total': sum(len(row['holes']) for row in rows)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
