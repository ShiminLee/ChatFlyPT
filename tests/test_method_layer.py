"""Method-layer artifacts stay present and well-formed.

These are the documentary-linguistics borrowings: an elicitation battery, a
negative registry, a data statement, a corpus index and an agreement report.
They are evidence, so their schemas are asserted rather than trusted.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'reports/corpus'
ELICITATION = ROOT / 'reports/elicitation'
REQUIRED_NEGATIVE_KEYS = {'id', 'claim_tested', 'result', 'status', 'evidence', 'consequence'}


@pytest.mark.parametrize('run_id', ['retinal-axes-v3', 'conspecific-v4'])
def test_elicitation_battery_covers_every_condition(run_id):
    path = ELICITATION / f'{run_id}-battery.json'
    if not path.is_file():
        pytest.skip(f'{run_id} battery not generated yet')
    report = json.loads(path.read_text(encoding='utf8'))
    assert report['schema'] == 'chatflypt.elicitation-battery.v1'
    assert report['questions']
    for entry in report['questions']:
        assert entry['conditions'] and entry['contrast'] and entry['criterion']
    # No condition may be collected without a question attached to it.
    assert report['uncovered_conditions'] == []


def test_negative_registry_is_complete_and_unique():
    path = CORPUS / 'negative-registry.json'
    if not path.is_file():
        pytest.skip('negative registry not written yet')
    registry = json.loads(path.read_text(encoding='utf8'))
    assert registry['schema'] == 'chatflypt.negative-registry.v1'
    entries = registry['entries']
    assert len(entries) >= 10
    assert len({entry['id'] for entry in entries}) == len(entries)
    for entry in entries:
        assert REQUIRED_NEGATIVE_KEYS <= set(entry)
        assert entry['status'] in {'refuted', 'unqualified', 'unmapped', 'superseded', 'retracted'}
        assert entry['evidence']


def test_data_statement_keeps_the_required_sections():
    path = CORPUS / 'data-statement.md'
    if not path.is_file():
        pytest.skip('data statement not written yet')
    text = path.read_text(encoding='utf8')
    for heading in ('整理理由', '标注者', '局限', '不能回答的问题'):
        assert heading in text


@pytest.mark.parametrize('run_id', ['retinal-axes-v3'])
def test_agreement_report_separates_abstention_from_disagreement(run_id):
    path = CORPUS / f'{run_id}-agreement.json'
    if not path.is_file():
        pytest.skip(f'{run_id} agreement report not generated yet')
    report = json.loads(path.read_text(encoding='utf8'))
    assert report['schema'] == 'chatflypt.readout-agreement.v1'
    assert report['sessions'] > 0 and report['windows'] > 0
    qualified = set(report['qualified_labels'])
    for label, value in report['cohens_kappa_by_label'].items():
        if label in qualified:
            assert value is None or isinstance(value, float)
        else:
            assert isinstance(value, dict) and 'abstained' in value['status']
    for key in ('qualified_labels', 'all_labels'):
        assert 0. <= report['session_accuracy'][key]['min'] <= 1.
        assert 0. <= report['session_accuracy'][key]['max'] <= 1.


@pytest.mark.parametrize('run_id', ['conspecific-v4', 'conspecific-v5'])
def test_semantic_map_keeps_holes_explicit_and_columns_measured(run_id):
    path = CORPUS / f'{run_id}-semantic-map.json'
    if not path.is_file():
        pytest.skip(f'{run_id} semantic map not generated yet')
    report = json.loads(path.read_text(encoding='utf8'))
    assert report['schema'] == 'chatflypt.semantic-map.v1'
    assert report['contexts'] and report['signals']
    for row in report['signals']:
        # Every signal lists every context, and a context it never occurred in is
        # an explicit hole - never silently treated as a meaning.
        assert set(row['contexts']) == set(report['contexts'])
        assert sum(1 for value in row['contexts'].values() if value is None) == len(row['holes'])
        assert row['reactions'], 'a signal must carry its measured reactions'
    # The map may not carry a human-message column of any kind.
    text = json.dumps(report, ensure_ascii=False).lower()
    for forbidden in ('human_message', 'intended_stimulus', 'input_plan', 'llm'):
        assert forbidden not in text


def test_corpus_index_keeps_the_four_layers_separate(run_id='retinal-axes-v3'):
    path = CORPUS / f'{run_id}-index.json'
    if not path.is_file():
        pytest.skip(f'{run_id} index not generated yet')
    index = json.loads(path.read_text(encoding='utf8'))
    assert index['schema'] == 'chatflypt.corpus-index.v1'
    assert index['summary']['windows'] == len(index['windows']) > 0
    assert index['summary']['interventions'] == 0
    assert index['summary']['llm_calls'] == 0
    window = index['windows'][0]
    for layer in ('context', 'measured', 'internal', 'behaviour', 'metadata'):
        assert layer in window
    assert 'delivered_items' in window['measured']
    assert window['context']['input_mode'] == 'environment_only'
