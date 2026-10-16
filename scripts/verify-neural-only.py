"""Verify the live fly interface without a model adviser or command feedback."""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/neural-expression-only-2026-09-20'


def request(route, payload=None):
    req = urllib.request.Request('http://127.0.0.1:8765' + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.load(response)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = request('/api/status')
    assert not status['llm_in_pathway'] and not status['external_action_control']
    cases = [
        ('right_foreleg_touch', {'message': '模拟右前腿触碰'}, ['foreleg_touch:right'],
         'e87ec9b359ed4b9babbce04d1b499a96'),
        ('five_senses', {'kind': 'words', 'words': ['vinegar','yeast','sugar','warm','bright']},
         ['vinegar','yeast','sugar','warm','bright'], 'a9f035ce13394e3698cd24835d6c5224'),
        ('blank', {'kind': 'stimuli', 'items': []}, [], None),
    ]
    rows=[]
    for name, payload, labels, baseline_id in cases:
        started=time.perf_counter()
        # Deliberately imitate a stale page which used to enable model control.
        payload.update(seed=81000, backend='configured', feedback_enabled=True)
        result=request('/api/converse',payload)
        (OUT/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
        dialogue=result['dialogue']; observation=result['observation']
        assert dialogue['status']=='neural_only' and dialogue['max_feedback_rounds']==0
        assert not dialogue['llm_called'] and not dialogue['external_action_control']
        assert not dialogue['feedback_enabled'] and len(result['observations'])==1
        assert [m['kind'] for m in result['messages']]==['sensation']
        assert set(observation['decoded']['words'])==set(labels)
        assert result['reply']==observation['fly_text']
        assert 'model_response' not in dialogue and 'adviser_request' not in dialogue
        unchanged=None
        if baseline_id:
            baseline=json.loads((ROOT/f'reports/dialogue/{baseline_id}.json').read_text(encoding='utf8'))['observation']
            with np.load(ROOT/baseline['raw_record']['file']) as before, np.load(ROOT/observation['raw_record']['file']) as after:
                for field in ['counts','spike_indices','spike_steps','driven_indices']:
                    np.testing.assert_array_equal(before[field],after[field])
            assert observation['scene']['body']['command_summary']==baseline['scene']['body']['command_summary']
            unchanged=True
        question=request('/api/chat',{'message':'你感受到了什么？','backend':'local','observation_id':observation['id']})
        assert question['reply']==observation['fly_text'] and not question['llm_called']
        assert not question['neural_simulation_run']
        row={'case':name,'passed':True,'reply':result['reply'],'observation_id':observation['id'],
             'seconds':time.perf_counter()-started,'original_neural_and_body_drive_unchanged':unchanged,
             'body_status':observation['scene']['body']['status']}
        rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    # Human direction is an input translation, not model-originated advice.
    translated=request('/api/communicate',{'message':'左转','backend':'local','feedback_enabled':True,'seed':81000})
    assert translated['handled'] and len(translated['observations'])==1
    assert translated['input_plan']['items'][0]['side']=='right'
    assert translated['input_plan']['translation']['forces_body_action'] is False
    assert translated['reception']['description']=='模拟果蝇的右前腿被触碰'
    assert translated['reply']=='我感到右前腿被碰到了。'
    assert not translated['dialogue']['llm_called'] and not translated['dialogue']['feedback_enabled']
    (OUT/'human-left.json').write_text(json.dumps(translated,ensure_ascii=False),encoding='utf8')
    report={'passed':True,'cases':rows,'human_direction_translation_preserved':True,
            'autonomous_model_action_feedback_removed':True,
            'legacy_enabled_options_cannot_restore_adviser':True}
    (OUT/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
