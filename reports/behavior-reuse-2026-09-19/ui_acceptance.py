"""Audit saved responses from the two real browser form submissions."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = [('ground', '45ea825e4857468e8166fb5d4451f978'),
         ('tethered_flight', 'd919872880944f67b35328e307bb0e17')]
rows = []
for context, identifier in CASES:
    path = ROOT / 'reports/general-sensory/interactive' / f'{identifier}.json'
    response = json.loads(path.read_text(encoding='utf-8'))
    obs = response['observation']
    body = obs['scene']['body']
    assert response['reception']['intent'] == '左侧冷刺激'
    assert response['reply_source'] == 'neural_readout'
    assert body['controller_id'] == 'badapple_hybrid_v1'
    assert body['context'] == context and body['status'] == 'ok'
    assert len(body['trajectory']) == 51 and body['duration_ms'] == 500.
    assert body['summary']['finite_state'] and body['summary']['physics_warnings'] == 0
    for frame in body['trajectory']:
        if frame['neural_command_active']:
            assert frame['command_available_at_ms'] < frame['time_ms']
        if context == 'tethered_flight':
            assert frame['position_mm'] == [0., 0., 2.]
            assert frame['commands']['left'] == frame['commands']['right'] == 0.
    rows.append(dict(context=context, observation_id=identifier, seed=obs['seed'],
                     response_file=str(path.relative_to(ROOT)),
                     response_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     spikes=obs['spikes'], motor_spikes=body['motor_spikes'],
                     frames=len(body['trajectory']), summary=body['summary'],
                     command_summary=body['command_summary']))
report = dict(cases=rows, assertions_passed=True,
              browser_checks=['Chinese message form to neural response and Hybrid replay',
                              'Timeline changed from 500 ms to recorded 0 ms',
                              'Side/top view and fullscreen controls operated',
                              'Explicit tethered mode selected via experiment settings',
                              'Ground restored as next-run default'],
              note='Browser tests used separate UI-generated seeds; this is not the same-record controller comparison.')
(HERE/'ui-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
