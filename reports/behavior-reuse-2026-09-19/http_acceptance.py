"""Exercise the running default backend without any chat API call."""
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = 'http://127.0.0.1:8765'


def call(route, payload=None):
    request = urllib.request.Request(BASE + route,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    cases = [
        ('right_foreleg_touch', '/api/communicate', {'message': '模拟右前腿被触碰', 'seed': 81000}),
        ('legacy_sugar_ground', '/api/observe', {'words': ['sugar'], 'seed': 42}),
        ('legacy_sugar_tethered', '/api/observe', {'words': ['sugar'], 'seed': 42, 'body_context': 'tethered_flight'}),
        ('blank', '/api/stimulate', {'items': [], 'seed': 5900007}),
    ]
    results = []
    for name, route, payload in cases:
        response = call(route, payload)
        (HERE / f'http-{name}.json').write_text(json.dumps(response, ensure_ascii=False), encoding='utf-8')
        obs = response.get('observation', response)
        body = obs['scene']['body']
        assert body['controller_id'] == 'badapple_hybrid_v1', body
        assert body['context'] == payload.get('body_context', 'ground')
        assert body['geometry_asset'] == '/assets/neuromechfly-behavior-scene.json'
        assert body['status'] in ('ok', 'no_motor_activity'), body
        record = ROOT / 'data/derived/body-trials' / obs['id']
        drive = json.loads((record / 'behavior-drive.json').read_text(encoding='utf-8'))
        assert drive['source_spikes_sha256'] == body['source_spikes_sha256']
        if body['physics_run']:
            assert body['summary']['finite_state'] and body['summary']['physics_warnings'] == 0
            assert body['trajectory'][0]['time_ms'] == 0
            assert body['initialization']['warmup_active_posture_control']
            assert body['controls']['neural_feedback'] is False
        else:
            assert not body['trajectory'] and not drive['has_command_in_window']
        if name == 'blank':
            assert obs['spikes'] == 0 and body['status'] == 'no_motor_activity'
        row = dict(name=name, route=route, request=payload, observation_id=obs['id'],
                   dataset=obs['dataset'], spikes=obs['spikes'], raw_record=obs['raw_record'],
                   status=body['status'], controller=body['controller_id'], context=body['context'],
                   frames=len(body['trajectory']), motor_spikes=body['motor_spikes'],
                   source_spikes_sha256=body['source_spikes_sha256'], command_summary=body['command_summary'],
                   physical_summary=body['summary'], reply=obs['response_text'])
        results.append(row)
        print(json.dumps({k: row[k] for k in ('name', 'status', 'spikes', 'motor_spikes', 'frames')}, ensure_ascii=False), flush=True)
    assert results[1]['source_spikes_sha256'] == results[2]['source_spikes_sha256']
    asset = call('/assets/neuromechfly-behavior-scene.json')
    assert asset['schema'] == 'chatflypt.body-scene.v1'
    assert len(asset['geoms']) == 69
    try:
        call('/api/stimulate', {'items': [], 'body_context': 'free_flight'})
        raise AssertionError('Invalid body context was accepted')
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
    weights = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (ROOT/'data/derived/state-sensory').glob('*-decoder.npz')}
    report = dict(cases=results, invalid_context_http_status=400,
                  geometry_count=len(asset['geoms']), decoder_sha256=weights,
                  assertions_passed=True, llm_calls=0)
    (HERE / 'http-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
