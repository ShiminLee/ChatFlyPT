"""Summarize retained two-model measurements without further model calls."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'reports/local-model-2026-09-19'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    protocols = [read(BASE/'benchmark-protocol.json'), read(BASE/'qwen3-4b/benchmark-protocol.json')]
    if protocols[0] != protocols[1]:
        raise RuntimeError('Comparative protocols differ')
    models = []
    for directory in (BASE, BASE/'qwen3-4b'):
        summary = read(directory/'benchmark-summary.json')
        rows = read(directory/'benchmark-results.json')['rows']
        review = read(directory/'semantic-review.json')
        assert len(rows) == len(review['rows']) == 24
        judgments = [r['judgment'] for r in review['rows']]
        assert judgments.count('pass') == review['acceptable_suggestions']
        assert judgments.count('fail') == review['forbidden_action_responses']
        assert judgments.count('concern') == review['other_grounding_or_policy_concerns']
        tokens_per_second = [r['http']['response']['timings']['predicted_per_second'] for r in rows]
        models.append({**summary, 'semantic_review': review,
                       'generation_tokens_per_second_median': statistics.median(tokens_per_second),
                       'raw_responses_sha256': hashlib.sha256((directory/'benchmark-results.json').read_bytes()).hexdigest()})
    comparison = {
        'same_frozen_protocol': True,
        'protocol_sha256': hashlib.sha256((BASE/'benchmark-protocol.json').read_bytes()).hexdigest(),
        'models': models, 'selected_experimental_default': 'chatflypt-qwen3-4b-instruct-2507',
        'selection_basis': 'Better JSON compliance and blank-signal abstention, not reliable semantic planning.',
        'reliable_autonomous_advice_demonstrated': False,
        'current_runtime': read(BASE/'runtime.json'), 'startup_checks': read(BASE/'startup-checks.json'),
        'service_left_running': True, 'simultaneous_gpu_model_services': 1,
        'integration_review': read(BASE/'integration-semantic-review.json'),
    }
    (BASE/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'protocols_identical': True,
        'models': [{'model': m['model'], 'protocol': m['protocol_passes'],
                    'manual_acceptable': m['semantic_review']['acceptable_suggestions'],
                    'median_s': m['latency_seconds']['median']} for m in models],
        'service': comparison['selected_experimental_default']}, indent=2))


if __name__ == '__main__':
    main()
