"""Summarize frozen test predictions alongside their actual physical records.

This is a reporting step after evaluation. It neither runs a decoder nor edits
weights, thresholds, scenes, targets, or physics. Failed predictions remain.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatflypt.paths import ROOT, dump, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.run_id):
        raise ValueError('Use an existing lowercase run identifier')
    reports = ROOT / 'reports/unified-readout/runs' / args.run_id
    trials = ROOT / 'data/derived/unified-readout/runs' / args.run_id / 'trials'
    evaluation_path = reports / 'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text(encoding='utf8'))
    rows = []
    for prediction in evaluation['test_predictions']:
        folder = trials / prediction['episode'] / f"turn-{prediction['turn']}"
        manifest = json.loads((folder / 'observation.json').read_text(encoding='utf8'))
        body_path = folder / 'measured-body.json'
        if sha256(body_path) != manifest['body_sha256']:
            raise ValueError('Physical evidence changed: ' + str(body_path))
        body = json.loads(body_path.read_text(encoding='utf8'))
        expected, predicted = set(prediction['expected']), set(prediction['predicted'])
        controls = body['controls']
        rows.append({**prediction, 'false_positive': sorted(predicted - expected),
            'missed': sorted(expected - predicted), 'exact_set': expected == predicted,
            'physical_record': str(body_path.relative_to(ROOT)),
            'physical_record_sha256': manifest['body_sha256'],
            'physics': body['summary'], 'active_motor_neurons': body['active_motor_neurons'],
            'motor_spikes': body['motor_spikes'],
            'controller_program_steps': controls['effective_program_steps'],
            'command_ranges': body['command_summary'],
            'behavior_goal_success': None})
    dump(reports / 'physical-test-summary.json', {
        'evaluation_sha256': sha256(evaluation_path), 'rows': rows,
        'scope': 'Frozen predictions and actual simulation records; no test-set retuning',
        'behavior_goal_success_evaluated': False, 'biologically_calibrated': False})
    lines = ['# 冻结测试的实际接收、神经读出与身体记录', '',
        f"批次 `{args.run_id}`。只汇总冻结后测试集，不改变阈值或隐藏失败。",
        '实际接收标签来自独立保存的受体事件；预测只来自内部神经记录。位移、路径长度和控制器分支被激活不等于完成找糖、逃避或其他行为目标。', '',
        '| 场景 / 窗口 | 标签整组一致 | 误报 | 漏报 | 身体路径 mm | 生效控制分支 |',
        '|---|---|---|---|---|---|']
    for row in rows:
        programs = ', '.join(k for k, n in row['controller_program_steps'].items() if n) or '姿态维持'
        lines.append('| ' + ' | '.join([
            f"{row['episode']} / {row['turn']}", '是' if row['exact_set'] else '否',
            ', '.join(row['false_positive']) or '无', ', '.join(row['missed']) or '无',
            f"{row['physics']['path_length_mm']:.4f}", programs]) + ' |')
    failures = sum(bool(row['physics']['physics_warnings']) or not row['physics']['finite_state'] for row in rows)
    lines += ['', f'物理失败窗口：{failures}/{len(rows)}。完整预期/预测、控制幅度、记录路径和哈希见 `physical-test-summary.json`。',
        '控制分支名称沿用复用控制器，仅描述它执行了哪个程序，不把程序名当作已观测到的自然行为。', '']
    (reports / 'physical-test-summary.md').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps({'windows': len(rows), 'physics_failures': failures,
        'exact_sets': sum(row['exact_set'] for row in rows),
        'report': str((reports / 'physical-test-summary.md').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
