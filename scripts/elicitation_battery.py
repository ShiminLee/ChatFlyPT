"""Elicitation battery: each preregistered condition as a question with a contrast.

Field practice for underdescribed languages replaces "how do you say X" with a
frame plus a minimal pair and an explicit decision criterion. This script writes
that registry for the readout protocols and refuses to hide any condition that no
question covers, so the battery can never silently omit material.

A criterion is what would count as an answer: a qualified label with the
preregistered precision/recall, or an explicitly reported non-difference. A
non-difference is a result, not a failure to be tuned away.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatflypt.paths import ROOT, dump

OUT = ROOT / 'reports/elicitation'

# question, conditions it covers, the contrast that makes it a minimal pair, criterion
BATTERY = [
    ('视觉：光照的有无能否读出？', ['bright', 'dark'],
     '同一场地、只改照明；两段独立会话',
     'bright:left/right 在验证集达到 precision>=0.9 且 recall>=0.75'),
    ('嗅觉：酵母场能否在无光条件下读出？', ['yeast', 'dark_yeast', 'dark'],
     '同一酵母几何，开灯/关灯互为对照，dark 为无气味基线',
     'yeast:left/right 合格；仅在开灯时可读即为视觉混杂'),
    ('嗅觉侧别：单侧气味入口能否被分别读出？', ['unilateral_yeast_left', 'unilateral_yeast_right'],
     '左右各一次固定小范围气味源，实际事件决定侧别',
     'yeast:left 与 yeast:right 分别合格'),
    ('温度：升温/降温能否读出？', ['heat', 'cold', 'unilateral_heat_left', 'unilateral_heat_right',
                                   'unilateral_cold_left', 'unilateral_cold_right'],
     '同几何热源与冷源、左右各一；冷热符号在受体处先合成',
     'warm:* 与 cold:* 合格；单侧条件给出侧别证据'),
    ('味觉：几何相同的糖平台与普通平台能否分开？', ['leg_sugar', 'leg_block_control',
                                                    'dark_leg_sugar_control', 'dark_leg_block_control'],
     '两种平台的几何完全一致，只有材质不同；暗条件再排除视觉',
     'leg_sweet_local/leg_appetitive_ascending 合格，且 block 条件不产生味觉标签'),
    ('看得到但碰不到：远处可见的糖是否会被当成尝到？', ['far_sugar_control'],
     '糖在视野内但不在接触范围',
     '不产生 sugar_specific 标签（阴性对照必须为阴性）'),
    ('口器接触能否代替唇部味觉？', ['mouth_sugar', 'mouth_high_salt', 'mouth_block'],
     'rostrum/haustellum 实际接触；官方身体没有已注册的 labellum 表面',
     '不产生 sugar_specific/high_salt 标签；记录为未映射（预期阴性）'),
    ('触碰：真实接触能否读出，侧别是否来自实际碰到的腿？', ['touch_left', 'touch_right'],
     '物理探针 + 撤除各两窗；探针侧别与实测接触腿分离',
     'foreleg_touch:left/right 合格；撤除窗不产生标签'),
    ('同种触碰：相同总接触时间下的节奏能否区分？',
     ['tap1x180_left', 'tap1x180_right', 'tap3x60_left', 'tap3x60_right',
      'tap6x30_left', 'tap6x30_right'],
     '每次 180 ms 总接触时间固定，只改敲击次数（1/3/6 次）与间隔',
     '在同一侧、同一总接触时间的三种节奏之间报告是否出现可区分的读出或行为差异；无可区分差异时如实报告为不可区分'),
    ('cVA 信息素场能否读出，远源是否不误报？', ['pheromone_left', 'pheromone_right', 'far_pheromone_control'],
     '实际 funiculus 位置的小场，左右各一；外加超出范围的远源对照',
     'cva:left/right 合格，且远源对照不产生标签'),
    ('求偶鸣声场能否读出，远源是否不误报？', ['song_left', 'song_right', 'far_song_control'],
     '实际 pedicel 位置的小场（约翰斯顿氏器所在），左右各一；外加远源对照',
     'auditory:left/right 合格，且远源对照不产生标签'),
    ('单侧光照的侧别能否读出（原配准混杂的对照）？',
     ['unilateral_bright_left', 'unilateral_bright_right'],
     '对侧眼遮光，只改一侧视网膜输入',
     'bright:left 与 bright:right 分别合格；与双侧亮条件对照'),
    ('换用上游羽流场后，cVA 能否被读出？',
     ['pheromone_plume_left', 'pheromone_plume_right', 'pheromone_left', 'pheromone_right'],
     '同一落点几何，只改场模型：固定有限距离场（v4）对比借用的上游羽流网格',
     'cva:left/right 在验证集达到 precision>=0.9 且 recall>=0.75；固定场条件下不合格即为"投递受限"的证据'),
    ('换用上游羽流场后，求偶鸣声能否被读出？',
     ['song_plume_left', 'song_plume_right', 'song_left', 'song_right'],
     '同一落点几何，只改场模型：固定有限距离场（v4）对比借用的上游羽流网格',
     'auditory:left/right 合格；否则记录为投递仍不足'),
    ('源跟随受体（伙伴贴身）能否让同种信号读出合格？',
     ['pheromone_partner_left', 'pheromone_partner_right',
      'song_partner_left', 'song_partner_right'],
     '保留声明的偏移几何，但每个采样时刻源锚定到实测受体位姿；与固定源对照',
     'cva:*/auditory:* 合格；跟随是"伙伴一直在"的物理主张，须与固定源分别报告'),
    ('对侧前腿的多回合敲击能否产生可被读出的运动/转向？',
     ['nudge_left', 'nudge_right', 'nudge_alternating', 'touch_left', 'touch_right'],
     '单次 300 ms 探针（touch_*）对比 3x60 ms、间隔 140 ms、强度 1.5 的对侧敲击序列',
     '报告实测 >=5 度的转向窗数量与运动分量是否合格；转向分量不达标时如实弃权'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v4')
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol or (ROOT / 'reports/unified-readout/runs' / args.run_id / 'protocol.json')
    if not protocol_path.is_file():
        raise SystemExit('No protocol at ' + str(protocol_path))
    plan = json.loads(protocol_path.read_text(encoding='utf8'))
    conditions = sorted({episode['condition'] for episode in plan['episodes']})
    entries, covered = [], set()
    for question, names, contrast, criterion in BATTERY:
        present = [name for name in names if name in conditions]
        missing = [name for name in names if name not in conditions]
        covered.update(present)
        entries.append({'question': question, 'conditions': present,
                        'contrast': contrast, 'criterion': criterion,
                        'not_in_this_protocol': missing})
    uncovered = [name for name in conditions if name not in covered]
    report = {'schema': 'chatflypt.elicitation-battery.v1', 'run_id': plan.get('run_id'),
              'protocol': str(protocol_path.relative_to(ROOT)),
              'scope': 'Preregistered questions; a non-difference is a reportable result, not a failure to tune away.',
              'qualification': plan.get('qualification'),
              'questions': [entry for entry in entries if entry['conditions']],
              'uncovered_conditions': uncovered,
              'notes': ['Conditions are declared before collection; the criterion is fixed with them.',
                        'Planned side or scene name is never a target: only actually delivered receptor events count.',
                        'Genuine no-response and unqualified labels remain in the record.']}
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / f'{args.run_id}-battery.json', report)
    lines = [f'# 诱导题库：{args.run_id}', '',
             '每个条件都是一次提问。对照是"最小对立对"（只改一个因素），判定标准在采集前固定。',
             '没有差异同样要如实记录，不能为了好看去调参数。', '']
    for entry in report['questions']:
        lines += [f"## {entry['question']}", '',
                  f"- 条件：{'、'.join(entry['conditions'])}",
                  f"- 最小对立：{entry['contrast']}",
                  f"- 判定标准：{entry['criterion']}", '']
    if uncovered:
        lines += ['## 尚未登记问题的条件', '', '、'.join(uncovered), '',
                  '（这些条件仍被执行和记录，但还没有对应的提问；属于待补的题库条目。）', '']
    (OUT / f'{args.run_id}-battery.md').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps({'questions': len(report['questions']), 'uncovered': uncovered}, ensure_ascii=False))


if __name__ == '__main__':
    main()
