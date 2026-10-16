"""Publish measured coverage separately from anatomical mapping and decoding."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from .paths import ROOT


def build():
    directory=ROOT/'reports/general-sensory'
    cat=json.loads((ROOT/'data/derived/sensory/catalogue.json').read_text(encoding='utf8'))
    audit=json.loads((directory/'channel-delivery.json').read_text(encoding='utf8'))
    models={key:json.loads((directory/f'{key}-readout.json').read_text(encoding='utf8'))
            for key in ['flywire','malecns']}
    if audit['status']!='completed':raise ValueError('Delivery audit is unfinished')
    specs=cat['channels']
    main=[s for s in specs.values() if not s['advanced'] and s['status']!='unmapped']
    main_fail=[r for r in audit['cases'] if not r['input_delivery_passed'] and not specs[r['channel']]['advanced']]
    lines=['# 通用输入扩展：实测结果','',
        '本轮将输入改为感觉通道、入口侧、强度、部位和时序的通用组合。没有为最终保留句编写特例。',
        '',f"目录含 **{len(specs)} 条记录**，其中 **{len(main)} 个主要输入通道**；其余为高级注释群及明确功能缺口。",'',
        '| 数据集 | 模型内感觉细胞 | 已审计 | 可执行入口细胞 | 暂不可执行 |',
        '|---|---:|---:|---:|---:|']
    for name,row in cat['coverage'].items():
        lines.append(f"| {name} | {row['sensory_cells_in_engine']:,} | {row['accounted_cells']:,} | {row['executable_cells']:,} | {row['not_executable_cells']:,} |")
    lines += ['', '这里的全部覆盖指本机两份连接组内的感觉细胞注释，不能外推为果蝇全部感受器、生理输入或自然感知已经复原。',
        '', '## 默认输入的真实传播', '',
        f"对 **{audit['total']} 条可执行通道**各进行一次独立全图仿真：**{audit['delivery_passed']} 条**通过投递及内部响应检查，**{audit['total']-audit['delivery_passed']} 条**未通过。失败项保留原参数与原始记录。",'',
        f"主要通道中 **{len(main)-len(main_fail)}/{len(main)}** 在默认参数下通过。未通过的主要通道：",'']
    for row in main_fail:
        lines.append(f"- {specs[row['channel']]['label']}（`{row['channel']}`）：{row['non_sensory_spikes']} 个非感觉脉冲；{row.get('failure_reason','未通过')}。")
    lines += ['', '零下游放电不能解释为该生物学感觉不存在，也不能算成功感知。这暴露了当前输入代理与 LIF 动力学的适用限制。',
        '', '[逐通道记录](channel-delivery.json) · [感觉细胞覆盖](../full-pathway/sensory-coverage.json)',
        '', '## 内部活动的盲读出', '',
        '| 模型 | 总试次 | 独立测试试次 | 标签集合全对 | 微平均 F1 | 单通道集合全对 | 混合集合全对 | 较弱/短刺激集合全对 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in models.values():
        lines.append(f"| {model['dataset']} | {model['trials']} | {model['test']['trials']} | {model['test']['exact_set_accuracy']:.1%} | {model['test']['micro_f1']:.3f} | {model['single']['exact_set_accuracy']:.1%} | {model['mixture']['exact_set_accuracy']:.1%} | {model['ood']['exact_set_accuracy']:.1%} |")
    lines += ['', '监督目标是仿真刺激类别和入口侧。特征排除全部感觉细胞与直接输入细胞；每次推断只读取脉冲计数，不接收原文、目标类别或指定侧别。刺激组合及随机种子按预先固定协议分组，阈值仅在验证集选择。',
        '', '求偶包络与普通声音共享感觉群，统一读出为听觉，不能据此声称识别求偶歌。糖/水候选群保持合并。总计数读出不解码先后顺序；高级注释群没有训练语义读出，只报告真实内部活动。',
        '', '| 对照 | FlyWire 微平均 F1 | MaleCNS 微平均 F1 |',
        '|---|---:|---:|']
    for key,title in [('shuffled_observations','打乱观测'),('zero_observations','零内部观测')]:
        lines.append(f"| {title} | {models['flywire']['controls'][key]['micro_f1']:.3f} | {models['malecns']['controls'][key]['micro_f1']:.3f} |")
    lines += ['', '[FlyWire 完整统计](flywire-readout.json) · [MaleCNS 完整统计](malecns-readout.json)',
              '', '逐侧测试 F1 同时显示在界面通道卡中；较低或零 F1 的标签不能当作可靠读出。指标来自人工刺激试次，不是自然感知准确率。',
              '', '初次保留验收发现一个额外视觉标签误报。随后采用通用验证门槛：标签必须在每个针对它的独立单通道验证试次中，有内部放电且被正确检出，才可进入中文感觉回应。门槛不读取测试句、测试目标或通道名称；原始分数仍保留。上表为加入门槛后的同一固定语料结果，不能当作修复后的新盲测。',
              '', '## 工程回归和最终保留验收', '']
    xml=directory/'pytest.xml'
    if xml.exists():
        tree=ET.parse(xml).getroot()
        suites=list(tree.iter('testsuite'))
        totals={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
        lines.append(f"回归：{totals['tests']} 项，失败 {totals['failures']}，错误 {totals['errors']}，跳过 {totals['skipped']}。覆盖解析、事件时序/叠加、数据集隔离、真实脉冲重放、无目标泄漏、零输入和不可执行输入不调用 API。")
    held=directory/'held-out-acceptance.json'
    if held.exists():
        result=json.loads(held.read_text(encoding='utf8'))
        lines += ['', '保留句只在实现、训练和开发回归冻结后运行。锁定文件记录了当时的代码、目录与模型摘要。',
                  '', '**首次保留验收未通过**：正确识别了感觉候选通道和侧别，但额外误报视觉标签。首次返回：',
                  '', result['result'].get('reply','没有回应'),
                  '', f"验收检查：`{json.dumps(result.get('checks',{}),ensure_ascii=False)}`。", '',
                  '[完整保留验收记录](held-out-acceptance.json) · [验收前锁定](acceptance-lock.json)']
    else:lines.append('最终保留句尚未运行。')
    fresh=directory/'fresh-neural-check.json'
    if fresh.exists():
        data=json.loads(fresh.read_text(encoding='utf8'))
        lines += ['', '修复后另外生成新的随机种子及未出现过的组合，模型在这些新脉冲产生前固定。下表包含全部注册标签，未通过验证门槛的视觉标签也保留为召回失败，未从分母删除：', '',
                  '| 新试次 | 数量 | 全部标签集合全对 | 全部标签微平均 F1 |', '|---|---:|---:|---:|']
        for key,row in data.items():
            lines.append(f"| {key} | {row['trials']} | {row['exact_set_accuracy_all_registered_labels']:.1%} | {row['micro_f1_all_registered_labels']:.3f} |")
        lines += ['', '[修复后新神经试次](fresh-neural-check.json)']
    regression=directory/'known-case-regression.json'
    if regression.exists():
        result=json.loads(regression.read_text(encoding='utf8'))
        lines += ['', '**已知句复测**（不是新的独立盲验）：'+result['result'].get('reply','没有回应'),
                  '', '复测检查：'+json.dumps(result.get('checks',{}),ensure_ascii=False),
                  '', '[已知句复测原始记录](known-case-regression.json) · [首次失败的代码与模型快照](initial-acceptance-snapshot/snapshot.json)']
    lines += ['', '## 尚未完成', '',
        '有明确目录缺口：'+ '、'.join(s['label'] for s in specs.values() if not s['advanced'] and s['status']=='unmapped')+'。',
        '', '现实空间到外周感受器的转导、自然浓度/力/光强/关节角的校准、动作动力学、持续状态与学习尚未接通。跨数据集输入分别运行，不能冒充一只整合后的果蝇。',
        '', '[生物学依据与方法](../../research/general-sensory-inputs.md) · [重运行脚本](../../scripts/reproduce-sensory.ps1)', '']
    path=directory/'RESULTS.md';path.write_text('\n'.join(lines),encoding='utf8')
    return path


if __name__=='__main__':
    print(build())
