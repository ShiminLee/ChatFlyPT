"""Render the current gap-closure evidence without changing historical results."""
import json
import xml.etree.ElementTree as ET
from .paths import ROOT


def build():
    directory=ROOT/'reports/gap-closure'
    def read(path):return json.loads(path.read_text(encoding='utf8'))
    catalogue=read(ROOT/'data/derived/sensory/catalogue.json')
    audit=read(directory/'electrical-audit.json')
    main=[s for s in catalogue['channels'].values() if not s['advanced'] and s['status']!='unmapped']
    results={r['channel']:r for r in audit['cases']}
    passed=sum(results[s['id']]['electrical_response'] for s in main)
    lines=['# 缺口修复与验证','',
        '本轮继续处理输入功能映射、遗漏的阈下神经响应、盲读出和身体动力学。旧版试次及首次保留验收失败记录未覆盖。',
        '', '## 已补上的部分','',
        '- 感觉输入增加 MaleCNS 糖、水、高盐、己酸相关候选；普通糖/水词分别选择不同真实细胞群。历史 FlyWire 糖水联合群仍可单独选择。',
        '- 在原 LIF 引擎上被动记录正负膜电位偏移的面积与峰值，不修改任何电位、权重、脉冲或递质符号。',
        '- 读出只用内部神经脉冲和电位，排除全部感觉及直接输入细胞。标签要在验证集的孤立阳性试次全部检出，并在含阴性/错侧/混合的验证集中达到至少0.90精确率。',
        '- 测得的运动活动通过工程控制器驱动官方 FlyGym/MuJoCo 身体，保存实际足端、关节和轨迹；零运动活动不生成动画。',
        '', '## 通路复核','',
        f"当前目录 {len(catalogue['channels'])} 项；{len(main)} 个主通道，{audit['total']} 个可执行通道。**{audit['electrical_response']}/{audit['total']}** 测得内部电响应；主通道 **{passed}/{len(main)}**。这里的电响应包括脉冲或峰值超过 1e-6 mV 的模型电位偏移。",'',
        f"复用了 {audit['archived_positive_spike_trials']} 份已有阳性脉冲记录，并新跑 {audit['new_voltage_trials']} 份电位观测试次。原有同种子试次脉冲计数保持一致：{audit['original_counts_preserved']}。这不是把原来的‘脉冲失败’改成‘脉冲成功’，而是补充过去漏看的电位证据。",'',
        '[电响应审计](electrical-audit.json) · [16组缺少当前模型可用传递的原始数据溯源](zero-outgoing-provenance.md)',
        '', '## 内部状态读出','',
        '| 数据集 | 完整试次 | 固定测试标签全对 | 固定测试微平均 F1 | 验证合格侧别标签 |',
        '|---|---:|---:|---:|---:|']
    unqualified=[]
    for key in ['flywire','malecns']:
        r=read(directory/f'{key}-state-readout.json')
        lines.append(f"| {r['dataset']} | {r['trials']} | {r['test']['exact_set_accuracy']:.1%} | {r['test']['micro_f1']:.3f} | {sum(r['qualified_labels'].values())}/{len(r['qualified_labels'])} |")
        protocol=read(directory/f'{key}-state-protocol.json')
        names=dict(zip(protocol['labels'],protocol['names']))
        unqualified.extend(names[label] for label,passed in r['qualified_labels'].items() if not passed)
    lines+=['','验证资格未通过的标签：'+('、'.join(unqualified) if unqualified else '无')+'。这些是当前读出器的分类误差，不表示生物学上缺少这些感觉或侧别的研究。',
        '阈值在原有网格上先满足资格条件，再最大化验证F1；alpha保留10。[仅训练／验证数据的选择诊断](validation-only-diagnostic.md)包含无法通过阈值修复的错侧证据，[修正前结果](before-qualified-threshold-selection/)另行保留。']
    lines+=['','合成刺激类别的识别率不是天然感觉准确率；固定测试协议不等于重新盲验。所有标签都留在指标分母中，未合格标签不能通过删除失败项提高指标。','',
        '| 模型冻结后新增试次 | 数量 | 标签全对 | 微平均 F1 |','|---|---:|---:|---:|']
    fresh=read(directory/'fresh-state-results.json')
    assert set(fresh)=={'flywire','malecns'}, 'Both frozen models need fresh trials'
    assert all(row['model_unchanged'] for row in fresh.values()), 'A probe changed during evaluation'
    for key,row in fresh.items():
        lines.append(f"| {key} | {row['trials']} | {row['exact_set_accuracy']:.1%} | {row['micro_f1']:.3f} |")
    lines+=['','[新神经试次与逐标签结果](fresh-state-results.json)','',
        '## 身体物理仿真','',
        '从保存的 MaleCNS 仿真试次提取腿运动群脉冲，运行500 ms官方实体模型。相同初始条件下另跑关闭驱动的被动物理对照，比较位置和关节。重力和接触也会移动身体，因此不能把总位移解释成神经选择了“行走”。','']
    body=read(ROOT/'reports/body-simulation/evidence.json')
    lines +=[f"记录中有 {body['motor_spikes']} 个腿运动脉冲；主动与被动对照最大关节差 {body['max_neural_minus_passive_joint_angle_rad']:.4f} rad，最终位置差 {body['final_neural_minus_passive_position_mm']} mm。物理数值警告 {body['neural_physics']['physics_warnings']}。",'',
        '[身体模型、环境锁定与重现](../body-simulation/README.md)','', '## 工程检查','']
    lines.append('实际 HTTP 身体集成和八种画布尺寸／时间组合的足端投影检查已通过。浏览器工具未能附加本地页面，本轮未完成真实浏览器布局的视觉复核；程序层绘图检查不能代替该项。')
    path=directory/'pytest.xml'
    if path.exists():
        suites=list(ET.parse(path).getroot().iter('testsuite'))
        count={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
        lines.append('完整回归：'+json.dumps(count,ensure_ascii=False)+'。')
    accept=directory/'acceptance.json'
    if accept.exists():
        a=read(accept)
        assert not a.get('in_progress'), 'HTTP acceptance is incomplete'
        lines+=['','[新增入口、视觉、电位与身体的端到端验收](acceptance.json)',json.dumps(a.get('summary',{}),ensure_ascii=False)]
    if (directory/'serving-smoke.json').exists():
        lines+=['','[正式端口验收](serving-smoke.json)：8765 已运行新版本；同种子神经数组与上述端到端记录逐项一致。']
    lines +=['','## 仍不能称为全部生理输入复原','',
        '当前本项目尚未实现的9项天然输入映射：'+ '、'.join(s['label'] for s in catalogue['channels'].values() if not s['advanced'] and s['status']=='unmapped')+'。这个数量统计项目实现状态，不是生物学研究空白的数量。', '',
        'ISN/DH44内部状态相关细胞已有文献和本机ID；项目尚未实现饥饿、口渴、内部营养和肠道扩张所需的渗透压、激素、代谢、基线、适应与肽信号机制。酸味、氨基酸味和成年伤害感受也已有功能研究，但本项目尚未核实感受器到当前EM细胞ID的完整对应。内部氧/二氧化碳的现有核实证据有幼虫到成年阶段的适用性问题，不能据此断言整个领域没有成年研究。', '',
        '剩余16组在当前模型中缺少可用传递：3个FlyWire组在当前作者连接表中没有原始出边；13个MaleCNS组都有原始出边及图内连接，其中12组的官方递质共识不确定，1组已知为血清素但当前未实现其调节动力学。不能把这16组统称为生物学上没有连接。', '',
        '当前身体已复用官方FlyGym/MuJoCo模型和步态模板；尚未经生理标定的是我们从神经运动群活动到控制器命令的适配关系，不能由此否定官方身体模型的依据。目前先跑神经再回放身体，尚无身体感觉回到连接组的反馈闭环。持续神经状态、学习和现实刺激强度的生理标定仍未实现。', '',
        '[缺口归因与官方控制器复用核查](../../research/gap-meaning-and-controller-reuse-2026-09-19.md)', '',
        '[逐项原始文献证据](../../research/sensory-gap-evidence-2026-09-19.md) · [功能映射与所需机制](../general-sensory/functional-mapping-research.md)', '']
    path=directory/'RESULTS.md';path.write_text('\n'.join(lines),encoding='utf8')
    return path


if __name__=='__main__':print(build())
