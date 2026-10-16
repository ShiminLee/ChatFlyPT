"""Produce a readable results document and an exportable scientific figure."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .paths import REPORTS
from .demo import summary
from .stimuli import WORDS


def main():
    acceptance=summary()
    def read(name):
        return json.loads((REPORTS/name).read_text(encoding='utf8'))
    physiology=read('mn9-evaluation.json')
    readout=read('readout-evaluation.json')
    first=read('first-sentence.json')
    command=read('auditory-command.json')
    main_metrics=readout['models']['all_non_sensory']['metrics']
    motor_metrics=readout['models']['descending_motor']['metrics']
    lines=['# ChatFlyPT 独立复现：实际运行结果','',
           '这是基于公开描述的独立功能重建，未取得 @chetaslua 原始代码、词表或评测集，不能称为原实验精确复现。','',
           '## 展示例子验收','',
           '| 例子 | 公开演示/报道的参照 | 本机实测 |','|---|---|---|',
           f"| 糖 / 糖+苦的 MN9 | 78 → 3 Hz | {physiology['sugar_mean_hz']:.1f} → {physiology['sugar_bitter_mean_hz']:.1f} Hz；10 次、每次 1 秒 |",
           f"| 16 词单标签识别 | 100% | {main_metrics['test']['single_word_top1_accuracy']:.2%}；独立定义词表、96 次新试次 |",
           f"| 仅下行/运动神经元读出 | 85%（媒体转述，未独立核实作者原文） | {motor_metrics['test']['single_word_top1_accuracy']:.2%} |",
           f"| 醋+酵母+糖+温暖+明亮句子 | 展示 1 句 | {first['exact_sentence_successes']}/{first['total']} 次完整标签/句子一致 |",
           f"| 原生 DNa01/DNa02 方向性 | 原帖缺少可核验测量 | 左编码试次中左右活动相同 {sum(r['larger_side']=='tie' for r in command['anatomical_laterality'] if r['input_command']=='left')}/37，未证明原生左转 |",'',
           '历史上的“听觉编码 → 下流传出 → 玩具环境左转/觅食”与真实大模型建议回传实验已于 2026-09-28 从代码树移除；当前界面不含该通路，身体动作只读取实际神经活动。','',
           '不能把独立词表、独立输入参数和独立环境下的数字当成作者原评测结果。作者原文与转载的区分见 [来源复核](../../research/chetaslua-investigation.md)。',
           '2026-09-19 生物学复核已将 humid 从误用的冷觉 HRN_VP1l 修正为 HRN_VP5 湿觉细胞，并重新生成语料、训练和评测；旧结果仅在 biological-audit/previous-v1 留档。','',
           '## 多标签和泛化','',
           '| 评测 | exact-set accuracy | micro F1 | macro F1 |','|---|---:|---:|---:|']
    for split in ['train','validation','test','ood']:
        m=main_metrics[split]
        lines.append(f"| {split} | {m['exact_set_accuracy']:.2%} | {m['micro_f1']:.3f} | {m['macro_f1']:.3f} |")
    lines+=['','OOD 同时包含新组合和不同于训练范围的 0.6×/1.4× 强度；因此结果不是单纯强度变化的因果隔离。','',
            '## 神经信号负对照','']
    for name in ['shuffled_neural_observations','zero_neural_observations']:
        m=readout['controls'][name]
        lines.append(f"- {name}：单词 top-1 {m['single_word_top1_accuracy']:.2%}，多标签 micro F1 {m['micro_f1']:.3f}。")
    lines+=['','## 第一条展示的实际输出','',first['trials'][0]['sentence'],'',
            '该句来自固定 seed=81000 的首次测试，没有选择最好的一次。完整 20 次结果见 [first-sentence.json](first-sentence.json)。',
            '提问结尾是模板；感官词来自盲读出。醋/酵母/光/温度的输入使用文档中明确列出的代理假设。','',
            '## 方法与证据','',
              '- [方法和假设](../../research/reconstruction-method.md)',
              '- [固定协议](protocol.json)',
              '- [全图引擎对照](engine-verification-full.json)',
              '- [读出评测](readout-evaluation.json)',
              '- [听觉指令测试和真实 DNa 活动](auditory-command.json)',
              '- [机器可读验收](acceptance.json)','',
              '下游分类器恢复工程指令不等于原生左转神经机制。这里没有身体物理或主观体验的验证。','',
              '![实测结果](results.png)','']
    (REPORTS/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={'width_ratios':[1,1.6,1]})
    trials=physiology['trials']
    for trial in trials:
        axes[0].plot([0,1],[trial['sugar_mn9_hz'],trial['sugar_bitter_mn9_hz']],color='#8db1a2',alpha=.5)
    axes[0].scatter(np.zeros(len(trials)),[x['sugar_mn9_hz'] for x in trials],color='#327d5b',s=24)
    axes[0].scatter(np.ones(len(trials)),[x['sugar_bitter_mn9_hz'] for x in trials],color='#bd783d',s=24)
    axes[0].set(xticks=[0,1],xticklabels=['Sugar','Sugar + bitter'],ylabel='MN9 firing rate (Hz)',title='10 independent 1-second trials',ylim=(0,105))
    x=np.arange(len(WORDS));axes[1].bar(x,[main_metrics['test']['per_word_f1'][w] for w in WORDS],color='#327d5b')
    axes[1].set(xticks=x,xticklabels=WORDS,ylim=(0,1.08),ylabel='F1',title='Held-out sensory-proxy labels')
    axes[1].tick_params(axis='x',rotation=60)
    labels=['Real\nobservations','Shuffled\nobservations','Zero\nobservations']
    values=[main_metrics['test']['micro_f1']]+[readout['controls'][k]['micro_f1'] for k in ['shuffled_neural_observations','zero_neural_observations']]
    axes[2].bar(labels,values,color=['#327d5b','#99afa5','#cedbd5'])
    axes[2].set(ylim=(0,1.08),ylabel='Micro F1',title='Readout negative controls')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('ChatFlyPT: independent reconstruction, not the original author\'s benchmark',fontsize=13)
    fig.tight_layout();fig.savefig(REPORTS/'results.png',dpi=180);fig.savefig(REPORTS/'results.pdf');plt.close(fig)
    print(REPORTS/'RESULTS.md',flush=True)


if __name__=='__main__':main()
