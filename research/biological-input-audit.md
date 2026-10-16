# 人类消息 → 果蝇接收：生物学复核

日期：2026-09-19。本轮核查自然信号的接收通路、细胞映射、脉冲参数和内部读出方法。确认了可用的自然接收入口，发现并修正一处实际映射错误；尚不能把全部输入标成已标定的生理信号。

## 自然社会信号

| 信号 | 原始生物学证据 | 当前入口与核查结论 |
|---|---|---|
| 同伴触碰腿部 | Ramdya 2015：末端腿部机械感觉感器参与接触诱发行走；沉默/光激活及 NOMPC 对照支持因果作用 | MaleCNS 前腿 bristle + ProLN + rootSide；右 100、左 124。类别有依据；脑与腹神经索 LIF 已接通，末端感器子集与生理动力学校准仍待完成 |
| cVA 信息素 | Datta 2008：Or67d 感觉细胞投射 DA1；下游社会反应有性别差异 | FlyWire v783 ORN_DA1，126 个。接收类别正确；不固定翻译为“你好”“危险”或某个动作 |
| 求偶鸣声 | Clemens 2018：触角 Johnston 器官听觉细胞处理声波，并表现快速适应 | 注明 auditory 的 JO-A/JO-B，341 个。入口合理；当前包络尚不等于真实声学转导 |

原始资料：[Ramdya 2015](https://www.nature.com/articles/nature14024)、[Datta 2008](https://www.nature.com/articles/nature06808)、[Clemens 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5760620/)。表内数量来自当前连接组注释筛选，不是论文测得的响应细胞总数。

## 触碰能否作为方向提示

已渲染并查看 Ramdya 论文第 3 页（期刊页 235）的图 3c、d。不同触碰部位对应不同轨迹分布，右前腿 R1 黄色轨迹主要偏向图左，左前腿 L1 粉色轨迹主要偏向图右。这支持继续验证“对侧触碰作为方向提示”；图上分布不能提供普遍的 100% 转向保证，也未直接证明 PFL3/DNa02 是必经通路。

主体实验用羽化后 2–4 天的雌蝇，当前腿细胞来自雄性 MaleCNS，需验证跨性别转用。piezo-Gal4 是实验驱动系，不能据此认定 Piezo 通道为必需受体；论文的关键温和触觉缺陷来自 NOMPC。

界面保留“向左 → 模拟右前腿触碰”候选方案并显示方向待验证；不会预先写死转向。旧 16 词的 BM_Taste 是口器触毛代理，不能充当前腿触碰。当前较宽的前腿候选群也不能冒称为已精确定位的末端感器。

## 已修正湿觉映射

旧 humid 错选 HRN_VP1l 的 13 个细胞，其 cell_sub_class 是 cooling；Marin 2020 的受体表达证据同样支持推定冷觉，不能根据 HRN 名称前缀认定湿觉。

现改为 HRN_VP5 且 cell_sub_class=moist 的 **16 个细胞**。Ir68a／VP5 湿觉有生理、遗传与连接研究支持。干觉 VP4、热觉 VP2、冷觉 VP3a/b 的类别选择与资料一致。150 Hz 仍为工程刺激强度，不代表某个已标定湿度值。

依据：[Marin 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443704/)、[Knecht 2017](https://elifesciences.org/articles/26654)。修改 `chatflypt/stimuli.py` 并新增实际细胞注释回归检验。清单版本为 `2026-09-19-v2-moist-VP5`；旧清单、读出器和报告留在 `reports/biological-audit/previous-v1/`，旧 humid 结果已明确标为失效。修正后重新采样、训练、评测，当前数字见 `reports/reconstruction/RESULTS.md`。

## 输入正确性的具体检验

1. **接收类别**：三种社会信号均有原始研究支持。
2. **精确细胞**：cVA/听觉按注释定位，触碰仍为候选；Gal4 名称不能直接换成连接组 body ID。
3. **输入脉冲**：现有通道均采用工程事件率，力、气味浓度、声学粒子速度不能直接等同为放电 Hz。
4. **下游响应**：cVA/鸣声已在当前脑模型产生下游脉冲；前腿目前只有编码及解剖连接核查。
5. **行为效果**：需要神经活动与身体模型产生后再测量，不能用分类器识别成功替代实际转向。

声音的具体差异：Clemens 的人工鸣声为 250 Hz 载波、4.6 ms 包络参数、36 ms 脉冲间隔。现实现仅保留 36 ms 间隔，用 5 ms 门宽、150 Hz 事件率；没有载波、相位响应和适应模型。这些工程参数已在清单内标明。

后续验证应针对末端触毛对应、物理刺激到受体放电标定、连接组内传播和行为轨迹分布；无需先解决“果蝇所有内部活动意味着什么”的泛化问题。

## 作者项目与本地读出

详见 [作者原文复核](chetaslua-investigation.md)。本地已经实现“已知刺激 → 内部活动 → 受监督标签读出 → 句子”。本轮沿用并重测该层；自由文本 Transformer 和蘑菇体/中央复合体单独选区均不是当前复现的前置要求。

输入标签可以给脑活动提供可测试的操作性含义。新增社会行为标签仍应来自独立行为测量。明确的感觉消息能落到有依据的接收类别，抽象方向词目前是有证据的社会刺激提示方案。

## 验证文件

- `reports/biological-audit/validation.json`：本轮映射检查、测试和重训摘要。
- `reports/reconstruction/readout-evaluation.json`：修正后留出结果及负对照。
- `reports/reconstruction/first-sentence.json`：五感句子的 20 个固定试次结果。
- `data/derived/social-input/catalogue.json`：输入细胞、参数、文献和逐层证据状态。
- `reports/social-input/verification.json`：此前六次社会信号代理的下游响应；本轮未改变这些通道的参数，不把旧结果当作新的生理标定。

触碰论文公开作者版 PDF：`research/snapshots/ramdya-nature2015.pdf`；图 3 页面渲染：`tmp/pdfs/ramdya-page3.png`。
