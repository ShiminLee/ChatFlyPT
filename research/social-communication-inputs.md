# 同种果蝇通信 → 神经输入

调查与实现日期：2026-09-19。

当前采用用户指定的路线：先找同种果蝇实际使用的通信信号，再重放到接收方的感觉通路。关联学习不作为这条路线的前置条件。实现见 `chatflypt/social.py`。

## 查到的直接证据

| 通信方式 | 实验支持 | 接收入口 | 当前数据映射 |
|---|---|---|---|
| 个体之间腿/翅接触 | 接触引起与触碰部位有关的行走响应，可形成群体避让；不能嗅到刺激的个体也可被带动 | 腿末端机械感觉感器神经元，依赖 NOMPC | MaleCNS 前腿触觉毛候选细胞，右 100、左 124 |
| cVA 信息素 | 雄性产生的化学信号，影响具有性别与情境差异的社会反应 | Or67d 感觉神经元 → DA1 | FlyWire v783 `ORN_DA1`，126 个 |
| 求偶鸣声 | 翅振产生脉冲/正弦鸣声，接收方处理其时间与频谱结构 | 触角 Johnston 器官的听觉 JONs | FlyWire v783 注明 auditory 的 JO-A/JO-B，341 个 |

触碰的直接依据是 Ramdya et al., Nature 2015。该研究不仅观察接触，还用机械探针、神经沉默和光遗传激活检验感觉入口；补充视频 3、4 展示右前腿的个体接触和人工机械刺激。该论文支持把这类触碰作为自然社会信号的输入起点。**论文没有给出适用于全部姿态和情境的“右触＝抽象向左指令”词典，也没有证明这条响应必须经过 PFL3/DNa02。** 后续应复现论文中的具体刺激位置和轨迹分布。

- [Ramdya et al., 2015：原论文及补充视频](https://www.nature.com/articles/nature14024)
- [作者实验室公开全文](https://www.epfl.ch/labs/ramdya-lab/wp-content/uploads/2024/05/Ramdya_Nature2015.pdf)

化学与声音两条入口依据：

- [Datta et al., 2008：cVA → Or67d → DA1，及性别差异](https://www.nature.com/articles/nature06808)
- [Wang & Anderson, 2010：cVA / Or67d 与雄性攻击行为](https://www.nature.com/articles/nature08678)
- [Clemens et al., 2018：JON 声音编码、适应与 36 ms 间隔的人工脉冲鸣声](https://www.nature.com/articles/s41467-017-02453-9)

## 已落地的转换

明确的人类消息 → 社会信号类型 → 有来源的接收细胞集合 → 带时间的外部神经输入事件。

```powershell
.\.venv\Scripts\python.exe -m chatflypt.social build
.\.venv\Scripts\python.exe -m chatflypt.social encode --message "模拟右侧触碰"
.\.venv\Scripts\python.exe -m chatflypt.social encode --message "释放cVA信息素"
.\.venv\Scripts\python.exe -m chatflypt.social encode --message "发出求偶脉冲鸣声"
.\.venv\Scripts\python.exe -m chatflypt.social verify
.\.venv\Scripts\python.exe -m chatflypt.social audit-touch
```

`data/derived/social-input/catalogue.json` 保存选择条件、全部神经元 ID、文献、参数依据和注释文件哈希。每条输入另存 `.npz` 事件文件与 `.json` 说明。事件文件包含 ID 命名空间、数据版本、刺激细胞、事件细胞 ID、事件时间步、0.1 ms 时间精度和随机种子；中文消息留在独立说明文件中，不进入神经引擎。

目前中文接口支持有限语法的感觉描述、左右触碰、cVA 和脉冲鸣声；并非任意语义到神经的翻译器。未知句子和否定句不被悄悄改写为刺激。底层编码器拒绝裸方向词；界面另将“向左／向右”接到对侧前腿触碰提案，并明确显示“方向映射待验证”，尚未声称产生转向。

## 映射与校准的实际边界

**触碰细胞已定位到前腿候选群，但尚未定位到论文的末端感器子集。** 选择条件是 MaleCNS `vnc_sensory / mechanosensory_tactile / mechanosensory bristle / ProLN / rootSide=L或R`。侧别使用感觉根的 `rootSide`，不猜测缺失的胞体侧别。左、右细胞数不同是当前注释筛选结果，不解释为生物功能强弱。Gal4 驱动系名称并不自动等于连接组中的某批 body ID。

现有模型里的旧 `touch` 使用 `BM_Taste` 口器味觉毛机械感觉代理；新模块独立建立前腿通道，不修改旧实验或拿口器细胞代替腿部细胞。MaleCNS 的 body ID 和 FlyWire 的 root ID 完全隔离。

**输入时序已经可以生成，生理强度还未校准。** 触碰暂用 150 Hz 峰值、15 ms 衰减的事件率；cVA 暂用 100 Hz；鸣声暂用 36 ms 间隔、5 ms 门宽、150 Hz 门内事件率。只有鸣声间隔取自文献示例，其余值是明确的工程设定。36 ms 声学脉冲间隔不等于 36 ms 神经放电间隔；当前实现只保留包络，还未实现声学载波、相位锁定或适应。所有通道采用条件独立的离散 Bernoulli 事件采样，与现有神经引擎时间步一致，不冒充实测感觉神经记录。

**腿部通道现已接入独立 MaleCNS 脑与腹神经索 LIF 引擎。** 166,700 个有分类注释的细胞、24,559,135 条带符号建模连接参与传播。内部读出用独立试次训练，感觉细胞及直接输入细胞全部排除；测试 39/40，强度 OOD 26/32。参数从 Shiu 脑模型移植，尚未生理标定；仍需校准末端触毛映射和身体物理，才能检验自然避让轨迹。程序对跨数据集输入报错。详见 [完整通路结果](../reports/full-pathway/RESULTS.md)。

进一步细分的原始资料：[Elabbady et al., 2026 论文](https://pubmed.ncbi.nlm.nih.gov/41966690/)与[作者代码和数据](https://github.com/tuthill-lab/elabbady_bristles_2026)。该研究提供 FANC 左前腿触毛体表映射，可用于定位研究；FANC 与 MaleCNS 的细胞 ID 仍须建立同源对应，不能直接复制。

## 已执行的检验

首轮模块检验时原复现与新增编码测试合计 **15 项通过**；后续加入界面和生物学回归检验，最新运行见 `reports/biological-audit/validation.json`。覆盖固定种子重复性、安静期、输入群体、事件时序、ID 命名空间隔离、不支持/否定消息拒绝、异常参数拒绝，以及原有动力学和读出器检查。

FlyWire 中每个新通道做 3 次独立 500 ms 仿真：

| 输入 | 3 次非感觉神经元脉冲数 | 3 次被激活的非感觉细胞数 |
|---|---|---|
| cVA 代理输入 | 196734 / 196315 / 196565 | 7823 / 7847 / 7893 |
| 求偶脉冲包络代理 | 34 / 49 / 35 | 16 / 18 / 15 |

所有试次刺激前均为 0 脉冲；无输入对照为 0。上述接收检验直接计数实际仿真脉冲。后续另外训练内部活动读出器，用于把响应写成文字：cVA/鸣声/空白测试 30/30；不改变引擎产生的活动。结果不证明完整自然神经编码、求偶行为或句子理解。响应用原始神经元 ID 和脉冲时刻保存，可以继续分析。

机器可读报告：`reports/social-input/verification.json`；腿触碰候选群的真实连接表核查：`reports/social-input/foreleg-connectivity.json`。

右侧 100 / 左侧 124 个候选细胞全部在 MaleCNS 原始连接表中具有输出边。原始边表含小片段与未分类体，报告分别列出原始目标体数和具有 superclass 注释的目标数，不能把全部原始目标都称为已识别神经元。
