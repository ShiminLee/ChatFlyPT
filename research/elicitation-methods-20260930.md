# 记录语言学的方法怎么用在这里（2026-09-30）

用户要求："看看落后语言是怎么研究的，你借鉴一下"。这里记的是**查到什么、我们借了什么、这一步新借了什么**，以及哪些做法我们不借（因为我们没有母语人）。

## 一、文献里的做法（要点与出处）

| 做法 | 出处 | 一句话 |
|---|---|---|
| 诱导 = **刺激 + 任务 + 反应**三件套，共七种诱导技术 | Bohnemeyer 2015, *A Practical Epistemology for Semantic Elicitation*（Bochnak & Matthewson 编 *Methodologies in Semantic Fieldwork*） | 先分清你给的是什么刺激、让对方做什么任务、收回来的回答算哪一种证据 |
| **黄金规则**：只有当说话人对任务与刺激的理解、以及回答的预期解释都被确认之后，回答才算数据 | 同上 | 不确认解释，就没有数据点 |
| 语料库天然欠覆盖；**没人说过的形式要靠定向诱导**才出现（Movima 案例） | Haude（CNRS/SeDyL）关于 elicit 出的未记录结构 | 小语料不能证明"没有"，只能证明"还没采到" |
| 定向故事板 / 问卷式刺激集 | Burton & Matthewson（targeted construction storyboards）；MPI 各类 questionnaire | 用固定的刺激集把同一对照在多个会话里重复出来 |
| **负面证据也是数据**：elicited data 能给出"未出现/不成立"的信息 | Lüpke 2010 等语言记录方法综述 | 采不到不是空白，是要写下来的结论 |
| 会话式诱导优于清单式提问；刺激顺序、语域、正字法都会影响结果 | Crane & Fleisch 2019；哈萨克元音和谐的采集方法比较 | 采集条件本身要记录、要固定 |
| 数据声明、归档规范、可复现的元数据 | Thieberger（*Oxford Handbook of Linguistic Fieldwork*）、Meakins/Green/Turpin | 语料要带着它的采集条件一起存档 |

## 二、本项目已经借到的（上一轮方法层，仍然有效）

- **语料索引四层分离**（`scripts/index_corpus.py`）：刺激层 / 神经层 / 反应层 / 说法层分开，标签层是注释，不等于译文。
- **诱导弹药库**（`scripts/elicitation_battery.py`）：问题—最小对照—判据三件套；**没有被覆盖的条件直接拒绝出题**，不假装能测。
- **负面登记表**（`reports/corpus/negative-registry.json`，21 条）：把"没测出来/被测伪/故意不做"写成条目，不许事后改写成能力。
- **数据声明**（`reports/corpus/data-statement.md`）+ 每个窗口的来源哈希。
- **会话间一致性 κ，弃权单列**（`scripts/readout_agreement.py`）：像记录语言学里对同一说话人多次采集做一致性检查。
- **语义地图**（`scripts/semantic_map.py`）：信号 × 情境 × 反应，空洞显式标出，而不是补平均值。

## 三、这一步新借的（覆盖率驱动的定向诱导）

### 1. 先做**覆盖率矩阵**，再设计采集（对应"语料库欠覆盖 + 定向诱导"）

`scripts/readout_coverage.py` → `reports/unified-readout/coverage-conspecific-v5.json`：每个标签在每个条件、每个分裂下**实际投递**了多少个窗口。v5 的结果把 12 个未合格标签分成两类：

- **四个是"没被真正诱导出来"**：`yeast:left`、`warm:left`、`cold:left`、`cold:right`。它们并不缺窗口——正例大多来自**双侧同时投递**的场景，等于没做出"左/右"这个对照。这正是 Haude 说的情形：语料看着有，其实没有对照。
- **八个是"没有诱导路径"**：`body_touch:*`、`wing_touch:*`（`environment.py` 明确拒绝把翼膜/整胸碰撞登记到对应感器群）、`sugar_specific:*`、`high_salt:*`（身体没有已注册的唇瓣表面）。这不是缺数据，是**当前器械测不到**，按"负面证据也是数据"写进登记表。

### 2. 定向诱导轮（v6）：补采做出来了，改动标注做不出来

`extend_v6_protocol` 加了两类条件：
- **跟随式单侧气味/温度源**（funiculus / arista 锚定，各侧各 3 次重复 × 3 分裂）：试图让"同一种气味，左 vs 右"这个最小对照真的各自有正例。
- **左转命令家族的补采**：v5 里带左转命令的窗口集中在 song 场/partner、前足轻推、单侧冷，各补采若干会话——因为 43:1 的右/左不对称本身就是**未充分诱导**的对照。

**测下来的结果分开说**（数字见 `reports/unified-readout/coverage-conspecific-v6.json` 与 `reports/corpus/negative-registry.json`）：

- **左转命令这一半成了**：v6 已完成窗口里 `steer > 0` 占 10/36（28%），v5 只有 27/294（9%）；超过信息阈值的左转窗从 1 个开始变多。补采确实把对照做出来了。
- **气味/温度左右这一半没成，而且查明原因不在会话数**：窗口分辨率下双侧窗口的每侧投递事件数比值中位数是 0.82（酵母）、0.85（升温）、0.77（降温），**接近对称**；294 个窗口里严格单侧的只有 6/6/5 个。我又按"少数服从多数"的转录思路写了一条**声明式主导规则**（一侧事件 ≥2 且 ≥2 倍另一侧才算这一侧），实测把四个侧标签的正例从 5/5/5、3/3/3 压到 0–1 → **规则被证伪并弃用**（代码保留，注释写明负面结果，v7 协议删除）。跟随式锚定几何也没有更好：三类群各 12 个双侧窗、0 个严格单侧窗。

对应记录语言学的结论：这是"**用这套器械elicitation不出这个对照**"——不是缺语料，也不是转录约定不好，而是刺激几何本身把两侧天线几乎同等照射。要拿到左右气味/温度，得改**器械**（让源的作用半径小于两触角间距），那是另一个立项。

### 3. 借"旧记录不重新解释"的归档纪律，做**扩展链**

语料不是重采一遍（那要 4 小时），而是在冻结语料上做**可验证的扩展**（`extension_chain`）：

- 新协议必须是旧协议的**超集**：每个旧集号在 id、分裂、条件、轮次、环境上逐字一致；不一致就拒绝运行。
- 每一行仍然带着**它自己那一轮**的协议哈希；拟合时两轮都接受，但要能说清哪一行来自哪一轮。
- 种子差异（35 个旧集号）**不要求相同，但必须报出来**：旧记录用它们当时采集的种子，不被重新生成。

对应记录语言学的做法：早期会话保留原始元数据，不能悄悄重录后混在一起。

## 四、我们不借的

- **没有母语人**：不能问"这句能不能说"。我们的替代是**物理投递**（实测受体事件才作数）+ **验证集判据**（每标签 precision/recall 门槛，等于"可接受性判断"，但由固定规则给出，不由人给）。
- **不做开放式语义诱导**：词汇表是有界的 17 通道 × 左右；有界性写在契约里，不因为"能诱导出更多"就宣称开放语义。
- **不把翻译者的猜测当回答**：LLM 不在回路里，读出弃权就弃权，问题路由也不许从问句本身推答案。

## 五、出处链接

- [Bohnemeyer 2015（语义诱导的实用认识论）](https://www.researchgate.net/publication/283640940_A_practical_epistemology_for_semantic_elicitation_in_the_field_and_elsewhere)
- [Bochnak & Matthewson 编《Methodologies in Semantic Fieldwork》](https://global.oup.com/academic/product/methodologies-in-semantic-fieldwork-9780190212339)
- [Haude：小语料与 elicit 出的未记录结构（Movima）](http://www.univ-paris3.fr/medias/fichier/abstract-katharina-haude-labex-efl-06-02-2014_1389888081479.pdf)
- [Thieberger 编《The Oxford Handbook of Linguistic Fieldwork》](https://academic.oup.com/edited-volume/38620)
- [Meakins, Green & Turpin《Understanding Linguistic Fieldwork》（开放获取）](https://www.taylorfrancis.com/chapters/oa-mono/10.4324/9780203701294-12/)
- [Crane & Fleisch 2019：会话式诱导而非清单式提问](https://www.degruyterbrill.com/document/doi/10.1515/lingty-2019-0018/html)
