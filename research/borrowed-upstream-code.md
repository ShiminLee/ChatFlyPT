# 借用的上游代码与数据（含许可与未借用清单）

日期：2026-09-29。本文件记录"看别人怎么做的、直接抄开源代码"这一轮实际借用了什么、从哪借、许可如何、以及**刻意没借**的东西。原则与仓库既有做法一致：能借就借，但必须留证、必须能说清哪部分不是我们写的。

## 1. 真借了：上游羽流数据 + 索引算法

| 项 | 内容 |
|---|---|
| 来源 | `vendor/flygym-gymnasium/`（**Apache-2.0**，已在 `research/manifests/sources.lock.json` 锁定） |
| 数据 | `flygym_gymnasium/data/test_data/plume_tracking/plume_short.hdf5`，sha256 `40a76057…`，`plume` 网格 60×320×480（文件存 float16），值域 0–0.789，另有 `inflow_pos/radius/scaler` |
| 算法 | `examples/olfaction/plume_tracking_arena.py` 的取帧与取格方式：`frame = int(t·fps)`、`column = x_mm/cell_mm`、网格外为 NaN |
| 上游模型 | 预计算的二维 Navier–Stokes 羽流 + Brownian 入流（生成脚本 `examples/olfaction/simulate_plume_dataset.py`） |
| 我们的落地 | `chatflypt/plume_field.py`（重写为可导入的纯 NumPy/h5py 实现，保留出处注释与许可标注）；经 `environment.py` 的 `field='plume'` 使用 |
| 为什么是重写而非直接 import | 上游 arena 依赖 `dm_control` 与 flygym 1.x 的飞行模型，两个 venv 都没有装；直接 import 不可行，**只有数学与数据可移植** |
| 我们的工程选择（已标注） | 上游是 160×240 mm、0.5 mm/格；我们的场地只有几毫米，因此用显式偏移 `ORIGIN_CELL=(160,23)` 把果蝇放在羽流填充区核心；上游只录了 0.3 秒，超过就**循环取帧**（标注为重复录制而非更长的仿真）；采样点出网格 → “无输入”而不是 0 |
| 使用边界 | 不是 cVA 浓度、不是实测量纲；受体仍是注释层候选群 |

## 2. 借了"形"没借"码"：连续向量目标的解码头

| 项 | 内容 |
|---|---|
| 来源 | `vendor/flyvis/`（**MIT**，已锁定）：`flyvis/task/decoder.py` 的 `DecoderGAVP`、`flyvis/task/objectives.py` 的 `epe`/`l2norm` |
| 借的是什么 | **设计形状**：把任务变量当作**连续向量**回归，用向量目标评分；而不是把我们自己的离散标签分类器调参 |
| 落地 | `scripts/train_intent_readout.py --mode continuous`：回归（偏航变化, 路径长度），资格用验证集的相关系数 + 转向符号一致率 + 位移阈值门的 precision/recall；规则在 v5 语料存在之前写进 `reports/intent-readout/protocol.json` |
| 没借什么 | **没有复制任何 flyvis 代码，也没有用它的权重**（flyvis 依赖 torch，未安装；且它的权重是在 Sintel 自然影片上训练的，与我们的模拟果蝇不是一回事） |

## 3. 只作参照、未接入（但已用来交叉验证）

- `vendor/flm/`（**MIT**，代码开源、无权重）：其 `scripts/prepare_graph.py` 用与我们相同的公开 feather 文件独立重建 MaleCNS v1.0 图。我们**跑通了它**（`data/local/flm_crosscheck.py`，把输出目录补到临时目录以免污染 vendor 树，两个源文件用硬链接并以它们自己断言的 sha256 校验），得到：

  | 项 | 上游独立实现 | 我们的 `graph.json` |
  |---|---|---|
  | 神经元 | 166,700 | 166,700 ✓ |
  | 注释边行 | 25,582,938 | 25,582,938 ✓ |
  | 突触接触 | 124,177,617 | 124,177,617 ✓ |
  | 建模（带符号快突触）边 | — | 24,559,135 = 25,582,938 − 1,023,803 ✓ |

  即：**我们的图构建被一份独立的第三方实现逐项复现**，而我们建模的边正好等于注释边减去"没有快突触递质模型"的 1,023,803 条——这条差额也因此在外部实现下得到确认。它的 `Reservoir` 是随机接口而不是解码器（作者自己在文档里说明 `B/P` 不是解剖通路），因此没有接入主通路。
- `vendor/shiu-brain/`（MIT）：我们的 LIF 方程与数据的来源，早已按"数据+公式"方式使用（`chatflypt/brain.py`），不是这一轮新借的。

## 4. 刻意没借（含理由）

| 候选 | 许可 | 为什么没借 |
|---|---|---|
| **CEBRA**（EPFL Mathis lab） | 0.4.0 起 Apache-2.0 | **有专利**（"Dimensionality reduction of time-series data"，2025 年授权，经 EPFL TTO 授权），且需要 torch；在一个以"可复核、可发布"为前提的仓库里引入专利约束的依赖不合适 |
| `vendor/eon-fly-brain/` 基准框架 | **GPL-2.0-or-later** | 强 copyleft；其 Shiu 子目录虽为 MIT，但与已用的 `shiu-brain` 重复。现仅用于 AST 提取刺激 ID 字面量（不复制代码） |
| `vendor/flygym/` 2.1.0 | — | 检索确认**没有任何**气味/风/声音代码（0 命中），无物可借 |
| flygym-gymnasium 的 arena **代码** | Apache-2.0 | 依赖 `dm_control` 与 1.x 模型，不可 import；只借数据与数学 |
| flyvis 权重 | — | 仓库里没有预训练检查点（上游需另下载），且任务不对口 |

## 5. 这一轮没有改变的分工

读出的主体仍是本项目自己的线性探针（`unified_readout.py`、`intent_readout.py`），上游解码器只提供**设计参照**。任何借用都必须通过同一套纪律：物理呈现 → 实测接收 → 盲读出 → 合格才说，不合格弃权；借用项一律在代码与文档里注明出处与许可。
