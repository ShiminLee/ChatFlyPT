# FLM 单独实验：连接组 + 预训练语言模型（2026-09-30）

原构想里"高维生成（目标）：用 Transformer 输入神经脉冲序列、输出自由文本；参考 FLM 项目"。本轮把 vendored 的 `vendor/flm`（MIT）**真的跑了一遍**，下面是它给什么、不给什么，以及为什么它只能作为**单独标注的实验**。

## 跑的是什么

| 项 | 内容 |
|---|---|
| 代码 | `vendor/flm`（MIT，pin `7251a892…`），未改一行源码 |
| 语言模型 | `LiquidAI/LFM2.5-1.2B-Instruct`，pin `0f604ada…`，权重 2237 MB（**LFM Open License v1.0**，非 MIT） |
| 训练语料 | `HuggingFaceTB/smoltalk` 的 `everyday-conversations` 两个 parquet（**CC BY 4.0**），64 训练 / 16 验证 / 24 测试对话 + 32 条自写风格样例 |
| 连接组 | 他们自己的 `prepare_graph.py` 用公开 MaleCNS v1.0 feather 重建（166,700 神经元 / 25,582,938 边 / 124,177,617 接触），与我们的图逐项一致（见 `research/borrowed-upstream-code.md`） |
| 训练 | 3 epoch、batch 32、lr 3e-4、预计算 LM 隐状态后只训 278k 参数的适配器；RTX 4060 Laptop（torch 2.6.0+cu124）**约 6 分钟** |
| 输出 | `data/local/flm-runs/conversation-v2/`（刻意不写进 vendor 树） |

遇到并解决的两个环境问题：HuggingFace 下载被 TLS 吊销检查挡住（`httpx CERTIFICATE_VERIFY_FAILED` / `curl CRYPT_E_NO_REVOCATION_CHECK`）——用 `curl --ssl-no-revoke` 取文件；`transformers 5.17` 与它的 chat template 不兼容（报 "Training/inference prefix mismatch"）——按它自己的 `requirements.txt` 降到 `4.57.6`。

## 结果（它自己的评测输出）

| 模式 | 留出对话 NLL | 困惑度 |
|---|---|---|
| base（不接连接组） | 1.3368 | 3.807 |
| **fly_adapter（连接组条件化）** | **1.3128** | **3.716** |
| direct_input_adapter（参数相同的直接输入对照） | 1.3132 | 3.718 |
| relabeled_wiring（打乱布线对照） | 1.3355 | 3.802 |
| no_edges（无连接） | 1.3368 | 3.807，`exact_base_identity: true` |

对话演示（`chat.py --prompt "Invent a tiny museum exhibit for a jar of pond water." --seed 42`）给出的是一段关于池水罐展柜的流畅英文建议——**与果蝇无关**。它自己的备注也写明："24 held-out conversations, not a broad chat benchmark；语料可能出现在骨干模型的预训练里；布线特异性需要更强的对照。"

## 结论：它给什么、不给什么

- **给**：一条可运行的开源"连接组状态 → 适配器 → LM logits"管线，含 no_edges 恰好等于基线的性质测试、以及一个参数匹配的对照——工程上是干净的参照实现。
- **不给**：
  1. **不是解码器**。它的适配器只把连接组状态作为条件去微调 LM 的 logits；语言来自预训练模型（其系统提示就写着"你是软件……不要声称自己是活体果蝇"）。
  2. **与我们的实测状态无关**。它用的是自己的图 + 自带的随机输入/输出接口（作者注明不是解剖通路），而不是我们 MaleCNS 会话里 10 ms 步进的实测活动。
  3. **连接组带来的增益很小且不特异**：相对基线 +0.024 NLL，而**参数相同的直接输入对照几乎打平**（1.3132 对 1.3128）；打乱布线的对照回到基线附近。也就是说，这批数据上看不出"结构本身"的可辨贡献。
  4. **内容不受我们实测事实约束**：它回答的是用户提示的议题，不是"果蝇测到了什么"。

## 与项目契约的界线（为什么它不进聊天）

项目红线是"不调用模型代替果蝇说话"、聊天里的话必须来自盲读出。用 LM 生成句子即使以连接组状态为条件，语言能力仍来自模型——按契约只能作为**单独标注的实验**存在，必须明确写"语言来自预训练模型"。因此：

- 正常聊天的"果蝇 · 神经读出"仍由**接地组合式渲染器**产出（槽位全部来自盲读出，零语言模型），见 `reports/generative-renderer/README.md`。
- 若今后要做"LM 只负责措辞"的版本，唯一在契约内的形态是：**事实槽位仍由盲读出决定，LM 仅把已定句子改写得更自然**，并在界面上标注"措辞由预训练模型生成"，同时审计改写是否引入了新事实（本项目的 `audit_grounding()` 正好可复用为该项的验收工具）。这一形态**尚未实现**，本文件只记录可行性与其边界。
