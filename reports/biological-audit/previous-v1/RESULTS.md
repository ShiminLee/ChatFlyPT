# ChatFlyPT 独立复现：实际运行结果

这是基于公开描述的独立功能重建，未取得 @chetaslua 原始代码、词表或评测集，不能称为原实验精确复现。

## 展示例子验收

| 例子 | 作者公开报告 | 本机实测 |
|---|---|---|
| 糖 / 糖+苦的 MN9 | 78 → 3 Hz | 85.9 → 9.4 Hz；10 次、每次 1 秒 |
| 16 词单标签识别 | 100% | 100.00%；独立定义词表、96 次新试次 |
| 仅下行/运动神经元读出 | 85% | 90.62% |
| 醋+酵母+糖+温暖+明亮句子 | 展示 1 句 | 20/20 次完整标签/句子一致 |
| 听觉编码后左转 | 37/37 | 自建玩具环境 37/37；下游外部读出驱动 |
| 原生 DNa01/DNa02 方向性 | 原帖缺少可核验测量 | 左编码试次中左右活动相同 37/37，未证明原生左转 |
| 觅食 | 未找到食物 | 自建环境 food_found=False，64 步 |
| 真实 ChatGPT | 作者称已调用 | unavailable；实际 OpenAI 协议调用=False |

不能把独立词表、独立输入参数和独立环境下的数字当成作者原评测结果。

## 多标签和泛化

| 评测 | exact-set accuracy | micro F1 | macro F1 |
|---|---:|---:|---:|
| train | 100.00% | 1.000 | 1.000 |
| validation | 100.00% | 1.000 | 1.000 |
| test | 99.40% | 0.999 | 0.998 |
| ood | 84.38% | 0.982 | 0.982 |

OOD 同时包含新组合和不同于训练范围的 0.6×/1.4× 强度；因此结果不是单纯强度变化的因果隔离。

## 神经信号负对照

- shuffled_neural_observations：单词 top-1 5.21%，多标签 micro F1 0.150。
- zero_neural_observations：单词 top-1 6.25%，多标签 micro F1 0.000。

## 第一条展示的实际输出

I smell vinegar and yeast. I taste sugar. It is warm and bright. What should I do?

该句来自固定 seed=81000 的首次测试，没有选择最好的一次。完整 20 次结果见 [first-sentence.json](first-sentence.json)。
提问结尾是模板；感官词来自盲读出。醋/酵母/光/温度的输入使用文档中明确列出的代理假设。

## 真实大模型记录

尚未完成真实 ChatGPT 调用，不能把模板或规则策略记为大模型回答。
配置 `.env` 后运行：`.\.venv\Scripts\python.exe -m chatflypt.demo --llm-only --provider openai`。

## 方法与证据

- [方法和假设](../../research/reconstruction-method.md)
- [固定协议](protocol.json)
- [全图引擎对照](engine-verification-full.json)
- [读出评测](readout-evaluation.json)
- [听觉指令测试和真实 DNa 活动](auditory-command.json)
- [闭环轨迹](closed-loop.json)
- [机器可读验收](acceptance.json)

下游分类器恢复工程指令不等于原生左转神经机制。这里没有身体物理或主观体验的验证。

![实测结果](results.png)
