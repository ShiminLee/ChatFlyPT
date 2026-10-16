# 真实声学读出组件（2026-09-24）

已将 DAS 作者发布的果蝇鸣声模型在本机运行，并完成数值转换、完整录音和无虚构表达测试。**完整感知和自由聊天仍未完成。此模型分类声音类型，不解码任意句意；当前物理身体没有发声信号生成模型。**

## 实际实现

- `chatflypt/acoustic_readout.py`：10 kHz 单声道波形 → 作者原始模型 → noise/sine/pulse 逐采样分类和区间。输入不含人类消息、感觉类别、物品身份或运动意图。
- 原始权重来自 [DAS 模型库](https://github.com/janclemenslab/das-menagerie/releases/tag/Dmel_single_v1.0)，未重新训练。下载文件及 SHA-256 在 `research/outward-signals/das-pretrained/sources.json`。
- `scripts/convert_das_model.py`：读取 HDF5 的图结构和数字权重，显式转换为 ONNX。旧 Python 3.7 Lambda 字节码不反序列化；归一化用作者公开的 `x / (max(abs(x), axis=2) + 1e-5)` 运算。
- 网络感受野左 1,440、右 1,443 个采样。分析分块只拼接有完整上下文的内部结果；全局首尾不分类。需要 144.3 ms 未来声音，不能用于声称零延迟的因果控制。
- 声学输出可以形成“检测到与果蝇持续鸣声模型相符的声学片段”这样的检测说明；不输出“我想吃糖”“请向左”等未经验证的句意。
- `expression.py` 与服务状态显示声学分类器已经安装、语义解码器和仿真身体声源尚缺；聊天没有新增虚构的果蝇发言，也不调用 LLM。

## 检验结果

[数值与录音验收](das-acceptance.json)，[自动化测试](tests.xml)。

50 项声学、对话、环境及启动依赖测试通过；最后的 21 项声学、HTTP 对话隔离及推理依赖测试也通过，两组有重叠，不相加。WAV 命令行已实际分析 5–7 秒外部录音，结果在 `cli-test.json`，未产生聊天消息。当前本地服务状态已验证分类器安装且 `semantic_decoder`、`body_signal_producer`、`llm_in_pathway` 均为 false。

- 3 组输入（含真实声音与静音），ONNX 与独立 NumPy 图计算最大误差 **8.3447e-7**。没有运行原 TensorFlow 版本；这是两个数值实现的比较。
- 相移重叠窗口在有效感受野内误差 **0**。
- 作者发布的 112.4352 秒录音，排除首尾不完整上下文后：

| 声学类别 | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|
| 持续鸣声 sine | 95.27% | 97.00% | 0.9613 |
| 脉冲鸣声 pulse | 87.72% | 80.87% | 0.8416 |
| 背景 noise | 99.05% | 98.91% | 0.9898 |

这里按**采样点**比较，不是按单个脉冲事件计分。录音与作者训练集的重叠情况未知，不能据此宣称独立动物泛化。静音和一组白噪声控制的有效窗口均没有鸣声阳性；这也不构成任意噪声下无误报的保证。

作者示例里只有声学标签，没有自然语言语义标签。[DAS 文档](https://janclemenslab.org/das/unsupervised/flies.html) 与 [模型库](https://github.com/janclemenslab/das-menagerie) 分别给出原始数据来源及模型适用种类。

## 复现

可选依赖在 `requirements-research.txt`。服务启动不导入 ONNX Runtime、HDF5 或训练库。

```powershell
.venv\Scripts\python.exe scripts/fetch_das_model.py
.venv\Scripts\python.exe scripts/convert_das_model.py
.venv\Scripts\python.exe scripts/verify_das_readout.py
.venv\Scripts\python.exe scripts/analyze_fly_audio.py recording.wav --output analysis.json
```

WAV 接口只接受 10 kHz 单声道，采样率不同会明确报错，不暗中改变时间或选择声道。外部录音来源始终写为外部数据，不能冒充模拟身体发出的信号。

## 失败基线与数据解释

保留最初的简单森林基线和报告（`scripts/acoustic_baseline.py`、`scripts/train_acoustic_readout.py`、`acceptance.json`），方便追溯。它按 0–60/60–80/80–112.4352 秒切分，没有跨分区窗口重叠，但脉冲精确率没有达到阈值，因此脉冲类被拒识，没有启用。

后来取得作者模型附带的逐采样标注，确认其波形等于先前示例的 float16 版本。此前形状聚类示例中的 pulse/sine 时间点不能保证覆盖所有事件，把其他点当负类会污染评价；旧基线的“未标注背景”不能解释为真静音。基线不作为可部署读出器，也没有以反复调整测试集参数来补分数。

## 尚未解决的上游与语义缺口

1. 神经活动必须经真实的翅肌/翅运动/声学模型产生模拟声信号；当前身体没有这个模型。运动命令中写了“song”或有展翅，不等于已经有鸣声。
2. 化学信号还缺释放动力学、环境传播和传感；基底振动还缺接触力到介质传播及接收模型。
3. 即便识别出鸣声类型，也不能推导开放的人类句意。需要与具体交流功能对应的可验证数据；当前没有“糖在哪里”等句意的输出监督依据。
4. 视觉空间配准、味觉受体及物理标定、内部代谢/内分泌等感知缺口见 `reports/persistent-session/remaining-capabilities.json`。此次声学组件不改变这些状态。

这不是把全部连接组下载下来就自然消失的缺口。原始 [Shiu 等全脑模型论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/) 明确说明其简化模型省略了非脉冲神经元、内部状态、长程神经肽等机制，且精确动力学可能表现不佳。现有模型在部分感觉运动任务上的验证，不构成完整感觉能力或开放语言语义的验证。
