# 建议回传工程码本：四动作实测

`CommandFeedback` 复用现有 `command-decoder-non_sensory.npz` 和 `stimuli.json:auditory_commands`，没有重训。每个条件使用新的 seed，真正执行 `Brain.run(..., 300 ms, record_spikes=True)`；本验收没有运行身体或把解码标签转成身体命令。

| 工程输入 | seed | 输入事件 | 全脑 spikes | 非感觉解码特征 spikes | 实际解码 |
|---|---:|---:|---:|---:|---|
| left | 9200001 | 6448 | 6749 | 374 | command:left |
| right | 9200002 | 5188 | 5806 | 664 | command:right |
| forward | 9200003 | 11697 | 12419 | 848 | command:forward |
| stop（静默） | 9200004 | 0 | 0 | 0 | words=[]，command=stop，status=silence |

这是四次新神经试次的码本读出结果，不是自然语言理解、总体准确率或实际运动成功率。静默 trial 也独立运行；空 `words` 避免把没有活动说成“听到了停下”。解码器只接受计数，未接收输入 action、seed 或事件元数据。

固定参数：左侧 212、右侧 175 个已注释听觉细胞；单侧或双侧 100 Hz，持续 300 ms；stop 无输入。解码器 273 个特征均核对为非感觉、非直接输入细胞。模型 SHA256 在四次运行前后保持 `f4b3e1407ca5aa0b18f61a83fec9d64ed40035b5dfbfd7dd13baf99fa3f24bbf`。

20 项单元测试通过，包括：输入标签为 left 而固定神经记录解码为 right 的隔离测试；直接输入计数改变不影响解码；偏置偏向 left 时全零仍返回 silence；非法 action/seed 在运行脑模型之前拒绝；感觉特征或 manifest 输入细胞混入特征时拒绝；事件记录与同种子的引擎输入时间表一致。

完整记录：`acceptance.json` 含每次原始脉冲和输入事件 NPZ 的 SHA256、模型/码本/注释来源、实际分数和接收描述；`*-neural.npz` 保存计数及脉冲时序，`*-inputs.npz` 保存实际确定性 Poisson 输入时间表。所有公开神经 ID 均为字符串，整数索引另外标明命名空间。

> 归档说明（2026-09-28）：本文档记录的历史通路已从代码树移除——`chatflypt/command_feedback.py`、`tests/test_command_feedback.py` 与顾问建议回传链路（`demo.py` 的闭环/LLM 阶段、`dialogue.py` 的 adviser 协议）均已删除。下方命令不再可运行；本目录保留为历史记录，`acceptance.json` 与 `*-neural.npz` 仍是当时的实测证据。聊天文本直接编码为神经磁带的旧入口（`server.receive`）同时移除。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_command_feedback.py -q
.\.venv\Scripts\python.exe reports/command-feedback-2026-09-20/acceptance.py
```

生产 API 的 HTTP 接线、可视化和真实 MuJoCo 动作结果由独立集成验收覆盖；本模块只返回新的 `trial`、输入、解码器及溯源。
