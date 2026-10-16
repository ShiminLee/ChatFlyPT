# 身体控制统计修正及轨迹一致性验证

2026-09-24。只修改报告统计，保留原始命令、vendor 控制器、神经映射、控制增益、步进次序与物理状态更新。

`command_execution` 对照固定版本原程序报告可执行分支：escape/landing/grooming 必须大于 .01；left/right、feeding/song/flight 和胫节目标偏移没有该死区。字段表示程序是否接受了输入，不宣称产生了行为，也不把滤波尾迹说成新脉冲。

- 原始 `commands` 完整保留，新增 `requested_command_nonzero` / `requested_command_in_window` / `requested_command_steps`。
- `neural_command_active`、`motor_activity_in_window` 与 `powered_steps` 表示上述可执行分支，另提供逐程序计数及 `no_effective_command_steps`。它们不是执行器功率测量。
- `BodyControlStatistics` 逐积分步记录实际使用的命令与关节偏移，因此包含窗口起点持有的前一完成窗命令，不含终点才可用的下一命令。
- 所有 min/max 和力峰值独立于其他消息窗口；没有零命令步的窗口，`max_force_during_zero_commands=null`。新增 `max_force_without_effective_commands` 可显示尾迹未被执行时仍有姿态维持力。
- `joint_summary` 同步修复边界持有及人为塞入零的问题。
- `proprioception.py` 去掉固定“100 ms”陈旧断言，明确 10/100 ms 离散采样均尚未验证高频振动转导。

验收命令：

```
.venv/Scripts/python.exe -m pytest tests/test_body_statistics.py tests/test_behavior_body.py tests/test_joint_motor.py tests/test_proprioception.py -q
```

结果：**50 passed in 42.29s**。覆盖阈值边界、小信号无死区通道、被动模式、实际零命令姿态力、跨消息持有、窗口力不串入、真实身体与原控制器一致性、胫节几何符号和本体感觉来源。

另用已归档驱动进行物理重放，**没有运行大脑**：

1. 修改前保存 `before.json`，修改后保存 `after.json`，0–100 ms 的 11 帧位置、旋转、关节、接触、控制值和力逐值相同，身体 summary 相同。见 `trajectory-verification.json`。
2. 用同一持续身体依次重放原 `unified-pathway-20260924/turn-0..2` 驱动，100/200 ms 加入及移除同一糖方块。三个窗口共 33 个保存帧的物理量与原归档逐值相同，见 `persistent-verification.json` 及 `persistent-turn-0..2`。

| 窗口 ms | 原始非零命令步 | 实際可执行分支步 | 零命令力峰值 | escape 实际 min/max |
| --- | ---: | ---: | --- | --- |
| 0–100 | 600 | 600 | 40 | 0 / 0.3934693402873666 |
| 100–200 | 1000 | 1000 | null（无零命令步） | 0.011881744533584322 / 0.08779487691181714 |
| 200–300 | 1000 | 0 | null（无零命令步） | 0.0009753129836887861 / 0.007206642350491874 |

第三轮的小数命令仍完整传入原控制器，只是报告不再把被原程序忽略的尾迹标为已执行动作。窗口切换只新建统计器，没有 reset 身体或控制器。

未完成行为生理校准；此次一致性验证证明统计修正没有改变这些复测轨迹，不能扩大为全部场景的生理正确性。
