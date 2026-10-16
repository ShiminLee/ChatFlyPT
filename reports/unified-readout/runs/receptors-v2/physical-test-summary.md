# 冻结测试的实际接收、神经读出与身体记录

批次 `receptors-v2`。只汇总冻结后测试集，不改变阈值或隐藏失败。
实际接收标签来自独立保存的受体事件；预测只来自内部神经记录。位移、路径长度和控制器分支被激活不等于完成找糖、逃避或其他行为目标。

| 场景 / 窗口 | 标签整组一致 | 误报 | 漏报 | 身体路径 mm | 生效控制分支 |
|---|---|---|---|---|---|
| test-bright / 0 | 是 | 无 | 无 | 0.6405 | escape |
| test-bright / 1 | 是 | 无 | 无 | 0.9993 | escape |
| test-dark / 0 | 是 | 无 | 无 | 1.1774 | escape |
| test-dark / 1 | 是 | 无 | 无 | 0.8897 | escape |
| test-yeast / 0 | 是 | 无 | 无 | 1.0788 | locomotion, feeding, song, joint_targets |
| test-yeast / 1 | 是 | 无 | 无 | 0.5890 | locomotion, feeding, song, joint_targets |
| test-dark_yeast / 0 | 是 | 无 | 无 | 0.9852 | feeding, song, joint_targets |
| test-dark_yeast / 1 | 是 | 无 | 无 | 0.5900 | feeding, song, joint_targets |
| test-heat / 0 | 是 | 无 | 无 | 1.0944 | escape, feeding, song, joint_targets |
| test-heat / 1 | 是 | 无 | 无 | 0.5570 | feeding, song, joint_targets |
| test-cold / 0 | 是 | 无 | 无 | 0.9172 | grooming, feeding, song, joint_targets |
| test-cold / 1 | 是 | 无 | 无 | 0.6516 | locomotion, grooming, feeding, song, joint_targets |
| test-mouth_block / 0 | 是 | 无 | 无 | 2.5537 | locomotion, escape, joint_targets |
| test-mouth_block / 1 | 否 | 无 | foreleg_touch:right | 0.7716 | locomotion, escape, joint_targets |
| test-mouth_sugar / 0 | 是 | 无 | 无 | 1.1915 | escape, joint_targets |
| test-mouth_sugar / 1 | 否 | 无 | foreleg_touch:right | 2.4229 | escape, joint_targets |
| test-mouth_high_salt / 0 | 是 | 无 | 无 | 1.6501 | locomotion, escape, joint_targets |
| test-mouth_high_salt / 1 | 是 | 无 | 无 | 0.6998 | locomotion, escape, joint_targets |
| test-leg_sugar / 0 | 是 | 无 | 无 | 2.8858 | locomotion, grooming, joint_targets |
| test-leg_sugar / 1 | 是 | 无 | 无 | 2.1218 | locomotion, joint_targets |
| test-touch_left / 0 | 是 | 无 | 无 | 0.5220 | escape, joint_targets |
| test-touch_left / 1 | 是 | 无 | 无 | 0.4657 | escape, joint_targets |
| test-touch_left / 2 | 是 | 无 | 无 | 0.3878 | joint_targets |
| test-touch_left / 3 | 是 | 无 | 无 | 0.6929 | escape, joint_targets |
| test-touch_right / 0 | 否 | 无 | foreleg_touch:right | 2.3216 | escape |
| test-touch_right / 1 | 是 | 无 | 无 | 0.2222 | escape |
| test-touch_right / 2 | 是 | 无 | 无 | 1.7822 | locomotion, grooming, escape, joint_targets |
| test-touch_right / 3 | 是 | 无 | 无 | 1.5140 | locomotion, escape, joint_targets |

物理失败窗口：0/28。完整预期/预测、控制幅度、记录路径和哈希见 `physical-test-summary.json`。
控制分支名称沿用复用控制器，仅描述它执行了哪个程序，不把程序名当作已观测到的自然行为。
