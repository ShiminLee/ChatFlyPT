# 冻结测试的实际接收、神经读出与身体记录

批次 `conspecific-v5`。只汇总冻结后测试集，不改变阈值或隐藏失败。
实际接收标签来自独立保存的受体事件；预测只来自内部神经记录。位移、路径长度和控制器分支被激活不等于完成找糖、逃避或其他行为目标。

| 场景 / 窗口 | 标签整组一致 | 误报 | 漏报 | 身体路径 mm | 生效控制分支 |
|---|---|---|---|---|---|
| test-bright / 0 | 是 | 无 | 无 | 0.6405 | escape |
| test-bright / 1 | 是 | 无 | 无 | 0.9993 | escape |
| test-dark / 0 | 是 | 无 | 无 | 1.1774 | escape |
| test-dark / 1 | 是 | 无 | 无 | 0.8897 | escape |
| test-yeast / 0 | 否 | 无 | yeast:left | 1.0788 | locomotion, feeding, song, joint_targets |
| test-yeast / 1 | 否 | 无 | yeast:left, yeast:right | 0.5890 | locomotion, feeding, song, joint_targets |
| test-dark_yeast / 0 | 否 | 无 | yeast:left | 0.9852 | feeding, song, joint_targets |
| test-dark_yeast / 1 | 否 | 无 | yeast:left | 0.5900 | feeding, song, joint_targets |
| test-heat / 0 | 否 | 无 | warm:left | 1.0944 | escape, feeding, song, joint_targets |
| test-heat / 1 | 否 | 无 | warm:left | 0.5570 | feeding, song, joint_targets |
| test-cold / 0 | 否 | 无 | cold:left, cold:right | 0.9172 | grooming, feeding, song, joint_targets |
| test-cold / 1 | 否 | 无 | cold:left, cold:right | 0.6516 | locomotion, grooming, feeding, song, joint_targets |
| test-mouth_block / 0 | 是 | 无 | 无 | 2.5537 | locomotion, escape, joint_targets |
| test-mouth_block / 1 | 是 | 无 | 无 | 0.7716 | locomotion, escape, joint_targets |
| test-mouth_sugar / 0 | 是 | 无 | 无 | 1.1915 | escape, joint_targets |
| test-mouth_sugar / 1 | 是 | 无 | 无 | 2.4229 | escape, joint_targets |
| test-mouth_high_salt / 0 | 是 | 无 | 无 | 1.6501 | locomotion, escape, joint_targets |
| test-mouth_high_salt / 1 | 是 | 无 | 无 | 0.6998 | locomotion, escape, joint_targets |
| test-leg_sugar / 0 | 是 | 无 | 无 | 2.8858 | locomotion, grooming, joint_targets |
| test-leg_sugar / 1 | 是 | 无 | 无 | 2.1218 | locomotion, joint_targets |
| test-touch_left / 0 | 是 | 无 | 无 | 0.5368 | escape, joint_targets |
| test-touch_left / 1 | 是 | 无 | 无 | 2.9020 | escape, joint_targets |
| test-touch_left / 2 | 是 | 无 | 无 | 1.0847 | locomotion, escape, feeding, joint_targets |
| test-touch_left / 3 | 是 | 无 | 无 | 0.6761 | locomotion, escape, feeding, joint_targets |
| test-touch_right / 0 | 是 | 无 | 无 | 1.8055 | escape |
| test-touch_right / 1 | 是 | 无 | 无 | 0.6700 | escape |
| test-touch_right / 2 | 是 | 无 | 无 | 1.8979 | escape |
| test-touch_right / 3 | 是 | 无 | 无 | 0.3982 | escape |
| test-leg_block_control / 0 | 是 | 无 | 无 | 1.6194 | locomotion, grooming, song, joint_targets |
| test-leg_block_control / 1 | 是 | 无 | 无 | 1.7997 | locomotion, song, joint_targets |
| test-far_sugar_control / 0 | 是 | 无 | 无 | 2.4514 | escape |
| test-far_sugar_control / 1 | 是 | 无 | 无 | 0.6394 | escape |
| test-dark_leg_sugar_control / 0 | 否 | 无 | leg_appetitive_ascending:left, leg_appetitive_ascending:right, leg_sweet_local:left, leg_sweet_local:right | 2.0886 | locomotion, grooming, feeding, joint_targets |
| test-dark_leg_sugar_control / 1 | 是 | 无 | 无 | 1.4904 | locomotion, feeding, joint_targets |
| test-dark_leg_block_control / 0 | 是 | 无 | 无 | 1.9356 | locomotion, grooming, joint_targets |
| test-dark_leg_block_control / 1 | 是 | 无 | 无 | 1.8646 | locomotion, joint_targets |
| test-unilateral_bright_left / 0 | 是 | 无 | 无 | 1.7326 | escape |
| test-unilateral_bright_left / 1 | 是 | 无 | 无 | 1.5940 | escape |
| test-unilateral_yeast_left / 0 | 否 | 无 | yeast:left | 1.0033 | feeding, song, joint_targets |
| test-unilateral_yeast_left / 1 | 是 | 无 | 无 | 0.6011 | feeding, song, joint_targets |
| test-unilateral_heat_left / 0 | 是 | 无 | 无 | 1.2890 | escape |
| test-unilateral_heat_left / 1 | 是 | 无 | 无 | 0.5280 | escape |
| test-unilateral_cold_left / 0 | 否 | 无 | cold:left | 1.5697 | grooming, escape, feeding, joint_targets |
| test-unilateral_cold_left / 1 | 是 | 无 | 无 | 0.6537 | feeding, song, joint_targets |
| test-unilateral_bright_right / 0 | 是 | 无 | 无 | 2.1889 | escape |
| test-unilateral_bright_right / 1 | 是 | 无 | 无 | 1.1238 | escape |
| test-unilateral_yeast_right / 0 | 是 | 无 | 无 | 0.8701 | locomotion, escape, feeding, joint_targets |
| test-unilateral_yeast_right / 1 | 是 | 无 | 无 | 0.6998 | locomotion, feeding, joint_targets |
| test-unilateral_heat_right / 0 | 否 | 无 | warm:right | 4.4882 | escape |
| test-unilateral_heat_right / 1 | 是 | 无 | 无 | 1.0815 | escape |
| test-unilateral_cold_right / 0 | 否 | 无 | cold:right | 0.4324 | escape |
| test-unilateral_cold_right / 1 | 是 | 无 | 无 | 0.7792 | escape |
| test-tap1x180_left / 0 | 是 | 无 | 无 | 2.3277 | escape, joint_targets |
| test-tap1x180_left / 1 | 是 | 无 | 无 | 1.4486 | escape, joint_targets |
| test-tap1x180_right / 0 | 是 | 无 | 无 | 0.5383 | escape |
| test-tap1x180_right / 1 | 是 | 无 | 无 | 1.6996 | escape |
| test-tap3x60_left / 0 | 是 | 无 | 无 | 2.3544 | locomotion, escape, joint_targets |
| test-tap3x60_left / 1 | 是 | 无 | 无 | 0.0977 | locomotion, joint_targets |
| test-tap3x60_right / 0 | 是 | 无 | 无 | 0.7030 | escape |
| test-tap3x60_right / 1 | 是 | 无 | 无 | 0.5216 | escape |
| test-tap6x30_left / 0 | 是 | 无 | 无 | 1.9135 | locomotion, grooming, escape, joint_targets |
| test-tap6x30_left / 1 | 是 | 无 | 无 | 1.1069 | locomotion, grooming, joint_targets |
| test-tap6x30_right / 0 | 是 | 无 | 无 | 1.4073 | escape |
| test-tap6x30_right / 1 | 是 | 无 | 无 | 0.0769 | 姿态维持 |
| test-pheromone_left / 0 | 否 | 无 | cva:left, cva:right | 1.7635 | locomotion, grooming, escape, feeding, joint_targets |
| test-pheromone_left / 1 | 是 | 无 | 无 | 0.5864 | locomotion, grooming, feeding, joint_targets |
| test-pheromone_right / 0 | 否 | cva:left | cva:right | 1.1319 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_right / 1 | 是 | 无 | 无 | 0.5405 | locomotion, feeding, song, joint_targets |
| test-far_pheromone_control / 0 | 是 | 无 | 无 | 0.7043 | escape |
| test-far_pheromone_control / 1 | 是 | 无 | 无 | 0.7786 | escape |
| test-song_left / 0 | 否 | 无 | auditory:left, auditory:right | 1.3599 | escape |
| test-song_left / 1 | 否 | 无 | auditory:right | 1.3302 | escape |
| test-song_right / 0 | 否 | 无 | auditory:left, auditory:right | 2.6510 | locomotion, escape |
| test-song_right / 1 | 是 | 无 | 无 | 1.3995 | locomotion, grooming, escape, joint_targets |
| test-far_song_control / 0 | 是 | 无 | 无 | 0.8068 | escape |
| test-far_song_control / 1 | 是 | 无 | 无 | 0.0690 | escape |
| test-pheromone_plume_left / 0 | 是 | 无 | 无 | 1.4947 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_plume_left / 1 | 是 | 无 | 无 | 1.0096 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_partner_left / 0 | 是 | 无 | 无 | 1.2412 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_partner_left / 1 | 是 | 无 | 无 | 0.9890 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_plume_right / 0 | 是 | 无 | 无 | 1.5220 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_plume_right / 1 | 是 | 无 | 无 | 0.9901 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_partner_right / 0 | 是 | 无 | 无 | 1.4697 | locomotion, grooming, feeding, song, joint_targets |
| test-pheromone_partner_right / 1 | 是 | 无 | 无 | 0.9352 | locomotion, grooming, feeding, song, joint_targets |
| test-song_plume_left / 0 | 是 | 无 | 无 | 3.4949 | locomotion, grooming, escape, feeding, song, joint_targets |
| test-song_plume_left / 1 | 是 | 无 | 无 | 3.9692 | locomotion, grooming, escape, feeding, song, joint_targets |
| test-song_partner_left / 0 | 是 | 无 | 无 | 2.8524 | locomotion, grooming, escape, joint_targets |
| test-song_partner_left / 1 | 是 | 无 | 无 | 3.2381 | locomotion, grooming, escape, feeding, joint_targets |
| test-song_plume_right / 0 | 是 | 无 | 无 | 7.5109 | locomotion, grooming, escape, feeding, song, joint_targets |
| test-song_plume_right / 1 | 是 | 无 | 无 | 3.2459 | locomotion, grooming, escape, feeding, song, joint_targets |
| test-song_partner_right / 0 | 是 | 无 | 无 | 3.4625 | locomotion, escape, joint_targets |
| test-song_partner_right / 1 | 是 | 无 | 无 | 3.4832 | locomotion, grooming, escape, joint_targets |
| test-nudge_left / 0 | 是 | 无 | 无 | 3.6636 | escape, joint_targets |
| test-nudge_left / 1 | 是 | 无 | 无 | 1.5392 | locomotion, escape, joint_targets |
| test-nudge_right / 0 | 是 | 无 | 无 | 2.3354 | escape, joint_targets |
| test-nudge_right / 1 | 是 | 无 | 无 | 2.3629 | locomotion, grooming, joint_targets |
| test-nudge_alternating / 0 | 是 | 无 | 无 | 1.9029 | locomotion, escape, joint_targets |
| test-nudge_alternating / 1 | 是 | 无 | 无 | 1.5767 | locomotion, escape, joint_targets |

物理失败窗口：0/98。完整预期/预测、控制幅度、记录路径和哈希见 `physical-test-summary.json`。
控制分支名称沿用复用控制器，仅描述它执行了哪个程序，不把程序名当作已观测到的自然行为。
