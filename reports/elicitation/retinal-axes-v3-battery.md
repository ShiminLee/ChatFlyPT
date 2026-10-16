# 诱导题库：retinal-axes-v3

每个条件都是一次提问。对照是"最小对立对"（只改一个因素），判定标准在采集前固定。
没有差异同样要如实记录，不能为了好看去调参数。

## 视觉：光照的有无能否读出？

- 条件：bright、dark
- 最小对立：同一场地、只改照明；两段独立会话
- 判定标准：bright:left/right 在验证集达到 precision>=0.9 且 recall>=0.75

## 嗅觉：酵母场能否在无光条件下读出？

- 条件：yeast、dark_yeast、dark
- 最小对立：同一酵母几何，开灯/关灯互为对照，dark 为无气味基线
- 判定标准：yeast:left/right 合格；仅在开灯时可读即为视觉混杂

## 嗅觉侧别：单侧气味入口能否被分别读出？

- 条件：unilateral_yeast_left、unilateral_yeast_right
- 最小对立：左右各一次固定小范围气味源，实际事件决定侧别
- 判定标准：yeast:left 与 yeast:right 分别合格

## 温度：升温/降温能否读出？

- 条件：heat、cold、unilateral_heat_left、unilateral_heat_right、unilateral_cold_left、unilateral_cold_right
- 最小对立：同几何热源与冷源、左右各一；冷热符号在受体处先合成
- 判定标准：warm:* 与 cold:* 合格；单侧条件给出侧别证据

## 味觉：几何相同的糖平台与普通平台能否分开？

- 条件：leg_sugar、leg_block_control、dark_leg_sugar_control、dark_leg_block_control
- 最小对立：两种平台的几何完全一致，只有材质不同；暗条件再排除视觉
- 判定标准：leg_sweet_local/leg_appetitive_ascending 合格，且 block 条件不产生味觉标签

## 看得到但碰不到：远处可见的糖是否会被当成尝到？

- 条件：far_sugar_control
- 最小对立：糖在视野内但不在接触范围
- 判定标准：不产生 sugar_specific 标签（阴性对照必须为阴性）

## 口器接触能否代替唇部味觉？

- 条件：mouth_sugar、mouth_high_salt、mouth_block
- 最小对立：rostrum/haustellum 实际接触；官方身体没有已注册的 labellum 表面
- 判定标准：不产生 sugar_specific/high_salt 标签；记录为未映射（预期阴性）

## 触碰：真实接触能否读出，侧别是否来自实际碰到的腿？

- 条件：touch_left、touch_right
- 最小对立：物理探针 + 撤除各两窗；探针侧别与实测接触腿分离
- 判定标准：foreleg_touch:left/right 合格；撤除窗不产生标签

## 单侧光照的侧别能否读出（原配准混杂的对照）？

- 条件：unilateral_bright_left、unilateral_bright_right
- 最小对立：对侧眼遮光，只改一侧视网膜输入
- 判定标准：bright:left 与 bright:right 分别合格；与双侧亮条件对照
