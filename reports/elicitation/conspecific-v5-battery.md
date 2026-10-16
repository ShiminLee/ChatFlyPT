# 诱导题库：conspecific-v5

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

## 同种触碰：相同总接触时间下的节奏能否区分？

- 条件：tap1x180_left、tap1x180_right、tap3x60_left、tap3x60_right、tap6x30_left、tap6x30_right
- 最小对立：每次 180 ms 总接触时间固定，只改敲击次数（1/3/6 次）与间隔
- 判定标准：在同一侧、同一总接触时间的三种节奏之间报告是否出现可区分的读出或行为差异；无可区分差异时如实报告为不可区分

## cVA 信息素场能否读出，远源是否不误报？

- 条件：pheromone_left、pheromone_right、far_pheromone_control
- 最小对立：实际 funiculus 位置的小场，左右各一；外加超出范围的远源对照
- 判定标准：cva:left/right 合格，且远源对照不产生标签

## 求偶鸣声场能否读出，远源是否不误报？

- 条件：song_left、song_right、far_song_control
- 最小对立：实际 pedicel 位置的小场（约翰斯顿氏器所在），左右各一；外加远源对照
- 判定标准：auditory:left/right 合格，且远源对照不产生标签

## 单侧光照的侧别能否读出（原配准混杂的对照）？

- 条件：unilateral_bright_left、unilateral_bright_right
- 最小对立：对侧眼遮光，只改一侧视网膜输入
- 判定标准：bright:left 与 bright:right 分别合格；与双侧亮条件对照

## 换用上游羽流场后，cVA 能否被读出？

- 条件：pheromone_plume_left、pheromone_plume_right、pheromone_left、pheromone_right
- 最小对立：同一落点几何，只改场模型：固定有限距离场（v4）对比借用的上游羽流网格
- 判定标准：cva:left/right 在验证集达到 precision>=0.9 且 recall>=0.75；固定场条件下不合格即为"投递受限"的证据

## 换用上游羽流场后，求偶鸣声能否被读出？

- 条件：song_plume_left、song_plume_right、song_left、song_right
- 最小对立：同一落点几何，只改场模型：固定有限距离场（v4）对比借用的上游羽流网格
- 判定标准：auditory:left/right 合格；否则记录为投递仍不足

## 源跟随受体（伙伴贴身）能否让同种信号读出合格？

- 条件：pheromone_partner_left、pheromone_partner_right、song_partner_left、song_partner_right
- 最小对立：保留声明的偏移几何，但每个采样时刻源锚定到实测受体位姿；与固定源对照
- 判定标准：cva:*/auditory:* 合格；跟随是"伙伴一直在"的物理主张，须与固定源分别报告

## 对侧前腿的多回合敲击能否产生可被读出的运动/转向？

- 条件：nudge_left、nudge_right、nudge_alternating、touch_left、touch_right
- 最小对立：单次 300 ms 探针（touch_*）对比 3x60 ms、间隔 140 ms、强度 1.5 的对侧敲击序列
- 判定标准：报告实测 >=5 度的转向窗数量与运动分量是否合格；转向分量不达标时如实弃权
