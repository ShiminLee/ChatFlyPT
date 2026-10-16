# 同种信号通道：cVA 信息素与求偶鸣声（MaleCNS 侧）

日期：2026-09-29。本条把"果蝇之间怎么沟通"的两个模态接进**同一个 MaleCNS 个体**，与已有的腿触碰通道并列。方法与边界如下。

## 1. 生理落点（用实际注释与模型位置，不猜）

| 信号 | 受体 | 模型采样位置 | MaleCNS 选群 | 证据 |
|---|---|---|---|---|
| cVA 信息素 | 触角嗅觉受体神经元（Or67d → DA1  glomerulus） | `l_funiculus` / `r_funiculus`（嗅觉感器所在节） | `class=='olfactory' & type=='ORN_DA1'`：204 个细胞，可执行 156（左 51 / 右 105） | Datta 2008（cVA→Or67d→DA1）、Wang & Anderson 2010；注释名按 glomerulus，无受体基因列 |
| 求偶鸣声 | 约翰斯顿氏器（触角梗节内的听觉弦音器） | `l_pedicel` / `r_pedicel`（JO 所在节） | `class=='mechanosensory' & subclass=='auditory'`：114 个细胞，可执行 77（左 60 / 右 17），37 个无模型出边被审计拒绝 | Clemens 2018（JON 编码、36 ms 脉冲间隔示例）；官方 `subclass=='auditory'` 注释 |

**明确排除**：MaleCNS 里 425 个 `subclass=='chordotonal organ'` 的细胞全部是**腿**本体感觉（MetaLN/MesoLN/ProLN/ProCN），**不得**当作听觉代理；`JO-FV/EV/ED` 属风/重力类，也不算听觉。

## 2. 物理呈现与落点验证

- 两个信号都做成**非实体工程场**（`pheromone`、`song`），不能碰撞身体。
- 场在**实际受体节**采样：cVA 在 funiculus、鸣声在 pedicel。近侧偏移 0.08 mm、工程半径 0.20 mm。
- 落点检查（`scripts/check_conspecific_placements.py` → `reports/unified-readout/conspecific-v4-placement.json`）：
  - 零接触、初始身体位姿逐位不变（`max_initial_body_position_change_mm = 0.0`）；
  - 初始场**单侧**：同侧 0.36、对侧 0.0（四组：cVA 左右、鸣声左右）。
- 鸣声的**时间结构**由受体编码器施加（`pulse_envelope`：36 ms 脉冲间隔、5 ms 门宽，与 FlyWire 侧 `courtship_pulse` 通道同一文献依据），**不是**靠移动声源实现；未实现载波、相位锁定与适应。

## 3. 人类语言到这两个模态

- "释放信息素" → 通道 `cva` → 在指定方位放置 `pheromone` 源；"播放求偶鸣声" → 通道 `courtship_pulse` → 放置 `song` 源。左右可指定（"在左边放信息素源"）。
- 放置后**是否收到由物理采样决定**：场强低于阈值就没有事件；远源对照不产生标签。
- 与聊天契约一致：这些词只改变物理环境，不编码任何神经事件；没有实体的旧通道（醋、声波等）仍明确不执行。

## 4. 工程参数与未标定项

- 场强不是浓度、不是声压；`strength` 只缩放工程场。
- cVA 的 100 Hz 与鸣声的 150 Hz 峰值是工程包络，不是实测放电率。
- 受体群是**注释层面的候选群**（群身份有依据，单细胞调谐未测）；因此新通道一律 `candidate` 状态、`advanced` 标记，读出必须重新训练（不复用 FlyWire 权重或标签）。
- 侧别来自各受体节的实际左右位置采样，不来自"计划侧别"。

## 5. 闭环（第二只果蝇）

`scripts/conspecific_loop.py` 用两个独立的 MaleCNS 会话串起最小闭环：A 的记录运动活动里 `song` 组大于零 → 在 B 的**实际 pedicel** 位置放 `song` 场 → B 实测接收、盲读出、实测身体结果。这是**声明的工程约定**（不是解码出的"歌"），也未证明 A 有交流意图或 B 理解了什么；A 若在采样窗内没有唱歌，脚本如实报告"未产生同种信号"。

首次运行结果：A 在 6 个采样窗内 `song` 组始终为 0，因此**没有向 B 呈现信号**，脚本报告无响应（`reports/conspecific-loop/latest.json`）。

## 6. 读出结果：通路通、标签未合格

`conspecific-v4` 语料（108 会话 / 228 窗，含 6 个鸣声条件与 6 个信息素条件）的实测投递证明**物理通路确实工作**：`mc_cva` 在 9 个窗口、`mc_auditory` 在 15 个窗口收到了事件。但两者都没有通过读出资格（每 split 只有 1–4 个正例，少于门槛要求）：固定源放在受体的**初始**位置，自由行走的果蝇很快离开 0.20 mm 的场。

结论（已记入 `reports/corpus/negative-registry.json`）：`cva:*` / `auditory:*` 保持**显式弃权**；要让它们合格，需要让源跟随受体（或加大场/延长呈现/专门采集更多投递窗），**不是**改门槛。

## 7. 与旧 FlyWire 侧通道的关系

旧 `social.py` 的 `cva` / `courtship_pulse` 通道在 **FlyWire v783** 上，属历史独立实验，仍可用于该侧的实验面板；它们**不得**与 MaleCNS 合并成同一个体（ID 命名空间与个体规则）。本次新增的是 MaleCNS 侧的同源群（`mc_cva`、`mc_auditory`），使正常聊天可以在单一个体上接收这两个模态。
