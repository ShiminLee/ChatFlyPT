# 同一神经记录：旧六腿适配器与 Bad Apple Fly 动作方案对照

2026-09-19。本次固定四份已有神经仿真记录，未重跑、训练或调参神经网络，未增加合成脉冲。旧、新通路使用同一 NPZ、相同 seed 和同一 0–500 ms 时间窗。

冷刺激和糖刺激经新方案产生了非零命令，均完成真实 MuJoCo 回放；触碰及空白返回 `no_motor_activity`，没有生成主动轨迹。四份新旧脉冲来源摘要完全相同；旧回放还与原 HTTP 记录中的身体轨迹逐帧完全一致。

## 结果

水平位移和朝向来自真实轨迹的首尾差，不等于行为正确率。每次非空回放 500 ms、51 帧；所有实际运行均为有限状态，MuJoCo 警告为 0。‘未运行’并非零位移测量。

| 固定用例 | 数据集 / seed | 神经总 spikes | 旧选中 spikes | 新 13 组 spikes | 旧 / 新水平位移 mm | 旧 / 新 yaw 变化 ° | 新结果 |
|---|---|---:|---:|---:|---|---|---|
| 右前腿触碰 | MaleCNS v1.0 / 81000 | 434 | 2 | 0 | 0.18987 / 未运行 | -0.56071 / 未运行 | `no_motor_activity` |
| 左侧冷刺激 | FlyWire v783 / 5900005 | 196395 | 36 | 91 | 1.35526 / 1.48232 | -17.02029 / 5.61659 | `ok` |
| 左侧糖刺激 | MaleCNS v1.0 / 5900001 | 10259 | 305 | 27 | 0.42351 / 0.16493 | -1.27617 / -0.21082 | `ok` |
| 空白 | FlyWire v783 / 5900007 | 0 | 0 | 0 | 未运行 / 未运行 | 未运行 / 未运行 | `no_motor_activity` |

新冷刺激 / 糖刺激采样最大胸部倾角分别为 3.085° / 5.039°，未出现胸部垂直轴倾斜 ≥90° 的采样帧。此指标不是完整的跌倒检测；仅证明这两个 500 ms 回放未观察到翻覆。

触碰的新 13 组共 105 个已匹配细胞，实际全部为 0 spike；旧六腿群有 2 个 spike。不能将此结果描述为‘脑没有运动神经活动’。旧 MaleCNS 映射统计 `vnc_motor` 的腿别和侧别，新方案读取作者指定的 DN / MN9 / pIP10 群；两者不等价。因此旧通路有轨迹而新通路没有，并非集成失败。

## 13 组实际命中

以下每格为左 / 右 spike 数；细胞数采用精确类型及已知侧别匹配。触碰与糖使用同一 MaleCNS 注释；冷与空白使用同一 FlyWire 注释。所有直接刺激细胞均按接口规则排除。

| 功能组（作者指定类型） | MaleCNS 细胞数 L/R | FlyWire 细胞数 L/R | 触碰 spikes L/R | 冷 spikes L/R | 糖 spikes L/R | 空白 spikes L/R |
|---|---:|---:|---:|---:|---:|---:|
| forward (DNg100, DNg97) | 2 / 2 | 2 / 2 | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 |
| backward (MDN) | 2 / 2 | 2 / 2 | 0 / 0 | 0 / 0 | 3 / 2 | 0 / 0 |
| turn_inside (DNa02) | 1 / 1 | 1 / 1 | 0 / 0 | 22 / 0 | 0 / 0 | 0 / 0 |
| turn_outside (DNg13) | 1 / 1 | 1 / 1 | 0 / 0 | 28 / 20 | 0 / 0 | 0 / 0 |
| context_walk (DNp09) | 1 / 1 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| antenna (DNg62, DNge078) | 2 / 2 | 2 / 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| rub (DNg11) | 3 / 3 | 3 / 3 | 0 / 0 | 0 / 0 | 3 / 1 | 0 / 0 |
| anterior (DNg12) | 21 / 21 | 21 / 21 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| feeding (MN9) | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 7 / 1 | 0 / 0 |
| song (pIP10) | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 0 |
| flight (DNg02) | 15 / 14 | 13 / 12 | 0 / 0 | 0 / 0 | 2 / 3 | 0 / 0 |
| landing (DNp07, DNp10) | 2 / 2 | 2 / 2 | 0 / 0 | 18 / 3 | 1 / 2 | 0 / 0 |
| escape (DNp01) | 1 / 1 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

FlyWire 缺少精确 `MN9` / `pIP10` 类型标签，本次未猜测别名补齐。`DNg12` / `DNg02` 仅采用原作者明确的下划线子类型家族规则。逐细胞 ID、每组细胞数与实际计数见 `functional-groups.csv` 及各 `new-drive.json`。

## 实际命令与零值原因

全部使用 `ground`；20 ms 半开脉冲 bin、40 ms 平滑、100 Hz 参考率，保持作者 `motor_commands` 公式和增益。以下范围只包括 `available_at_ms < 500` 的命令；第 500 ms 才完成的最后一个 bin 保留在文件中，不能提前作用于这次身体回放。

| command | 冷刺激 min / max | 糖刺激 min / max |
|---|---:|---:|
| `left` | 0.00000000 / 0.40310468 | -0.14081938 / 0.04918367 |
| `right` | 0.00000000 / 0.48356216 | -0.14081938 / 0.04918367 |
| `antenna` | 0.00000000 / 0.00000000 | 0.00000000 / 0.00000000 |
| `rub` | 0.00000000 / 0.00000000 | 0.00000000 / 0.13115645 |
| `anterior` | 0.00000000 / 0.00000000 | 0.00000000 / 0.00000000 |
| `feeding` | 0.00000000 / 0.00000000 | 0.00000000 / 0.25639747 |
| `song` | 0.00000000 / 0.00000000 | 0.00000000 / 0.09836734 |
| `escape` | 0.00000000 / 0.00000000 | 0.00000000 / 0.00000000 |
| `flight` | 0.00000000 / 0.00000000 | 0.00000000 / 0.00000000 |
| `landing` | 0.00000000 / 0.00000000 | 0.00000000 / 0.00000000 |

冷刺激的 `forward`、`backward`、`context_walk` 全为 0；`DNa02` 左侧 22 spike 本身不能创造基础步行驱动。原公式还存在 `DNg13` 的对侧加性项：`left = ... + 0.6 × turn_outside_R`、`right = ... + 0.6 × turn_outside_L`，本次 DNg13 左 28 / 右 20 spike 实际产生非零命令。因此这份冷记录不是‘只有 DNa02’的零命令情形。

糖记录命中 `forward`、`backward`、`rub`、`feeding`、`song`，对应命令非零。冷/糖的 `landing`，以及糖的 `flight` 虽有 spike，但原公式在 ground 环境将飞行和降落命令固定为 0；不会因此补出飞行动画。触碰和空白的 10 个命令全部为 0。

冷和糖首次非零命令都在 100 ms 可用。每次 5000 个 0.1 ms 积分步中，1000 步神经命令为零、4000 步非零。原版 Hybrid 零命令期间仍主动维持姿态及接触反馈，两次记录的零命令最大执行器力均为 40 个原模型单位；不能将所有早期位移或姿态保持算作连接组输出。完整 25 行 × 10 命令见各 `new-commands.csv`。

## 控制器、身体和时间对应

旧方案：MaleCNS 按 VNC 腿运动细胞群，FlyWire 按 DNa01 / DNa02 侧别共享到三条腿，驱动 CPG / 预计算步态；弱增益、无原版 50 ms 预热。新方案：未修改的 Bad Apple Fly `Body`，官方 FlyGym `HybridTurningController`，原作者口器、翅膀和梳理定性动作程序。它额外启用头部、口器、翅膀关节，保留原版姿态执行器、增益与接触反馈。

| 实际物理模型 | 旧 | 新 |
|---|---|---|
| FlyGym / MuJoCo | 2.1.0 / 3.9.0 | 2.1.0 / 3.9.0 |
| nq / nv / nu | 73 / 72 / 48 | 88 / 87 / 63 |
| joint 数 / mesh geom 数 | 66 / 69 | 81 / 69 |
| 网格 | simplified_max2000faces | simplified_max2000faces |
| 前端资产 | `neuromechfly-scene.json` | `neuromechfly-behavior-scene.json` |

新资产是实际模型的 69 个 geom，头部、口器和翼有各自正确的 body 归属；并未造出原模型不存在的独立 labellum。两次新物理轨迹共 102 帧均核对了资产 body/joint/geom 顺序，以及本地 geom 变换重建后的世界位置/旋转，最大误差小于 1e-10。实际资产路径、SHA256 和数值误差写入 `audit.json`。

新初始化严格保留作者流程：构造时 50 ms 预热状态被丢弃，再以该条原神经记录的 seed 调用 reset 并预热 50 ms，之后设神经回放 time=0；这两段不计入 500 ms 神经窗口。预热使用主动姿态控制，不是脑反应。触碰/空白因全窗无可用命令，未实例化身体或预热。旧新初始化、控制器、映射同时不同，因此本比较不是单变量实验，更不以位移变化作为生物校准。

作者 `backend/app.py` 每个 chunk 先完成脑仿真，再让身体按该 chunk 读出推进整段。适配器保留原公式和身体，但采用 completed-bin 因果时序：bin 末尾之前不可使用该 bin；100 ms 完成的命令只能影响从 100 ms 开始的积分步骤。两个实际新轨迹的全部 102 帧均独立核查命令来源 bin，无提前读取未来脉冲。没有为了得到更大动作把命令向前移动。

这些结果验证‘保存的神经脉冲 → 原作者功能群/公式 → 原作者身体程序 → 实际 MuJoCo 状态’可执行；没有验证生物动作正确率。身体局部接触反馈没有回送连接组；ground 不含飞行气动力；官方雌蝇几何没有与 MaleCNS 雄蝇个体配准。命令命名不证明果蝇具有该意图，程序性进食/求偶/梳理模板也不等于测得行为。

## 来源、文件与复现

Bad Apple Fly 仓库 `kevinlinxc/badapple-fly`，固定 commit `6726892e7ebd93e6c5ffab056ab8db09433c28a6`；FlyGym 本地官方树 commit `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`。原 `body.py` 与 `evidence.py` 均按固定 SHA256 加载。完整注释文件、来源 manifest 和旧适配器 SHA256 见 `protocol.json`，新适配器 SHA256 见 `new-summary.json`。

| 用例 | observation ID | 原始 NPZ SHA256 |
|---|---|---|
| 右前腿触碰 | `7972d97a4ddb4d06a404f9781d6aa09f` | `1f5cbc613e82b2c3bbd2afe17b27064b61fbbaa48ef51c0e24aa490eba7cf634` |
| 左侧冷刺激 | `301732e8b7334dd9b267ce880cee77b5` | `ed926f5251da43af54537188ab78428ef2deea83a9cc22c5368b875292867d47` |
| 左侧糖刺激 | `9c5281bc5fda4816b89af0812a26958e` | `407861ded8ae4ae9987dfe143892b863931e2c6e7cd3ef4d03a54e4519fa9b02` |
| 空白 | `26b566cee24d4c77b4945fdb91e4c739` | `cc5f82fc09152fa5a79b76675cfe7f391d67b12755e0c54b730edf814a0ab6bb` |

冷刺激原 HTTP response 还含另一套 MaleCNS 观测，本报告仅选指定 FlyWire observation；不拼接两个数据集。原 HTTP response 的路径/摘要也固定在协议中。

- `compare.py`：读取现存 NPZ/HTTP records，旧、新驱动及实际物理回放。
- `old-summary.json` / `new-summary.json`：完整运行摘要、命令范围、功能组、模型和时序检查。
- `audit.py` / `audit.json`：仅从保存结果复核来源、逐帧因果时序、网格配准并生成本文。
- 四个用例目录的 `old-drive.json` / `new-drive.json`、`old-result.json` / `new-result.json`：完整驱动与真实轨迹。
- `functional-groups.csv` / 各 `new-commands.csv`：便于复查的细胞群及命令值。

在仓库根目录使用现有 `.venv` 和 `.venv-body`：

```powershell
.\.venv\Scripts\python.exe reports/behavior-reuse-2026-09-19/compare.py --prepare --old
.\.venv\Scripts\python.exe reports/behavior-reuse-2026-09-19/compare.py --new
.\.venv\Scripts\python.exe reports/behavior-reuse-2026-09-19/audit.py
```

本轮比较读取器首次将 FlyWire 冗余 root_id 列经 reindex 转成浮点，被严格 ID 校验拒绝；已仅在比较脚本改用精确字符串索引。该失败未运行对应身体，也未改神经数据、生产模块或刺激条件。随后四份新侧完整通过。

先前独立官方 Hybrid 外部命令基线见 `../controller-reuse-2026-09-19/README.md`：各 1 秒前进、左右转和零命令姿态保持成功。它们明确是外部控制命令，不并入本报告四份神经记录，也不作为连接组自主控制证据。本轮模拟感觉输入已有两份产生非零原作者命令，未额外扩展外部条件。
