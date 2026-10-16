# 神经记录驱动的实体动力学适配器

已在本机 Windows x64、Python 3.12.14 上运行官方 FlyGym 2.1.0 / MuJoCo 3.9.0。不是周期函数生成的动画：身体位置、朝向、关节角、足端、接触数量来自 MuJoCo 积分后的状态。完整代码在 `chatflypt/body_simulation.py`，普通应用环境只需 NumPy；物理依赖位于独立 `.venv-body`，已加入 `.gitignore`。

这补上了**实际神经活动 → 工程控制器 → 物理身体**这一段，但没有完成神经元到肌肉的生理校准，也没有把身体感觉反馈回连接组。不能据此声称预测真实果蝇动作，或准确复现未公开的原作者实现。

## 已运行的证据

输入直接读取已有神经记录 `data/derived/pathway-trials/902c68d5bf3e4fe9a01ef36623886b87.npz`。没有读取用户语句、刺激名称、目标方向、读出标签或 API 设置。记录的 SHA-256、注释 SHA-256、分组计数和检查结果见 [evidence.json](evidence.json)。

| 验证项目 | 实测结果 |
|---|---|
| MaleCNS 腿运动群 | 381 个已注释细胞，42 个活跃，53 个脉冲 |
| LF / LM / LH / RF / RM / RH 脉冲数 | 4 / 8 / 9 / 5 / 6 / 21 |
| 神经记录与物理时间 | 都为 500 ms，没有时间拉伸；51 帧 |
| MuJoCo 模型 | nq=73、nv=72、nu=48、dt=0.1 ms |
| 完整物理状态 | 有限，数值/物理警告 0；最多 11 个接触 |
| 没有神经驱动的执行器 | 最大主动执行器力严格为 0 |
| 全部执行器关闭的被动物理对照 | 最大主动执行器力严格为 0 |
| 神经试次与被动物理对照最大关节差 | 0.8495954893 rad |
| 两者最终胸部位置差 | [0.140181, -0.032731, -0.037304] mm |
| 全零神经试次 | `no_motor_activity`、空轨迹、不启动物理进程 |
| 运行时间 | 神经回放含图片约 5.02 秒；被动物理约 8.30 秒，随机器负载变化 |

神经试次胸部总位移为 [0.330342, -0.033124, -1.575430] mm，被动物理对照也有明显下沉和位移。因此本次结果只能称为**神经活动调制下的物理姿态/关节响应**，不能把全部位移归因于神经驱动，也不能称为已经验证“步行”或“转向”。

本地生成文件：

- `data/derived/body-simulation/measured-drive.json`：六腿驱动、完整来源细胞 ID、时间和脉冲计数。
- `data/derived/body-simulation/neural/body-result.json`：实际 MuJoCo 每帧状态。
- `data/derived/body-simulation/neural/body-final.png`：检查过的物理模型渲染图，能完整看到模型及地面。
- `data/derived/body-simulation/neural/body.gif`：实际物理帧的回放。精确物理时间以 JSON 为准，GIF 播放器可能限制帧率。
- `data/derived/body-simulation/passive/body-result.json`：同初始状态、同种子、同持续时间，全部执行器关闭的对照。

## 信号如何进入身体

1. 检查 `Trial.counts` 与逐脉冲记录严格一致；缺少时间记录时拒绝构造动作。
2. MaleCNS 只选择 `superclass=vnc_motor` 且 `subclass=fl/ml/hl` 的腿运动神经元，以 `somaSide=L/R` 分组。直接被输入刺激的细胞剔除。翼、头、腹运动细胞、感觉细胞不用于腿控制。未知侧别不猜测。
3. FlyWire 只选择已注释的 DNa01/DNa02。因其缺少 VNC，活动按侧别复制到同侧三腿，明确标为工程连接；没有把其他脑运动群当作腿运动群。
4. 每 10 ms 分箱，脉冲数除以该群**全部已注释细胞数**和实际箱宽，得到平均事件率。以固定 20 Hz 工程参考归一化、截断到 0–1，用 30 ms 指数滤波。只有已经结束的箱可用，最多延迟 10 ms，不访问未来脉冲。
5. 官方 FlyGym 12 Hz tripod CPG 和官方记录的关节模板产生控制目标。每腿测得的幅值同时调制该腿步幅以及全部位置/黏附执行器的增益。初始步幅为零，零输入的腿主动执行器力为零，没有隐藏的预热行走。
6. MuJoCo 处理惯性、重力、被动关节、地面碰撞与接触。返回积分后的实际状态。物理状态不是由目标方向、语义标签或预设轨迹生成。

`somaSide` 与实际支配肌肉侧别之间尚未逐细胞核验。CPG、关节模板、增益、速率参考和群体合并都是未校准的工程选择。身体源自雌蝇，MaleCNS 源自雄蝇；尚未配准。当前不包含飞行、喂食、求偶、每个肌肉的活动模型，也没有从自然行为数据验证动作预测。

## 集成接口

```python
from chatflypt.body_simulation import build_motor_drive, run_body

drive = build_motor_drive(
    trial, brain.ids, brain.rows,  # MaleCNS；FlyWire 则传 NeuralView.rows
    dataset="MaleCNS v1.0",
    driven_indices=stimulated_cell_indices,
)
body = run_body(drive, max_duration_ms=500, sample_ms=10, timeout=90)
```

`build_motor_drive` 只接真实试次和注释，不接自然语言。注释行必须与 ID 对齐。主环境无需导入 FlyGym，也不修改主环境依赖。166700 个注释细胞的驱动构建在本机测得约 0.77 秒；只拷贝所需注释列，避免复制大量解剖坐标。

`run_body` 默认截取**从神经时间 0 开始的前 500 ms**，超过部分明确返回 `truncated=true`。不把短神经记录复制成更长动作。默认超时 90 秒，终止超时子进程并返回空轨迹。`render=False` 不需要窗口或可视化服务，Windows 子进程使用隐藏窗口标志。

返回状态：

- `ok`：真实物理运行完成。
- `no_motor_activity`：当前时间窗内无可用驱动，`physics_run=false`、`trajectory=[]`。
- `unavailable`：隔离物理环境尚未安装。
- `error` / `timeout`：空轨迹并附 `reason`；不能将失败当作静止预测。
- `passive_control`：仅在显式 `passive_control=True` 的验证对照使用，所有主动执行器关闭，不应作为神经响应显示。

`body.trajectory` 每帧：

| 字段 | 内容 |
|---|---|
| `time_ms` | 实际物理时间，毫秒 |
| `position_mm` | 胸部实际位置，3 元数组，毫米 |
| `quaternion_wxyz` / `yaw_rad` | 胸部实际姿态 / 水平朝向 |
| `feet_mm` | 六个 tarsus5 足端位置，6×3 毫米；顺序 LF、LM、LH、RF、RM、RH |
| `joint_angles_rad` | 实际关节角，按顶层 `joint_names` 排序 |
| `body_positions_mm` / `body_quaternions_wxyz` | 身体全部片段实际位置/旋转，按 `body_names` 排序 |
| `contact_count` | 实际接触数 |
| `drive` | 该积分步实际使用的六腿工程幅值 |
| `max_actuator_force_model_units` | 当前最大主动执行器力，采用 FlyGym 模型单位 |

顶层另含 `engine`、`controls`、`summary`、`units`、`limitations`、来源脉冲 SHA-256、运动细胞计数、源时间/实际时间、截断状态以及 `biologically_calibrated=false`。

## 重现

源码使用已保存的官方归档，commit 为 `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`。归档 SHA-256 为 `3bac51eb25b8c6715172eb898665ceff5a91aa78b088715d7077443faa84380a`，236 个归档文件已逐一与本地 `vendor/flygym` 比对，全部一致。版本、文件哈希和平台见 [environment.json](environment.json)，全部 47 项安装记录见 [requirements-lock.txt](requirements-lock.txt)。无需 GPU、RL 依赖或旧版 dm_control。

在仓库根目录执行 PowerShell：

```powershell
.\.venv\Scripts\python.exe -m venv .venv-body
.\.venv-body\Scripts\python.exe -m pip install -r reports/body-simulation/requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest tests/test_body_simulation.py -q
.\.venv\Scripts\python.exe reports/body-simulation/reproduce.py
```

默认重现脚本只读取上述已有神经记录，也可传入另一个同格式的 MaleCNS NPZ 路径。它不请求聊天模型，不读取密钥，不运行任何保留验收语句。

官方依据：[FlyGym 源码](https://github.com/NeLy-EPFL/flygym/tree/38c8ec61034cd59bc5ba0de20688d4a3c0000d60)、[官方安装说明](https://neuromechfly.org/installation/)、[MuJoCo 3.9.0 Windows wheel](https://pypi.org/project/mujoco/3.9.0/)。具体身体与 CPG 使用的是上述已核对的本地官方代码。
