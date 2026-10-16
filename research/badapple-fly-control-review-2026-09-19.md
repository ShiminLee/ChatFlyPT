# Bad Apple Fly：作者、视图协议与神经—身体控制核查

核查日期：2026-09-19。研究部分为官方源码静态审查、来源与许可核查；随后经用户授权新增了独立动作读出适配器，具体变更与验证见文末。未运行该项目神经模型。

**结论：用户描述的 Fly Lab / Bad Apple / NeuroMechFly / Ground contact physics 标签与此仓库的界面源码直接吻合。身体和脑图可以独立接入现有观测记录。原项目的动作由指定细胞群的平滑放电率，经人工规定的连续控制公式，驱动 CPG 步态与姿态程序；不是一个学会理解任意神经活动含义的行为模型。**

## 固定版本与作者

- Bad Apple Fly 官方仓库：[kevinlinxc/badapple-fly](https://github.com/kevinlinxc/badapple-fly)。本次固定到 [6726892e7ebd93e6c5ffab056ab8db09433c28a6](https://github.com/kevinlinxc/badapple-fly/tree/6726892e7ebd93e6c5ffab056ab8db09433c28a6)，提交时间为 2026-09-13 07:41:26 UTC，GitHub commit API 的 author 为 Kevin Lin、账号为 kevinlinxc。当前本地 23 份文件及 SHA-256 清单在 `research/manifests/badapple-fly-source.json`。
- 作者[公开 GitHub 资料](https://github.com/kevinlinxc)自链接 kevinlinxc.com 及 X / YouTube 的 @linguinelabs。本次没有找到足以把他与 @chetaslua 认定为同一人的一手依据。这个仓库不能直接写成“@chetaslua 的原始代码”。
- 上游身体依赖在项目 [pyproject.toml](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/pyproject.toml) 固定为 FlyGym 2.1.0、MuJoCo 3.9.0。FlyGym v2.1.0 标签解析为 [ca65a510c2afe6ac61c51df4f274c8d190c2f95f](https://github.com/NeLy-EPFL/flygym/tree/ca65a510c2afe6ac61c51df4f274c8d190c2f95f)。本次从该固定提交读取的 hybrid_controller、turning_controller、cpg_controller、preprogrammed、neuromechfly 和 LICENSE，与本机已有对应文件逐字节一致。

## 界面匹配与一个重要显示差异

`frontend/index.html` 的标题、品牌和身体窗口分别包含 MaleCNS Fly Lab、Fly Lab、NeuroMechFly，并提供 ground / tethered_flight 两种环境。`main.js` 创建 Bad Apple 预设，在回放 ground 记录时把身体副标题设为 Ground contact physics。这是源代码层面的明确匹配；本次没有对用户截图做像素比对，也没有据此推定截图发布者身份。[HTML](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/index.html#L3-L8)、[身体窗口](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/index.html#L59-L67)、[回放标题](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/main.js#L219-L232)。

原脑图显示的是 soma 三维点云，不是神经元树突/轴突或逐条连接。它能把刺激与放电分别画出来，但视频回放默认切换为 `pattern` 模式；此模式直接隐藏 `fireGeo` 放电层，保留 `stimulated_mask` 图案。**因此视频在“脑里播放”的画面主要表示人为选择的输入位置，不能视作网络自行理解或生成该图案。**[点云加载](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/main.js#L148-L164)、[分层与模式切换](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/main.js#L183-L191)、[视频默认模式](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/main.js#L223-L226)。

## 神经活动如何成为动作

### 1. 输入和网络

原项目读取 MaleCNS v1.0 的注释、连接权重与递质表，保留 superclass 非空的注释细胞及其相互连接，没有再按边权设阈值。突触权重使用连接表计数；乙酰胆碱为正，GABA、谷氨酸、组胺为负，调质与未知递质的快速突触贡献为零。这个符号约定不包含逐受体建模。[数据来源](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/scripts/download_data.py#L8-L13)、[构图和符号](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/scripts/prepare_data.py#L15-L61)。

神经元为简化 LIF：0.1 ms 时间步，膜时间常数 20 ms、突触衰减常数 5 ms，静息/复位 −52 mV，放电门槛 −45 mV，2.2 ms 不应期，1.8 ms 化学突触延迟。默认突触增益 0.275。人工刺激用独立 Bernoulli 事件近似 Poisson 输入，每次事件直接增加 68.75 mV，输入率上限 1,000 Hz；silence 将指定细胞膜电位和突触状态钳制。参数属于计算实现，不能当作该 MaleCNS 标本的实测生理。[积分器](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/brain.py#L8-L42)、[初始状态及输入](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/brain.py#L61-L103)。

Bad Apple 输入将有限 soma 坐标的 X/Y 投影到视频像素；Y 翻转，保持比例，忽略 Z 深度。RGB 均值小于 96 的像素对应细胞被直接刺激，未限制为视觉感觉细胞。因此下降/输出细胞也可能处于直接刺激集合中，之后被动作读出器读取；这不是“电影经复眼进入脑”的实验。[完整投影实现](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/video.py#L19-L50)、[刺激注入](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/app.py#L184-L193)。

### 2. 细胞群读出、时间滤波与连续控制

每个计算窗口先统计各细胞 spikes，再将瞬时率 `spikes / 窗口秒数` 做时间常数 **40 ms** 的指数平滑。窗口基础间隔为 **20 ms**；刺激边界和视频帧边界进一步切分。读出器按精确 type 及 **somaSide** 取左右群体均值；DNg12、DNg02 额外包含 `类型名_` 前缀的子类型。不存在训练分类器或“哪个动作得分最高就执行哪个”的竞争选择。[滤波](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/brain.py#L73-L103)、[窗口划分](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/app.py#L196-L209)、[群体读出](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/evidence.py#L61-L86)。

以下是**作者实际编码的映射**，不是本次重新证实了所有细胞类型的生物学作用。作者为各项附有原始论文链接、功能解释及限制，全部列于 [evidence.py](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/evidence.py#L2-L59)。

| 控制变量 | 指定细胞类型 | 程序中的作用与边界 |
|---|---|---|
| forward | DNg100、DNg97 | 向前步态驱动；不是经拟合的 spikes→速度关系 |
| backward | MDN | 减去向前驱动，产生反向步态；使用反向遍历的步态轨迹 |
| turn_inside | DNa02 | 按胞体侧降低同侧步幅 |
| turn_outside | DNg13 | 按胞体侧增强对侧步幅 |
| context_walk | DNp09 | 只采用促行走解释，未实现威胁情境下的停止/冻结 |
| antenna | DNg62、DNge078 | 触角梳理姿态；合并 aDN1/aDN2 注释 |
| rub | DNg11 | 前足互搓姿态 |
| anterior | DNg12 家族 | 前部梳理，与 antenna/rub 共同混合；家族推断 |
| feeding | MN9 | 口器伸展；不模拟吞咽、摄食量或奖励 |
| song | pIP10 | 单侧翼展示；没有声音或实测鸣声波形 |
| flight | DNg02 家族 | 仅 tethered_flight 下调节翼振幅；没有自主起飞 |
| landing | DNp07、DNp10 | 仅 tethered_flight 下伸腿；没有降落轨迹 |
| escape | DNp01 | ground 下中足伸展；没有真实弹跳/起飞动力学或电突触 |

归一化率为 `n = clip(群体均值 / reference_hz, 0, 1.5)`，`reference_hz` 默认 100 Hz。令每组左右归一化率的平均为 `m`，则净步态驱动是 `d = max(m_forward, m_context_walk) − m_backward`。左右驱动分别为：

```text
left  = [d × (1 − 0.75 n_inside,L) + 0.60 n_outside,R] / (1 + g)
right = [d × (1 − 0.75 n_inside,R) + 0.60 n_outside,L] / (1 + g)
g     = max(m_antenna, m_rub, m_anterior)
```

这些增益、上限与混合方式是人工控制规则。没有生物学验证过的“达到 100 Hz 才选中动作”门槛，100 Hz 是缩放参考值。不同运动程序可同时混合。tethered_flight 会把左右步行输入直接置零。[完整控制公式](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/evidence.py#L88-L106)。

### 3. CPG、姿态和物理

步行调用 FlyGym `HybridTurningController`。左右驱动的**绝对值**分别成为每侧三条腿的 CPG 目标幅度，**正负号**决定该侧 CPG 相位推进方向。默认六振荡器采用 tripod 相位关系，基础频率 12 Hz；周期相位经样条查询预录的七自由度单腿轨迹，生成位置执行器目标。它不是直接用 MaleCNS 的完整 VNC 运动神经元群逐肌肉控制身体。[左右控制](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym_demo/complex_terrain/turning_controller.py#L14-L36)、[CPG 默认值](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym_demo/complex_terrain/cpg_controller.py#L166-L186)、[步态轨迹](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym_demo/complex_terrain/preprogrammed.py#L39-L143)。

其他动作在步态目标上混合手写关节姿态：梳理有 6 Hz 正弦项，求偶展示固定使用左翼并含 30 Hz 项，系留振翅含 80 Hz 项；口器伸展为关节偏移。梳理/着陆/逃逸姿态存在 0.01 的工程触发判断，梳理强度超过 0.2 时关闭前足黏附。这些角度、频率和阈值均不代表刺激特异的实测运动学。[姿态代码](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/body.py#L65-L108)。

身体使用雌性标本来源的 NeuroMechFly mesh，关节位置执行器增益 45、力范围 ±65，足黏附增益 40，胫节和跗节可与平地接触。地面模式的自由身体位移来自接触和执行器动力学，而非前端按“走路”标签平移图标。系留模式则每个物理步强制固定根位置与姿态，没有空气动力学。[身体配置](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/body.py#L21-L63)、[推进与系留](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/body.py#L100-L117)。

**有局部身体反馈，没有身体→MaleCNS 的神经感觉反馈。** 控制器每个物理步读取胸部高度、足端高度、胫节/跗节对地接触力与朝向，做悬腿回撤和绊脚校正。因此不能把原 README 的“no sensory feedback”扩大解释为控制器不读取物理状态；其准确边界是这些状态不回注神经网络。[身体观测与校正](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym_demo/complex_terrain/hybrid_controller.py#L29-L102)、[神经到身体调用链](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/app.py#L226-L238)。

每个窗口先完成脑积分，再用该窗口末的平滑率推进整段身体时间，属于分块耦合。它没有按每个 spike 的精确时刻即时控制关节，不能从动画读取真实毫秒级神经—运动延迟。

## 视图协议：可以接入现有 ChatFlyPT 记录的部分

### 身体几何与帧

| 数据 | 格式与含义 |
|---|---|
| `/api/scene` | `geoms` 数组、初始 `frame`、`joint_names`、`body_source`、`units`、物理能力标志 |
| `geoms[i]` | `name`、平坦 xyz `vertices`、平坦三角索引 `faces`、RGBA `color` |
| `frame.positions[i]` | 第 i 个 mesh 的世界位置，三分量，单位 mm |
| `frame.rotations[i]` | 第 i 个 mesh 的世界旋转矩阵，行优先的九个数 |
| `frame.thorax` | 根/胸部世界位置，镜头用其增量跟随 |
| `frame.displacement_mm` | 相对于 warmup 后原点的根位移 |
| `frame.joint_angles` | 关节角记录；渲染主体直接使用各 mesh 世界变换 |

后端从 MuJoCo 导出真实 mesh 顶点/面与变换。前端生成 `BufferGeometry`，逐帧更新 mesh 位置和由旋转矩阵得到的四元数。世界 Z 轴向上；左右身体坐标来自模型，不应和 somaSide 或图像左右混为一谈。前端按 mesh 名重新设身体/眼睛/翅膀颜色，并未直接使用返回的所有 RGBA。[几何输出](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/body.py#L119-L137)、[Three 身体视图](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/main.js#L81-L97)。

最小接入只需要一份几何和有序身体帧，不必同时复用原神经积分器、实验表单、动作解释、视频资源。若本项目暂时只有静态 mesh 或示意关节动画，应标出实际能力；Ground contact physics 只能用于后端确实提供 MuJoCo 接触结果的模式。

### 脑图与同步回放

- `/api/positions` 返回有限 soma 位置的 `ids` 与 `pos`。前端将坐标包围盒居中、统一缩放到显示空间，构造共用点云 BufferAttribute。
- 每帧有 `time_ms`、`body`、`readouts`、`commands`，以及 `active_mask` 与 `stimulated_mask`。两种 mask 必须分别对应计算放电和施加刺激，不能互相替代。
- mask 是 `numpy.packbits(..., bitorder='little')` 的 Base64。位序对应结果里的完整 `activity_ids`，不直接等于可视 soma 点云顺序；浏览器通过 ID 映射，忽略没有坐标的细胞。
- 记录每 100 帧压缩为 gzip JSON 块，manifest 提供时间范围与 URL。前端保留两个块，按目标时间选择不晚于它的最近记录帧；不是在浏览器重新计算物理，身体帧之间也未插值。
- 渲染循环限制约 30 fps，仿真结果先计算、再播放；屏幕播放流畅不代表后端能实时模拟整张图。

[帧与结果结构](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/app.py#L222-L255)、[压缩与位序](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/recording.py#L8-L35)、[mask 解码](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/activity.js)、[回放选择](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/recording.js)、[渲染循环](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/main.js#L196-L217)。

接入 ChatFlyPT 时，模型观测和文本回应仍只由实际神经活动读出；视图中的 `stimulated_mask`、输入侧别、原文和 reception 不应变成输出解码特征。FlyWire 的长 root ID 应以字符串对齐，不能直接沿用原 MaleCNS 短整数的 JavaScript Number 假设。阈下状态需要新增独立视觉层/图例，原项目并没有这种显示。

## 源码、数据与资产许可边界

| 部分 | 核实结果 | 具体来源/复用记录 |
|---|---|---|
| Bad Apple Fly 主项目的 Python、HTML、CSS、JS | 固定提交的完整树中未发现根 LICENSE、COPYING、NOTICE；pyproject 没有 license 字段。不能从仓库公开可读推定其主代码为 MIT/Apache | 保留 commit、文件 SHA-256、作者来源和这项缺失信息；Three 的许可不能外推到作者的布局/控制代码。[固定仓库树](https://github.com/kevinlinxc/badapple-fly/tree/6726892e7ebd93e6c5ffab056ab8db09433c28a6) |
| Three.js / OrbitControls | vendored 目录附 MIT，版权为 Three.js authors | 复用时连同版权与许可文本保留。[THREE-LICENSE.txt](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/frontend/vendor/THREE-LICENSE.txt) |
| FlyGym 控制器与默认 NeuroMechFly 身体资产 | FlyGym 2.1.0 仓库整体声明 Apache-2.0；默认简化 STL 与 rigging 随包提供，未见该目录独立覆盖条款 | 可直接从许可明确的 FlyGym 获取身体几何、配置和控制器，保留上游 LICENSE、来源和修改说明。[LICENSE](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/LICENSE)、[默认模型加载](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym/compose/fly/neuromechfly.py#L15-L105)、[简化网格](https://github.com/NeLy-EPFL/flygym/tree/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces) |
| MaleCNS v1.0 数据 | 官方下载页面链接 CC-BY-4.0 | 保留 MaleCNS 数据集、版本、论文与数据来源归属。[官方页面](https://male-cns.janelia.org/download/) |
| `assets/badapple-small.mp4` | 仓库含 7,424,067 字节视频，但所核查源码/README 未附单独视频版权许可或素材授权链 | 本地源码清单未下载此视频；复用身体与脑图不需要它。不能将其与 Three/FlyGym 的代码许可混同 |
| 神经分组颜色 JSON | 仓库的 `prepare_visual_groups.py` 从 MaleCNS 注释生成；显示归类是代码加工，细胞信息源自 MaleCNS | 本轮未下载原 JSON；可基于本项目已有真实注释生成自己的分组显示。[生成脚本](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/scripts/prepare_visual_groups.py) |

身体资产具体来源是 NeuroMechFly 上游，非 Bad Apple Fly 作者另行制作的果蝇。默认使用每段最多约 2,000 面的简化 mesh，并镜像左侧网格生成右侧；无需为了相同视图下载高分辨率外部 mesh。雌性身体与 MaleCNS 雄性神经数据的组合应在模型信息中说明。[上游默认参数](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym/compose/fly/neuromechfly.py#L48-L105)、[原项目身体来源字段](https://github.com/kevinlinxc/badapple-fly/blob/6726892e7ebd93e6c5ffab056ab8db09433c28a6/backend/body.py#L135-L137)。

## 研究交付与动作读出实现

已给主线程交付几何/帧/mask 协议、动作群体与侧别规则、40 ms 滤波及连续缩放公式、CPG 和局部反馈机制、Bad Apple 默认显示输入图案的区别，以及固定来源和许可清单。这部分是源码实现核查，不是生物学验证，也不是原项目运行质量测试。

用户授权采用这套动作方案后，新增 `chatflypt/behavior_readout.py`，接口为：

```python
build_behavior_drive(trial, neuron_ids, annotations, *, dataset,
                     driven_indices=(), context='ground', bin_ms=20.,
                     smoothing_ms=40., reference_rate_hz=100.)
```

适配器在加载时核对本地 `evidence.py` 的固定 SHA-256，直接调用原文件的 `MAPPINGS` 与 `motor_commands`；没有另写第二份动作公式，也没有训练或改变脑模型。默认时间窗为 20 ms，Hz 平滑常数为 40 ms，缩放参考为 100 Hz，保留原负方向值。以下差异明确记录在输出中：

- 神经输入来自当前系统的真实 spike 索引和 0.1 ms 时间记录，而不是重跑 Bad Apple Fly 积分器。
- 直接受刺激的细胞从输出群体的计数和分母中移除，保留被排除的字符串 ID、数量和 spikes。
- MaleCNS 采用精确 `type/somaSide`，FlyWire 采用精确 `cell_type/side`。除原作者 DNg12/DNg02 的下划线家族规则外，不猜名称别名；未知侧别和缺失类型显式记录。
- 每个半开区间 `[start, end)` 内的 spikes 只在 `end` 时刻成为可用命令，身体不得提前使用。最后一窗在整个试次结束时才可用，因此不计入该试次内的主动命令判定。不同于原项目用窗口末率回放同一窗口的分块方式。
- `chatflypt.behavior-drive.v1` 包含逐窗原始群体率、平滑读出率、spike 计数、10 个动作命令、13 组细胞来源、spike 摘要及注释映射摘要。原有整数数组记录的 spike SHA-256 算法保持兼容；字符串 ID 使用稳定 JSON 编码而非对象内存地址。
- `_validate_behavior_drive` 独立检查时间窗、计数/分母/滤波、正放电率、带符号命令、原公式一致性和零 spike 禁止非零命令。这是记录内部一致性验证；原始神经记录仍需留存才能外部核验。

`tests/test_behavior_readout.py` 的 34 项测试通过，覆盖全部 13 组和两个身体 context 的原公式精确一致、时间边界与部分末窗、平滑先于限幅、直接刺激排除、零信号、左右转向及后退、长 ID 与未知侧别、忽略文本/输入标签，以及畸形时间/率/命令拒绝。测试使用确定的合成 spike 记录，不调用脑模型或身体模型。本子任务仅新增动作读出模块、专用测试并更新此报告；server、body 和前端的集成由主线程及对应代理完成。
