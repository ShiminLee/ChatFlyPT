# 唇瓣几何与感器配准后续核查

2026-09-24，只读核查。核心、vendor、物理模型、感觉编码和训练输入均未修改；没有运行神经或身体仿真。证据快照与只读Blender解析脚本位于 `research/labellum-registration-20260924/`。

## 结论

**当前默认 NeuroMechFly 没有可直接使用的独立 labellum 网格、关节或味毛位置表；原始 Blender 的空“Labellum”顶点组也不能提供区域掩膜。** 因此不能依据已检查的默认模型，把整块 rostrum/haustellum 碰撞重新当作已定位的唇部味觉，也不能自行划一个远端百分比当真值。

但不能说官方生态没有更细口器模型：**当前 FlyGym 同时集成的实验性 FlyBody 后端，已有作者提供的左右独立末端口器网格、关节和接触椭球**，名为 `l_labrum/r_labrum`。它提供了可追溯的几何起点；尚没有证明这些网格内每个接触点对应 labellar 外侧味毛、内侧味peg或特定GRN。它也不是当前 NeuroMechFly 网格的现成标注，可复用模型并不等于可以直接复制它的局部坐标到另一个身体。

## 版本与查找范围

在线核对的官方默认分支头与本地锁定版本一致，没有找到比本地更晚、可直接启用的默认NMF唇瓣更新：

| 官方资源 | 核查版本 | 结果 |
|---|---|---|
| NeLy-EPFL/flygym | `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`，2026-06-28 | 本地相同；最新发布 `v2.1.0`，2026-06-24 |
| NeLy-EPFL/flygym-gymnasium | `d285260a1c8a7b3494150cd1590f2c9fe4b5e06b`，2026-04-01 | 旧API同样仅Rostrum/Haustellum口器网格 |
| NeLy-EPFL/NeuroMechFly | `6fdc6212bcf5b71fcd47bca119876c92092c2cd6`，2024-10-11 | 已归档；另检查原始Blender模型 |
| TuragaLab/flybody | `d015e9bfe441bd90ae431bac24c55cb74bdbce26`，2025-07-30 | FlyGym替代模型的原始作者来源，含左右labrum几何 |

已保存四个未截断的GitHub递归目录树、发布元数据。对7个关键本地FlyGym文件按Git blob算法逐项验证，全部匹配官方头版本；结果见 `local-upstream-verification.json`。完整官方高分辨率NMF资产前缀 `neuromechfly_fullsize_meshes_20260623a` 的S3清单也已检查：40项、未截断，口器仅 `c_rostrum.stl` 与 `c_haustellum.stl`。没有为了检查而替换本地高分辨率网格。[官方发布](https://github.com/NeLy-EPFL/flygym/releases/tag/v2.1.0)，[锁定FlyGym源码](https://github.com/NeLy-EPFL/flygym/tree/38c8ec61034cd59bc5ba0de20688d4a3c0000d60)

## 当前默认 NeuroMechFly：能定位身体段，不能定位唇瓣感器面

`src/flygym/anatomy.py:220` 的口器链只有 `rostrum`、`haustellum`；父子关系为 `c_head → c_rostrum → c_haustellum`。官方关节部位坐标写在 `assets/model/neuromechfly/rigging.yaml`：

| 身体段 | 父段局部位置，mm | 四元数wxyz | 可以据此确定什么 |
|---|---|---|---|
| c_rostrum | `[0.43, 0, -0.274]` | `[1,0,0,0]` | 相对head的段原点/关节几何 |
| c_haustellum | `[-0.371, 0, -0.0196]` | `[1,0,0,0]` | 相对rostrum的段原点/关节几何 |

这些坐标必须经父链和实际姿态变换才是世界坐标，不能当作味毛位置。`base_fly.py` 的解剖joint-site API创建的是关节位姿站点，不是已测得的味觉感器；允许用户创建site也不提供生物学坐标依据。[官方解剖链](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/anatomy.py#L220)，[官方rigging](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/assets/model/neuromechfly/rigging.yaml#L9)

默认简化STL：`c_haustellum.stl` 为1998面，`c_rostrum.stl` 为2000面；官方高分辨率STL分别为4311面与3786面。提高分辨率可以改善形状，STL本身仍没有唇瓣/感器标签。简化元数据文件只记录网格简化误差，不能解释某一面属于哪种感器。

## 原始作者 Blender：发现“Labellum”名，但它不是可用的区域配准

检查了原始仓库的 `data/design/blender/neuromechfly_full_model.blend`，不是仅看文件名或Git LFS指针。文件为32,612,588字节，SHA-256与作者LFS指针一致：

`935780666ab35ca1d8b66e33b928d9f1d10caaf83eff5ed19993b8ebc0291111`

文件头为`BLENDER-v291`。只读解析SDNA、对象、网格、骨骼及顶点组，未运行文件内脚本、未改姿态或模型。结果保存为 `original-blender-datablocks.json`，可由 `inspect_blend.py`复算。

- 68个对象、65个网格、65个骨骼；与口器有关的对象和骨骼只有`Rostrum`与`Haustellum`，没有独立`Labellum`对象/骨骼。
- `Haustellum`对象有2174个顶点，确实存在名称为`Labellum`的顶点组；但**没有任何顶点权重记录**，因此不能选出唇瓣区域。
- 读取器的阳性对照：`Rostrum`对象的`Proboscis`组有1792条顶点权重记录，与其1792个顶点对应。因此不是因为所有顶点组都被读取为空。

空组名提供了作者建模命名线索，但不能据此给当前haustellum任意面赋味觉。这里没有做肉眼形态猜测、没有从形状自动切远端区域，也没有声称已穷尽所有作者私有标注。[原始作者Blender文件](https://github.com/NeLy-EPFL/NeuroMechFly/blob/6fdc6212bcf5b71fcd47bca119876c92092c2cd6/data/design/blender/neuromechfly_full_model.blend)

## FlyBody：可复用的左右口器几何起点

FlyGym的 `compose/fly/flybody.py` 明确链接作者模型 [TuragaLab/flybody](https://github.com/TuragaLab/flybody/tree/d015e9bfe441bd90ae431bac24c55cb74bdbce26)，并标明此后端为实验性。其解剖链在haustellum之后增加 `l_labrum`、`r_labrum`；rigging保留原始名 `labrum_left`、`labrum_right`。[官方FlyGym适配器](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/compose/fly/flybody.py)，[官方FlyBody解剖链](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/flybody/anatomy_flybody.py#L214)

原作者文件 `flybody/fruitfly/assets/fruitfly.xml` 明确给出：

| 项目 | 左侧 | 右侧 |
|---|---|---|
| mesh | `labrum_left_lower.obj` | `labrum_right_lower.obj` |
| 相对haustellum的body位置，原始cm | `[-0.00263,0.0182,0]` | `[0.00263,0.0182,0]` |
| 同位置转为FlyGym mm | `[-0.0263,0.182,0]` | `[0.0263,0.182,0]` |
| body四元数wxyz | `[0.741,-0.0142,-0.591,0.319]` | `[0.699,0.00274,0.642,-0.316]` |
| 独立关节轴／范围 | `[1,0,0]`／`[-0.00524,1.05]` rad | 同左 |
| 接触几何 | `labrum_left_lower_collision`，ellipsoid | `labrum_right_lower_collision`，ellipsoid |
| 接触椭球半轴，原始cm | `[0.0035,0.00875,0.0131]` | 同左 |
| 接触椭球相对本体位置，原始cm | `[-0.000603,0.0106,-0.00438]` | `[0.00116,0.0106,-0.00414]` |

各自网格变换、碰撞椭球四元数以及本地FlyGym完整rigging副本已保存于 `flybody-mouth-inventory.json`。原始XML用cm，FlyGym作者转换为mm；不能漏掉10倍转换或忽略各body四元数。源码另提供口器黏附执行器，但黏附参数不构成味觉或进食量模型。[原始XML](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/fruitfly/assets/fruitfly.xml)，[FlyGym口器rigging](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/assets/model/flybody/rigging.yaml#L71)

作者论文称此处为`labrum (mouth)`，说明其口器接触用于喂食/求偶建模。名称与真实解剖labellum并不能直接互换；所查XML中这两个body没有味觉感器sites，所查源码没有味毛/味peg到面片和GRN的注册。作者的模型形态来自共聚焦图像；原始模型与数据可用于更细核实，并不等于这个问题没有可利用的资料。[Vaxenburg等，Nature 2025](https://www.nature.com/articles/s41586-025-09029-4)

## 是否足以实施下一步

| 目标 | 当前证据判断 |
|---|---|
| 在现有默认NMF上直接恢复准确labellar味觉 | **不足。** 原点/整段网格/空组都不是感器表面注册。 |
| 给现有haustellum规定任意远端范围并叫“真实唇瓣” | **不成立。** 无作者坐标或区域掩膜支撑该阈值。 |
| 构造作者定义的左右末端口器、测量其真实接触 | **有具体实现材料。** 可以后续评估复用FlyBody口器；它需要处理不同身体坐标、网格、关节和控制兼容，不属于本次冻结输入。 |
| 将上述接触送入标记明确的整个口器感觉候选 | **可研究的后续候选，不是已校准结果。** 先核实labrum/labellum的作者命名与对应表面，再保留候选级别并区分外侧味毛、内侧味peg。 |
| 单个触毛／味毛到精确MaleCNS细胞及浓度响应 | **本次所查模型未提供。** 需受体解剖数据和表面注册，不应让群体代理掩盖这一缺口。 |

下一步最实质的证据是作者的口器表面注释或固定版本的解剖图像—模型配准：若可以确认FlyBody左右末端网格覆盖labellar感器面，便可用作者几何与实际接触建立明确标记的候选入口；若继续使用默认NMF，则需要作者模型之间的几何注册或明确的面片掩膜。未经这一步，不重新启用粗口器糖/盐代理。本报告没有发消息给作者、没有替用户联络第三方。
