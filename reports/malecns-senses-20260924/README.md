# MaleCNS 同一连接组的多感觉入口

日期：2026-09-24。本项完成了新的同脑候选人口选择、执行侧别/出边核查及神经层连通验证；**没有宣称完整自然转导、视网膜空间配准、真实行为或开放语言完成**。正常聊天的物理呈现与一体化运行由服务端和具身层单独验收，本报告的 5 条神经试验是明确标记的独立直接干预诊断，不冒充环境输入实验。

同日后续已经实现保留局部小眼信息的候选空间桥接，不再局限于执行全眼均值。它利用实际感光细胞出边推断主柱，并映射到官方 Retina 采样几何，仍然没有完成真实生理视野配准。详见 `spatial-vision.md`；下文 5 条神经诊断为新增空间桥接前的独立群体输入记录，不能冒充新空间通路验收。

## 接口与隔离

- `chatflypt/malecns_senses.py::CHANNEL_MAP` 将已有环境模态名解析到独立的 `mc_*` 通道，目标均为 `MaleCNS v1.0 / malecns_body_id`。不复制 FlyWire ID。
- `target_channel(catalogue, channel, dataset)` 返回当前连接组可执行目标；无映射时拒绝，不回退其他脑。
- `remap_environment_inputs(catalogue, items, dataset)` 返回新输入列表和明确排除项；不修改原值、不生成事件。
- `remap_environment_sample(catalogue, sample, dataset)` 深拷贝完整实测样本，更新输入和已有通道证据、保留传感数据，增加 `sensory_projections` 和 `neural_dataset`。它必须在物理采样之后调用。
- 新群均为 `advanced=True / status=candidate / readout_status=untrained_new_population`，`readout_group` 是自身名称。旧训练词表和签名不吸收它们；新的盲读出须另行训练并做独立环境验证。
- 现有 MaleCNS 口器味觉、腿部味觉和触觉输入保持原始 ID。湿度、cVA、CO₂ 等登记同脑入口不等于正常聊天已有对应物理源；缺少物理呈现时仍不得直接注入兜底。

## 本地实际清单

人口来自官方 `body-annotations-male-cns-v1.0-minconf-0.5.feather`，按精确 `superclass`、`class`、`type` 选取。侧别仅用 `rootSide`。执行还要求 ID 存在于实际 signed graph 且有至少一条建模出边。全部原始 ID、unknown 侧别及无出边 ID 仍在 `populations.json` 留档；没有用数组顺序或 somaSide 补左右。

| 入口 | 注释类型 | 全部细胞 | 可执行左/右 | 未定侧别 | 可执行建模出边 |
|---|---|---:|---:|---:|---:|
| mc_bright | R1-R6 | 3377 | 1112 / 2247 | 0 | 14785 |
| mc_vinegar | ORN_DM1, ORN_VA2 | 157 | 68 / 80 | 9 | 17252 |
| mc_yeast | ORN_DM2 | 54 | 24 / 29 | 1 | 4682 |
| mc_geosmin | ORN_DA2 | 48 | 21 / 20 | 7 | 3976 |
| mc_cva | ORN_DA1 | 204 | 51 / 105 | 48 | 19643 |
| mc_co2 | ORN_V | 55 | 16 / 39 | 0 | 3533 |
| mc_warm | TRN_VP2 | 7 | 4 / 3 | 0 | 787 |
| mc_cold | TRN_VP3a, TRN_VP3b | 7 | 3 / 4 | 0 | 594 |
| mc_moist | HRN_VP5 | 12 | 4 / 8 | 0 | 811 |
| mc_dry | HRN_VP4 | 28 | 13 / 14 | 1 | 2833 |
| mc_evaporative_cooling | HRN_VP1d | 18 | 10 / 8 | 0 | 2231 |

R1–R6 原来右侧 2265 个中有 18 个没有当前建模出边，因此不进入执行选择；没有删掉原始清单，更没有虚构缺失左侧细胞。其余群均有出边；unknown 细胞另行排除。

## 功能证据与边界

官方 MaleCNS 下载站提供注释、连接权重、预测递质、突触空间位置及骨架；这些支持身份和连接核实，不等于直接测得自然刺激转导。[MaleCNS 官方数据](https://male-cns.janelia.org/download/)

DM1/VA2 对低浓度苹果醋的吸引有生理/行为研究，但醋实际激活更多嗅球，且组合随浓度变化。沿用两群只作为部分代理；DM2 的酵母入口明确保留工程代理属性，不把 DM2 名称等同完整酵母气味。[Semmelhack 与 Wang，Nature 2009](https://pubmed.ncbi.nlm.nih.gov/19396157/)

VP2、VP3、VP4、VP5 的升温、冷却、干燥、潮湿归属由感觉投射及受体表达研究支持；VP1d 的蒸发冷却仍为候选。由跨样本的类型/投射对应选取 MaleCNS ID，不代表对本批 EM 细胞进行了受体表达测量；当前输入率不是摄氏度或相对湿度响应曲线。[Marin 等，Current Biology 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443704/)，正文通过 [Europe PMC 的原文 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7443704/fullTextXML) 复核。

下文历史诊断使用分眼群体输入；现在正常具身通路使用 `spatial-vision.md` 所述局部小眼候选桥接，保留官方 histamine 递质符号。均一 LIF 事件模型没有完整 graded 电位或经过标定的 lamina 自发基线，因而光输入可能只有抑制/阈下响应，不能为得到运动或句子而改成兴奋性。气味、温湿度的输入率继续是未经生理标定的工程值。

## 视网膜空间资源已核实

本地注释实际含 `assignedOlHex1`、`assignedOlHex2`，23,720 个细胞同时具有两字段，分布在 L1/L2/L3/L5/C2/C3/Mi1/Mi4/Mi9/T1/Tm1/Tm2/Tm4/Tm9/Tm20 共 15 种列细胞。**6,091 个视觉感觉细胞无一直接具有两字段**，不能把这些列坐标当作现成的 721 小眼 → 感光细胞对应表。完整检查在 `retinotopy-inventory.json`。

官方视觉论文介绍根据突触空间构建列中心线和 15 种细胞的列归属，同时说明体积边界导致 R1–R6 重建不全。官方代码的 hex 坐标文档说明网格定义与任意外部原点；这些提供了空间配准材料，尚不等于 NeuroMechFly 实际双眼相机朝向、视野和连接组眼网膜已经配准。[Nern 等，Nature 2025](https://www.nature.com/articles/s41586-025-08746-0)，[作者代码](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code)，[作者坐标说明](https://raw.githubusercontent.com/reiserlab/male-drosophila-visual-system-connectome-code/main/docs/coordinate-systems.md)

进一步的眼结构与视野映射资源为 [Eye structure shapes neuron function in Drosophila motion vision](https://www.nature.com/articles/s41586-025-09276-5) 及 [作者 eyemap_T4 代码](https://github.com/reiserlab/eyemap_T4)。本轮确认了该资源和分析脚本，没有安装 R 分析环境、运行该眼图或声称得到现成配准。

## 验证与实际神经结果

执行 `tests/test_malecns_senses.py`、`tests/test_sensory_functional_mapping.py`、`tests/test_sensory_model_loading.py`：11 项通过。涵盖精确官方类型/根侧别/出边、旧词表隔离、单 ID 命名空间编码、未知模态拒绝、证据/输入一致且不修改原始样本，以及既有味觉与读出加载约束。

`audit.py` 运行 5 条独立 200 ms 神经诊断，每条起始静息，100 ms 输入。均未改递质、未加背景驱动。记录和膜电位保存在 `neural-smoke-*.npz`，汇总 `neural-smoke.json`。

| 左侧诊断入口 | 输入事件 | 内部脉冲 | 有膜电位变化的内部细胞 |
|---|---:|---:|---:|
| 视觉 | 1176 | 0 | 1292 |
| DM2 气味代理 | 255 | 84441 | 126304 |
| VP2 升温 | 55 | 80369 | 127684 |
| VP3 冷却 | 36 | 83457 | 127488 |
| 视觉 + DM2 + VP2 + 前腿触觉 | 1691 | 84611 | 126963 |

所有膜电位统计有限，均在同一个 MaleCNS 图内传播，计数排除了所有感觉及直接输入细胞。该结果说明神经层已可接受同一连接组的多模态输入，**不证明强广泛响应符合活体、不证明模态整合正确，也不是新感觉标签读出成绩**。温度与气味引起的广泛持续活动需要生理动力学与行为验证，不能因有出边和大量脉冲而算“真实果蝇大脑已完成”。
