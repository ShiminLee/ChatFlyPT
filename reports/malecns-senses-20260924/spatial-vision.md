# 保留局部眼图信息的候选空间桥接

> **2026-09-24 后续更正：本页下面的 bbox 地图已作为历史基线归档，不是当前实现。** 当前使用作者公开的 MaleCNS 光轴、同 body ID 列交叉表、实际编译眼相机和官方 Retina 像素射线，接受2322个R1–R6（左713、右1609），拒绝1055个；不再使用bbox、左眼手动镜像或视场外边缘回退。150次真实MuJoCo亮斑渲染中40/40作者列光轴抽样照亮对应小眼。几何核查不等于生理校准，仍保留候选限定。完整来源、版本/光轴不确定性与复现脚本见 [`retinal-registration-followup-20260924/README.md`](../../research/retinal-registration-followup-20260924/README.md)。旧npz/json和代码保存于 `reports/unified-pathway-20260924/retinal-bbox-baseline/`。

2026-09-24，继同脑入口审计之后新增。实现文件 `chatflypt/spatial_vision.py`，构建产物位于 `data/derived/malecns/spatial-vision/receptor-map.npz` 和 `receptor-map.json`。**局部空间信息现在可以进入 MaleCNS；真实视野角和个体视网膜配准仍未完成。** 该桥接不把左右眼平均值倒灌未定位细胞。

## 实际做了什么

1. 在当前原始符号不变的 MaleCNS 图中，读取每个 R1–R6 感光细胞到同侧 L1/L2/L3 的实际出边。这些下游细胞已有官方 `assignedOlHex1/2`；对每一柱累加相应突触权重，最多的一柱占这些同侧目标权重至少 50%、至少 3 个突触且无并列时，记录为候选主柱。光感细胞侧别仍只用 `rootSide`；下游 L1/L2/L3 的 `somaSide` 仅用于验证其属于哪一个 optic lobe，不用于编造感觉入口左右。
2. 将主柱的官方 hex 坐标投影到二维：q=hex1 向右上，p=hex2 向左上；使用作者坐标示意中的原点 [18,19]。利用同侧全部已标注视柱的范围归一化，不把重建不完整的 R1–R6 覆盖区域拉伸成完整眼睛。
3. 从实际加载的官方 FlyGym `compound_eye.npz` 中读取每个小眼 ID 覆盖的像素，计算几何质心，按候选投影选最近质心。**没有按神经元编号或小眼 ID 排序硬配对。** 运行时只读取这一个小眼测得的亮度，不读取物品名、人的目标或标签。
4. `project_spatial_vision(vision, catalogue)` 给出每细胞实测局部亮度、使用的小眼、逐眼接受/拒绝情况和映射哈希。`encode_spatial_vision(projection, seed, window_ms)` 产生独立的同 MaleCNS 事件磁带；环境层排除重复的均值事件磁带，接收框仍可保留 `mc_bright` 作为模态摘要。

## 列方向、镜像与比例的依据和假设

官方 [列坐标文档](https://raw.githubusercontent.com/reiserlab/male-drosophila-visual-system-connectome-code/main/docs/coordinate-systems.md) 和 [列坐标图](https://raw.githubusercontent.com/reiserlab/male-drosophila-visual-system-connectome-code/main/docs/assets/column_coord.png) 明确给出 q/p 轴及原点；本地保存了所读的 `column_coord.png`。感受器→L1/L2/L3 的实际图和列标签来自 [MaleCNS 官方数据](https://male-cns.janelia.org/download/)。官方视觉论文说明列标签是由重建和空间分析得到，也说明 R1–R6 不完整。[Nern 等，Nature 2025](https://www.nature.com/articles/s41586-025-08746-0)

本机实际 FlyGym 来自 `vendor/flygym/src/flygym`。官方 `vision.yaml` 的左右相机方向分别为 `[1.57,0,-0.47]` 和 `[-1.57,3.14,0.47]`，Retina 的像素行向下。代码将 hex 上方转为画面上方，并以右相机为横轴约定、对左相机作双侧反射。**解剖前后方向与相机横轴的对应没有独立验证；这个横轴基准、镜像和覆盖范围缩放是工程假设。** 参数完整保存在映射 JSON，`registration_calibrated=false`，不能将此说成官方提供了 MaleCNS 与 NeuroMechFly 的精确配准。[FlyGym 作者代码](https://github.com/NeLy-EPFL/flygym)，[NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y)

进一步真实眼图匹配可使用 [Zhao 等，Nature 2025](https://www.nature.com/articles/s41586-025-09276-5) 和 [eyemap_T4 作者代码](https://github.com/reiserlab/eyemap_T4) 的 micro-CT、视柱及 lens 匹配。已核实其 `proc_eyemap.R` 是基于实际眼结构/手动参考点进行视网膜与 medulla 对齐，不是可以无条件复制的 ID 对照表。当前工程桥接没有声称运行过这套眼图或使用了它拟合的光轴。

## 覆盖与未覆盖

| 眼 | 原注释 R1–R6 | 候选定位 | 拒绝 | 对应视柱 | 实际取样小眼 |
|---|---:|---:|---:|---:|---:|
| 左 | 1112 | 1106 | 6 | 300 | 237 / 721 |
| 右 | 2265 | 2223 | 42 | 522 | 401 / 721 |

拒绝 48 个细胞，包含原图中无建模出边的 18 个以及没有合格列关联的细胞。每个已定位细胞的目标 ID、各柱突触支持、置信占比、小眼索引和几何距离均存档。多个细胞同属一个视柱、多个柱映射到同一个粗采样小眼是显式的降采样结果；没有添加缺失感光细胞或为未采样空间编造输入。

该不均匀覆盖会影响两眼响应，不能为了看起来对称而复制另一侧细胞。亮度仍用既有 10 Hz 事件近似，未加入虚构背景兴奋，也未改变 histamine 抑制符号；graded 受体/lamina 基线等生理问题仍然存在。

## 验证

`tests/test_spatial_vision.py` 六项通过：官方来源/同侧列证据、hex 轴方向、局部输入可区分、完全黑暗不使用伪造平均值兜底、拒绝细胞保持拒绝、错误形状/非有限值拒绝以及种子可重放。

验收将两张**全眼平均亮度相同**的合成眼图分别点亮两个不同的小眼。对应的驱动感光细胞集合非空且互不相交，实际事件也只投递到相应细胞；右眼保持无输入。完整输入 ID、事件计数、映射哈希与每眼覆盖见 `spatial-acceptance.json`。这个实验说明程序已经保留局部空间差异，**不证明解剖配准正确、物体身份可识别或果蝇会朝该方向走**。真实环境和统一闭环验收另由具身主通路执行。

复建：`.venv\Scripts\python.exe -X utf8 -m chatflypt.spatial_vision`。构建只使用已有官方本地注释、图、Retina 资产；不修改原数据、模型符号或 vendor 文件。

真实闭环后续修正：官方 Retina 的像素平均会出现 `1.000000000000004` 这样的浮点累加误差。输入校验现允许 `1e-10` 容差，仅通过边界校验后截取至 `[0,1]`，不修改原始测量、不改变映射或实际光强。新增针对这一真实边界的回归 1 项通过，同时确认 `1+1e-8`、`-1e-8` 和无穷值仍拒绝。此前六项空间验证记录保留。
