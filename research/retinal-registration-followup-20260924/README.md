# MaleCNS 光轴配准复核与离线候选，2026-09-24

**结论：找到并解析了作者公开的 MaleCNS 双侧列号—光轴对应表，已经能纠正当前 bbox 候选的方向和尺度；不是只有数据下载。** 离线构建将当前 R1–R6 的实际同侧 L1/L2/L3 主柱证据，接到作者光轴和实际 FlyGym 眼相机。2322 个细胞通过，原候选中其余1007个因列/视场证据不足拒绝。运行了150个真正的 MuJoCo 亮斑渲染对照，其中40个分层抽样的作者列光轴全部照亮候选对应小眼。

**后续集成状态：** 父代理确认旧bbox基线84窗与fit完成后，本轮获得明确许可并已将候选接入正式 `spatial_vision.py`，新增 `retinal_registration.py`。当前正式地图2322接受/1055拒绝，SHA256 `71fbfa7bb4f45272f641044473b988f913a676349b9b9c00fb45aa08cb56e756`。旧地图及代码完整保存到 `reports/unified-pathway-20260924/retinal-bbox-baseline/`；离线比较脚本现在读取这份固定历史快照。正式构建从原始图重新计算主柱，不读取旧bbox结果。

这些结果验证的是几何实现和可审计的作者图谱配准。没有运行神经网络，没有用行为目标决定映射，不能称为生理响应、对象识别或完整视觉动力学校准。

## 1. 固定来源与新发现

| 来源 | 固定版本 | 作用 |
|---|---|---|
| [Zhao 等原始 `eyemap_T4`](https://github.com/reiserlab/eyemap_T4/tree/99d2a43123db636cedb55af9ff31a59657e7d17e) | `99d2a43123db636cedb55af9ff31a59657e7d17e` | FAFB Mi1、lens/cone、原始 microCT 光轴和坐标定义；20个源文件 |
| [Arthur Zhao 的 `eyemap-archive`](https://github.com/artxz/eyemap-archive/tree/503c7f055d5491a48b60b49ade8c71798d24d8f1) | `503c7f055d5491a48b60b49ade8c71798d24d8f1` | **关键新来源：MaleCNS 左右 `pqxyztp_*.xlsx`，实际还带 hex1/hex2；**34个文件 |
| [Reiser lab `visualpathways`](https://github.com/reiserlab/visualpathways/tree/23f6ac131529b5f56894c6eeb9b88b17894fc00d) | `23f6ac131529b5f56894c6eeb9b88b17894fc00d` | 双侧 `ME*_columnar-cells_location.xlsx` 中 body ID—列号见证；67个代码/结构化数据文件 |
| [MaleCNS v1.0 官方补充列类型表](https://github.com/flyconnectome/2025malecns/blob/67767d2233657983993ff6c2be48e836a935863c/supplemental_data/optic-column-type-assignments-v1.0.xlsx) | `67767d2233657983993ff6c2be48e836a935863c` | 独立复核当前列号/L1 body ID；不替代光轴表 |
| [Berg 等 Cell 2026](https://doi.org/10.1016/j.cell.2026.08.015) | 正式论文 | Methods 已说明将 Zhao 眼图扩展到 MaleCNS medulla column ROIs |

逐文件下载URL、字节数、SHA256分别在 `source-manifest.json`、`eyemap-archive-manifest.json`、`visualpathways-manifest.json`、`column-v1-manifest.json`。未执行下载的 R/Python 分析代码、未加载远程 pickle。RData 通过 `rdata` 数据解析器读取。

作者 archive README 明确生成管线 `R_eyemap-DIY` 为私有，公开的是预计算结果。因此可使用并审计对应表，**不能声称重跑过作者完整配准优化、知道其每列不确定度，或把作者已配准的独立样本当作 MaleCNS 同一只动物的实测眼光学。** `20240701` 是源样本标签；归档首次提交于2026-08-20，hex字段于2026-08-28加入，不应将样本标签当成文件发布日。

作者数据许可为 CC BY-SA 4.0（`eyemap-archive/LICENSE`），派生的本目录光轴表/候选地图按同许可保留署名和修改说明。作者网页代码为MIT；`eyemap_T4` 原始分析代码为GPL-3.0。参考 [Zhao 等 Nature 2025](https://www.nature.com/articles/s41586-025-09276-5) 和 [Nern 等 Nature 2025](https://doi.org/10.1038/s41586-025-08746-0)。

## 2. 真正读取了什么字段

`rdata-schema.json` 记录全部已解析的变量名、维度、行列标签；导出 CSV 保留数字原精度。

| 原始文件/变量 | 实际维度与含义 | 能否直接 join 当前 MaleCNS |
|---|---|---|
| `eyemap.RData:eyemap` | 778×2，FAFB Mi1 行索引、右眼局部 lens 索引 | **不能**；不是 MaleCNS bodyId |
| `med_ixy.RData:med_ixy` | 779×3：`ind,x,y`，FAFB 右 medulla 的局部网格 | **不能**直接当 `assignedOlHex1/2` |
| `eyemap.RData:ucl_rot_sm` | 778×3，对应匹配 lens 的单位光轴 | 有方位，但需要样本间列配准 |
| `eyemap.RData:med_xyz` | 778×3，FAFB Mi1 原生位置 | 位置未用于当前光轴投影；不在本报告替它臆定长度单位 |
| `eyemap.RData:lens_ixy` | 852×3，当前文件内部使用的右 lens 网格 | 右眼局部 lens ID |
| 单独 `lens_ixy.RData:lens_ixy` | **786**×3，与上行不是同一表 | 不能因为同名就拼接；本次不使用 |
| `microCT/20240701.RData` | 1709 lens、1709 cone、`i_match`、`ind_left_lens`、`ucl_rot_sm`等 | 独立眼样本：左857、右852 |
| `eyemap_aux/ucl_rot_aux` | 817行，其中39个辅助边界点 | 辅助外推点，不能当新增实测受体 |
| archive 常规 MaleCNS `pqxyztp_left/right.xlsx` | **850/846**行，`p,q,x,y,z,theta,phi,hex1,hex2` | 有明确的列坐标，但仍须核实版本 |
| archive `lens_med.rda` | `lens_med_right`898×2、`j7_right`7个种子索引 | 无左列或 body ID 列，不能用它假补左列证据 |

原始 FAFB 工作重建的是女蝇 FAFB 右视叶 Mi1，CT 是独立扫描；这些事实与新增 MaleCNS 预计算图谱应分开叙述。原论文和作者图谱提供配准的科学根据，不代表当前代码从一个图谱建立每细胞真实感受野的生理验证。

本次导出了 `fafb-right-778-lens-mi1-axes.csv`、`microct-1709-lens-axes.csv`、`eyemap-embedded-852-lens-grid.csv` 和两份 `author-atlas-*-full-precision.csv`。最后两份来自原始xlsx；网页CSV将光轴四舍五入至6位小数，构建不依赖该精度损失。

## 3. 坐标系：不是凭左右外观翻转

原作者 `proc_uCT.R:37–81,141–182` 从视网膜赤道/镜片点建立前向 `vf`、左向 `vleft = cross(vn,vf)`、背向 `vn`，将光轴表示为 **+x前、+y左、+z背**。archive的单位向量长度误差小于3×10⁻¹⁶，与原CT同侧光轴最近差：左中位0.158°、最大0.626°；右中位0.160°、最大0.747°。这支持坐标轴继承，**最近邻相似本身不证明列身份**。

实际字段满足 `p = hex2 - 19`、`q = hex1 - 18`。`theta = acos(z)`，单位度；xlsx中的 `phi = -atan2(y,x)`（折返到±180°）。作者网页另从xyz重新计算标准atan2并为Mollweide显示翻x。**本候选只使用xyz，不把显示phi的正负当相机左右，不再手工对左眼乘−1。** 检查数字在 `source-geometry-checks.json`。

`camera_scan.py --capture` 在独立身体实例原版初始化之后，直接读取 MuJoCo `cam_xmat/cam_xpos` 和 head `xmat/xpos`，计算 `head_rotation.T @ camera_rotation`。头部与世界坐标、左右相机不同旋转全部保存。初始化仅原控制器50ms预热，没有神经运行或注入；该快照不更改活跃会话。

MuJoCo 相机看向局部−z，图像右向+x、上向+y，[官方说明](https://mujoco.readthedocs.io/en/3.9.0/modeling.html#cameras)。按实际157° fovy、512×450分辨率和编译后双眼外参，将像素中心射线转至头坐标。官方 Retina 的鱼眼 `zoom=2.72`、`distortion=3.8`、`+1e-6`、整数截断都逐项复现，再对**实际属于每个小眼的有效像素**求平均光轴。实际 Retina 足迹半径中位2.78°，P95 3.87°；不能将721个理想六角中心当成均匀球面网格。

## 4. 同一 body ID 揭示了左侧列版本差异

双侧作者列中心表包含L1/L2/L3/Mi1 body ID。右侧3674个有效身份/类型/侧别/列号与本地注释完全一致。左侧2635项中1564项同号、1071项移位。它是跨版本列编号差异，不能以“都是hex1/hex2”直接join。

离线候选使用明确交叉表：每个**当前列**必须至少有两个不同列类型的同一 body ID 见证；所有见证必须指向同一个作者列，逆向也必须唯一；冲突或证据不足拒绝。这样左872列通过，其中352列坐标改变、5列冲突拒绝；右892列通过、无坐标变化。逐细胞见证在 `body-id-column-check-*.json`，逐列见证/拒绝在 `column-crosswalk-*.json`。候选每个细胞同时保留 `hex`（当前）与 `author_hex`（作者图谱）。

例：当前左列(36,28)的L1 26947、L2 29782、Mi1 146878在作者旧表同处(36,30)，因此使用作者(36,30)的光轴。不能继续拿作者(36,28)的光轴给它们。

独立官方v1.0补充表复核：右892个L1全一致；左868一致、4项异常，其中3个缺当前hex、1个从(4,16)到(5,16)不同。这些额外证据在 `v1-official-L1-check-*.json`，没有无声覆盖主注释。

**剩余图谱版本不确定性：** archive提供常规和`DRA`命名两套MaleCNS图，右眼846列数值完全相同；左图常规850行、DRA828行，共有816列的光轴差中位4.82°、最大9.07°，不是单纯“仅多一块背缘”。本候选明确使用常规完整眼图，不混版本；私有生成管线未给出两版本的详细选择规则。作者列中心表与光轴表之间基于公开同一MaleCNS hex定义，缺失的私有生成版本/注册残差仍须保留为不确定性。因此跨body-ID校对提高可信度，但不是个体眼球重新标定。

## 5. 候选覆盖、拒绝与旧图比较

不更改神经元身份、真实图出边、主柱最低3突触/占比≥0.5/平票拒绝规则。先沿当前主柱交叉到作者列；确认作者光轴落入真实相机和官方compound-eye mask，才选择角距离最近的小眼平均光轴。**超出实际视场的列不吸附到边缘，也不使用眼均值或旧bbox回退。**

| 项目 | 左眼 | 右眼 |
|---|---:|---:|
| 作者常规图列 | 850 | 846 |
| 作者列在当前Retina可见范围 | 742 | 725 |
| 原先接受的R1–R6细胞 | 1106 | 2223 |
| 新候选接受细胞 | **713** | **1609** |
| 对应不同官方小眼 | 171/721 | 333/721 |
| 新拒绝（原候选内） | 393 | 614 |
| 同批通过细胞新角误差中位/P95 | 1.637° / 2.658° | 1.734° / 2.819° |
| 同批通过细胞旧bbox角误差中位/P95 | 44.510° / 111.932° | 51.821° / 111.724° |

左拒绝175出复眼mask、157出矩形相机、45缺作者列、15在眼相机背面、1没有唯一body-ID交叉；右拒绝分别210、228、97、79、0。旧图此前已拒绝的48个细胞仍保持拒绝，不能并入新地图。整合后总拒绝1055（48+1007），接受2322。

上述“角误差”是作者光轴与实际小眼采样平均方向的夹角。它衡量几何重采样，不是生物观测残差，且旧/新只比较共同接受的相同细胞。没有按误差最小挑选作者地图版本。

## 6. 真实渲染验收

`camera_scan.py` 复用实际编译眼相机的世界位置/旋转，创建无其他遮挡物的控制光学场景，在已知头/世界方向放置真实白色发光球，半径0.6mm、距被测眼20mm（角半径1.72°）。MuJoCo实际渲染经过官方 `Retina.correct_fisheye` 和 `raw_image_to_hex_pxls`，没有直接写“该小眼应亮”到测量数组。

- 110个预先定义的方向网格：每眼方位−150°…150°、步长30°，仰角−60°…60°、步长30°。
- 40个按列号均匀抽样的作者真实光轴（每眼20）：**40/40映射小眼确实收到亮斑**。
- 总150次渲染，其中82可见；可见峰值对应方向误差中位1.84°、P95 2.86°、最大3.59°。
- 水平+90°（左侧）只左眼可见，−90°只右眼可见；前方0°双眼可见，水平±150°当前眼相机均不可见。这与模型的实际外参一致，未用未知区域平均补光。

逐例真实世界坐标、测量强度、峰值索引、亮起小眼列表保存在 `renderer-scan.json`；水平前/左/右的原始和复眼PNG在 `scan-*.png`。这是**光学投影隔离测试**：使用实际相机及官方Retina，但不包含动物/地面遮挡。真实任务中的遮挡仍由正式环境渲染决定。本次也不声称阴影、光谱、偏振、适应、graded电位或方向选择性已经补齐。

## 7. 可复现步骤和集成边界

从仓库根目录运行，输出全部限定在本目录：

```powershell
.venv-body/Scripts/python.exe research/retinal-registration-followup-20260924/camera_scan.py --capture
.venv/Scripts/python.exe research/retinal-registration-followup-20260924/export_sources.py
.venv/Scripts/python.exe research/retinal-registration-followup-20260924/registration.py
.venv-body/Scripts/python.exe research/retinal-registration-followup-20260924/camera_scan.py
```

`candidate-receptor-map.npz/.json` 继续保存离线候选，正式地图由 `.venv/Scripts/python.exe -m chatflypt.spatial_vision` 独立重建。`registration.py` 读取历史快照仅为了沿用原有图出边主柱证据和做同细胞对照。运行时`project/encode`仍只取实测(2,721,2)数据，用当前body ID、eye index、ommatidium index映射，不使用物品身份。保留 `candidate_spatial_projection` 与 `registration_calibrated=False`。

正式加载已校验作者数据、交叉表来源、当前官方眼资产和眼相机代码等24个源文件的哈希，源文件stat参与缓存键，缓存不会掩盖修改。构建还检查固定作者manifest的commit和数据哈希；相机快照保存实际几何源哈希，资产变化后必须重新capture，不能仅重建地图把旧快照重新盖章。逐ID大表只存独立provenance，每10ms报告中的`transforms`为1669字节紧凑摘要。源路径经 `spatial_source_paths()` 提供给统一读出contract；新分布仍须独立读出验收，不能直接继承旧模型成绩。

## 8. 同时发现并修复的真实身体位置缺陷

实际初始化复现：FlyGym的静态body融合删除了 `fly/l_eye`，原 `_internal_bodyids_by_fly` 保留−1，原 `get_body_positions` 因而把右后腿末节 `rh_tarsus5` 位置报告成左眼。右眼body ID虽有效，解剖body的原点位于头部内部，也不是光学传感器位置。触角arista/funiculus和六腿的ID/位置核查正常。

新增 `body_measurement.py` 只修读取，不改MuJoCo或vendor：有有效body ID者逐值读取；被融合的静态body按原MjSpec局部位姿与存活父级的实际位姿恢复；缺少运动body或无法恢复则报错，绝不负索引取最后一行。正常初始化/持续反馈/独立物理观察共用安全访问器；旧独立身体回放入口也接入同一访问器，模型重编译后继续读取新model。视觉暗区采样改为实际 `cam_xpos`，输出 `vision.eye_positions_mm` 与来源字段。

中性seed1、原50ms预热并刷新运动学后，两眼相机分别为左 `[0.783100777957532,0.37997980677321014,1.1712471110161704]`、右 `[0.7831051771176779,-0.3800197963731954,1.1720219450071947]` mm。测试在左眼位置放0.1mm暗区，左眼测量归零而右眼逐值不变；修改环境后两眼/body帧保留；其余68个原有效body帧与原读取逐值一致。

100ms、1000个物理步的零命令/escape命令对照中，修复前后每步全部qpos/qvel逐位相同。修的是位置报告和环境光采样，不是让身体产生期望动作。证据在 `reports/unified-pathway-20260924/body-measurement-repair.json`，回归在 `tests/test_body_measurement.py`。身体/环境源变更后已经重新capture相机、重建正式map并重跑亮斑扫描；原列匹配数和40/40光轴接收不变。
