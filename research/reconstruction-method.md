# ChatFlyPT 独立复现方法与验收边界

本项目按 @chetaslua 的公开演示描述独立实现，尚未取得作者的完整源码、16 词词表、读出器权重、刺激映射和测试集。因此“相同功能的独立实验”和“作者实验的精确复现”分别记录，不能互换。

## 已固定的实验设计

- 神经后端：Shiu 论文 LIF 方程，保留静息 -52 mV、阈值 -45 mV、膜时间常数 20 ms、突触时间常数 5 ms、不应期 2.2 ms、延迟 1.8 ms、权重系数 0.275 mV、步长 0.1 ms。
- 解码用 FlyWire v783 的 138,639 个神经元、15,091,983 条带符号连接行。固定全图，未训练连接组或引入可塑性。
- MN9 检查独立采用论文 v630 和原论文明确给出的糖/苦输入 ID；每个条件 10 个随机种子、1 秒模拟、150 Hz 输入。两条件配对种子，统计配对差和 bootstrap 区间。
- 自建 Numba 事件传播实现与 Brian2 使用同一输入事件带，比较脉冲时间/神经元及终末电位。小型含抑制反馈回路和全图测试均保存报告。两套引擎不同的随机数流不要求产生同一随机样本。
- 每个样本从静息状态开始。尚未实现跨试次生物记忆或持续大脑状态。

## 16 个标签是什么

独立定义：糖、苦、水、低盐、醋、酵母、土臭素、信息素、温暖、寒冷、潮湿、干燥、明亮、触碰、声音、风。

它们不是作者未公开词表的恢复结果，也不是真实果蝇活动—语言语料。标签指代我们设计的刺激干预；训练答案来自仿真实验配置。

`reports/reconstruction/stimuli-summary.json` 包含每个刺激的完整字符串神经元 ID、频率、证据来源和代理假设。特别注意：

- 醋使用 DM1+VA2 的部分代理；DM1/VA2 参与醋气味吸引，但完整气味刺激涉及多个小球和浓度响应，不等价于我们的恒定 Poisson 激活。
- 酵母使用 DM2 通道作为工程代理，DM2 不特异于酵母。标签读出成功不能证明识别真实酵母。
- 视觉使用确定性抽样的 128 个 R1-6 感光细胞，用统一 LIF 模型近似；不包含真实视网膜转导、光子数或完整 FlyVis 电压动力学。
- 温度、湿度、声音也采用标注细胞群的恒定激活，没有摄氏温度、声波或真实湿度传感器校准。
- 2026-09-19 生物学复核发现旧 humid 通道误选 HRN_VP1l（cooling）。已改为 HRN_VP5（moist，16 个），依据 Marin 2020 和 Knecht 2017；重新采样、训练及评测。旧结果保留在 `reports/biological-audit/previous-v1/`，不能继续当作湿觉识别结果。
- 最新 v783 的苦味注释与论文 Ir94e 列表存在交集；不删去不方便的数据。混合条件中共享细胞的输入率显式相加，并记录在清单中。

## 读出与数据切分

协议在采样前保存到 `reports/reconstruction/protocol.json`。776 个独立试次，每次 300 ms。全部随机种子唯一；同一混合标签组合不跨 train/validation/test/ood；单标签测试允许相同刺激类别的新随机试次。五标签展示组合独立保留，未进入任何训练或阈值选择数据。

所有 `sensory`、`sensory_ascending` 和直接刺激的细胞都在训练前排除。另训练仅 `descending`/`motor` 神经元的读出器。输入是脉冲计数的平方根变换，之后为线性 ridge 回归（alpha=10），各标签阈值只用 validation 选择。

报告同时给出单标签 top-1 准确率、多标签 exact-set 准确率、micro/macro F1、逐标签 F1，以及未见组合/不同强度表现。“16 个词全部正确”与“任意组合都正确”并非同一指标。

负对照包含神经观测随机换序和全零神经输入。输入感官细胞活动任意改动不应改变解码结果，另有自动测试。尚未做保持度分布的随机重连大脑对照，因此不宣称真实连接结构优于随机网络。

## 展示例子的检查

1. MN9：报告真实的多试次均值，不为了匹配 78→3 Hz 调参。
2. 16 词：报告独立词表、独立测试集的准确率，分别与公开报道 100%/85% 对照。
3. 第一句话：20 个事先固定种子，输入五种工程代理，逐试次保存完整脉冲。读出器只接收神经计数，文本模板只接收解码标签。不能将输入配置直接放进句子。最后的提问是明确的固定模板。
4. 真实大模型：用户已选择 ChatGPT。通过本机 `.env` 或 Windows 用户环境变量配置 OpenAI API；不使用 Claude 代替，不编造离线模型回复。保存真实返回的模型 ID、响应 ID 和用量。
5. 听觉回传：左/右编码分别激活左/右听觉细胞，前进激活双侧，停止为静默。单独训练外部线性读出器，从下游活动恢复指令；每类 37 个独立测试试次。该码本是工程约定，不是证明果蝇理解英语。
6. 转向和觅食：建立明确公开的 7×7 玩具环境，通过下游读出器驱动位置/朝向。可重放真实大模型建议并用不同神经试次测试 37 次转向；不是 37 次新 API 请求，也不是原作者未公开环境。另列 DNa01/DNa02 活动左右差异，不将外部分类器的决策冒充原生运动反应。没有 NeuroMechFly 身体物理。

所有成功与失败进入 `reports/reconstruction/acceptance.json`；报告不会把未运行、未配置或未达到的项目写成通过。

## 来源

- 作者原帖：<https://x.com/chetaslua/status/2098387725972123825>（已通过官方 oEmbed 与公开页面取得原文；数值为作者自报）
- 作者后续线性读出实验：<https://x.com/chetaslua/status/2098819578994507891>（HTML 任务，不能反推为聊天实验的完整训练细节）
- 温湿觉通路：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7443704/>；<https://elifesciences.org/articles/26654>
- Shiu 模型及数据：<https://github.com/philshiu/Drosophila_brain_model>
- FlyWire 神经元标注：<https://github.com/flyconnectome/flywire_annotations>
- v783 糖输入列表：<https://github.com/eonsystemspbc/fly-brain>
- 醋气味回路：<https://www.nature.com/articles/nature07983>
- DM2 与酵母相关气味：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11766258/>
- Brian2 不应期：<https://brian2.readthedocs.io/en/2.5.1/user/refractoriness.html>
- OpenAI 文本接口：<https://developers.openai.com/api/docs/guides/text>

各上游固定提交和许可证仍见 `research/manifests/sources.lock.json`。本项目没有复制 @chetaslua 未公开代码，也不自称得到作者认可。
