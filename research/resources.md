# ChatFlyPT 资源调查与选择

核查日期：2026-09-19。标记“源码审查”不表示已独立复现论文或社区作者的全部结果。所有本地上游源码均保留原许可；锁定提交和归档校验值见 `manifests/sources.lock.json`。

## 数据集

| 资源 | 本次取得内容 | 用途与限制 |
|---|---|---|
| [MaleCNS v1.0 官方下载](https://male-cns.janelia.org/download/) | 神经元注释、神经递质预测、完整连接权重表；共 1,109,008,094 字节 | 雄性 CNS，含脑和腹神经索。CC-BY-4.0。无需账户即可下载这三张静态表；neuPrint 交互 API 则需要 token |
| [Shiu 论文仓库](https://github.com/philshiu/Drosophila_brain_model) 中 FlyWire v630 | 127,400 个模型节点，14,687,178 条连接表记录；CSV+Parquet | 原论文计算基线，模型和数据可一起运行 |
| 同仓库 FlyWire v783 | 138,639 个模型节点，15,091,983 条连接表记录 | 公开版本扩展；原 v630 刺激神经元 ID 不能原样照搬 |
| [FlyWire 原始发布归档](https://zenodo.org/records/10676866) | 已保存完整下载清单、大小、MD5 与 CC-BY-4.0 许可元数据 | 783.0 的权威静态归档；未再下载 9.5 GB 全突触表，当前模型输入已具备 |
| [FlyWire 系统注释](https://github.com/flyconnectome/flywire_annotations) | 固定提交的注释仓库 | 用于细胞类型/功能群筛选。注释版本与连接组 materialization 版本须分别记录；遵循仓库引用要求 |
| [FlyVis](https://github.com/TuragaLab/flyvis) | 官方视觉模型代码和 618,148 字节的连接组 JSON | 特定视觉回路，非全脑 SNN；预训练权重另行下载，本次未下载 |

### MaleCNS 实测过滤结果

`reports/data-audit.json` 是对下载文件实读的结果：

- 注释表有 **211,577** 行，不等于筛选后模型神经元数。
- 采用明确的 `superclass` 非空、`status != Glia` 过滤，保留 **166,700** 个节点。
- 原始连接表有 **151,856,684** 行，含未被上述节点集保留的端点。
- 两端均落在保留节点集时剩 **25,582,938** 条连接记录，其接触计数合计 **124,177,617**。
- 神经递质表关联键是 `body`，注释表键是 `bodyId`；预测与实测标注不可混称为确定的递质身份。

这些数字由文件和过滤规则决定，不可用“全脑多少突触”一个数字混写。MaleCNS `weight` 是解剖接触数，不是已经训练好的、有确定正负号的生理突触权重。FlyWire 论文输入的带符号权重则已经包含上游建模选择。

### 本地路径

- `data/raw/malecns-v1.0/`：三张官方 Feather 表。
- `vendor/shiu-brain/`：630/783 配套 CSV、Parquet 与原模型。
- `vendor/flywire-annotations/supplemental_files/`：细胞注释。
- `vendor/flyvis/flyvis/connectome/fib25-fib19_v2.2.json`：视觉连接组。
- `research/manifests/malecns.lock.json`：官方 GCS 对象 generation、大小、MD5、本地 SHA-256。

## 源码及采用意见

| 项目/本地目录 | 已核验内容 | 采用意见 |
|---|---|---|
| [Shiu / shiu-brain](https://github.com/philshiu/Drosophila_brain_model) | MIT；Brian2 LIF、刺激/静默实验、原始 notebook | **首选科学计算基线**；已做本地短时执行验证。固定权重，不自带学习或语言功能 |
| [Eon / eon-fly-brain](https://github.com/eonsystemspbc/fly-brain) | GPL-2.0；Brian2、PyTorch、其他加速后端；有 v783 刺激列表 | 后续 GPU 加速候选。声明的平台是 Linux/WSL；本次没有运行其 GPU 后端 |
| [FLM / flm](https://github.com/nftechie/flm) | 原创代码 MIT、上游资源另有条款；图状态到语言输出的适配器 | 用作结构对照参考。图状态不是动作电位；其 README 报告的对照未建立果蝇拓扑优势 |
| [lixiang1076 / fly-brain-chat](https://github.com/lixiang1076/fly-brain) | 声明 MIT；关键词到刺激、输出模板、两种不同的学习实现 | 接口参考。不是 @chetaslua 源码；部分数据是 LFS 指针；复用前核对继承代码许可 |
| [TuragaLab / flyvis](https://github.com/TuragaLab/flyvis) | MIT；官方视觉回路、训练和分析工具 | 后续视觉模块优先。Python `>=3.9,<3.13`；源码包依赖版本生成，安装需按官方方式处理 |
| [NeLy-EPFL / flygym](https://github.com/NeLy-EPFL/flygym) | Apache-2.0；2.1.0；Python `>=3.12,<3.15` | 新身体仿真候选；资源按需加载，取到源码不等于所有 mesh 已下载 |
| [NeLy-EPFL / flygym-gymnasium](https://github.com/NeLy-EPFL/flygym-gymnasium) | Apache-2.0；1.3.2；Python `>=3.10,<3.13` | 旧教程和感知示例的对应版本；与 2.x 分环境 |
| [Neuromorphicism/fly-brain-snntorch](https://github.com/Neuromorphicism/fly-brain-snntorch) | 已检查公开 README 与许可元数据；未下载/运行 | 可训练 SNN 备选，数值语义及延迟/不应期须与论文模型比较，不能仅凭速度声明替换基线 |

FlyGym 的版本迁移说明见 [官方 migration](https://neuromechfly.org/migration/)。FlyVis 的[预训练下载说明](https://turagalab.github.io/flyvis/install/)对应 `flyvis download-pretrained`；源码包含下载器及校验值，权重尚未在本机取得。

## 需要保留的空缺

1. @chetaslua 演示源码未找到，追查范围与原帖入口见 `chetaslua-investigation.md`。
2. 没有取得经生物实验验证的“神经活动—自由对话文本”配对数据；这需要另行设计，而不是从连接组文件直接读取。
3. 本次仅验证安装与短时运行，不包含语言模型、全论文统计复现、学习训练或实时身体闭环。
4. v630 的一个糖刺激 ID 和一个苦刺激 ID 不在 v783 模型名单中。严格检查会拒绝运行。Eon 提供 v783 的独立糖刺激列表，可显式选择；尚不能据此声称已完成完整跨版本细胞映射。

## 关键论文入口

- [Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9)：全脑 LIF 计算模型；是本项目基线的论文依据。
- [Dorkenwald et al., Nature 2024](https://www.nature.com/articles/s41586-024-07558-y)：成体雌蝇脑连接组。
- [Schlegel et al., Nature 2024](https://www.nature.com/articles/s41586-024-07686-5)：全脑细胞类型注释。
- [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3)：连接组约束视觉模型。
- [NeuroMechFly v2, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02497-y)：身体和感知仿真框架。
- [MaleCNS 官方首页及发布记录](https://male-cns.janelia.org/)：引用当前正式论文/版本时以官方页面为准，避免混用早期预印本与新发布数据。
