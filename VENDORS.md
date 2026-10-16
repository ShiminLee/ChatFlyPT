# 第三方依赖与数据（VENDORS）

本仓库**不 redistribut** 任何第三方代码或授权数据。所有依赖按 commit + sha256 钉版，
精确哈希记录在各模型快照的 `contract.json`、`*_provenance.json` 与 `research/snapshots/` 中。
重建环境请按下表获取，并用哈希核对。

| 依赖 | 上游 | 钉版记录位置 | 许可 | 本仓库用法 |
|---|---|---|---|---|
| MaleCNS v1.0 连接组与注释 | [male-cns.janelia.org](https://male-cns.janelia.org/download/) | `data/derived/malecns/graph.json`（graph_sha256）、`data/raw/*/​*.gcs.json` 侧车文件 | 见上游数据页的 data license | 唯一神经图；`data/raw/` 不入库 |
| NeuroMechFly 身体/资产 | [neuromechfly.org](https://neuromechfly.org/) | 空间配准 provenance（`receptor-map.json`） | 官方模型/资产 Apache-2.0 | MuJoCo 身体、复眼资产 |
| flygym | github.com/NeLy-EPFL/flygym | `research/snapshots/NeLy-EPFL__flygym*.json`（commit） | Apache-2.0 | 仿真与 Hybrid 控制器 |
| flygym-gymnasium | github.com/NeLy-EPFL/flygym-gymnasium | `research/snapshots/NeLy-EPFL__flygym-gymnasium__commit.json` | Apache-2.0 | 借用的羽流网格 `plume_short.hdf5`（sha256 40a76057…） |
| badapple-fly | github.com/kevinlinxc/badapple-fly @ `6726892e` | `chatflypt/behavior_body.py` 与 `behavior_readout.py` 头部常量（body/evidence sha256） | **未找到根许可文件**——因此本仓库不 redistribut 其代码，只按哈希引用 | 钉版 Body 与运动公式 |
| flyvis | github.com/TuragaLab/flyvis | `research/snapshots/TuragaLab__flyvis__commit.json` | MIT | 读出头形状的出处（仅思路借用） |
| FLM | github.com/nftechie/flm | `research/snapshots/nftechie__flm__commit.json` | MIT | 独立实验（已归档，不在聊天通路） |
| eyemap 档案 | 视网膜配准上游 | `research/retinal-registration-followup-20260924/`（manifest） | 见 manifest | 光轴配准输入；`research/archives/` 不入库 |

## 为什么 vendor/ 不入库

1. **许可**：badapple-fly 无根许可文件，redistribut 有法律风险；其余虽为 Apache/MIT，
   但钉版哈希已足以精确重建，复制反而增加漂移面。
2. **体积**：vendor/ 3.87 GiB（含模型缓存），远超仓库需要。
3. **可核对**：本仓库的契约哈希体系本来就以 sha256 钉版——缺的文件按上表获取后
   会自动通过校验，拿错则显式报错。
