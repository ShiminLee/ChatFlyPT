# SETUP：从克隆到能对话的页面

（2026-10-04 验证。每条命令都在开发机上实际执行过；全新的空机器端到端走查
标注见文末。前置要求：Windows 10+、Python 3.12（`py -3.12` 可用）、
curl（系统自带）、约 6 GiB 磁盘、联网。）

## 快速路径（五条命令）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
.venv\Scripts\python.exe scripts\acquire_sources.py
.venv\Scripts\python.exe scripts\acquire_malecns.py --include-connections
.venv\Scripts\python.exe -m chatflypt.malecns build
powershell -ExecutionPolicy Bypass -File scripts\setup-body-env.ps1
```

然后启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-chatflypt.ps1
```

打开 <http://127.0.0.1:8765>，发"右侧有酵母"，应得到感觉读出（参见
`docs/screenshots/ui-overview.png` 的样子）。

## 每一步做什么、怎么核对

### 1. 主环境 `bootstrap.ps1`

创建 `.venv`（Python 3.12）并按 `research/requirements-baseline.lock.txt` 安装依赖。
**校验**：脚本自带的 `pip check` 通过即成功。

### 2. 第三方源码 `acquire_sources.py`

下载 9 个钉版仓库的 tarball 到 `research/archives/`（gitignored），解压到 `vendor/`。
commit 与归档 sha256 记录进 `research/manifests/sources.lock.json`。
**校验**：最后一行输出 `Ready badapple-fly: 8,146,051 archive bytes`；共 9 行 Ready。
注意：badapple-fly 上游无根许可文件，仓库**不分发**其代码，只按钉版 commit
（`6726892e…`）取不可变归档；`behavior_body.py`/`behavior_readout.py` 载入时
会校验 `body.py`/`evidence.py` 的文件 sha256，取错版本会显式报错。

### 3. MaleCNS 数据 `acquire_malecns.py --include-connections`

下载三张官方表到 `data/raw/`（注释、递质、连接表约 1.05 GiB），MD5 钉版。
**校验**：`data/raw/malecns-v1.0/` 下三个 feather 文件存在，且各有一个
`.gcs.json` 侧车记录来源与哈希。`--include-connections` 必须加——第 4 步构建
神经图需要完整连接表。

### 4. 神经图 `python -m chatflypt.malecns build`

从 raw 表构建 `data/derived/malecns/graph.npz`（166,700 神经元、约 2,450 万条
带符号建模边）与 `graph.json` provenance。
**校验**：`graph.json` 里 `graph_sha256` 存在；构建过程打印
`MaleCNS graph {n}/{总批数} batches` 后正常结束。
若 `data/derived/malecns/spatial-vision/` 的配准图校验失败（报
"Spatial registration source changed"），运行
`.venv\Scripts\python.exe -m chatflypt.spatial_vision` 重建，期望
`map_sha256 = 71fbfa7b…`（本机两次重建均位级相同）。

### 5. 身体环境 `setup-body-env.ps1`

创建 `.venv-body`（Python 3.12），按 `research/requirements-body.lock.txt`
（47 个钉版，含 MuJoCo 3.9.0）安装依赖，再把 `vendor/flygym` 以 `--no-deps`
可编辑方式装入，最后复制 WMI shim 到 site-packages（防御主机 WMI 服务降级
导致的 worker 崩溃，见负面登记表）。
**校验**：脚本最后一行输出 `flygym … | mujoco 3.9.0`。

## 四份解码器权重已随仓库携带

`data/derived/*/malecns-*.npz`（v9 契约、与当前代码逐位匹配）在白名单里，
clone 即有，**无需重训**。若权重缺失或契约不匹配，读出会如实显示
"尚未校准"而不是拿旧权重充数。

**例外**：`data/derived/state-sensory/` 下两份"实验设置"直刺激面板用的历史解码器
（FlyWire 16 词 62.1 MiB、MaleCNS 触觉 49.9 MiB）体积过大未入库，挂在
[Release decoder-weights](https://github.com/ShiminLee/ChatFlyPT/releases/tag/decoder-weights)。
需要跑直刺激实验时下载后放入 `data/derived/state-sensory/`；只用正常聊天则不需要它们。

## 已知限制（照实说）

- 全新空机器的端到端走查**未执行**：每条命令都在开发机上验证过（包括
  acquire_sources 对新增 badapple-fly 条目的真实下载），但没有在擦净的
  机器上从头跑完一遍。第一个跑通的人请把踩到的坑补回本文件。
- 空间配准图若在你的机器上重建出不同的 sha256（理论上不该），不要强行
  覆盖——那说明上游几何输入变了，先查 `research/retinal-registration-followup-20260924/`。
- 视觉/嗅觉/触觉的入口映射是候选且未标定；页面上的读出是候选标签，
  会弃权，不构成生物学声明。
