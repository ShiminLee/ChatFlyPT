# 打包上 GitHub：步骤与待决定项

（2026-10-04。目标：把本仓库作为**可运行的代码 + 完整的验收记录**发布，
不含授权数据、第三方代码与大体积运行记录。）

**克隆者怎么跑起来**：见 [SETUP.md](SETUP.md)——五条命令从克隆到能对话的页面，
每条都有校验点。发布前请先通读一遍确认与你决定的许可/可见性一致。

## 已就绪

- `.gitignore` 已补齐：排除 venv、vendor、授权数据、大体积报告、模型快照、LLM 权重、
  Nature 论文 PDF（版权）等；保留全部代码、测试、research 笔记、小型评估 JSON、
  以及四份已发布解码器权重（`data/derived/**/malecns-*.npz`，均 < 数 MB）。
- `VENDORS.md` 已写：每个第三方依赖的来源、钉版哈希位置、许可状态与获取方式。

## 待你决定（4 项）

1. **许可**：你自己的代码用什么许可？MIT / Apache-2.0 / 暂不放许可（默认保留所有权利）。
   不写许可的公共仓库别人不能合法复用，只能看。
2. **可见性**：public 还是 private？
3. **仓库名**：建议 `chatflypt` 或 `fly-cns-conversation`。
4. **解码器权重**：data/derived 默认被忽略。四个 `.npz`（合计 < 10 MiB）可以
   (a) 解除忽略直接入库，或 (b) 挂在 GitHub Release assets。二选一。

## 你需要在本机做的一次性动作

1. 安装 [Git for Windows](https://git-scm.com/download/win)（当前机器没有 git）。
2. `gh auth login`（GitHub CLI 已装在
   `C:\Users\33927\AppData\Local\Programs\GitHub CLI\`，只是未登录）。

## 决定之后我来执行的步骤

1. `git init` → `git add -A` → 检查 `git status`：确认无 >100 MiB 文件（GitHub 硬限制）、
   无 `data/raw`、无 Nature PDF、无 vendor 代码。
2. 按你的决定写入 `LICENSE`。
3. 首次 commit；`gh repo create <名字> --<public|private> --source . --push`。
4. 推送前最后核对一遍 `.gitignore` 生效情况
   （`git ls-files | measure`，并抽查 `git ls-files | findstr /i "pdf gguf feather"`）。

## 体积预估

| 内容 | 约计 |
|---|---|
| chatflypt/（代码+网页资产） | 11 MiB |
| scripts/ + tests/ | 3 MiB |
| research/（不含 archives/pdf/wheels） | ~40 MiB |
| reports/（仅 README、评估 JSON、验收记录） | ~30–60 MiB |
| 四份解码器权重 | < 10 MiB |
| **合计** | **约 0.1 GiB 以内** |

## 发布形态的实话

它不是论文，也不会有很多 star。它是这类项目里少见的**诚实样本**：
预注册指标、负面登记表、契约哈希、可复现语料——正好是
[业内批评文](https://www.neuroai.science/are-flies-playing-beat-saber)
点名缺失的东西。受众是少数认真的人，但对他们它是有用的。
