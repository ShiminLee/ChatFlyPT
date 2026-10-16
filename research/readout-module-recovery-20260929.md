# 读出器模块的删除与恢复记录（2026-09-29）

## 起因

清理契约禁止的“感觉读出 → LLM 建议 → 回灌神经”通路残留时，计划把 `chatflypt/touch_readout.py` 与 `chatflypt/social_readout.py` 一并删除，但附带条件写的是“无任何调用方和测试引用时才删除”。当时的检索使用 `findstr` 多模式匹配，漏报了 `tests/test_malecns.py:11` 对 `touch_readout` 的导入与 `scripts/reproduce-pathway.ps1:60-62` 的两个 `train` 阶段，因此删除时条件实际并不成立。删除后又确认这批文件不在任何模型指纹内，但工作目录不是 git 仓库，删除不经回收站，源码无法按常规方式回滚。

## 恢复方式

- `chatflypt/touch_readout.py`：从本会话子代理的转录（ZCode 会话数据库 `part.data` 字段）中提取到逐行带编号的完整文件读取记录，按编号重建，共 263 行。
- `chatflypt/social_readout.py`：转录中没有该文件的完整读取记录（命中的最长文本是旧 `server.py`）。该模块只保留 `chatflypt/__pycache__/social_readout.cpython-312.pyc`（2026-09-19 编译，与源码最后修改时间一致），因此按字节码结构重建：模块常量、函数名、局部变量名、常量池逐项对照，逻辑用恢复出的 `touch_readout.py` 作同源模板。

## 保真度校验

对原始 `.pyc` 与重建源码逐项比较代码对象（函数集合、`co_names`、`co_consts`、`co_varnames`、`argcount`）：

- `touch_readout`：16 个函数全部一一对应，模块级 53 个名字与全部字符串常量完全一致；`protocol()` 输出与字节码版本逐字相同；用既有 `touch-decoder.npz` 对 8 组随机计数向量解码结果完全一致；`readout_mask` 与字节码版本逐元素相同（148,763 个允许细胞）。
- `social_readout`：7 个函数全部对应，`SocialDecoder` 加载的 indices/weights/thresholds 与既有产物完全一致，8 组随机计数向量的解码结果一致。残留差异只有局部变量命名与整数字面量写法等不改变语义的部分。

决定性校验：用录制的计数矩阵 `data/derived/social-input/social-counts.npz` 重跑 `train()` 的全部仿真后步骤（掩码、特征选择、Ridge 拟合、阈值、序列化），得到的模型 SHA256 为 `20b4c26d742ce653aacc7ec87d61dec9597a9ea0ca075f4ce2c2a12e55d75c57`，与 `reports/full-pathway/social-readout.json` 记录值及磁盘上的既有产物**完全相同**；特征数 8248 也与记录一致。即 `social_readout` 的整条拟合管线可复现发布时的模型。

`tests/test_malecns.py` 的 4 项测试通过，覆盖恢复后的 `touch_readout` 接口与盲读出隔离。

## 结论与边界

两个模块已恢复为可读、可编辑的源码并放回原位；`__pycache__` 中的 `.pyc` 保留为字节码参照。上述校验证明恢复件在既有产物与记录上行为等价，但不等于逐字节还原原始源码文本：`social_readout.py` 的局部命名与表达式写法是重建而非复制，`touch_readout.py` 是从转录重建（与字节码结构一致）。复算与校验脚本保存在 `data/local/`（`extract_sources.py`、`verify_reconstruction.py`、`verify_social_refit.py`、`check_recovery*.py`）。

## 教训（用于后续清理）

多模式 `findstr` 检索不可靠（本次同时漏报了 `test_malecns.py` 的导入与 `tests/test_llm_response_failures.py` 的 `ADVISER_SYSTEM` 导入，后者由后续 `Select-String` 检索补上）。删除源码前应改用 `Select-String` 或等价工具全仓复核，并保留可回滚副本；本项目无版本控制，删除即不可逆。
