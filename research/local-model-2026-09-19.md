# ChatFlyPT 本地语言模型调查与实测

> 归档说明（2026-09-28）：本次调查所评测的顾问协议及其代码已从项目移除（`dialogue.ADVISER_SYSTEM`、`build_adviser_batch`、`parse_adviser_response` 与相关脚本均已删除）。下方记录保留为当时的实测证据；当前应用不含模型建议或神经回传通路。

调查于 2026-09-19 启动，下载与实测在本机 2026-09-20 完成。数据与原始响应在 `reports/local-model-2026-09-19/`。仅新增独立运行器、权重和评测脚本；未安装到主 Python 环境，也没有读取或改写远程 API 密钥。

机器是 i9-12900HX（16 核、24 线程）、31.74 GiB 内存、RTX 4060 Laptop 8188 MiB 显存。常见安装路径及 PATH 未发现 Ollama、LM Studio 或 llama.cpp；Hugging Face 缓存只有 BERT/DistilBERT，不适合生成中文建议。完整硬件与调查范围见 `environment.json`，不是全磁盘私有文件扫描。

首先选了 [Qwen 官方 Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)：1.7B 参数、Q8_0、1,834,426,016 字节，Apache-2.0；支持非思考模式。这是轻量基线，不宣称是最新或最强的 Qwen。也查阅了更新的 [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)，但本轮优先使用可直接追溯的现成 GGUF。

运行器使用 [llama.cpp 官方 b10964](https://github.com/ggml-org/llama.cpp/releases/tag/b10964) Windows CUDA 12.4 包，MIT 许可；CUDA 依赖单独来自同一 release。`v0.4.1` 官方 latest release 的 `nightly-tag.txt` 指向该 build。三份首批文件共 2,479,937,294 字节，均按发布方 SHA256 校验。具体下载 URL、版本、哈希和本地路径在 `artifacts.json`；[官方 server 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)说明兼容 chat completions 接口。

使用 127.0.0.1:8081，4096 上下文、全层 GPU offload、8 CPU 线程、关闭 thinking，温度 0.7、top_p 0.8、top_k 20、min_p 0、presence penalty 1.5、seed 84217。请求完全复用生产 `LLM.complete`、`dialogue.ADVISER_SYSTEM`、`build_adviser_batch` 与 `parse_adviser_response`，没有评测专用 JSON grammar、提示后缀或额外模型可见信息。

24 项协议在首次调用前冻结：20 项合成盲解码输出边界案例，另有 4 项已保存神经仿真的真实盲读出。模型只收到生产白名单构造的感觉证据，未收到刺激输入、reception、身体轨迹、旧回复或期望动作。原始 HTTP payload/response、模型文本、延迟都保留。评测脚本首次误将 `LLM.complete` 的元数据字典当成字符串；已保留初始错误报告，并对同一批原响应离线重评，没有为此重发模型请求或挑选结果。

1.7B 首轮：24/24 请求成功；19/24 通过生产严格解析器；10/24 是不带 Markdown 的纯 JSON（解析器允许单个完整 JSON fence）。5 个禁止回传的案例仍输出动作，被解析器拒绝。中位延迟 0.499 秒，P95 0.573 秒，首次基准调用 0.926 秒，中位生成速度约 73 token/s。所有响应都有自然中文，但人工逐条审核只有 8/24 无明显依据或策略问题；5/24 是禁止动作，11/24 有无依据的方向选择、动作语义误解或忽略不确定性。没有观察到“已经完成动作”的捏造。人工标准及逐条理由在 `semantic-review.json`，不把格式成功称为可靠推理，也不把这套有限样本的工程动作建议视为生物学正确答案。

第二候选采用 [Qwen3-4B-Instruct-2507 官方原模型](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)对应的 [llama.cpp 组织 ggml-org 的 Q8_0 转换](https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF)。明确区分发布方：原模型是 Qwen，GGUF 转换由 ggml-org 发布，并非 Qwen 账户。官方模型卡说明是 4B 非思考指令模型，Apache-2.0。文件 4,280,403,520 字节，固定 revision 与哈希见 `artifacts-4b.json`。它已替换 1.7B 服务，使用完全相同的 24 项冻结请求完成评测。

4B 实测 24/24 请求成功且为纯 JSON，21/24 通过正式解析器，中位 0.848 秒、P95 1.157 秒；人工审核 9/24 无明显依据/策略问题，3/24 禁止动作被拒绝，12/24 有语义疑虑。保留 4B 为实验选项，理由是格式与无信号处理略好，不是已经达到可靠自主建议。两次都没有额外 JSON grammar，没有改变感知模型或训练数据。

主流程后续 3 项完整神经 HTTP 通路均形式通过，但触碰避让方向与五感“适宜觅食/活动”仍是未验证推断；空白 none 正确。完整通路能传递并读回工程码，不证明建议符合生物学或身体已经完成目标。启动脚本与完整复现说明见 `reports/local-model-2026-09-19/README.md`；现仅 4B loopback GPU 服务保留运行。
