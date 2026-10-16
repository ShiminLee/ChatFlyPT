# API 配置与真实请求验证

日期：2026-09-19。

界面右上角新增 **API 设置**，支持地址、密钥、模型和 Responses / Chat Completions 协议；可以读取模型、发送真实测试请求、保存后立即生效。模型可以从列表选择或手动填写。

当前保存地址为 `https://api.requestme.top/v1`，请求模型为 `gpt-5.5`，协议为 Chat Completions。网关返回的模型名称仅为服务方声明，未独立验证其底层模型身份。

密钥没有写入代码、测试报告或浏览器存储。Windows 使用当前账户的 DPAPI 加密保存在 Git 忽略的 `data/local/api-settings.json`；读取设置只返回是否已配置密钥，不返回密钥内容。留空密钥可以保留现有值，更换地址时需要填写新地址使用的密钥。保存的配置优先于 `.env` 与环境变量。

## 验证

- 全量 **94 项测试通过**，0 失败、0 跳过；见 `pytest.xml`。新增覆盖加密保存/读取、留空保留、更换目标不转发旧密钥、HTTP 设置读写、跨站与非本机 Host 拒绝、兼容协议响应解析以及错误不泄漏密钥。
- 浏览器实际打开了设置窗口，读取到 21 个模型，验证了留空密钥保存且即时生效；输入框没有回填密钥。
- 代码与报告的密钥扫描无命中。
- 聊天框另执行了“五感神经观测 → 请求模型解释”的真实请求，网关在 55 秒内未返回文本，界面正确显示超时；没有假造神经观测解释。
- `gpt-5.4-mini`（两个协议）和 `gpt-5.4`（Responses）测试返回 HTTP 502；未冒充成功。`gpt-5.5` 返回了真实文本，之后仍观察到间歇 502。
- 五感展示获得了真实建议；两种导航情形分别返回预期的 `left`、`right`。`forward` 和 `stop` 请求遇到网关 HTTP 502；其中 forward 经一次补测仍失败。此次五例验收为 **partial**，不计作全部通过。

原始调用、模型声明、响应 ID、服务方用量与失败记录见 `reports/reconstruction/live-llm.json`；首次运行记录另存 `live-llm-first-attempt.json`。`connection-test.json` 保存最初成功连接的返回，不表示后续请求均可靠。失败没有生成离线替代答案。

## 本地小模型

现有线性读出器承担神经活动到感觉标签的识别，有限感觉标签的中文表达可以直接由模板完成。自由对话与简单建议可以接本地语言模型，再通过同样的观测约束与案例测试评估。设置支持本机 HTTP OpenAI 兼容地址和自定义模型，不会自动下载或安装本地模型。

接口字段依据：[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[Models](https://developers.openai.com/api/reference/resources/models/methods/list)。
