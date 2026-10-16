# ChatFlyPT 三维身体视图替换

2026-09-19 已在本机正式端口 `http://127.0.0.1:8765` 生效。

右下模拟空间已采用截图对应的 Kevin Lin / kevinlinxc 的 Bad Apple Fly 视图实现：Three.js 场景、金色身体/红眼/半透明翼材质、地面网格、灯光、侧视/俯视、OrbitControls 旋转缩放、镜头跟随与地面轨迹。新增展开/收起和当前身体记录时间，保持原聊天、神经图、接收说明布局。

原项目固定提交 `6726892e7ebd93e6c5ffab056ab8db09433c28a6`。复用的是其身体视图；本次没有把它的神经输入、动作映射或 Bad Apple 图案模式替换进 ChatFlyPT。完整源代码核查见[作者与控制方式](../../research/badapple-fly-control-review-2026-09-19.md)。

## 身体数据如何接入

- 浏览器本地加载69个官方NeuroMechFly网格：54,905顶点、115,158三角面，约3.25MB。无需外部CDN或额外在线模型接口。
- 本项目已经保存各刚体位置与WXYZ四元数；渲染器按实际刚体名称对齐，组合网格局部变换。对MuJoCo合并到胸部的头部等静态网格，使用真实owner body而非按名称猜测。
- 神经图和身体回放共用时间轴。身体选择当前时间以前最近的实测帧；不生成步态插帧。若身体记录只覆盖前500ms，末端姿态保持在最后实测帧。
- 无动作记录时只显示官方初始姿态；加载失败、WebGL中断或记录缺少完整姿态时显示错误状态。

## 验证

- 8个页面/模块/模型资源均HTTP 200，MIME正确：[资源校验](http-assets.json)。
- 184项Python测试通过，0失败：[回归报告](pytest.xml)。JavaScript语法检查通过。
- 新增几何对齐测试使用独立MuJoCo与现有身体worker的真实20ms记录，所有网格世界顶点最大误差小于9×10⁻⁹mm。
- 从实际浏览器聊天框发送“模拟右前腿被触碰”：434次神经脉冲、259个活跃细胞，身体仿真成功，0–500ms共51帧、69个身体部位与网格资产一致。[端到端记录](end-to-end.json)
- 在真实浏览器目视检查初始侧视、俯视以及仿真结束帧；侧/俯选择、重置、展开/收起、共享时间轴归零均已通过UI操作。展开后的实际仿真帧显示完整立体身体；网页恢复到聊天布局。

这些验证证明视图及记录接入正确，不改变既有神经→运动控制器的生理标定状态。原项目控制公式及反馈边界见上述源码核查报告。

## 可维护文件与来源

- [视图模块](../../chatflypt/web/body-view.js)：标注原作者、提交与改编范围。
- [模型导出脚本](../../scripts/export_body_scene.py)：使用同一官方FlyGym身体构建流程，可重建资产。
- [网格资产](../../chatflypt/web/assets/neuromechfly-scene.json)：包括模型来源、官方commit、文件哈希、Apache-2.0许可与验证记录。
- [原项目源码锁定清单](../../research/manifests/badapple-fly-source.json)。Three.js/OrbitControls保留其[MIT许可](../../chatflypt/web/vendor/THREE-LICENSE.txt)。原项目应用代码未附项目级LICENSE，不将其标成MIT或Apache；许可状态分别记录在源码核查报告中。
