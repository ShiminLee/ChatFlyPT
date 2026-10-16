# 三维环境与接收/表达分离 · 2026-09-20

当前本机页面已接入八类可编辑物品：糖、酵母、高盐、热源、冷源、光源、暗区、可站立方块。编辑器支持场景旋转/缩放、物品 XY 拖动、XYZ 坐标、三轴尺寸、Z 旋转及强度/半径。它不是完整 Blender 编辑器。

草稿在浏览器中以虚影显示，不送入仿真；已应用物品被移动/删除时，原位置另留淡色线框。撤销可恢复已应用版本。应用经过物理初始化检查和本轮神经/身体试次，成功后原子保存到 `data/local/environment.json`。版本冲突或初始化失败不覆盖原环境。聊天中显示“环境改动”及版本/物品数量，可展开查看坐标。

## 真实感知与实现范围

物体的外观、挥发物和接触化学感觉是不同输入，不能因为视觉中出现了糖块就激活糖味标签。

| 通道 | 本轮实现 | 尚未完成 |
|---|---|---|
| 物体视觉 | 固体几何进入 MuJoCo 官方双眼渲染，官方 Retina 每眼 721 小眼采样；接收框可看双眼画面 | 连接组当前只接入分眼平均亮度代理，没有小眼到视网膜列的空间配准或物体身份识别 |
| 气味 | 在实际身体触角位置采样酵母相关工程场，按距离衰减 | 非真实气流/扩散模型；DM2 不是酵母全部挥发物的完整编码 |
| 糖/高盐味觉 | MuJoCo 口器接触实体时，驱动已有唇部相关候选感觉群；远处不触发味觉 | 真实果蝇腿部也有味觉，本实现没有相应腿部味觉神经入口配准；口器碰撞也是粗粒度代理 |
| 热/冷 | 触角位置的有限范围工程场，转为已有升温/低温通道 | 无热传导或生理温度标定，强度不是摄氏度 |
| 光/暗 | 光源进入物理渲染；暗区衰减区域内的双眼读数 | 暗区不是完整遮挡几何/光传输模型，也未单独实现 OFF 通路 |
| 方块 | 真实 MuJoCo 几何、碰撞及摩擦，能改变接触和身体高度 | 并不自动生成爬上方块的神经意图 |

所有采样在独立试次起点进行。身体后续接触被记录，但尚未连续回送同一个连接组状态。相互独立的数据集不会被拼成一个体。

## 接收与表达

`dialogue.sensory_report` 的感觉文字已移入“信息如何被接收”。`expression.py` 单独定义对外沟通表达协议。当前缺少发声、化学释放、基底振动的实测输出和语义解码，因此其 `text` 为 null、`messages` 为空；聊天仅有系统状态，不生成虚构的果蝇发言。

LLM 和建议回灌仍关闭；运动指令只来自原有运动相关活动读出。返回值明确标记 `expression_translation_completed=false`、`closed_loop=false`，不能以页面成功回放声称完整自由沟通已完成。

## 验证

- `final-tests.xml`：85 项通过，覆盖真实连接组通路、输出隔离、原控制器、实际碰撞和视网膜、环境版本/失败隔离。
- `environment-final-tests.xml`：补充环境专项验证，含酵母/热/冷的距离与侧别，以及缺少侧别映射不得代用另一侧。
- 网页实际添加八个草稿物品、应用、修改坐标/旋转、拖动方块、撤销、再应用均已验证；浏览器无 console error/warn。
- `revision-2.json`：真实网页提交的八物品环境。500 ms 神经和身体记录，MuJoCo `status=ok`、零数值警告、`llm_called=false`。只有当前位置实际采样非零的光/酵母输入进入神经系统；未接触糖/高盐时没有其味觉输入。
- 远处糖块相对空环境改变了 721 小眼视觉数组，但未激活糖味；暗区可将双眼亮度置零；实际方块接触抬高身体；实际口器碰撞可触发当前糖味候选入口。
- 所有这些结果是软件一致性与物理连接验证，不是行为或感知准确率标定。

## 原始资料

- [NeuroMechFly 官方视觉教程](https://flygym.readthedocs.io/latest/tutorials/vision_basics.html)：双眼相机、复眼分箱与场景物体视觉。
- [当前 FlyGym Simulation API](https://neuromechfly.org/api_reference/flygym/simulation/)：`get_raw_vision` 与 `get_ommatidia_readouts`。
- [NeuroMechFly 官方嗅觉教程](https://flygym.readthedocs.io/latest/tutorials/olfaction_basics.html)：在感觉器官位置采样环境气味。
- [糖味感觉器官功能分工，Nature Communications 2016](https://www.nature.com/articles/ncomms10678)：腿部糖感觉参与选择，不同感觉器官并非可随意替换。
- [食物气味的吸引/回避，Nature 2009](https://www.nature.com/articles/nature07983)：食物气味与嗅小球反应。
- [温湿度连接组，Current Biology 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443704/)：温湿度入口及下游通路。

核心代码：`chatflypt/environment.py`、`behavior_body.py`、`expression.py`、`conversation.py`、`server.py`、`web/environment-editor.js`。上游固定版本 `vendor/badapple-fly/backend/body.py` 未改动。
