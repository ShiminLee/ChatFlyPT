# 实际身体测量到 MaleCNS 本体感觉：有来源的有限群体模型

2026-09-24。代码：`chatflypt/proprioception.py`。本次不是把 1,454 个本体感觉细胞同时激活；也不从人的要求、运动命令、预期动作或文字读出生成感觉。入口来自实际关节几何、关节速度和接触记录。

## 已经能执行的部分

`encode_proprioceptive_feedback(frame, seed, duration_ms=100., inventory=None, peak_rate_hz=50.)` 返回 `(event_tape, reception_report)`。磁带使用已有的 `dataset/id_namespace/input_neuron_ids/event_steps/event_neuron_ids` 结构，只含 MaleCNS 中有实际建模出边的候选入口。报告记录测量、输入细胞、发表的活性函数、工程事件增益和未覆盖部分。

具身主循环应把此磁带作为 **环境／身体实测反馈** 合并到 `environmental`，不能并入人的直接干预。仅在神经引擎成功接收后才能把报告的 `neural_feedback_applied` 从 `false` 改为 `true`。事件需纳入累计磁带、首次输入时刻和整段重放；新本体感觉入口也必须排除在盲读出特征和运动输出直接刺激计数之外。

`measure_proprioception(frame)` 单独返回不施加输入的测量与候选入口清单，供审计使用。缺失测量使用 `null`，不冒充实测零。

## 采用的定量来源

[Dallmann 等，Nature 2025](https://doi.org/10.1038/s41586-025-09554-2) 的作者 [MIT 开源函数](https://github.com/chrisjdallmann/feco-inhibition/blob/e1233f4a987c532c9f1ab42273af21a0a6a50393/code/utils/imaging_predict_gcamp.m) 给出了由关节运动预测 FeCO 群体钙信号的活性函数。该函数拟合或依据 [Mamiya 等，Neuron 2018](https://doi.org/10.1016/j.neuron.2018.09.009) 的实际数据。源码、许可证和 SHA-256 已保存在 `data/raw/proprioception-dallmann2025/`，固定提交 `e1233f4a987c532c9f1ab42273af21a0a6a50393`。

- **claw：** 输入为实际三维腿段位置计算出的 femur–tibia 夹角，单位度。以被动运动数据的 90° 为中心，复用作者五个四次多项式系数。作者后来的行为追踪数据用 80° 中心，论文解释为追踪差异；本实现没有把该偏移直接套到 MuJoCo qpos。
- **club：** 输入为实际膝关节角速度的绝对值，单位 °/s；采用论文被动运动模型的 ±5 °/s 阈值。两方向均可激活，不猜测 hook 的方向身份。
- 作者的 30 ms/300 ms 双指数核用于模拟 **GCaMP 测量**，不代表感觉传入的神经延迟，因此没有把该成像滤波插进神经输入。

发表的输出是无量纲群体活性，不是 Hz。当前将活性整流、限制到 0–1，再乘以本项目此前本体感觉实验所用的 **50 Hz 工程事件尺度**，生成可重放的 Bernoulli 磁带。该增益不是作者拟合值，也不是活体放电标定；不能宣称“真实放电率已经还原”。接口可用 0–150 Hz 做明确增益敏感性对照，不能以调出期望行为为通过标准。

同族候选细胞使用同一群体活性，是明确的群体近似。它保留姿势变化和移动依赖，但不提供每细胞角度调谐、claw 屈伸细分、钙信号到脉冲率的实测转换或异质适应。该论文还研究了行为相关的突触前调制；本模块不会从“正在走”这个行为标签编造抑制信号，动态相互作用交给现有连接组，其生理正确性仍需验证。

## 本地身份和建模图检查

本地 MaleCNS v1.0 注释 SHA-256：`2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`。

总计 1,454 个 `mechanosensory_proprioceptive` 感觉细胞。其中 645 个具有明确 `ProLN/MesoLN/MetaLN` 主腿入口和 L/R `rootSide`。809 个没有此明确组合或属于其他器官，保留在排除清单，不假配到腿。

| 腿 | claw 候选 | club 候选 | hook 候选（未投递） |
|---|---:|---:|---:|
| 左前 lf | 4 | 8 | 4 |
| 左中 lm | 22 | 29 | 11 |
| 左后 lh | 22 | 31 | 10 |
| 右前 rf | 1 | 3 | 6 |
| 右中 rm | 17 | 31 | 17 |
| 右后 rh | 25 | 30 | 12 |

可执行 claw + club 共 **223 个**，均有当前神经模型出边。仅 33 个细胞具有直接 `FeCO claw/hook/club` 同义标签，且本版本都在右后腿；其余族身份按同版本中相同精确类型名称推定，明确标为 `type_congruence_candidate`，不改写为直接证实。复合类型名不拆开强配。

完整入口列表在 `load_proprioceptor_inventory()` 返回值中，分别保留解剖候选 ID、当前有出边 ID、身份依据、侧别、入口神经和缺失参数。不同连接组 ID 不混用。较大的前后腿数量差异也是覆盖局限，不假造细胞补平。

[Lee 等的作者数据与代码](https://github.com/sagrawal/Lee_2024) 提供 FANC 的 FeCO 分型连接分析，可用于后续核对。它不是 MaleCNS 的逐细胞活动标定表，不能直接替换 ID。更细的角度分布与外周机械结构数据见 [Mamiya 2023 数据集](https://doi.org/10.5061/dryad.dbrv15f6q) 和[原论文](https://doi.org/10.1016/j.neuron.2023.07.009)。数据下载本次网页端返回 403、API 下载返回 401；未把未取得的文件算作已分析数据。

## 物理输入约定

```python
frame = {
    'time_ms': 100.,
    'joint_positions_rad': {'fly/rh_trochanterfemur-rh_tibia-pitch': 1.2},
    'joint_velocities_rad_s': {'fly/rh_trochanterfemur-rh_tibia-pitch': 0.3},
    'body_names': [...],
    'body_positions_mm': [...],
    'body_contacts': [{
        'body_geom': 'fly/rh_tarsus5', 'other_geom': 'ground',
        'distance_mm': -0.001, 'active_constraint': True,
        'normal_force_model_units': 4., 'tangential_force_model_units': 2.
    }]
}
```

膝夹角来自 `trochanterfemur → tibia → tarsus1` 三个实测关节原点。raw qpos 另外保存，不将它冒充动物的绝对夹角。club 只用实际 tibia hinge 的速度大小；hook 的屈伸细胞归属未知，故不投递。接触必须约束激活且表面距离 ≤ 0；记录包括地面接触。接触力保持模型原生单位，不冒充牛顿、cuticle strain 或钟状感器放电。

## 验证和完成边界

`tests/test_proprioception.py` 本次 **18 项通过**。检查实际注释及图出边、侧别/入口排除、几何角与 qpos 分离、作者阈值边界、缺测不补零、力只测不编造转导、静止中立角零事件、物理角度/速度变化产生候选事件、固定 seed 重放一致、人的文字与目标动作不能改变磁带。

这是有出处且可执行的 **有限群体本体反馈**，仍需主通路真实身体—神经运行与累计重放验收。测试通过不是动物行为验证。静止时非中立膝角也可产生 claw 输入，因此加入后“空环境”不再天然等于零神经输入；必须改用自身空环境基线评估外界刺激的增量。

尚未覆盖：hook 屈伸方向身份、毛板的具体关节与触毛偏转、钟状感器负荷到角质层应变、平衡棒/翅等器官，以及高频基底振动。100 ms 采样不能分辨 100–2,000 Hz 的 FeCO 振动响应；不能把 club 运动门控改称振动模型。以上项目没有被空列表或任意增益伪装成完成。
