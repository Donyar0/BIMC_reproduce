# 文本到视觉单流：双向原型一致性

本实验沿用冻结 CLIP 和原始 BiMC 的 prompt＋描述融合文本原型 `T_c`。
真正的视觉原型 `V_c` 从当前阶段训练/支持图像的 CLIP 特征按类求均值，
不使用 BiMC 可选的新类视觉校准。Flow 是一套共享速度网络；基类以
`x_t=(1-t)T_c+tV_c` 和目标速度 `V_c-T_c` 训练。

训练中的第二项损失让同类的两条轨迹对应：

```text
T'_c = normalize(ODE_forward(T_c, 0 -> 1))
V'_c = normalize(ODE_reverse(V_c, 1 -> 0))
L = mean(||v(x_t,t)-(V_c-T_c)||²)
    + CONTRAST_WEIGHT * symmetric_InfoNCE(T'_c, V'_c)
```

对比损失使用批内其他类别作为负样本；只有一个类别的批次不贡献该项。
`CONTRAST_STEPS` 控制训练中两条轨迹的 Euler 步数，默认 4；
推理 `STEPS` 默认为 8。基类与增量优化器均按 epoch 使用余弦退火学习率。

提供两份仅在增量阶段不同的配置：

| 配置 | 增量阶段 | 原型库 |
|---|---|---|
| `bidirectional_flow_frozen.yaml` | Flow 冻结 | 旧类原型保持不变，只追加新类 |
| `bidirectional_flow_incremental.yaml` | 新类支持图像的类别均值继续训练 Flow | 用更新后的 Flow 重算全部已见类文本侧原型 |

增量训练只看当前新类的支持图像和对应文本；没有旧类图像回放。
为了避免新旧类别的生成原型来自不同版本的 Flow，增量训练版每阶段都
重算全部已见类的正向文本原型。两组实验分别保存独立运行结果，
不能在运行中途切换 `ADAPT`。

推理对每张测试图像特征 `q` 求逆向积分结果 `q'`，默认计算两路分数：

```text
query_to_generated    = cos(q,  T'_c)  # 视觉分数
reverse_to_text       = cos(q', T_c)   # 文本分数
```

默认 `SCORE_MODE=fused`，按 `SCORE_WEIGHTS=[1,1,0]` 等权平均视觉与文本分数。
第三路 `cos(q',T'_c)` 默认不计算，也不写入指标；如需旧三路实验可将
第三个权重改为正值。运行会在 `metrics.csv/json` 写出两路及融合后的各阶段
准确率。查询图像不会参与训练或原型库构建。

已有三路分数版本的完整运行目录可以只重评估，不重新训练。更新分支后：

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bidirectional_flow_incremental.yaml \
  --eval_run output/bidirectional_flow_incremental/cub200/seed_已有目录 \
  --trusted_checkpoints
```

冻结版运行则换用 `bidirectional_flow_frozen.yaml` 和对应的输出目录。
重评估要求数据、训练参数和积分步数与原运行一致，只允许改变分数设置。

在仓库根目录以 CUB200 为例：

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bidirectional_flow_frozen.yaml --use_swanlab

python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bidirectional_flow_incremental.yaml --use_swanlab
```

分别写入 `output/bidirectional_flow_frozen/cub200/seed_...` 和
`output/bidirectional_flow_incremental/cub200/seed_...`。`base_flow.pt`
是基类权重；增量训练版另有 `flow_session<N>.pt`；每阶段均保存
`bank_session<N>.pt` 和阶段指标。真实数据精度需实际运行验证。
