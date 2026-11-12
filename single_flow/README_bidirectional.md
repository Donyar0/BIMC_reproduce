# 文本到视觉单流：双向原型一致性

本实验沿用冻结 CLIP 和原始 BiMC 的 prompt＋描述融合文本原型 `T_c`。
真正的视觉原型 `V_c` 从当前阶段训练/支持图像的 CLIP 特征按类求均值，
不使用 BiMC 可选的新类视觉校准。Flow 是一套共享速度网络；基类以
`x_t=(1-t)T_c+tV_c` 和目标速度 `V_c-T_c` 训练。

训练中的第二项是更新后同类原型的速度匹配损失，替代旧版 InfoNCE。
正向更新的文本原型 `T'_c` 在视觉侧，反向更新的真实视觉原型 `V'_c`
在文本侧，因此第二项也沿文本到视觉方向计算：

```text
T'_c = normalize(ODE_forward(T_c, 0 -> 1))
V'_c = normalize(ODE_reverse(V_c, 1 -> 0))
z_s = (1-s)V'_c + sT'_c,  s ~ Uniform(0,1)
L_FM = mean(||v(x_t,t)-(V_c-T_c)||²)
L_bidir = mean(||v(z_s,s)-(T'_c-V'_c)||²)
L = L_FM + BIDIR_FM_WEIGHT * L_bidir
```

两条更新轨迹参与反向传播，损失不依赖其他类别作为负样本。
`BIDIR_STEPS` 控制训练中两条轨迹的 Euler 步数，默认与推理的
`STEPS` 一样为 8。基类与增量优化器均按 epoch 使用余弦退火学习率。

提供两份仅在增量阶段不同的配置：

| 配置 | 增量阶段 | 原型库 |
|---|---|---|
| `bidirectional_flow_frozen.yaml` | Flow 冻结 | 旧类原型保持不变，只追加新类 |
| `bidirectional_flow_incremental.yaml` | 新类支持图像的类别均值继续训练 Flow | 用更新后的 Flow 重算全部已见类文本侧原型 |

增量训练只看当前新类的支持图像和对应文本；没有旧类图像回放。
为了避免新旧类别的生成原型来自不同版本的 Flow，增量训练版每阶段都
重算全部已见类的正向文本原型。两组实验分别保存独立运行结果，
不能在运行中途切换 `ADAPT`。

推理对每张测试图像特征 `q` 求逆向积分结果 `q'`，默认计算三路分数：

```text
query_to_generated    = cos(q,  T'_c)  # 视觉分数
reverse_to_text       = cos(q', T_c)   # 文本分数
reverse_to_generated  = cos(q', T'_c)  # 更新图像和更新文本
```

默认 `SCORE_MODE=fused`，按 `SCORE_WEIGHTS=[1,1,1]` 等权平均。
运行会在 `metrics.csv/json` 写出三路及融合后的各阶段准确率。
设第三个权重为 0 时第三路不计算，也不参与融合。
查询图像不会参与训练或原型库构建。

这次更换了训练损失，旧 InfoNCE 训练权重不能当作新实验，需要重新训练。
完成新实验后，若只改变 `SCORE_WEIGHTS`，可以不重新训练而重评估：

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bidirectional_flow_incremental.yaml \
  --eval_run output/bidirectional_velocity_incremental/cub200/seed_已有目录 \
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

分别写入 `output/bidirectional_velocity_frozen/cub200/seed_...` 和
`output/bidirectional_velocity_incremental/cub200/seed_...`。`base_flow.pt`
是基类权重；增量训练版另有 `flow_session<N>.pt`；每阶段均保存
`bank_session<N>.pt` 和阶段指标。真实数据精度需实际运行验证。
