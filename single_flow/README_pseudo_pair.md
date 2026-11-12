# 伪视觉→伪文本速度与真实文本对齐实验

该实验基于 `unified-bidirectional-flow` 的图像→文本速度场，输入特征来自
冻结 CLIP，真实文本原型 `T_c` 是原始 BiMC prompt＋描述融合结果。
`V_c` 是同类别训练/支持图像的真实视觉均值。训练时沿当前 Flow
正向积分 `V_c` 得到伪文本 `\hat T_c`，沿同一 Flow 反向积分 `T_c`
得到伪视觉 `\hat V_c`。两条路径保持可微；对每个出现的类别，损失为：

```text
z_s = (1-s)*V_hat[c] + s*T_hat[c],  s ~ Uniform(0, 1-TIME_EPS)
L_velocity = mean_c ||v(z_s,s) - (T_hat[c]-V_hat[c])||²
L_text = mean_c (1-cos(T_hat[c], T_c))
L_total = PSEUDO_VELOCITY_WEIGHT*L_velocity + PSEUDO_TEXT_WEIGHT*L_text
```

两项分别默认权重 1.0 和 0.1。新配置明确设 `FM_WEIGHT=0`、
`CYCLE_WEIGHT=0`，因此原始图像→文本 FM 和旧 cycle 损失均不参与训练。
基类用基类训练图像，增量训练组用新类支持图像及旧类视觉/文本均值回放。
CLIP 保持冻结，两阶段均使用 warmup＋cosine 学习率调度。

`PSEUDO_STEPS=8` 控制训练时两条可微 Euler 路径；分类和 ODE 求解器
消融仍由 `STEPS`、`SOLVER` 控制。推理分类规则沿用该分支原有的
图像→文本分数与反向生成视觉原型分数融合，没有改动。

两份配置分别进行增量冻结与增量继续训练：

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/unified_flow_pseudo_pair.yaml --use_swanlab

python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/unified_flow_pseudo_pair_incremental.yaml --use_swanlab
```

分别输出到 `output/unified_flow_pseudo_pair_frozen/cub200/seed_...` 与
`output/unified_flow_pseudo_pair_incremental/cub200/seed_...`。新损失是
由当前模型生成端点的自监督约束，真实数据精度和训练稳定性需要实际实验验证。
旧实验权重不能当作这两项损失训练的权重使用，应重新训练。
