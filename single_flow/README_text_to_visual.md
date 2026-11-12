# 原始 BiMC 融合描述文本到视觉空间的单流实验

本分支的 Flow 文本起点统一使用原始 BiMC 的 `text_proto`：
`normalize((1-LAMBDA_T)*prompt_feature + LAMBDA_T*description_proto)`。
`TEXT_CALIBRATION=False` 时 BiMC 只使用类别 prompt 特征。
文本原型直接由 `BiMC.build_task_statistics()` 产生，不训练或调用文本 Adapter。
冻结 CLIP；Flow 仅在基类训练，增量阶段冻结。

| 配置 | 基类训练终点 `x1` | 训练配对 |
|---|---|---|
| `single_flow.yaml` | 原始 BiMC 文本＋视觉融合原型 | 每类一对 |
| `single_flow_visual_prototype.yaml` | 每类图像特征均值原型 | 每类一对 |
| `single_flow_visual_sample.yaml` | 每张基类训练图像的 CLIP 特征 | 每张图像一对 |

三组实验均以该类融合描述文本原型 `x0` 为起点，训练速度网络拟合
`x1-x0`。默认损失为
`mean(||v((1-t)x0+t*x1,t)-(x1-x0)||²)`，`CE_WEIGHT=0`。
视觉单图实验的每个 epoch 有更多训练对和优化步数，比较精度时应记录
更新步数，并做等步数对照。

增量阶段只需新类的融合描述文本原型；新类支持图像不作为 Flow 起点或终点，
也不更新网络。8 步 Euler 正向积分产生每类一个分类原型，旧类库保持不变。
测试图像经冻结 CLIP 编码后，与全部已见类的生成原型计算余弦分数。

以 CUB200 为例，在仓库根目录分别运行：

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/single_flow.yaml
python -u main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/single_flow_visual_prototype.yaml
python -u main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/single_flow_visual_sample.yaml
```

结果分别保存到 `output/single_flow`、`output/single_flow_visual_prototype`
和 `output/single_flow_visual_sample` 下。旧 Adapter 配置产生的模型和缓存
不属于这组对照；请重新运行实验。合成测试覆盖三种终点和增量生成，真实
数据集效果需要实际训练验证。
