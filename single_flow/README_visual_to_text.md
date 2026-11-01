# 单流视觉 → 文本：纯 FM 训练，完整 Flow 测试，无文本 Bank

本分支的默认方案：**冻结 CLIP、仅训练 Flow Matching**。Session 0 使用基类图像特征与 BiMC 融合文本原型训练；Session 1+ 使用当期 5-shot 支持图像与当期融合文本原型进行小学习率 FM 微调。

## 直接运行

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/visual_to_text.yaml --use_swanlab
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/visual_to_text.yaml
python main.py --data_cfg configs/datasets/miniimagenet.yaml --train_cfg configs/trainers/visual_to_text.yaml
```

配置默认值：`FLOW.INC.ENABLED=True`、`FLOW.ALPHAS=[1.0]`、`FLOW.STEPS=1`。**仅允许 alpha=1 的完整 Flow 输出**，不使用残差弱更新。若希望完整区间使用多个 Euler 积分步，可将 STEPS 改为 8。

## 训练与推理

FM：`x_t=(1-t)z+t*p_c`，`v*=p_c-z`，`L_FM=mean(sum((v_theta(x_t,t)-v*)**2,dim=-1))`。不使用交叉熵或边际损失；基类和增量两阶段均只优化这个目标。

测试：CLIP 提取每张测试图像视觉特征，从 t=0 到 1 完整积分 Flow，最后归一化输出；与**所有已见类的融合文本原型**计算余弦相似度，并做 argmax 分类。默认单步：`normalize(z+v_theta(z,0))`。保存 `metrics.csv`、`metrics.json`、`base_flow.pt` 和增量微调权重 `flow_sessionN.pt`。

**无需单独的文本原型 Bank。** 每个 session 在推理时从当时所有已见类别名与描述直接计算融合文本原型，作为本次评测的临时矩阵；不会创建或保存任何 `bank_session*.pt`。这保证各阶段仍与全部已见类别比较，但重复编码会增加一定运行开销。

`FLOW.INC.ENABLED=False` 可令增量阶段冻结 Flow（只添加文本候选类）；这与当前默认的增量 FM 微调不同。

## 直接使用独立模块（不建 Bank）

```bash
python -m single_flow.visual_to_text train --cache base.pt --out flow.pt --source sample
python -m single_flow.visual_to_text classify --checkpoint flow.pt --cache test.pt --candidate-cache all_seen_text.pt --steps 1 --out pred.pt
```

这里 `candidate-cache` 必须包含本阶段**全部已见类别**的 `class_ids`、`text_prototypes` 和 `encoder_id`，供独立 CLI 读取，不需要也不会输出 bank。主程序从 CLIP 即时计算候选原型，不依赖此 CLI 缓存接口。

从头运行实验时，使用默认的空 checkpoint 路径。实验结果与 CIFAR100/CUB/miniImageNet 的真实 GPU 精度尚未在这里执行验证。运行轻量测试：`python -m unittest discover -s single_flow -v`。
