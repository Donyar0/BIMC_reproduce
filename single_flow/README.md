# 复用 BIMC 原型的单流融合文本 → BiMC 图文融合原型

Flow 直接读取 BIMC 已计算的原型，不再加载 CLIP、编码文本或重新融合描述。
`fm.py` 来自 FlowComposer，保留上游 MIT LICENSE；训练脚本为任务适配实现。
环境：Python 3.10+、PyTorch 2.6+；运行 BIMC 还需仓库原有依赖、数据和 CLIP 权重。

## 推荐：像 BIMC 一样直接运行完整实验

现在 `main.py` 可自动完成原型提取、仅基类训练 Flow、每阶段追加 bank、
测试图像编码和全部已见类别评估，不需要手工准备 test.pt。
先设置各 dataset YAML 的 ROOT 为实际数据路径，并准备原有依赖、描述和 CLIP 权重。
从仓库根目录运行：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/single_flow.yaml
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/single_flow.yaml
python main.py --data_cfg configs/datasets/miniimagenet.yaml --train_cfg configs/trainers/single_flow.yaml
```

一键依次运行三个数据集（失败即停止，终端日志分别保存）：

```bash
bash scripts/run_single_flow_all.sh
# 可选 SwanLab：
bash scripts/run_single_flow_all.sh --use_swanlab
```

CIFAR100 和 MiniImageNet 各9阶段，CUB200 共11阶段（依据当前默认划分）。
每阶段输出 `Test acc mean`、`=> Task [i], Acc`、基类/增量类/调和准确率；
最后输出 `Final acc:[...]`、`Task-wise acc`、最后阶段准确率和阶段平均准确率。
所有准确率使用百分数，与原 BIMC 的 AccuracyEvaluator 完全共用。
增量准确率沿用原实现：各已见增量任务准确率的算术平均；基类和增量分组
均在全部已见类别竞争下统计。阶段0的增量/调和准确率按原实现记0。
`Final acc` 是各阶段总准确率列表，不是只含最后一个数。

结果自动保存至 `output/single_flow/<dataset>/seed_<seed>_<timestamp>/`：
- `metrics.csv`、`metrics.json`：各阶段指标及完成状态（每阶段更新）。
- `config.yaml`：完整实验配置。
- `base_flow.pt`：仅基类训练得到的网络。
- `bank_session*.pt`：每阶段全部已见类伪视觉原型。

只保存 bank 和模型即可评估，默认不保存逐样本训练缓存。启用
PROTOTYPE_CACHE_DIR 时，新 Flow runner 会按数据集/运行时间分目录导出缓存。
Flow 的训练终点改为 BiMC 图文融合原型，而不再是单纯视觉原型。
增量阶段生成时仅使用文本原型：新类真实视觉原型不用于 Flow 推理或二次融合。
分类只用生成的伪图文融合原型。

对比原 BIMC：同一 dataset YAML 改用 `configs/trainers/bimc.yaml` 即可。
使用相同种子、CLIP、数据划分；Flow 训练的 torch RNG 单独隔离，不影响后续
数据加载随机状态。结果需要在真实数据上运行，仓库没有预填实验准确率。
默认超参数未经三个数据集验证，应使用基类验证数据选参。

## 消融实验：直接预测终点，仍使用速度损失

增加了独立配置 `configs/trainers/single_flow_endpoint.yaml`，原配置
`single_flow.yaml` 仍使用速度输出，两者保持同样的文本起点、BiMC 图文融合终点、
网络、训练轮数、学习率、损失函数及 8 步 Euler 解算器。训练与推理一致：

- `PREDICTION: velocity`：网络直接输出 `v_hat`。
- `PREDICTION: endpoint`：网络直接输出 `x1_hat`，经
  `v_hat = (x1_hat - xt) / (1-t)` 换算为速度；
  **FM Loss 仍然是 `||v_hat - (x1-x0)||²`**。
  不直接对 `x1_hat` 做 MSE 监督。
- 可选 CE 损失用 `x1_hat` 或从速度还原的终点计算，默认两组均为 0。

因为终点参数化包含 `1/(1-t)`，仅 endpoint 模式的训练采样
`t ~ U[0, ENDPOINT_T_MAX)`，默认 `ENDPOINT_T_MAX=0.875`，
避免在 `t→1` 时速度换算数值放大。默认 8 步 Euler 的最后一步时间为
`t=7/8`，对应覆盖范围；更改步数时应重新检查时间范围。
最终积分、归一化和全部已见类别分类方式保持不变。
端点模式的第 8 次 Euler 更新等于当步预测的 `x1_hat`，是公式的直接结果。

推荐在三个数据集上对比完全相同设置的两组：

```bash
# 原速度输出（对照）
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/single_flow.yaml
# 新终点输出 + 速度监督
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/single_flow_endpoint.yaml
```

终点实验的结果单独保存在 `output/single_flow_endpoint`，不会覆盖原对照。
独立命令也支持 `python -m single_flow.s1_flow train ... --prediction endpoint --endpoint-t-max 0.875`；
模型 checkpoint 记录输出模式，生成过程根据 checkpoint 自动选择换算。
从旧版 checkpoint 读取时默认按速度输出以保持兼容。
整个新增实验尚需在真实 GPU 数据集上验证其效果。

## 原型来源

`models/bimc.py` 的 `fused_text_prototype()` 是 BIMC 原有的文本混合公式：
`(1-LAMBDA_T)*text_features + LAMBDA_T*description_proto`。
关闭 TEXT_CALIBRATION 时使用模板特征。build_task_statistics 将该混合向量
归一化后保存为 `text_proto`。它仍是文本原型，不包含视觉特征。
原 BIMC 分类中的 beta 图文融合继续使用未归一化的文本混合，保持原公式。

`image_proto` 直接沿用 BIMC 的输出：基类为图像特征均值原型；增量类在
VISION_CALIBRATION=True 时经 soft_calibration 校准。Flow 不重复计算或校准。

Flow 的新训练终点为 BiMC 原始分类器使用的融合原型：
`x1 = normalize(BETA * ((1-LAMBDA_T)*text_features + LAMBDA_T*description_proto) + (1-BETA)*image_proto)`
（关闭 TEXT_CALIBRATION 时直接使用 text_features）。注意这里的文本部分是
**归一化前**的文本混合，完全沿用 `forward_ours()`；增量视觉原型按 BiMC
设置校准，而基类视觉原型是原始图像均值原型，不会额外自校准。
缓存新增 `fused_prototypes`，旧的缓存必须重新生成。

## 导出缓存

在使用的 trainer YAML 的 TRAINER.BiMC 下设置：

```yaml
PROTOTYPE_CACHE_DIR: "output/bimc_prototypes"
```

然后从仓库根目录照常运行 BIMC，例如：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/bimc.yaml
```

Runner 在每个任务 build_task_statistics 后导出 `session_0.pt`、`session_1.pt` 等。
数据来自 source=train 的基类训练集/增量 support 集，不使用测试图像生成原型。
这个命令仍运行原 BIMC 分类评估；Flow 的训练和分类由下面的独立命令完成。
导出开关默认为空，不写缓存。每次实验应使用独立输出目录，以免覆盖旧缓存。

缓存字段：
- `class_ids`：按行排列的类别编号。
- `text_prototypes`、`visual_prototypes`：已经生成的 [C,D] 原型。
- `fused_prototypes`：按 BiMC 原始 BETA 融合且归一化后的 [C,D] 训练终点。
- `image_features`、`image_labels`：供可选 sample 模式使用。
- `encoder_id`：包含 backbone、精度和数据配置；使用自定义 CLIP 权重时需自行补充权重版本。
- `stage`：base 或 incremental；`split`：train。

导出接口 `single_flow.bimc_cache.from_task_statistics()` 只复制张量，不编码或融合。
它检查 class_index 与图像原型的排序一致。缓存中的阶段字段不能识别错误标注的数据来源。

## 训练、生成与分类

从仓库根目录运行：

```bash
python -m single_flow.s1_flow train --cache output/bimc_prototypes/session_0.pt --out output/base_flow.pt --target prototype
python -m single_flow.s1_flow generate --checkpoint output/base_flow.pt --cache output/bimc_prototypes/session_0.pt --out output/bank_base.pt
python -m single_flow.s1_flow generate --checkpoint output/base_flow.pt --cache output/bimc_prototypes/session_1.pt --existing-bank output/bank_base.pt --out output/bank_session1.pt
python -m single_flow.s1_flow classify --cache test.pt --bank output/bank_session1.pt --out output/predictions.pt
```

默认 prototype 模式直接将基类 text_prototypes 作为 x0、fused_prototypes 作为 x1；
这个模式只需要文本原型和图文融合原型，不需要逐样本特征。--target sample
按基类样本数重复上述类别级训练对，训练终点仍是 fused_prototypes，
并不将单个 image_features 作为终点。可选 CE 也使用 fused_prototypes 分类参照。
训练 xt=(1-t)x0+t*x1，监督速度 x1-x0；仅训练速度网络。
默认只有速度平方误差（维度求和、batch 求均值），--ce-weight 可开启视觉原型 CE。
prototype 模式每类每轮仅一对数据，默认100轮、batch64、24块只是未经真实实验验证的起点。

增量阶段不训练。generate 只读取 encoder_id、class_ids、text_prototypes；
缓存中的新类校准视觉原型仅作为 BiMC 原始统计量保留，
不参与增量阶段的 Flow 生成、分类融合或网络更新。
默认8步欧拉积分，最终归一化；每类一个确定性伪视觉原型。
追加 bank 时旧类原型逐元素保留，重复类别编号报错；各阶段应使用同一模型与步数。
模型/步数身份目前没有在 bank 中强制核对。

test.pt 需要同一冻结 CLIP 的 image_features、encoder_id 和可选 image_labels。
测试特征仍由现有图像提取流程准备；本次新增的是训练/support原型导出，不自动导出测试缓存。
分类使用测试图像特征与全部已见类伪视觉原型的余弦相似度。

旧版 template_features/description_features 缓存和 --alpha 参数已移除，请重新导出缓存并训练。
融合设置由 BIMC 的 TEXT_CALIBRATION/LAMBDA_T 统一控制，所有阶段保持相同配置。
不能用测试集选择这些超参数；真实分类精度与新类泛化尚未验证。

## 验证

```bash
python -m unittest discover -s single_flow -v
```

8项测试覆盖三个数据集配置、两阶段完整合成流程、全部已见类别竞争，以及：原型直接输入、导出行序、增量拒绝训练、旧 bank 保留、
非连续标签、sample/CE 路径，以及共用文本融合公式。未运行真实数据集或 GPU 训练。
