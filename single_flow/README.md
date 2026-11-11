# 复用 BIMC 原型的单流文本 → 伪视觉原型

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
当前仍不将增量视觉原型融合到分类器。Flow 分类只用生成的伪视觉原型。

对比原 BIMC：同一 dataset YAML 改用 `configs/trainers/bimc.yaml` 即可。
使用相同种子、CLIP、数据划分；Flow 训练的 torch RNG 单独隔离，不影响后续
数据加载随机状态。结果需要在真实数据上运行，仓库没有预填实验准确率。
默认超参数未经三个数据集验证，应使用基类验证数据选参。

## 原型来源

`models/bimc.py` 的 `fused_text_prototype()` 是 BIMC 原有的文本混合公式：
`(1-LAMBDA_T)*text_features + LAMBDA_T*description_proto`。
关闭 TEXT_CALIBRATION 时使用模板特征。build_task_statistics 将该混合向量
归一化后保存为 `text_proto`。它仍是文本原型，不包含视觉特征。
原 BIMC 分类中的 beta 图文融合继续使用未归一化的文本混合，保持原公式。

`image_proto` 直接沿用 BIMC 的输出：基类为图像特征均值原型；增量类在
VISION_CALIBRATION=True 时经 soft_calibration 校准。Flow 不重复计算或校准。

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

默认 prototype 模式直接将基类 text_prototypes 作为 x0、visual_prototypes 作为 x1；
这个模式只需要两种原型，不需要逐样本特征。--target sample 使用现有 image_features
作为终点，并用现成的 visual_prototypes 作为单样本训练模式的目标。
训练 xt=(1-t)x0+t*x1，监督速度 x1-x0；仅训练速度网络。
仅优化速度平方误差（维度求和、batch 求均值），没有额外分类损失。
prototype 模式每类每轮仅一对数据，默认100轮、batch64、24块只是未经真实实验验证的起点。

增量阶段不训练。generate 只读取 encoder_id、class_ids、text_prototypes；
缓存中的校准视觉原型被保留供后续实验使用，但目前不参与生成、分类融合或网络更新。
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
非连续标签、sample 路径，以及共用文本融合公式。未运行真实数据集或 GPU 训练。

## 双向 Flow：增强图像 → 原始 BiMC 融合文本

当前 `unified_flow` 的起点为增强图像经冻结 CLIP 提取并归一化的特征，
终点为原始 BiMC 文本原型：

```text
x0 = normalize(CLIP_image(augment(image)))
x1 = normalize((1-LAMBDA_T)*prompt_proto + LAMBDA_T*description_proto)
xt = (1-t)*x0 + t*x1
v_target = x1-x0
```

不使用 calibText Linear Adapter；`TEXT_CALIBRATION: True`、`LAMBDA_T: 0.5`
恢复原始 Prompt/Description 融合。CLIP 全程冻结。

只有基础阶段进行 Flow 训练。旋转（默认 ±15°）、水平翻转、随机遮挡
（默认概率 0.5、面积 2%–20%）和高斯噪声（概率 0.5、标准差 0.03）
在像素空间进行，再归一化为 CLIP 输入；每轮重新随机生成，不缓存固定增强视图。
支持集的真实原型与测试图像仍使用干净的测试变换。
基础训练沿用该阶段已经选择的图像子集，增强不会重新抽取支持样本。

增量阶段仅提取当前支持集统计、追加类别 Bank 和评估，**不微调 Flow**，
不回放旧样本，不训练文本 Adapter。所有 `flow_session*.pt` 的网络参数相同。
`ADAPT_STEPS` 必须为 0。

### 速度预测与终点预测对比

| 配置文件 | 网络输出 | 计算速度 | 损失 |
|---|---|---|---|
| `configs/trainers/unified_flow.yaml` | `v_hat` | 网络输出本身 | `mean(sum((v_hat-(x1-x0))**2))` |
| `configs/trainers/unified_flow_endpoint.yaml` | `x1_hat` | `(x1_hat-xt)/(1-t)` | 相同速度损失 |

终点预测模式不对网络输出归一化，也不额外加终点 MSE。
两组使用相同种子、网络、增强、优化器和时间分布；只改变输出参数化及结果目录。
为避免 `t=1` 的除零，两组训练均采样 `t∈[0,1-TIME_EPS)`（默认 `TIME_EPS=0.05`）。
推理在同一速度场接口中将时间上限限制为 `1-TIME_EPS`。
终点预测的速度损失等价于按 `1/(1-t)^2` 加权的终点误差，这个权重是换算公式的结果。

### 双向推理

图像查询从 `t=0` 正向积分到文本空间，与融合文本原型计算余弦分数；
文本原型从 `t=1` 反向积分生成视觉原型，再与真实支持图像均值融合构成视觉 Bank。
后者与干净查询图像计算视觉余弦分数。
`GAMMA` 融合视觉分数和文本分数；`ALPHA` 混合原图像特征与正向积分结果，
默认 1 表示完整使用图像到文本映射。`BETA_BASE/BETA_INC` 是生成视觉原型的权重。

反向生成使用负步长，时间顺序也随之反转，不再沿用旧的文本→视觉正向积分。
反向首步的 `t=1` 使用上述时间上限近似；终点预测在终点附近的反向生成
可能对 `TIME_EPS` 和积分步数敏感，需要在验证数据上检查。
不计算协方差、马氏距离、描述 KNN 或 BiMC Ensemble。
正数 `LOGIT_SCALE` 只改变 softmax 概率，不改变 argmax。

### 后台运行

先更新数据集 YAML 的 ROOT，在仓库根目录、已安装依赖的环境中执行：

```bash
mkdir -p logs
# 两组依次运行，避免在同一 GPU 同时占用显存。
nohup env CUDA_VISIBLE_DEVICES=0 bash scripts/run_image_text_ablation.sh \
  configs/datasets/cifar100.yaml --use_swanlab \
  > logs/image_text_ablation.log 2>&1 < /dev/null &
echo $!
tail -f logs/image_text_ablation.log
```

也可单独运行任一配置：

```bash
python -u main.py --data_cfg configs/datasets/cifar100.yaml \
  --train_cfg configs/trainers/unified_flow.yaml --use_swanlab
python -u main.py --data_cfg configs/datasets/cifar100.yaml \
  --train_cfg configs/trainers/unified_flow_endpoint.yaml --use_swanlab
```

两组结果分别写入 `output/unified_flow_image_text_velocity/` 和
`output/unified_flow_image_text_endpoint/`；每个目录按数据集、种子和时间区分运行。
保存 `config.yaml`、`base_flow.pt`、每阶段的 Flow/Bank、`metrics.csv` 和 `metrics.json`。
SwanLab 记录完整配置、Git 提交信息和阶段准确率。

### 重评估与兼容性

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml \
  --train_cfg configs/trainers/unified_flow.yaml \
  --eval_run output/unified_flow_image_text_velocity/cifar100/原训练目录
```

仅可重评估此版本产生的相同预测模式权重；旧的文本→图像/单向/calibText 权重
必须重新训练。重评估时使用训练时的数据、CLIP、文本融合参数、积分步数、
`PREDICTION` 与 `TIME_EPS`，可调整评分参数及 Beta；不会进行训练。
严格选参应使用独立验证数据，不能用测试集选择最优设置。
只加载可信权重；旧版 PyTorch 如需不受限 pickle 加载，显式使用 `--trusted_checkpoints`。

### 验证

`test_image_text_flow.py` 用合成特征和像素验证原始 BiMC 文本融合、
终点→速度公式及梯度、两方向积分、像素增强、冻结 CLIP、拒绝增量训练，
以及两组配置的两阶段运行、相同权重、旧 Bank 保留与重评估。
未执行真实数据集/GPU 准确率实验。
