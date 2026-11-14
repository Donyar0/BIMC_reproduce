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

## 双向 Flow：正常图像特征 → 原始 BiMC 融合文本

当前 `unified_flow` 的起点为基类训练图像经固定测试预处理、冻结 CLIP 提取并归一化的特征，
终点为原始 BiMC 文本原型：

```text
x0 = normalize(CLIP_image(clean_preprocess(image)))
x1 = normalize((1-LAMBDA_T)*prompt_proto + LAMBDA_T*description_proto)
xt = (1-t)*x0 + t*x1
v_target = x1-x0
```

不使用 calibText Linear Adapter；`TEXT_CALIBRATION: True`、`LAMBDA_T: 0.5`
恢复原始 Prompt/Description 融合。CLIP 全程冻结。

基础阶段训练 Flow；增量阶段是否继续训练由 `ADAPT_STEPS` 决定。训练图像使用与原型统计相同的固定预处理
（Resize、CenterCrop、Normalize）；CLIP 特征提取一次后缓存在内存中。
训练每轮随机抽取已缓存的正常图像特征，不进行旋转、遮挡、加噪声、
随机裁剪或随机翻转。基础阶段仍使用原本选定的训练图像集合。

默认 `ADAPT_STEPS: 0`，增量阶段只提取支持集统计、追加类别 Bank 并评估；
可选的增量训练组会以新类支持样本和每个旧类一个真实均值更新 Flow，
再重新生成全部已见类别的视觉原型。CLIP 和文本原型保持冻结。
两个阶段都只优化图像特征到原始 BiMC 融合文本原型的 FM 速度误差，
没有伪原型对齐、循环约束或分类损失。

### 速度预测与终点预测对比

| 配置文件 | 网络输出 | 计算速度 | 损失 |
|---|---|---|---|
| `configs/trainers/unified_flow.yaml` | `v_hat` | 网络输出本身 | `mean(sum((v_hat-(x1-x0))**2))` |
| `configs/trainers/unified_flow_endpoint.yaml` | `x1_hat` | `(x1_hat-xt)/(1-t)` | 相同速度损失 |

终点预测模式不对网络输出归一化，也不额外加终点 MSE。
两组使用相同种子、网络、正常图像特征、优化器和时间分布；只改变输出参数化及结果目录。
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

两组结果分别写入 `output/unified_flow_image_text_clean_velocity/` 和
`output/unified_flow_image_text_clean_endpoint/`；每个目录按数据集、种子和时间区分运行。
保存 `config.yaml`、`base_flow.pt`、每阶段的 Flow/Bank、`metrics.csv` 和 `metrics.json`。
SwanLab 记录完整配置、Git 提交信息和阶段准确率。

### 重评估与兼容性

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml \
  --train_cfg configs/trainers/unified_flow.yaml \
  --eval_run output/unified_flow_image_text_clean_velocity/cifar100/原训练目录
```

仅可重评估此版本产生的相同预测模式权重；旧的文本→图像/单向/calibText 权重
必须重新训练。重评估时使用训练时的数据、CLIP、文本融合参数、积分步数、
`PREDICTION` 与 `TIME_EPS`，可调整评分参数及 Beta；不会进行训练。
严格选参应使用独立验证数据，不能用测试集选择最优设置。
只加载可信权重；旧版 PyTorch 如需不受限 pickle 加载，显式使用 `--trusted_checkpoints`。

### 验证

`test_image_text_flow.py` 用合成特征验证原始 BiMC 文本融合、
终点→速度公式及梯度、两方向积分、正常图像特征缓存、
单损失 FM、增量冻结与继续训练、旧 Bank 保留与重评估。
未执行真实数据集/GPU 准确率实验。

### 推理阶段 ODE 求解器对比

`FLOW.SOLVER` 可选 `euler`（默认）、`midpoint`、`heun`、`rk4`。
每步速度场调用次数分别为 1、2、2、4，正向图像到文本与反向文本到视觉
使用同一个求解器和 `FLOW.STEPS`。训练损失与网络权重不受求解器设置影响。
`TIME_EPS` 的截断仍作用于终点预测模式和速度预测模式的速度场接口，
所以四种求解器遵循相同的时间边界规则。

已有本版本的完整训练目录可以直接重评估，无需重新训练。重评估会用所选
求解器从已保存的文本与真实视觉原型重新生成每阶段视觉 Bank；它不会直接
沿用原目录中由 Euler 生成的视觉原型。`FLOW.STEPS` 也可在重评估中修改。

在仓库根目录运行四组实验（默认按顺序执行）：

```bash
python scripts/run_ode_solver_ablation.py \
  output/unified_flow_image_text_clean_velocity/cifar100/你的训练目录 \
  --data_cfg configs/datasets/cifar100.yaml --use_swanlab
```

对于终点预测网络，将路径替换为
`output/unified_flow_image_text_clean_endpoint/cifar100/你的训练目录`。
脚本在该训练目录的 `ode_solver_ablation/` 中保存四份配置、各组日志、
评估结果，以及列出 `final_acc`、`avg_acc` 和整个重评估用时
`eval_seconds` 的 `summary.csv`。用时包含特征提取和 Bank 重建。
可以用 `--solvers euler rk4` 只比较指定求解器。四种求解器计算量不同，
比较准确率时也应关注推理时间。
自己训练且来源可信的旧版权重若需要 pickle 加载，可在脚本命令后添加
`--trusted_checkpoints`，该选项会传给每一次重评估。

### 单损失 FM：增量冻结与继续训练

`unified_flow.yaml` 在增量阶段冻结 Flow，`unified_flow_incremental.yaml`
在每个增量阶段用当前新类支持图像特征和每个旧类一个真实视觉均值回放，
继续优化同一个图像→真实融合文本的 FM 损失。更新后重新生成全部已见类的
视觉 Bank；每个 Session 保存自己的 `flow_session*.pt`。CLIP 始终冻结。

```text
x_t = (1-t)*x_image + t*T_class
L_FM = mean ||v_theta(x_t,t) - (T_class-x_image)||²
```

基础阶段的 `LR` 与增量阶段的 `ADAPT_LR` 是峰值学习率。
每次优化更新前，学习率先线性预热，再按余弦曲线下降到峰值的
`MIN_LR_RATIO`（默认 0.01）。`WARMUP_RATIO` 默认 0.05；
基础训练按 `EPOCHS × 每轮批次数` 调度，增量阶段按 `ADAPT_STEPS`
调度。实际学习率保存在 checkpoint 的 `lr_history` 和
`incremental_training[*].lr` 中。

两组配置可在仓库根目录分别运行：

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/unified_flow.yaml --use_swanlab

python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/unified_flow_incremental.yaml --use_swanlab
```

对应的输出目录分别是
`output/unified_flow_image_text_clean_velocity/`、
`output/unified_flow_image_text_fm_incremental/`。它们是两个独立训练实验；
伪原型双损失和循环约束产生的旧权重不能代替这次的单损失训练。
实际准确率和学习率超参数需用验证集检验。

## Gaussian text-to-visual Flow with distribution replay

Run `python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/tangent_flow.yaml` after setting the dataset root and installing dependencies. This is a separate `METHOD: tangent_flow`; existing BiMC and unified-flow experiments remain available.

For each class, the code estimates an isotropic text Gaussian around the BiMC fused text prototype from its GPT description embeddings. It estimates visual covariance from normalized support-image CLIP features. Base class covariance is shrunk toward the Base pooled covariance. New classes blend their few-shot empirical covariance with a similarity-weighted transfer of Base class covariance. Only `(text center, text sigma, visual mean, visual covariance)` are retained for replay; no past image examples are stored. Covariances have a diagonal ridge to ensure Cholesky sampling is defined.

Base training and each incremental adaptation sample fresh text and visual endpoints every optimizer step. The objective is the squared flow-matching velocity error. `STEPS_PER_EPOCH` fixes the Base number of updates; `ADAPT_STEPS` fixes incremental updates and `REPLAY_FRACTION` controls the share of old-class distribution samples. This is unbounded resampling from estimated Gaussians, not new independent observations. The 8-block architecture is retained.

Each session forwards new text centers into visual space and appends their fused prototypes to the Bank. Existing Bank rows are not rewritten when the model is adapted. At inference, image features score against the Bank and are also integrated backward to score against untouched text centers. A reverse-then-forward cycle error reduces the text branch weight through `CYCLE_SCALE`. The original BiMC-style shared-covariance Mahalanobis probability applies to Base columns, and Description KNN probability to incremental columns, combined using `DATASET.ENSEMBLE_ALPHA`.

A reversible ODE tends to yield low cycle error even for out-of-distribution inputs, so the gate must be checked on held-out validation data before interpreting it as an ambiguity detector. Hyperparameters and accuracy have not been validated on the real datasets. Checkpoints from the earlier tangent-flow version are incompatible.

### Task 1 scale and gate diagnostics

`tangent_flow.yaml` now sets `NOISE_SCALE: 0.25` and `CYCLE_GATE: False`. Each incremental adaptation batch has an exact 50/50 old/new split for the default even batch size; sampling within each side is uniform over its classes. During Task 1 evaluation, the log and `metrics.json` report Task 0 and Task 1 accuracies for visual Bank only, reverse text only, Flow fusion without Cov/KNN, and the full score. All variants compete against every seen class. Base and incremental test-set mean cycle errors are reported each session. `COSINE_TEMPERATURE` applies the same multiplier to visual, reverse-text, and Description KNN cosine logits before softmax; the default remains 1.0 so the initial ablation measures the original probability scale. Try 20 or 50 only as separate validation runs. Removing Cov/KNN and improving accuracy implicates the geometry-prior fusion, but by itself does not prove that temperature mismatch is the sole cause.

### Flow-only 11-session experiment

The current `tangent_flow.yaml` evaluates only the visual Bank and reverse-text Flow branches. The runner calls `classify_bidirectional(..., ensemble_alpha=1.0)` for every session, skips the Mahalanobis pseudoinverse, and never computes Cov/KNN scores. Visual covariances remain necessary for Gaussian endpoint sampling and distribution replay during training. Results go to `output/gaussian_text_visual_flow_only/` so earlier ensemble runs are not overwritten. Task 1 visual-only, text-only, and Flow-only diagnostics remain available. The `DATASET.ENSEMBLE_ALPHA` value does not affect this method's current inference.

### Pure Flow versus Full at temperature 25

`configs/trainers/tangent_flow.yaml` is the pure Flow baseline (`USE_GEOMETRY_PRIORS: False`, `COSINE_TEMPERATURE: 1.0`). `configs/trainers/tangent_flow_full_t25.yaml` enables the original Base Cov / incremental Description KNN probability blend and applies `COSINE_TEMPERATURE: 25.0` to visual, reverse-text, and KNN cosine logits before softmax. The covariance logits retain BiMC's `/512` scaling. The two configurations write to separate output directories. Run both with the same dataset config and seed, then compare `Final session Acc` and `Average session Acc`. Temperature can affect the mix of visual and text probability branches even without Cov/KNN, so the pure Flow run deliberately retains the previous value of 1.0.

### Frozen Base Flow, separate probability scales (from commit 607ac3c)

`configs/trainers/tangent_flow_frozen_t500.yaml` keeps the original Base distributional FM training and freezes the Flow at every incremental session (`ADAPT_STEPS: 0`). New-class distribution statistics and Bank rows are still computed. Flow visual and reverse-text cosine logits use temperature 500; Description KNN uses 25; Mahalanobis logits use divisor 2048. These are independent inference settings. Results are isolated in `output/gaussian_text_visual_frozen_t500_knn25_cov2048/`. This experiment has not been run on CUB-200 in the development workspace.

### Previous-session velocity distillation and session-balanced replay

`configs/trainers/tangent_flow_distill_t500.yaml` is the incremental-training counterpart to `tangent_flow_frozen_t500.yaml`. At each adaptation stage it deep-copies the prior checkpoint's Flow, switches the copy to eval mode, and freezes its parameters. The student minimizes the existing FM loss plus `DISTILL_WEIGHT` times squared student–teacher velocity difference on the same old-class `(x_t,t)` points. Each default batch has 32 new-class and 32 old-class samples. At Task 1, all 32 old samples come from Base. From Task 2 onward, 16 come from Base and 16 from earlier incremental sessions; the latter choose an earlier incremental session uniformly, then a class uniformly within that session. The frozen configuration remains unchanged for comparison; both use Flow temperature 500, KNN temperature 25, and Cov divisor 2048.

### Replay comparison: uniform old classes and stronger distillation

`configs/trainers/tangent_flow_class_uniform_distill2.yaml` samples each old replay class uniformly among all classes seen before the current session and uses `DISTILL_WEIGHT: 2.0`. It retains the 32/32 old/new split, Base training, inference scales, and remaining hyperparameters of `tangent_flow_distill_t500.yaml`. The latter explicitly uses `REPLAY_STRATEGY: base_incremental_half` and distillation weight 1.0. The two outputs are isolated. Changes in accuracy cannot distinguish replay strategy from weight strength without additional one-factor ablations.

### Matched Flow/KNN temperature and one-step visual contrast

The three `*t500.yaml` configurations now use `COSINE_TEMPERATURE: 500.0` for Flow visual/text scores and `KNN_TEMPERATURE: 500.0` for Description KNN. Their output directories have changed so earlier temperature-25 results are preserved. Base training and incremental adaptation add `CONTRAST_WEIGHT: 0.1` times a cross-entropy loss: normalize `x_t + (1-t)v_theta(x_t,t)` and classify it against the normalized real visual means of the unique classes occurring in that batch. Repeated examples of the same class share one target column. `CONTRAST_TEMPERATURE: 20.0` scales this training loss's cosine logits independently of inference temperature. The incremental teacher distillation and replay schedules remain configurable. These numerical choices need validation on CUB-200; matching temperatures does not guarantee calibrated probabilities.

### All-seen contrast versus distillation-only baseline

`configs/trainers/tangent_flow_distill_knn_t500.yaml` isolates the 16 Base + 16 historical-increment replay, previous-flow distillation weight 1.0, Flow/KNN temperature 500, and Cov divisor 2048; its contrast weight is zero. `configs/trainers/tangent_flow_distill_t500.yaml` has the same settings plus one-step endpoint cross-entropy against **all currently seen class** visual means, with weight 0.1 and contrast temperature 20. Target columns follow memory row indices, so sampled class rows map to their global seen-class prototypes. The configurations write to separate directories and must be run independently to compare whether the all-seen contrast helps.

### Task 1 new-class visual Bank check

Task 1 evaluation now prints `visual_real_new` and `visual_flow_new` accuracies for both Task 0 and Task 1. Both scores use only cosine similarity in visual space against all seen classes; Base Bank rows remain the same fused prototypes. The Task 1 rows are replaced by either the unblended five-shot real visual means or the unblended forward-generated Flow text prototypes. This isolates the new-class Bank contribution from reverse-text and Cov/KNN scoring. The all-seen CE configurations now use `CONTRAST_WEIGHT: 0.05` (the previous value was 0.1, not 1.0) and write to new output directories.
