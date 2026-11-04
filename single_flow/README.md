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
作为终点，并用现成的 visual_prototypes 作为可选 CE 的分类参照。
训练 xt=(1-t)x0+t*x1，监督速度 x1-x0；仅训练速度网络。
默认只有速度平方误差（维度求和、batch 求均值），--ce-weight 可开启视觉原型 CE。
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
非连续标签、sample/CE 路径，以及共用文本融合公式。未运行真实数据集或 GPU 训练。

## 双向共享 Flow 实验

使用 `configs/trainers/unified_flow.yaml` 运行：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/unified_flow.yaml
```

此配置以基类单样本视觉特征训练共享速度场；增量阶段混合当前支持集样本与旧类文本/真实视觉均值原型重放，更新 Flow 后重新生成全部已见类原型。分类时将生成原型与真实视觉均值按基类和增量类的权重融合，同时对测试特征进行一次反向 Euler 校准，并加权融合视觉与文本余弦分数。每阶段保存 Flow 和原型 bank；仅保留历史类均值，不保存历史图像特征。

`ADAPT_STEPS`、`ADAPT_LR`、`ALPHA`、`BETA_BASE`、`BETA_INC` 和 `GAMMA` 控制微调和融合，初值是实验起点，应使用验证数据选择。`ADAPT_STEPS: 0` 可关闭增量微调。实验需要 GPU、数据集、CLIP 权重及配置中的正确数据路径；该方法的效果需要真实数据验证。

双向余弦得分先在全部已见类别上做 softmax。启用 `TRAINER.BiMC.USING_ENSEMBLE` 时，沿用 BiMC 的融合方式：基类概率与马氏距离概率融合，增量类概率与描述特征 KNN 概率融合，权重为数据集配置的 `ENSEMBLE_ALPHA`。协方差来自各阶段真实训练/支持图像的 BiMC 收缩协方差，按阶段类别数加权；马氏距离使用双向 Flow 校准后的视觉原型。当前 Flow 对相同文本原型输出确定结果，重复生成不能提供可靠的类内协方差，因此不将重复伪原型用于估计协方差。

`FLOW.LOGIT_SCALE` 是双向余弦得分 softmax 前的正数乘数，默认 1 保持原行为；它与训练损失中的 `FLOW.TEMPERATURE` 相互独立。可在验证集比较 1、5、10 等值，并联合搜索 `FLOW.GAMMA`（例如 0.4、0.5、0.6、0.7）和 `FLOW.BETA_INC`（例如 0.3、0.4、0.5、0.6）。无 Ensemble 时正数缩放不改变 argmax；启用 Ensemble 时会改变 Flow 分支与协方差、KNN 概率的相对影响。softmax 温度不能保证各分支校准一致，需在验证集检查基类和增量类准确率。不要用测试阶段结果选择参数。
### 当前双向 Flow 默认分数与 SwanLab 追踪

`unified_flow.yaml` 现在默认设置 `USING_ENSEMBLE: True`；若要复现纯 UR-Flow 评分，将其设为 `False`。CIFAR100 的 `ENSEMBLE_ALPHA` 在 `configs/datasets/cifar100.yaml` 中配置。基类使用 UR-Flow + 马氏距离，增量类使用 UR-Flow + 类别描述最大相似度。马氏距离的协方差伪逆和原型二次项每个 Session 仅计算一次，结果与原始公式的 softmax 数值等价。

在仓库根目录运行，且先用 `swanlab login` 登录（首次使用）：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml \\
  --train_cfg configs/trainers/unified_flow.yaml \\
  --use_swanlab --swanlab_project BIMC_reproduce \\
  --swanlab_experiment UR-Flow-CIFAR100-seed1 \\
  --swanlab_mode online
```

SwanLab 配置中自动保存全部 YAML 配置、数据集和训练配置路径，以及 `git.git_branch`、`git.git_commit`、`git.git_dirty`、`git.git_remote_url` 和 `git.git_commit_url`。其中 GitHub 链接仅在 origin 为 GitHub 仓库时构建，避免记录包含凭证的远程 URL。每阶段的准确率、基类和增量类指标以及最终准确率由 `log_metrics` 上传；本地仍保存 `config.yaml`、`metrics.csv`、`metrics.json`。

运行前建议 `git status --short` 确认工作区干净，并在 SwanLab 配置里核对 `git.git_commit`。不要在运行中途更新分支，否则同一次实验中的代码版本可能变化。

### 不重新训练地评估已有权重

训练结果目录必须包含原始 `config.yaml` 以及每阶段的 `flow_session*.pt`、`bank_session*.pt`。在原训练配置的副本中修改 `ALPHA`、`GAMMA`、`BETA_BASE`、`BETA_INC` 或 `LOGIT_SCALE`（也可以切换 `USING_ENSEMBLE`），然后运行：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/unified_flow.yaml \
  --eval_run output/unified_flow/cifar100/原训练目录
```

可添加原有的 `--use_swanlab` 等参数。该入口不会训练或修改原结果，另建 `eval_...` 结果目录及 `source_run.txt`。它重新读取测试数据并提取特征；启用 Ensemble 时还会重新提取训练/支持集统计，并与已保存的真实原型核对。须使用训练时相同的数据、随机种子、配置与 CLIP 权重。结果仍然是测试集结果；严格选参需要独立验证集，当前入口没有验证集支持。只加载可信来源的权重文件。

如果旧版 PyTorch 的安全加载器无法读取自己训练的 checkpoint，可在上述命令添加 `--trusted_checkpoints`。该选项允许 pickle 执行代码，仅对来源可信的权重使用，不要用于下载的不明文件。

### 增量类马氏距离融合

启用 `USING_ENSEMBLE` 后，`FLOW.INC_ENSEMBLE_MODE` 可设为 `knn`（默认，原 BiMC 描述近邻）、`mahalanobis`（增量类使用共享协方差马氏距离），或 `blend`（两种辅助概率加权）。`INC_COV_WEIGHT` 控制 blend 中马氏概率比例；`MAHA_TEMPERATURE` 是所有类别马氏 logits 的正数除数，默认 512，调整它同时影响基类。三条分支先各自在所有已见类上 softmax，再按类组融合。协方差使用真实图像统计而非伪原型估计。改变这些评分设置可使用 `--eval_run` 重评已有权重；该入口仍使用测试集，应在独立验证集选参。
