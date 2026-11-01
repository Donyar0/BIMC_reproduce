# 2026-11-01：纯 Flow Matching + 完整 Flow 推理（当前分支）

当前默认流程为 **普通基类 FM 训练 + 真实增量 FM 微调 + 完整 Flow 推理**，无 CE、无伪增量：

- 伪增量模式默认 **关闭**（`FLOW.META.ENABLED: False`），不会执行任何 episode。保留的实验性代码不参与当前训练。旧实验功能说明：每个 episode 把基类划分为 50 个伪旧类、10 个伪新类。内循环只用新类 5-shot Support 的 FM 更新临时参数；外循环只计算新类 Query FM、旧类 Query FM 和原始参数上的旧类 FM，权重分别为 `NOVEL_FM_WEIGHT=1`、`OLD_FM_WEIGHT=0.5`、`BASE_FM_WEIGHT=1`。采用一阶元梯度；它不是仅设置 CE 系数为零的旧代码。
- 配置 `FLOW.META.ENABLED: False`：普通基类训练只用 FM；`FLOW.CE_WEIGHT` 必须为 0。
- 配置 `FLOW.INC.ENABLED: True`：每个真实增量 session 仍使用当前 5-shot 图像和融合文本进行 FM 微调；`FLOW.INC.CE_WEIGHT` 必须为 0。如果希望增量不训练，请设为 `False`。
- `FLOW.ALPHAS: [1.0]` 强制完整更新，不扫描或混合 alpha。默认 `FLOW.STEPS: 1`：`normalize(z + v_theta(z, 0))`；如需完整区间多步数值积分，可调整 `FLOW.STEPS` 为 8（仍是完整更新而非 0.1/0.2 的混合）。
- 结果输出方式不变：`base_flow.pt`、`flow_sessionN.pt`、`metrics.csv/json`、各阶段 ACC 和 Task-wise ACC。
- 注意 **必须重新训练** 才能比较 FM-only 权重；旧 checkpoint 不会因为修改 YAML 而自动删除其训练过的 CE 信息。加载旧 checkpoint 仅用于对照，不等于 FM-only 实验。

运行：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/visual_to_text.yaml --use_swanlab
```

本次仅完成代码与配置更新；尚未在真实 CIFAR100/CUB/miniImageNet 数据集上执行 GPU 训练和精度验证。

---

# 单流视觉 → 文本：基类训练 + 可选增量微调

这是独立的新方法，原 `single_flow` 文本 → 伪视觉原型方法保持不变。
共享原版 `fm.py`，无需属性流、物体流或 Composer。

## 一键完整实验

从仓库根目录运行，先设置 dataset YAML 中的数据路径并准备原有 CLIP 权重、描述和依赖：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/visual_to_text.yaml
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/visual_to_text.yaml
python main.py --data_cfg configs/datasets/miniimagenet.yaml --train_cfg configs/trainers/visual_to_text.yaml
```

可加原有 `--use_swanlab` 参数。结果在 `output/visual_to_text/<dataset>/<run>/`：
`base_flow.pt` 为基类 Flow；开启增量训练后 `flow_sessionN.pt` 为各阶段 Flow；`bank_session*.pt` 为累计的**文本**候选库；
`metrics.csv/json` 保存各阶段、各 α 的总准确率、基类/增量类/调和准确率，口径沿用 BIMC。

## 实验一：复用已有权重，降低 Flow 更新强度

不需要重新训练。将 `--flow_checkpoint` 设置为已有 `base_flow.pt`：

```bash
python main.py \
  --data_cfg configs/datasets/cifar100.yaml \
  --train_cfg configs/trainers/visual_to_text.yaml \
  --flow_checkpoint /path/to/base_flow.pt \
  --flow_alphas 0 0.1 0.2 0.5 1
```

如果旧版 PyTorch 加载**自己训练的可信权重**时出现
`Weights only load failed / Unsupported operand 71`，再添加
`--flow_trusted_checkpoint`，显式使用 `weights_only=False`。
默认不自动回退到普通 pickle 加载；来源不可信的文件不要开启此选项。

也可在 YAML 中设置 `FLOW.CHECKPOINT`、`FLOW.ALPHAS` 和
`FLOW.TRUST_CHECKPOINT`。命令行参数覆盖 YAML。
`CHECKPOINT` 非空时跳过训练，路径错误直接报错，不会偷偷重新训练；
除非显式启用了 `PROTOTYPE_CACHE_DIR` 导出缓存，否则也不提取基类训练图像。
`CHECKPOINT` 为空时保留原有训练流程：只训练一次，再用同一模型扫描所有 α。

令 `v = normalize(CLIP(image))`，`f = normalize(Flow(v))`，使用
`z_alpha = normalize((1-alpha)*v + alpha*f)` 与同一文本库计算余弦分数。
两个端点先各自归一化，避免特征长度改变 α 的含义。
每批图像只提取一次 CLIP 特征、积分一次完整 Flow；中间 α 共用该终点。
α 不修改 Euler 步长、积分时间或训练损失。仅评估 α=0 时不执行 Flow 网络。

| α | 含义 |
|---|---|
| 0 | 原始 CLIP 图像特征，对照组 |
| 0.1、0.2 | 较弱更新 |
| 0.5 | 两个归一化端点等权融合 |
| 1 | 原来的完整 Flow 输出 |

注意 α=0 仍沿用当前文本融合配置；若 `TEXT_CALIBRATION=True`，
它是“CLIP 图像 + 融合文本”的对照，并非关闭描述融合的标准单模板 CLIP。
复用权重时保持 CLIP、类别划分/随机种子、文本描述和融合配置与训练一致。
程序检查方向、encoder_id、基类 ID、种子和特征维度；旧 checkpoint 没有记录完整的
文本融合配置，无法自动验证这一部分。加载后以 checkpoint 中的网络结构和训练记录为准，
修改 YAML 的 EPOCHS/BLOCKS/CE_WEIGHT 不会改变已训练的权重。

输出仍在新的 `output/visual_to_text/<dataset>/<run>/`，不会覆盖旧权重：

- `metrics.csv`：每行是一个 `(task_id, alpha)`；CIFAR100 的 9 个阶段 × 5 个 α 共 45 行。
  `mean_acc / base_avg_acc / inc_avg_acc / harmonic_acc` 都是全部已见类别共同竞争的正式指标。
- `novel_only_acc`：所有已见增量类样本，只在所有已见增量类候选中分类。
  `current_only_acc`：当前新增类样本，只在当前新增类候选中分类。
  这两项使用标签限制候选集合，仅供诊断，不能替代正式 FSCIL 指标。
- `novel_to_base_pct`：全部已见新类样本预测为基类的百分比；
  `current_to_base_pct`：当前新增类的同类统计。
  `current_base_max_sim / current_novel_max_sim`：在当前新增类样本上的两组最大余弦分数均值。
  `current_class_acc` 保存当前每类在全部已见候选中的准确率。
- Task 0 或没有对应测试样本时，诊断准确率写为 JSON `null` / CSV 空值。
- `metrics.json` 的 `by_alpha` 保存各 α 的最终与跨阶段平均准确率。
  `evaluation.json` 保存来源 checkpoint、真实训练参数、结构、α 和推理配置。
  `elapsed_time` 是该阶段所有 α 共用的总时间，不能把 5 行时间相加。

优先比较 α=0 与 α=1，以及较小 α 的新类/基类/调和准确率和新类内部诊断。
扫描用于分析；不要用增量测试集挑选 α 后再将最大值作为无偏的正式成绩，正式选参用验证集。

## 数据与训练方向

- CLIP 始终冻结；文本复用 BIMC 的模板与描述编码及 `fused_text_prototype()`。
  使用数据集原有模板（不强制所有数据集改成同一句模板），描述仍是文本特征。
  `TEXT_CALIBRATION` / `LAMBDA_T` 控制是否融合及描述权重。
- 默认 `FLOW.SOURCE: sample`：每张**基类训练图像**归一化特征为 x0，
  其真实类别的归一化文本原型为 x1。`prototype` 可作对照：基类图像均值原型为 x0。
- 随机 t，xt=(1-t)x0+t*x1；网络输入 xt,t，输出 v。
  Lfm=mean_batch(sum_dim((v-(x1-x0))²))。
- 估计终点 normalize(xt+(1-t)*v)，与所有基类文本计算余弦分数/temperature，
  以真实类别计算 CE。总损失 Lfm+CE_WEIGHT*CE；默认 CE_WEIGHT=0.1，可设0消融。
  估计终点保留梯度；原始 CLIP 缓存不参与梯度。
- 只在 session 0 创建优化器并训练。增量阶段只编码新类文本、追加候选库，
  旧类文本逐元素保留，不读取增量 support 图像，不计算视觉校准或协方差。

## 推理与分类

每张测试图像先经冻结 CLIP，然后从 z0=normalize(image_feature) 出发，
欧拉更新 z_{k+1}=z_k+v(z_k,k/K)/K，最终归一化。
默认 STEPS=1，即 normalize(z0+v(z0,0))；可改为8等做多步对照。
中间步不归一化，与训练的直线插值一致。更新后的图像特征与**全部已见类文本**
计算余弦相似度，取最大值分类；测试标签只用于统计指标，绝不作为 Flow 输入。

这里更新的是每张测试图像的特征，不是提前存储的类别视觉原型。
本方法没有伪视觉原型库，不融合 BIMC 的 beta、协方差、KNN 分类器。

## 独立缓存接口

兼容现有 BIMC 导出的缓存。基类 sample 训练必需：
`stage='base', split='train', encoder_id, class_ids, text_prototypes [C,D],
image_features [N,D], image_labels int64 [N]`。
prototype 训练改用 `visual_prototypes [C,D]`，行序与 class_ids 一致。
文本已由 BIMC 融合，Flow 不再次编码或融合；各阶段必须使用同一 CLIP 和融合配置。
阶段/split 标记用于防误用，不能自动识别被错误标注的数据来源。

```bash
python -m single_flow.visual_to_text train --cache base.pt --out vtt.pt --source sample
python -m single_flow.visual_to_text build-bank --checkpoint vtt.pt --cache base.pt --out text_base.pt
python -m single_flow.visual_to_text build-bank --checkpoint vtt.pt --cache novel.pt --existing-bank text_base.pt --out text_seen.pt
python -m single_flow.visual_to_text classify --checkpoint vtt.pt --cache test.pt --bank text_seen.pt --steps 1 --alpha 0.1 --batch-size 256 --out predictions.pt
```

novel.pt 只需 `encoder_id, class_ids, text_prototypes`；每次仅包含新增类。
test.pt 需 `encoder_id, image_features`，`image_labels` 可选。
CLI 保存的 accuracy 为0–1；完整 runner 的指标为百分数。
权重有 direction 标记，新方法拒绝加载旧文本到视觉权重，必须重新基类训练。
`encoder_id` 沿用原缓存约定；自定义权重需自行加入精确版本标识。

## 验证与限制

```bash
python -m unittest discover -s single_flow -v
```

合成测试检查学习方向、真实图像输入分类、单/多步推理、冻结权重、
文本追加保留旧行、错误阶段/标签/权重拒绝、两阶段 runner、三个数据集配置；
并检查 α 两端和中间混合、单次 Flow 共用、可信加载显式开关、复用权重跳过训练图像与训练、
正式评估与新类内部诊断分离，以及多 α 结果持久化。
没有运行真实数据集/GPU 实验，默认参数不是已验证的最优参数；
基类到增量类的泛化和 sample/prototype 差异需实验验证。选参只使用基类验证数据。


## 新增：每阶段 5-shot 增量训练

默认 `FLOW.INC.ENABLED=True`。基类 Session 0 的训练完全保留；Session 1+
只读取**当前新增类别**的训练 support，默认每类 5-shot。CLIP 冻结，
以前一阶段 Flow 作为初始化，小学习率继续训练：
`L_inc = L_FM(new support) + INC.CE_WEIGHT * L_seen_CE`。
其中 FM 目标是当前类别 BiMC 融合文本原型；CE 比较的是
**所有已见类别**的融合文本原型（基类、此前增量类和当前新类），
不是仅当前 5 类。增量默认 EPOCHS=10，BATCH_SIZE=25，LR=1e-5，
CE_WEIGHT=0.1。CUB200 每阶段为 10 类 × 5 张，因此有 50 张 support。
每阶段微调结束保存 `flow_sessionN.pt`。不使用旧类别图像或测试图像，
也暂未增加旧类蒸馏约束，故旧类遗忘仍可能发生。

设置 `FLOW.INC.ENABLED=False` 恢复原来的增量冻结对照；
`INC.CE_WEIGHT=0` 用于 FM-only 消融。可使用
`--flow_checkpoint /path/to/base_flow.pt` 跳过基类训练、继续增量微调。
正式分类始终在全部已见类别之间进行；请比较 Base/Incremental/
Harmonic Acc 和 novel-to-base 比例，不用 current-only 代替正式指标。

## 方案二：伪增量 Episode 元训练（新增）

在 `configs/trainers/visual_to_text.yaml` 中将 `FLOW.META.ENABLED: True`。
这会**替换 Session 0 的普通 fit()**，从随机初始化开始，在真实基类训练集
中反复构造伪增量 episode。不是在已经见过全部 60 类的旧基类 Flow 上
划分伪新类；开启该模式时不能传 `--flow_checkpoint`。

每个 episode 在真实的 60 个基类中随机选 10 个**伪新类**，
剩下 50 个作为**伪基类**。每个伪新类严格分开
5-shot Support 和 5 张 Query（均来自基类训练图像，不接触测试集）。
伪基类抽样用于外层 FM 监督和旧类 Query，伪新类 Support 仅用于
内层 1 步虚拟 FM+Seen-CE 更新。其外层优化是在**虚拟更新后的参数**
上，计算新类 Query CE、旧类 Query CE，并组合初始参数上的伪基类 FM Loss：

```
inner = FM(pseudo-novel support) + inner_ce_weight * seen_CE(support)
theta_fast = theta - inner_lr * grad(inner)  # 梯度截断的 first-order MAML
outer = query_ce_weight * CE(theta_fast, novel query)
      + old_ce_weight * CE(theta_fast, pseudo-base query)
      + fm_weight * FM(theta, pseudo-base examples)
```

以上 Query 分类使用 t=0 更新后的图像特征，候选文本为本次
真实基类中的所有类别（包含伪新类）。为节省计算量使用**一阶近似**，
不计算二阶元梯度，因此它不是完整 MAML。所有 episode 每轮随机重选伪新类。
`meta_history` 记录训练损失，最终生成 `base_flow.pt`。
真实增量阶段依然执行原有的每阶段 5-shot FM+Seen-CE 微调
（`FLOW.INC.ENABLED: True`），所有常规指标与 SwanLab 记录保持原样。

**重要约束：** 新增的 Query 来自真实**基类训练集**，只在一个 episode 内
作为独立查询；在整个 meta 训练中所有 60 个基类会轮换角色。因此，
这些伪新类不是彻底留出的跨训练类别验证集，不能把它们的 Query ACC
当作真正未见类别泛化成绩。严谨评估仍以真实 FSCIL 增量测试类别为准。

运行（不要传旧的 `base_flow.pt`）：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml \
  --train_cfg configs/trainers/visual_to_text.yaml \
  --use_swanlab --swanlab_project BIMC_reproduce \
  --swanlab_experiment pseudo_incremental_meta
```

用于对照：将 `FLOW.META.ENABLED: False` 恢复普通基类训练，
同时保留 `FLOW.INC.ENABLED: True`。关闭增量微调则设
`FLOW.INC.ENABLED: False`。建议先用 EPOCHS=2、
EPISODES_PER_EPOCH=3 做短程冒烟测试；默认 20×30 episode，
每个 episode 包含多次 Flow 前向及反向，计算显著高于普通基类 FM。
