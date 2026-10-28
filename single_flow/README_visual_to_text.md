# 单流视觉 → 文本：基类训练，增量冻结

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
`base_flow.pt` 为冻结 Flow；`bank_session*.pt` 为累计的**文本**候选库；
`metrics.csv/json` 保存各阶段总准确率、基类/增量类/调和准确率，口径沿用 BIMC。

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
python -m single_flow.visual_to_text classify --checkpoint vtt.pt --cache test.pt --bank text_seen.pt --steps 1 --batch-size 256 --out predictions.pt
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
文本追加保留旧行、错误阶段/标签/权重拒绝、两阶段 runner、三个数据集配置。
没有运行真实数据集/GPU 实验，默认参数不是已验证的最优参数；
基类到增量类的泛化和 sample/prototype 差异需实验验证。选参只使用基类验证数据。
