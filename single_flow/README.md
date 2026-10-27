# 单流文本原型 → 伪视觉原型（基类训练版）

基于 FlowComposer 改写 s1_flow.py，fm.py 保持原样。独立运行，不依赖原仓库的 dataset.py、Troika、Composer 或属性/物体分支。
来源：https://github.com/HKUST-LongGroup/FlowComposer
fm.py blob SHA: 9fb977d3131943928750a0deb9b92c6d18f3824b
原训练脚本 blob SHA: 7b05248fea6e1b9e43e4b9ee15f9573c39268551
附上游 MIT LICENSE。新脚本是任务适配实现，不代表作者原方法。

## 环境
Python 3.10+、PyTorch 2.6+。在你的服务器安装适配 CUDA 的 PyTorch；此包不包含 CLIP 权重、数据集或已训练权重。

## 特征准备：始终冻结同一个 CLIP
使用你现有的 CLIP 加载器和官方匹配的图像预处理。文件中的 encode_class_text(model, tokenize, class_names, descriptions) 和 encode_images(model, images) 会执行 eval、requires_grad_(False)、no_grad。
前者兼容提供 encode_text 的 CLIP 和返回 token Tensor 的 tokenizer；其他库接口需包装。视觉描述过长时先拆成适合 CLIP 上下文长度的句子，不要静默截断。描述由类别语义提供，不能用测试图像生成。

文本融合：normalize(alpha * normalize(template) + (1-alpha) * normalize(mean(normalize(description_i))))。
alpha 默认 0.5，训练后保存在 checkpoint，增量沿用。所有特征必须出自同一套冻结编码器、投影空间与图像预处理。

示例缓存创建（在你的现有数据加载代码中）：

```python
import torch
from s1_flow import encode_class_text, encode_images
# clip_model/tokenize/image_loader 由你的 CLIP 与数据集代码提供
# class_ids 可不连续，但要用 Python int，且保持文本行的顺序一致。
class_ids = [10, 23]
names = ['apple', 'banana']
descriptions = [['A round fruit with a short stem.'], ['An elongated fruit with a curved shape.']]
template, description = encode_class_text(clip_model, tokenize, names, descriptions)
image_features, image_labels = [], []
for images, labels in base_train_loader:
    image_features.append(encode_images(clip_model, images))
    image_labels.append(labels.cpu().long())
torch.save({
    'stage': 'base', 'split': 'train',
    'encoder_id': '你的CLIP架构-权重版本-预处理版本',
    'class_ids': class_ids,
    'template_features': template,       # [C,D]
    'description_features': description, # [C,D]
    'image_features': torch.cat(image_features), # [N,D] 基类训练图像
    'image_labels': torch.cat(image_labels),     # [N] int64，全局 class_id
}, 'base_train.pt')
```

阶段字段是校验，不会自动识别你错误标注的测试或增量数据。你必须按基类训练划分构建缓存。不能用测试集选 epoch、融合权重、深度或积分步数。

## 基类训练
```bash
python s1_flow.py train --cache base_train.pt --out output/base_flow.pt --target prototype --epochs 100 --blocks 24
```
默认目标是归一化图像特征均值再归一化得到的类别视觉原型，每个类别每轮一对训练样本、随机一个时间；基类数少时每轮更新次数很少。--target sample 则以每个真实图像特征为终点，类别频次由样本数决定。
输入 x0 是融合文本原型；x1 是视觉目标；xt=(1-t)x0+t*x1；监督速度 x1-x0。所有缓存都 detach，只训练一个速度网络。
默认仅使用速度平方误差（维度求和、batch 求均值）。--ce-weight 0.1 可选加入新定义的视觉原型 CE：将估计终点归一化后与基类真实视觉原型计算温度缩放余弦分数。此项是可选实验，不是照搬作者的文本 CE。
没有测试集评估或最佳测试指标选模，保存最后一轮权重和训练损失历史。默认参数只是起始设置，未经你的数据验证。hidden 默认特征维度，blocks 可调；保留原始 24 块可能较大。

## 生成基类原型
```bash
python s1_flow.py generate --checkpoint output/base_flow.pt --cache base_train.pt --out output/bank_base.pt --steps 8
```
这里只读取文本字段；无需图像即可生成。采用欧拉法，最后归一化；--steps 1 是单步。中间步不归一化，以保持与训练的欧氏直线路径一致。

## 增量阶段：不训练
为当前新增类别编码同样的两种文本，保存文本缓存，只需 encoder_id、class_ids、template_features、description_features 四个字段，不需要图像。
```bash
python s1_flow.py generate --checkpoint output/base_flow.pt --cache session1_text.pt --existing-bank output/bank_base.pt --out output/bank_session1.pt --steps 8
```
加载冻结基类 Flow，没有优化器或 backward。追加新类原型，旧类原型原样保留；重复 class_id 会报错。后续阶段继续传上一阶段 bank。所有阶段使用同一个 checkpoint 和生成设置。
每类一个确定性原型；sample 模式也不是同文本随机生成多个样本。基类训练到新类的泛化需要实验，不能保证保留零样本性能或涨点。此版本不使用增量 support 图像；若要融合少样本视觉信息，那是额外机制。

## 分类
用冻结 CLIP 提取测试图像特征，另存 test.pt，字段 encoder_id、image_features，可选 image_labels。
```bash
python s1_flow.py classify --cache test.pt --bank output/bank_session1.pt --out output/predictions.pt
```
归一化测试特征与全部已见类伪视觉原型算余弦相似度，argmax 映射回全局 class_id。分类输出不会更新任何模型或原型。最终精度只用于报告。

## 测试
```bash
python -m unittest test_single_flow.py
```
测试使用小维度合成特征，覆盖学习下降、推理不更新权重、旧类缓存保留、非连续标签与非法训练划分。不能替代真实数据集复现。

