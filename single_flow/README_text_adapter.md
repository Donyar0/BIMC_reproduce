# 文到图 Flow 的 VDT 文本聚合 Adapter 消融实验

目标分支：`single-flow-text-to-visual`。基于 `feature/bimc-vdt-self-attention`
**原始 LinearTextAdapter**：由 CLIP 编码每个类别的
`"a photo of <class>." + description`，线性残差+LayerNorm+masked mean
得到 `T_c^adapter`；不是该分支的视觉 Self-Attention 模块。

**仅基类阶段额外训练 Text Adapter**：对基类训练图像做
`CE(100 * normalized_image @ adapted_text_prototypes.T, label)`，
20 epochs / batch 256 / lr 1e-4。训练完成立即冻结 Adapter。
CLIP 视觉和文本编码器全程冻结。

文到图 FM（保留速度参数化与原损失）：

- 训练起点 `x0=T_c^adapter`（取代 BiMC 原 Prompt/Description 融合文本）。
- 训练终点 `x1=normalize(BETA*T_c^adapter+(1-BETA)*V_c^BiMC)`
  （同时替换 BiMC 图文融合终点里的文本部分，不只是替换起点）。
- 速度监督始终 `||v_theta((1-t)x0+tx1,t)-(x1-x0)||^2`，
  默认 `TARGET=prototype`, `CE_WEIGHT=0`，无额外损失。
- 增量阶段 **Text Adapter、Flow 都冻结**；新类仅使用 Adapter
  由新类名称/描述生成 `T_c^adapter`，经 Flow 8 步积分生成分类原型。
  当前新类真实视觉原型只参与构建统计量，不用于生成或拼接分类器；
  历史生成原型保持不变。
- 分类方法仍为 CLIP 测试图像特征与已见类生成原型做余弦相似度
  和 argmax；无 BiMC ensemble/MGC/KNN。

```bash
git fetch github
git switch single-flow-text-to-visual
git pull --ff-only github single-flow-text-to-visual

# 原方案（不使用 Adapter，保持之前的速度预测和 BiMC 图文融合终点）
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/single_flow.yaml

# 新方案（VDT Adapter 替换文到图的全部文本原型）
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/single_flow_adapter.yaml

# 合成数据测试
python -m unittest discover -s single_flow -v
```

输出目录为 `output/single_flow_adapter/<dataset>/<run>`，保存
`base_text_adapter.pt`、`base_flow.pt`、`bank_sessionN.pt` 和
`metrics.*`；Flow 模型与 bank 记录 Adapter 权重的 SHA256，
防止将不同 Adapter 或旧文本原型与此 Flow 混用。
原 `single_flow.yaml` 不变。旧缓存需重新生成；只使用本配置启动
完整实验即可自动计算所需文本特征。尚未在真实数据集/GPU 验证。
