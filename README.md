# BiMC reproduce

## BiMC 与 BiMC + 10-Crop 同条件对比

在原有环境中安装 `requirements.txt`，并修改 `configs/datasets/*.yaml` 中的
`ROOT` 和 `GPT_PATH` 后运行：

```bash
python main.py \
  --data_cfg configs/datasets/cifar100.yaml \
  --train_cfg configs/trainers/bimc.yaml \
  --compare_ten_crop
```

CUB200、miniImageNet 可替换 `--data_cfg`。不加 `--compare_ten_crop` 时保持原有
BiMC 中心裁剪流程。也可在配置中设置 `DATALOADER.TEST.COMPARE_TEN_CROP: True`。

同一次运行中，每个 session 只抽取一次支持集、构建一次统计量；两组评测共享
few-shot 样本、CLIP 权重、文本描述、视觉原型、协方差、校准和融合参数以及测试样本。
支持集始终使用原来的中心裁剪，不把十个视角当成额外训练样本。

测试预处理沿用原来的 bicubic 短边缩放到 224：

- BiMC：224 × 224 中心裁剪。
- BiMC + 10-Crop：四角和中心五个裁剪，加上水平翻转后的五个裁剪；每个视角使用
  相同 CLIP 归一化和完整 BiMC 分类器，最后对十个最终预测概率取算术平均。

为保持原始对照条件，没有额外改为短边 256。对于 CIFAR100 等正方形图片，缩放后
五个空间裁剪会重合，因此实际只有原图与水平翻转两个独立视角；不要把它解释为
十个不同的空间区域。若实验要求 resize=256 的标准十裁剪，应同时调整两组测试
预处理，并与这里的原始 resize=224 结果分开报告。

10-Crop 按视角依次推理，模型单次 batch 大小保持不变；输入张量和数据加载内存
会增加，推理时间通常增加。`bimc.yaml` 保持 `USING_ENSEMBLE: False`；若选择
`bimc_ensemble.yaml`，两组都会开启相同的 ensemble，应单独报告该实验。

终端逐 session 输出两组准确率和百分点差值，以及最终准确率序列。
可追加 `--use_swanlab`：原有指标对应中心裁剪，`ten_crop/*` 对应十裁剪，包含
`mean_acc`、`base_avg_acc`、`inc_avg_acc`、`harmonic_acc`、`elapsed_time`、
`delta_mean_acc`、`final_acc` 和 `avg_acc`。

运行不需要真实数据或下载 CLIP 权重的回归测试：

```bash
python -m unittest discover -s tests -v
```
