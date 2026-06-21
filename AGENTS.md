# AGENTS.md

## 项目
从彩色验证码图像中识别红色字符并按从左到右顺序输出。图像含背景噪声、颜色干扰及几何变形。

## 环境

```bash
pip install -r requirements.txt
```

PyTorch 需 CUDA 版本：请根据你的 CUDA 版本从 https://pytorch.org 安装对应版本。

## 项目结构

```
├── config.py           # 超参数（可调 batch_size, lr, 增强参数等）
├── dataset.py          # 数据加载 + RandAugment + MixUp
├── models/
│   ├── model.py        # ViT-B/16 Encoder + Transformer Decoder
│   └── decoder.py      # Decoder layers, ColorHead, CharHead
├── losses.py           # 损失函数和准确率计算
├── train.py            # 训练脚本（支持断点续训）
├── inference.py        # 推理生成 submission.csv
├── checkpoints/
│   ├── best_model.pth           # 最优模型权重（推理用）
│   └── checkpoint_epoch100.pth  # epoch100 完整状态（续训用）
├── train/              # 训练数据（images/ + labels.csv）
├── test/               # 测试数据（images/）
├── submission_sample.csv
└── submission.csv      # 当前模型预测结果
```

## 数据

- `train/images/` — 50,000 张图像
- `train/labels.csv` — 列：`filename`, `color`, `all_label`
  - `color`：5 字符，`r`=红色，`u`=非红色
  - `all_label`：5 字符，全部字符
- `test/images/` — 5,000 张图像

## 使用方式

### 训练

```bash
python train.py
```

- 首次训练从零开始，checkpoint 每 5 epoch 保存到 `checkpoints/`
- 断点续训：`python train.py --resume checkpoints/checkpoint_epochXX.pth`
- 快速测试：`python train.py --quick`（3 epoch 小数据集）
- 日志输出到 `training.log`，进度表在 `progress.csv`

### 推理

```bash
python inference.py
```

- 默认使用 TTA（水平翻转平均），可 `--no-tta` 关闭
- 输出 `submission.csv`，格式 `id,label`

### 续训示例

```bash
# 中断后从 epoch 80 恢复
python train.py --resume checkpoints/checkpoint_epoch80.pth
```

## 当前结果

- 模型：ViT-B/16 pretrained + 4层 Transformer Decoder
- 验证集 sample_acc：**93.33%**
- 测试集 accuracy：约 60%（泛化差距待改善）

## 已知问题 / 待改进

1. **33% 泛化差距** — 验证 93% → 测试 60%
   - 可能原因：RandAugment m=9 过强 + MixUp 50% 导致训练分布偏离测试
   - 建议降低 `RANDAUG_M` 到 5，禁用 hue 变换，降低 `MIXUP_ALPHA` 或关掉 MixUp
2. **Windows 兼容** — `num_workers=0`，Windows 下训练速度比 Linux 慢
3. **RTX 4060 Laptop (8GB)** — 约 13 分钟/epoch，100 epoch ≈ 22 小时
