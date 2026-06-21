# 红色字符识别方案设计（Transformer 架构）

## 1. 问题分析

### 1.1 任务特点
- 每张图恰好 5 个字符，水平排列，位置相对固定——这是强结构先验
- 输出只关心**红色**字符，颜色识别和字符识别是**两个协同子任务**
- 图像含背景噪声、颜色干扰、几何变形——需要全局感受野来鲁棒建模
- 字符集 36 类（0-9 + A-Z），规模适中

### 1.2 为什么用 Transformer

| CNN 的局限 | Transformer 的优势 |
|-----------|-------------------|
| 局部感受野，需要堆深层才能建模长距离依赖 | Self-attention 天然全局感受野，一步捕获跨字符关系 |
| 空间归纳偏置强（平移等变性），对几何变形敏感 | 位置编码解耦空间信息，对几何扰动更鲁棒 |
| 特征图需手动池化/切分来关联 5 个位置 | Query tokens 通过 cross-attention 自动定位对应字符 |

### 1.3 可用的监督信号
训练标签给出了 `color` 和 `all_label`，任务可分解为：
1. 对 5 个位置各预测字符（36 分类）
2. 对 5 个位置各预测颜色（2 分类，r/u）
3. 用颜色预测对字符做 mask，输出红色字符字符串

---

## 2. 核心架构：Query-based ViT + Transformer Decoder

### 2.1 整体流程

```
输入图像 (H×W×3)
      │
      ▼
┌──────────────────────┐
│   Patch Embedding     │  patch_size=16, 展平为序列
│   + Position Embedding│
│   + [CLS] Token       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   ViT Encoder (L层)   │  Self-attention → 全局上下文特征
│   输出: patch tokens   │  shape = (N_patches, D)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Transformer Decoder │  M 层 (建议 M=2-4)
│                       │
│   Input:              │
│   - 5 个 Position     │
│     Query Tokens      │  ← 可学习参数, shape=(5, D)
│   - patch tokens      │  ← 来自 Encoder 的 KV
│                       │
│   每层:               │
│   - Self-attention    │  → queries 间交互
│   - Cross-attention   │  → queries 关注图像区域
│   - FFN               │
└──────────┬───────────┘
           │
    输出: 5 个 refined query tokens, shape=(5, D)
           │
    ┌──────┴──────┐
    ▼              ▼
┌───────┐     ┌───────┐
│颜色头  │     │字符头  │
│(共享)  │     │(共享)  │
└───┬───┘     └───┬───┘
    │              │
    ▼              ▼
 5×1 logits    5×36 logits
 (r/u prob)    (char prob)
    │              │
    └──────┬───────┘
           ▼
    mask 过滤, 从左到右拼接
           ▼
    "RJ2"
```

### 2.2 关键设计细节

#### Position Query Tokens
5 个可学习的 embedding 向量，每个负责对应**从左到右的一个字符位置**。初始化方式：

```python
# 方案 A: 零初始化, 让 Decoder 从零学习
query_tokens = nn.Parameter(torch.zeros(5, d_model))

# 方案 B (推荐): 用正弦位置编码按水平坐标初始化
# 第 i 个 query 对应水平位置 approximately x = (i + 0.5) / 5 * W
position_hints = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])  # 归一化 x 坐标
query_tokens_init = sinusoidal_pe(position_hints, d_model)
query_tokens = nn.Parameter(query_tokens_init)
```

**为什么 query 间用 self-attention？** 字符间存在上下文依赖（如相邻字符可能组成常见组合），self-attention 让 query 间交换信息，提升一致性。

#### Cross-Attention 的作用
每个 query token 通过 cross-attention 自动聚焦到图像中对应位置的区域，**隐式完成定位**，无需额外检测头或 ROI 操作。

```
CrossAttention(Q=query_tokens, K=patch_tokens, V=patch_tokens)
→ 每个 query 自适应地从全图 patch 中聚合信息
```

#### 输出头（颜色和字符分离）

```
                                  refined query (D)
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
             Linear(D, D//2)                 Linear(D, D//2)
             GELU + LayerNorm               GELU + LayerNorm
             Dropout(0.2)                   Dropout(0.2)
                          │                              │
                          ▼                              ▼
                Linear(D//2, 1)              Linear(D//2, 36)
                颜色 logit (5个)             字符 logit (5个)
```

两个输出头共享 query tokens 但**参数独立**。颜色头和字符头对所有 5 个 query **共享权重**——因为字符类别和颜色语义是位置无关的。

---

## 3. Encoder 选择

### 3.1 ViT-B/16（推荐）
- 参数量 ~86M，ImageNet-21K 预训练
- patch_size=16：对 224×224 输入产生 14×14=196 个 patch tokens
- 12 层 Transformer，d_model=768，12 heads
- 是 DETR 原论文使用的 encoder

### 3.2 其他候选

| Encoder | 特点 | 适用场景 |
|---------|------|---------|
| ViT-L/16 | 更大容量（~307M） | GPU 充裕、追求极致精度 |
| Swin-B | 层次化特征，多尺度 | 需要多尺度特征的变体 |
| DeiT-III | 更好的训练策略 | 如果从头训或弱预训练 |
| BEiT-v2 | MIM 预训练 | 对噪声/遮挡更鲁棒 |

### 3.3 多尺度特征融合（进阶）

标准 ViT 输出单一分辨率的 patch tokens。为提升对小字符和细粒度颜色差异的感知能力，可替换为 Swin Transformer，利用其金字塔特征：

```
Swin-Tiny
  Stage 1: 56×56 → 特征保留空间细节（颜色判断）
  Stage 2: 28×28
  Stage 3: 14×14
  Stage 4: 7×7  → 高层语义（字符识别）

→ 多尺度 FPN 融合 → 扁平化为 multi-scale token 序列
→ 送入 Decoder
```

---

## 4. 损失函数

### 4.1 主损失

```
char_loss  = CrossEntropyLoss(char_logits, char_labels)   # (B, 5, 36) → (B, 5)
color_loss = BCEWithLogitsLoss(color_logits, color_labels)  # (B, 5, 1)  → (B, 5)
total_loss = λ_char * char_loss + λ_color * color_loss
```

- `λ_char = λ_color = 1.0` 起步，可视验证集调整
- 颜色分支用 `pos_weight`（依据训练集 r/u 比例）

### 4.2 辅助损失（Decoder 各层输出监督）

类似 DETR，对 Decoder 的**每一层**都加监督，加速收敛：

```python
for layer_output in decoder_layers_outputs:  # [layer0, layer1, ..., layerM]
    char_loss_layer  = CrossEntropyLoss(layer_output.char_logits, labels)
    color_loss_layer = BCEWithLogitsLoss(layer_output.color_logits, labels)
    total_loss += λ_aux * (char_loss_layer + color_loss_layer)
```

`λ_aux = 0.3`

### 4.3 对比学习辅助（可选）

对颜色分支增加对比损失：同一 query 对红色/非红色样本的 embedding 在投影空间中拉远/拉近。

---

## 5. 数据增强

| 增强 | 参数 | 概率 | 说明 |
|------|------|------|------|
| RandAugment | n=2, m=9 | 1.0 | 自动搜索的最优增强组合 |
| RandomResizedCrop | scale=(0.6, 1.0) | 1.0 | 模拟不同缩放 |
| ColorJitter | b=0.2, c=0.2, s=0.2, h=0.02 | 0.6 | hue 幅度小以保护红色 |
| RandomPerspective | distortion=0.15, p=0.5 | 0.5 | 模拟几何变形 |
| GaussianBlur | kernel=3-7 | 0.2 | 模拟模糊 |
| MixUp | α=0.2 | 0.5 | 图像+标签同时混合 |
| CutMix | α=1.0 | 0.5 | 区域替换 |

**注意**：MixUp/CutMix 时，标签也需要软混合。对于 5 个位置的颜色和字符标签，按混合比例 λ 做线性插值（one-hot → soft label）。

### 预处理
```python
transform = Compose([
    Resize((224, 224)),          # ViT 标准输入
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3)  # 或 ImageNet 统计值
])
```

---

## 6. 训练策略

### 6.1 验证集
- 50,000 → 45,000 train / 5,000 val（9:1 分层，按红色字符数量分层）
- 建议 5-fold cross validation 得到稳健的精度估计

### 6.2 超参数

| 超参数 | 值 | 备注 |
|--------|-----|------|
| Optimizer | AdamW | Transformer 标配 |
| LR (encoder) | 1e-4 | 预训练 backbone 用低 LR |
| LR (decoder + heads) | 5e-4 | 随机初始化的部分用高 LR |
| Weight decay | 0.05 | ViT 训练常用值 |
| Batch size | 64 | A100:128 / RTX3090:64 / RTX4070:32 |
| Epochs | 100-150 | 配合 early stopping |
| LR schedule | Cosine + linear warmup 5 epochs | |
| Drop path | 0.1 (encoder), 0.0 (decoder) | Stochastic depth |
| Label smoothing | 0.1 | 仅用于字符分类 |
| Gradient clip | 1.0 | |

### 6.3 三阶段训练

```
Stage 1 (epochs 0-10):
  冻结 Encoder, 仅训练 Decoder + Heads
  LR: 1e-3, batch_size: 128

Stage 2 (epochs 11-80):
  解冻 Encoder, 全模型训练
  LR: 1e-4 (encoder) / 5e-4 (decoder)
  Cosine annealing

Stage 3 (epochs 81-120):
  用更大分辨率 (384×384) fine-tune
  LR: 1e-5 / 5e-5, batch_size: 32
  仅最后 40 epoch
```

### 6.4 处理空红色样本
训练集保证每张 ≥1 个红，但测试集可能有零红色样本。策略：
- 构造少量合成数据：从训练集中选取图像，将其 color 标签改为 `uuuuu`，让模型学习输出空 label
- 或依赖颜色头输出低概率时自动判为非红（阈值校准）

---

## 7. 推理

### 7.1 基本推理

```python
@torch.no_grad()
def predict(model, image):
    char_logits, color_logits = model(image)  # (5,36), (5,1)

    result = []
    for pos in range(5):
        if torch.sigmoid(color_logits[pos]) >= threshold:
            char_idx = char_logits[pos].argmax().item()
            result.append(IDX_TO_CHAR[char_idx])

    return ''.join(result)  # "" 如果没有红色字符
```

### 7.2 TTA（Test Time Augmentation）

```python
augmentations = [
    identity,
    hflip,           # 输出结果反转
    rotate(+3°),
    rotate(-3°),
]
# 每个增强推理一次, 对 char_logits 和 color_logits 取平均
# hflip 的结果需要 inverse hflip 后取平均
final_color_prob = color_probs.mean(dim=0)  # (5,)
final_char_prob  = char_probs.mean(dim=0)   # (5, 36)
```

### 7.3 阈值校准
在验证集上遍历 threshold ∈ [0.3, 0.7]，选字符级 F1 或样本级准确率最高的阈值。

---

## 8. 模型集成

### 8.1 异构模型组合

| 模型 | Encoder | Decoder | 输入 | 特点 |
|------|---------|---------|------|------|
| A | ViT-B/16 | 2 层 Transformer | 224×224 | 基线 |
| B | Swin-B | 2 层 Transformer | 224×224 | 多尺度特征 |
| C | ViT-L/16 | 4 层 Transformer | 384×384 | 大模型+高分辨率 |
| D | ViT-B/16 | 4 层 Transformer + 对比学习辅助 loss | 224×224 | 训练策略差异 |

### 8.2 投票策略
- 颜色：平均 sigmoid 概率 → >0.5
- 字符：平均 softmax 概率 → argmax
- 冲突时：降级为多数投票

---

## 9. 精度预估与迭代

| 阶段 | 配置 | 预期准确率 |
|------|------|----------|
| 基线 | ViT-B/16 + 2 层 Decoder + 标准增强 | 92-94% |
| + 强增强 | RandAugment + MixUp + CutMix | 94-96% |
| + 多尺度 | Swin-B + FPN | 95-97% |
| + TTA | 5 种增强取平均 | +0.5-1% |
| 3 模型集成 | ViT-B + Swin-B + ViT-L | 97-99% |

---

## 10. 代码框架

```
project/
├── config.py              # 超参数配置
├── dataset.py             # Dataset + 增强 pipeline
├── models/
│   ├── encoder.py         # ViT / Swin backbone
│   ├── decoder.py         # Transformer Decoder with cross-attention
│   ├── query_tokens.py    # Position Query Token 定义
│   ├── heads.py           # 颜色头 + 字符头
│   └── model.py           # 整体 model 组装
├── losses.py              # 主 loss + 辅助 loss
├── train.py               # 训练脚本 (含三阶段)
├── inference.py           # 推理 + TTA + 生成 submission.csv
├── ensemble.py            # 多模型集成
└── utils.py               # 指标计算、可视化
```

推荐依赖：
- `torch` + `torchvision`
- `timm` — ViT/Swin 预训练权重
- `albumentations` — 数据增强
- `einops` — tensor 重排（可读性）

---

## 11. 关键细节备忘

1. **标签对齐**：CSV 列名是 `filename` 而非 `id`，读取时注意；`color` 字符串的字符顺序 = 从左到右。

2. **Query 顺序保证**：Decoder 的 self-attention 是 permutation-equivariant 的，需依赖 query 的初始化位置编码来建立顺序。如果训练后发现 query 顺序混乱，给每个 query 加可学习的 position embedding 并在训练初期加一个 order loss 约束。

3. **空 label**：测试集可能有无红色字符的样本，输出 `""`。训练时混入人工构造的 `uuuuu` 样本（从现有数据修改 color 标签生成），比例约 5%。

4. **颜色保真**：RandAugment 中慎用大幅度 hue 变换，避免把红色变成其他颜色导致标签错误。

5. **梯度累积**：如果 batch_size 因显存受限，用 `gradient_accumulation_steps=2-4` 等效扩大 batch size。

6. **混合精度**：使用 `torch.cuda.amp` 加速训练，对 Transformer 通常无明显精度损失。
