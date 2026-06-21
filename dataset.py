import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_IMG_DIR, TRAIN_LABEL_FILE, TEST_IMG_DIR,
    CHAR_TO_IDX, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    RANDAUG_N, RANDAUG_M, MIXUP_ALPHA, CUTMIX_ALPHA,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pad_to_square(image, fill=0):
    w, h = image.size
    if w == h:
        return image
    max_side = max(w, h)
    new_img = Image.new("RGB", (max_side, max_side), (fill, fill, fill))
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img


def get_train_transform():
    return T.Compose([
        T.Lambda(lambda img: pad_to_square(img, fill=128)),
        T.Resize(IMG_SIZE),
        T.RandAugment(num_ops=RANDAUG_N, magnitude=RANDAUG_M),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    return T.Compose([
        T.Lambda(lambda img: pad_to_square(img, fill=128)),
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CaptchaDataset(Dataset):
    def __init__(self, img_dir: Path, label_file: Path, train: bool = True):
        self.img_dir = img_dir
        self.df = pd.read_csv(label_file)
        self.train = train
        self.transform = get_train_transform() if train else get_val_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        color_str = str(row["color"])
        label_str = str(row["all_label"])

        img = Image.open(self.img_dir / filename).convert("RGB")

        color = torch.tensor([1.0 if c == "r" else 0.0 for c in color_str], dtype=torch.float)
        char = torch.tensor([CHAR_TO_IDX[c] for c in label_str], dtype=torch.long)

        img = self.transform(img)
        return img, color, char


def mixup_data(x, color, char, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, color, char, color[index], char[index], lam


def cutmix_data(x, color, char, alpha=CUTMIX_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    rw = int(W * np.sqrt(1 - lam))
    rh = int(H * np.sqrt(1 - lam))
    x1 = max(cx - rw // 2, 0)
    y1 = max(cy - rh // 2, 0)
    x2 = min(cx + rw // 2, W)
    y2 = min(cy + rh // 2, H)
    actual_lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    return mixed_x, color, char, color[index], char[index], actual_lam


def create_dataloaders(val_split: float = 0.1, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    df = pd.read_csv(TRAIN_LABEL_FILE)
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=val_split,
        stratify=df["color"].str.count("r"),
        random_state=42,
    )

    train_dataset = CaptchaDataset(TRAIN_IMG_DIR, TRAIN_LABEL_FILE, train=True)
    val_dataset = CaptchaDataset(TRAIN_IMG_DIR, TRAIN_LABEL_FILE, train=False)

    train_dataset.df = df.iloc[train_idx].reset_index(drop=True)
    val_dataset.df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    return train_loader, val_loader


class TestDataset(Dataset):
    def __init__(self, img_dir: Path):
        self.img_dir = img_dir
        import os
        self.files = sorted(os.listdir(img_dir))
        self.transform = get_val_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img = Image.open(self.img_dir / filename).convert("RGB")
        img = self.transform(img)
        return img, filename
