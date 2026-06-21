import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime

from config import (
    CHECKPOINT_DIR, BATCH_SIZE, NUM_EPOCHS, LR_ENCODER, LR_DECODER,
    WEIGHT_DECAY, WARMUP_EPOCHS, GRAD_CLIP, MIXUP_ALPHA, CUTMIX_ALPHA,
    LABEL_SMOOTHING, VAL_SPLIT,
)
from models.model import CaptchaModel
from dataset import create_dataloaders, mixup_data, cutmix_data
from losses import compute_loss, compute_mixup_loss, compute_accuracy


LOG_FILE = Path(__file__).resolve().parent / "training.log"
SUMMARY_FILE = Path(__file__).resolve().parent / "progress.csv"

_log_fp = None


def _get_log_fp():
    global _log_fp
    if _log_fp is None:
        _log_fp = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    return _log_fp


def log(msg: str):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    fp = _get_log_fp()
    fp.write(line + "\n")
    fp.flush()


def init_summary():
    header = "Epoch\tTrain Loss\tVal Loss\tSample Acc\tChar Acc\tColor Acc"
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n")


def log_summary(epoch: int, train_loss: float, val_loss: float,
                sample_acc: float, char_acc: float, color_acc: float, best: bool = False):
    marker = " *" if best else ""
    line = f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{sample_acc:.4f}\t{char_acc:.4f}\t{color_acc:.4f}{marker}"
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_optimizer(model):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    return AdamW([
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": decoder_params, "lr": LR_DECODER},
    ], weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer, steps_per_epoch):
    warmup = LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS * steps_per_epoch
    )
    cosine = CosineAnnealingLR(optimizer, T_max=(NUM_EPOCHS - WARMUP_EPOCHS) * steps_per_epoch)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[WARMUP_EPOCHS * steps_per_epoch])


def train_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp, epoch):
    model.train()
    total_loss = 0.0
    metrics_sum = {"color_acc": 0.0, "char_acc": 0.0, "sample_acc": 0.0}
    total_steps = len(loader)
    log_every = max(1, total_steps // 20)  # log ~20 times per epoch

    pbar = tqdm(loader, desc=f"E{epoch} Train", dynamic_ncols=True)
    for step, (imgs, colors, chars) in enumerate(pbar, 1):
        imgs = imgs.to(device)
        colors = colors.to(device)
        chars = chars.to(device)

        use_mixup = MIXUP_ALPHA > 0 and np.random.rand() < 0.5

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_mixup:
                mixed_imgs, c_a, ch_a, c_b, ch_b, lam = mixup_data(imgs, colors, chars)
                outputs = model(mixed_imgs, return_aux=True)
                loss = compute_mixup_loss(outputs, c_a, ch_a, c_b, ch_b, lam)
            else:
                outputs = model(imgs, return_aux=True)
                loss = compute_loss(outputs, colors, chars)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        acc = compute_accuracy(outputs, colors, chars)
        for k in metrics_sum:
            metrics_sum[k] += acc[k]

        avg_metrics = {k: v / step for k, v in metrics_sum.items()}
        pbar.set_postfix(loss=f"{loss.item():.3f}", samp=f"{avg_metrics['sample_acc']:.3f}")

        if step % log_every == 0:
            log(f"E{epoch} [{step}/{total_steps}] loss={total_loss/step:.3f} "
                f"char_acc={avg_metrics['char_acc']:.3f} color_acc={avg_metrics['color_acc']:.3f} "
                f"sample_acc={avg_metrics['sample_acc']:.3f}")

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    metrics_sum = {"color_acc": 0.0, "char_acc": 0.0, "sample_acc": 0.0}

    pbar = tqdm(loader, desc="Val", dynamic_ncols=True)
    for imgs, colors, chars in pbar:
        imgs = imgs.to(device)
        colors = colors.to(device)
        chars = chars.to(device)

        outputs = model(imgs, return_aux=False)
        loss = compute_loss(outputs, colors, chars)

        total_loss += loss.item()
        acc = compute_accuracy(outputs, colors, chars)
        for k in metrics_sum:
            metrics_sum[k] += acc[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


def main(quick_test: bool = False, resume: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_dataloaders(val_split=VAL_SPLIT)
    log(f"Train samples: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = CaptchaModel().to(device)
    optimizer = build_optimizer(model)

    if quick_test:
        log("Quick test mode: 3 epochs on small subset")
        from torch.utils.data import Subset
        train_loader.dataset.df = train_loader.dataset.df.iloc[:128]
        val_loader.dataset.df = val_loader.dataset.df.iloc[:32]

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, steps_per_epoch)
    scaler = GradScaler() if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    best_sample_acc = 0.0
    start_epoch = 1
    max_epochs = 3 if quick_test else NUM_EPOCHS

    if resume:
        log(f"Resuming from {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_sample_acc = checkpoint.get("best_sample_acc", 0.0)
        start_epoch = checkpoint["epoch"] + 1
        scheduler = build_scheduler(optimizer, steps_per_epoch)
        for _ in range(checkpoint["epoch"] * steps_per_epoch):
            scheduler.step()
        log(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_sample_acc:.4f}")
    else:
        init_summary()
        log(f"=== Training started (fresh) ===")

    for epoch in range(start_epoch, max_epochs + 1):
        log(f"=== Epoch {epoch}/{max_epochs} start ===")
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, use_amp, epoch
        )
        log(f"E{epoch} Train - loss={train_loss:.4f} char_acc={train_metrics['char_acc']:.4f} "
            f"color_acc={train_metrics['color_acc']:.4f} sample_acc={train_metrics['sample_acc']:.4f}")

        val_loss, val_metrics = val_epoch(model, val_loader, device)
        is_best = val_metrics["sample_acc"] > best_sample_acc
        log(f"E{epoch} Val   - loss={val_loss:.4f} char_acc={val_metrics['char_acc']:.4f} "
            f"color_acc={val_metrics['color_acc']:.4f} sample_acc={val_metrics['sample_acc']:.4f}")
        log_summary(epoch, train_loss, val_loss,
                    val_metrics["sample_acc"], val_metrics["char_acc"], val_metrics["color_acc"],
                    best=is_best)

        if is_best:
            best_sample_acc = val_metrics["sample_acc"]
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            log(f"  >> New best model! sample_acc={best_sample_acc:.4f}")

        if epoch % 5 == 0:
            ckpt_path = CHECKPOINT_DIR / f"checkpoint_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_sample_acc": best_sample_acc,
            }, ckpt_path)
            log(f"  Checkpoint saved: {ckpt_path}")

    log(f"=== Training finished === Best val sample_acc: {best_sample_acc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test with 3 epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()
    main(quick_test=args.quick, resume=args.resume)
