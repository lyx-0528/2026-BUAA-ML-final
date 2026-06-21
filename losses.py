import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOSS_CHAR_WEIGHT, LOSS_COLOR_WEIGHT, AUX_LOSS_WEIGHT, LABEL_SMOOTHING


def char_loss_fn(logits, targets, label_smoothing: float = LABEL_SMOOTHING):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        label_smoothing=label_smoothing,
    )


def color_loss_fn(logits, targets):
    return F.binary_cross_entropy_with_logits(
        logits.view(-1),
        targets.view(-1),
    )


def compute_loss(outputs, color_targets, char_targets):
    loss = (LOSS_CHAR_WEIGHT * char_loss_fn(outputs["char"], char_targets) +
            LOSS_COLOR_WEIGHT * color_loss_fn(outputs["color"], color_targets))

    if "aux_char" in outputs:
        for aux_c, aux_col in zip(outputs["aux_char"], outputs["aux_color"]):
            loss = loss + AUX_LOSS_WEIGHT * (
                char_loss_fn(aux_c, char_targets) +
                color_loss_fn(aux_col, color_targets)
            )

    return loss


def compute_mixup_loss(outputs, color_a, char_a, color_b, char_b, lam):
    loss_a = compute_loss(outputs, color_a, char_a)
    loss_b = compute_loss(outputs, color_b, char_b)
    return lam * loss_a + (1 - lam) * loss_b


def compute_accuracy(outputs, color_targets, char_targets, threshold: float = 0.5):
    color_pred = (torch.sigmoid(outputs["color"]) >= threshold).long()
    char_pred = outputs["char"].argmax(dim=-1)

    color_acc = (color_pred == color_targets.long()).float().mean()
    char_acc = (char_pred == char_targets).float().mean()

    batch_size = char_targets.size(0)
    sample_correct = 0
    for i in range(batch_size):
        red_mask = color_pred[i].bool()
        true_red_mask = color_targets[i].bool()
        pred_chars = char_pred[i][red_mask]
        true_chars = char_targets[i][true_red_mask]

        if pred_chars.numel() == true_chars.numel() and (pred_chars == true_chars).all():
            sample_correct += 1

    sample_acc = sample_correct / batch_size

    return {
        "color_acc": color_acc.item(),
        "char_acc": char_acc.item(),
        "sample_acc": sample_acc,
    }
