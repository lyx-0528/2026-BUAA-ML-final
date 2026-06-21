import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from config import (
    CHECKPOINT_DIR, TEST_IMG_DIR, SUBMISSION_OUTPUT,
    BATCH_SIZE, NUM_WORKERS, COLOR_THRESHOLD, IDX_TO_CHAR,
)
from models.model import CaptchaModel
from dataset import TestDataset


@torch.no_grad()
def predict_tta(model, img, device, threshold=COLOR_THRESHOLD):
    model.eval()
    imgs = [img]

    flipped = torch.flip(img, dims=[-1])
    imgs.append(flipped)

    all_color_probs = []
    all_char_logits = []

    for aug_img in imgs:
        outputs = model(aug_img.unsqueeze(0).to(device))
        color_prob = torch.sigmoid(outputs["color"])
        char_logits = outputs["char"]
        all_color_probs.append(color_prob.cpu())
        all_char_logits.append(char_logits.cpu())

    orig_color = all_color_probs[0]
    flipped_color = all_color_probs[1].flip(dims=[-1])
    avg_color_prob = (orig_color + flipped_color) / 2.0

    orig_char = all_char_logits[0]
    flipped_char = all_char_logits[1].flip(dims=[-2])
    avg_char_logits = (orig_char + flipped_char) / 2.0

    is_red = avg_color_prob[0] >= threshold
    char_indices = avg_char_logits[0].argmax(dim=-1)

    result = ""
    for pos in range(5):
        if is_red[pos]:
            result += IDX_TO_CHAR[char_indices[pos].item()]

    return result


@torch.no_grad()
def predict_simple(model, img, device, threshold=COLOR_THRESHOLD):
    model.eval()
    outputs = model(img.unsqueeze(0).to(device))
    color_prob = torch.sigmoid(outputs["color"])[0]
    char_logits = outputs["char"][0]

    is_red = color_prob >= threshold
    char_indices = char_logits.argmax(dim=-1)

    result = ""
    for pos in range(5):
        if is_red[pos]:
            result += IDX_TO_CHAR[char_indices[pos].item()]

    return result


def main(use_tta: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CaptchaModel().to(device)
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint at {checkpoint_path}, using random weights")

    dataset = TestDataset(TEST_IMG_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    predict_fn = predict_tta if use_tta else predict_simple

    ids = []
    labels = []

    for imgs, filenames in tqdm(loader, desc="Inference"):
        for img, fname in zip(imgs, filenames):
            label = predict_fn(model, img, device)
            ids.append(fname)
            labels.append(label)

    df = pd.DataFrame({"id": ids, "label": labels})
    df.to_csv(SUBMISSION_OUTPUT, index=False)
    print(f"Saved {len(df)} predictions to {SUBMISSION_OUTPUT}")
    print(f"Empty labels: {df['label'].eq('').sum()} / {len(df)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()
    main(use_tta=not args.no_tta)
