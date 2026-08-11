"""Train the FP32 baseline on data/manifest.csv and export to ONNX.

Reports the honest numbers Section 8.3 of the doc asks for: accuracy/F1 on
the held-out val split, and FP32 model size in MB.
"""
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from features import logmel, load_audio, CLIP_SECONDS
from model import SiloSenseNet, count_params

MANIFEST = "data/manifest.csv"
MODEL_DIR = "models"
FP32_ONNX = os.path.join(MODEL_DIR, "silosense_fp32.onnx")
CHECKPOINT = os.path.join(MODEL_DIR, "silosense_fp32.pt")
METRICS_JSON = os.path.join(MODEL_DIR, "fp32_metrics.json")

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class ManifestDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        y = load_audio(r["path"], offset=float(r["start_offset_sec"]), duration=CLIP_SECONDS)
        feat = logmel(y)
        return torch.from_numpy(feat), int(r["label"])


def load_manifest():
    train_rows, val_rows = [], []
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            (train_rows if row["split"] == "train" else val_rows).append(row)
    return train_rows, val_rows


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(y.numpy().tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    return acc, f1


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_rows, val_rows = load_manifest()
    if not train_rows or not val_rows:
        raise SystemExit("manifest is empty — run prepare_dataset.py first")

    train_ds, val_ds = ManifestDataset(train_rows), ManifestDataset(val_rows)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = "cpu"
    model = SiloSenseNet().to(device)
    print("params:", count_params(model))

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    n_clean = sum(1 for r in train_rows if int(r["label"]) == 0)
    n_infested = sum(1 for r in train_rows if int(r["label"]) == 1)
    class_weights = torch.tensor(
        [len(train_rows) / (2 * n_clean), len(train_rows) / (2 * n_infested)], dtype=torch.float32
    )
    print(f"class weights (clean, infested): {class_weights.tolist()}")
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        acc, f1 = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={total_loss/len(train_ds):.4f} val_acc={acc:.4f} val_f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), CHECKPOINT)

    # Reload best checkpoint, final eval, export.
    model.load_state_dict(torch.load(CHECKPOINT))
    acc, f1 = evaluate(model, val_loader, device)
    print(f"FINAL best-checkpoint val_acc={acc:.4f} val_f1={f1:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            p = model(x.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(p.tolist())
            all_labels.extend(y.numpy().tolist())
    prec, rec, f1_per_class, support = precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
    print(f"per-class (clean, infested):")
    print(f"  precision: {prec.tolist()}")
    print(f"  recall:    {rec.tolist()}")
    print(f"  f1:        {f1_per_class.tolist()}")
    print(f"  support:   {support.tolist()}")

    from features import N_MELS, N_FRAMES
    dummy = torch.randn(1, N_MELS, N_FRAMES)
    torch.onnx.export(
        model, dummy, FP32_ONNX,
        input_names=["logmel"], output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_mb = os.path.getsize(FP32_ONNX) / 1e6
    print(f"exported {FP32_ONNX} ({size_mb:.3f} MB)")

    with open(METRICS_JSON, "w") as f:
        json.dump({
            "val_accuracy": acc, "val_f1": f1,
            "model_size_mb": size_mb,
            "n_params": count_params(model),
            "train_windows": len(train_rows), "val_windows": len(val_rows),
        }, f, indent=2)


if __name__ == "__main__":
    main()
