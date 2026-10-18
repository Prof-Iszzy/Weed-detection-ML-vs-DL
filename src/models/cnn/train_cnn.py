"""
Train the custom CNN (pretrained ResNet-50 + custom head + focal loss + class
weights + staged unfreezing) on DeepWeeds, with the SAME 5-fold protocol as the
classical ML baseline.

Phase 5 + 6 of project_execution_algorithm.md.

What makes this a *custom* model (handbook Section 4.3: "not just a library
call with default settings"):
  (a) a custom classification head (2 dense layers, BatchNorm, two Dropouts)
  (b) a custom loss: focal loss with per-class weights, aimed at the
      Negative-vs-weeds imbalance (9,106 vs ~1,000 per species)
  (c) a custom training schedule: backbone frozen for the first N epochs,
      then unfrozen with a lower learning rate for the backbone

Fair-comparison rules (same as train_ml.py):
  - only the 'train' and 'val' rows of data/deepweeds_splits.csv are used
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - metrics: accuracy + macro F1 / precision / recall, reported as mean +/- std
  - the 'test' split is NEVER touched here (frozen until Phase 7)

Designed for Google Colab (GPU) but also runs on CPU for small smoke tests.
Folds that are already finished are skipped on re-run, so a Colab disconnect
does not cost you the whole run.

Run (from the project root, where data/ lives):
    python -m src.models.cnn.train_cnn
Quick check that the code works (tiny, fast):
    python -m src.models.cnn.train_cnn --smoke
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

RANDOM_SEED = 42  # same seed as the split and the classical ML folds
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
class WeedDataset(Dataset):
    """Loads one image + its integer label each time the DataLoader asks."""

    def __init__(self, filepaths, labels, root, transform):
        self.filepaths = list(filepaths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.root / self.filepaths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def build_transforms(image_size):
    # Augmentation ONLY for training; validation/test see the clean image.
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# 2. Model: pretrained backbone + CUSTOM head
# ----------------------------------------------------------------------------
def build_model(arch, num_classes, pretrained, head_dropout):
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, arch)(weights=weights)
    in_features = model.fc.in_features  # 2048 for resnet50
    model.fc = nn.Sequential(           # <- the custom head
        nn.Dropout(head_dropout),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(head_dropout / 2),
        nn.Linear(512, num_classes),
    )
    return model


def set_backbone_frozen(model, frozen):
    """Freeze/unfreeze everything except the head (model.fc)."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = not frozen


# ----------------------------------------------------------------------------
# 3. Custom loss: class-weighted focal loss
# ----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Cross-entropy that down-weights easy examples by (1 - p_correct)^gamma, so
    training concentrates on the hard, often minority-class images.
    gamma=0 reduces exactly to (weighted) ordinary cross-entropy.
    """

    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        p_correct = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return (((1 - p_correct) ** self.gamma) * ce).mean()


def compute_class_weights(labels, num_classes):
    """Inverse-frequency weights, scaled so they average to 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------------------------------------------------------
# 4. Train / evaluate one fold
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        n += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets), total_loss / max(n, 1)


def make_optimizer(model, lr, backbone_lr_factor):
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    return torch.optim.AdamW([
        {"params": head, "lr": lr},
        {"params": backbone, "lr": lr * backbone_lr_factor},
    ], weight_decay=1e-4)


def train_one_fold(fold, args, df, class_names, trainval_idx, heldout_idx, device):
    num_classes = len(class_names)
    labels_all = df["label"].to_numpy()
    paths_all = df["filepath"].to_numpy()

    # Carve a small slice off the fold's training part for early stopping, so
    # the held-out fold is only ever used for the final score (no peeking).
    fit_idx, es_idx = train_test_split(
        trainval_idx, test_size=0.1, stratify=labels_all[trainval_idx], random_state=RANDOM_SEED)

    train_tf, eval_tf = build_transforms(args.image_size)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=device.type == "cuda")
    train_loader = DataLoader(WeedDataset(paths_all[fit_idx], labels_all[fit_idx], args.root, train_tf),
                              shuffle=True, drop_last=True, **loader_kw)
    es_loader = DataLoader(WeedDataset(paths_all[es_idx], labels_all[es_idx], args.root, eval_tf),
                           shuffle=False, **loader_kw)
    held_loader = DataLoader(WeedDataset(paths_all[heldout_idx], labels_all[heldout_idx], args.root, eval_tf),
                             shuffle=False, **loader_kw)

    model = build_model(args.arch, num_classes, not args.no_pretrained, args.head_dropout).to(device)

    weights = compute_class_weights(labels_all[fit_idx], num_classes).to(device) if args.class_weights else None
    if args.loss == "focal":
        criterion = FocalLoss(args.gamma, weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = make_optimizer(model, args.lr, args.backbone_lr_factor)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_path = Path(args.out_dir) / f"fold{fold}_best.pt"
    best_loss, bad_epochs, history = float("inf"), 0, []

    for epoch in range(args.epochs):
        set_backbone_frozen(model, epoch < args.freeze_epochs)  # staged unfreezing
        model.train()
        if epoch < args.freeze_epochs:  # keep frozen BatchNorm statistics frozen too
            for name, m in model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        run_loss, seen, t0 = 0.0, 0, time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)

        es_pred, es_true, es_loss = predict(model, es_loader, device)
        es_acc = accuracy_score(es_true, es_pred)
        history.append({"epoch": epoch + 1, "train_loss": run_loss / max(seen, 1),
                        "val_loss": es_loss, "val_acc": es_acc})
        print(f"  fold {fold} epoch {epoch + 1:>2}/{args.epochs}  "
              f"train_loss {run_loss / max(seen, 1):.4f}  val_loss {es_loss:.4f}  "
              f"val_acc {es_acc:.4f}  ({time.time() - t0:.0f}s)")

        if es_loss < best_loss - 1e-4:
            best_loss, bad_epochs = es_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience and epoch >= args.freeze_epochs:
                print(f"  early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))  # back to the best epoch
    t0 = time.time()
    pred, true, _ = predict(model, held_loader, device)
    ms_per_image = (time.time() - t0) / len(true) * 1000
    p, r, f1, _ = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
    return {
        "fold": fold,
        "accuracy": accuracy_score(true, pred),
        "f1_macro": f1, "precision_macro": p, "recall_macro": r,
        "inference_ms_per_image": ms_per_image,
        "epochs_run": len(history),
        "history": history,
    }


# ----------------------------------------------------------------------------
# 5. Main: 5-fold CV, resumable
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-csv", default="data/deepweeds_splits.csv")
    ap.add_argument("--root", default=".", help="folder that the CSV's relative filepaths start from")
    ap.add_argument("--out-dir", default="results/checkpoints/cnn")
    ap.add_argument("--results-csv", default="results/tables/cnn_results.csv")
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--backbone-lr-factor", type=float, default=0.1,
                    help="backbone lr = lr * this, once unfrozen")
    ap.add_argument("--head-dropout", type=float, default=0.4)
    ap.add_argument("--loss", default="focal", choices=["focal", "ce"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--class-weights", dest="class_weights", action="store_true", default=True)
    ap.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 2 folds, 2 epochs, no pretrained weights, resnet18, 64px")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.folds, args.epochs, args.freeze_epochs, args.patience = 2, 2, 1, 2
        args.no_pretrained, args.arch, args.image_size, args.batch_size, args.workers = True, "resnet18", 64, 8, 0

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    splits = Path(args.splits_csv)
    if not splits.exists():
        print(f"ERROR: {splits} not found. Run from the project root (or pass --splits-csv).")
        return
    df = pd.read_csv(splits)
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)  # test stays frozen
    class_names = sorted(df["species"].astype(str).unique())
    df["label"] = df["species"].astype(str).map({c: i for i, c in enumerate(class_names)})
    print(f"Loaded {len(df)} train+val images, {len(class_names)} classes: {class_names}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    for fold, (trainval_idx, heldout_idx) in enumerate(skf.split(df, df["label"]), start=1):
        done_file = out_dir / f"fold{fold}_result.json"
        if done_file.exists():
            print(f"Fold {fold}: already finished, loading saved result.")
            fold_results.append(json.loads(done_file.read_text()))
            continue
        print(f"\n{'=' * 60}\nFold {fold}/{args.folds}\n{'=' * 60}")
        result = train_one_fold(fold, args, df, class_names, trainval_idx, heldout_idx, device)
        done_file.write_text(json.dumps(result))
        fold_results.append(result)
        print(f"Fold {fold} held-out: acc {result['accuracy']:.4f}  F1 {result['f1_macro']:.4f}")

    # Summary in the same shape as classical_ml_results.csv
    n_params = sum(p.numel() for p in build_model(args.arch, len(class_names), False, args.head_dropout).parameters())
    summary = {"model": f"custom_{args.arch}"}
    for m in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "inference_ms_per_image"]:
        vals = [r[m] for r in fold_results]
        key = m if m != "inference_ms_per_image" else "inference_ms"
        summary[f"{key}_mean"], summary[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
    summary["params"] = n_params
    summary["settings"] = str({k: getattr(args, k) for k in
                               ["arch", "loss", "gamma", "class_weights", "lr", "backbone_lr_factor",
                                "batch_size", "freeze_epochs", "head_dropout", "image_size"]})
    pd.DataFrame([summary]).to_csv(args.results_csv, index=False)
    # Per-fold curves for the visualisation lead (loss/accuracy plots)
    (out_dir / "histories.json").write_text(json.dumps({r["fold"]: r["history"] for r in fold_results}))

    print(f"\nSaved results to {args.results_csv}")
    print(f"CNN accuracy {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}, "
          f"macro-F1 {summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}")
    print("This is your official custom-CNN result for Phase 7's comparison.")


if __name__ == "__main__":
    main()
