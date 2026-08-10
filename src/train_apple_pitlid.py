"""
PiTLiD reproduction — Apple 4-class small-sample experiment (PyTorch).

Reference: "PiTLiD: Identification of Plant Disease From Leaf Images Based on
Convolutional Neural Network" (IEEE, 2022). Official code:
https://github.com/zhanglab-wbgcas/PiTLiD (local clone: E:\\plant_disease\\PiTLiD).

2026-08-10: rewritten to match the official code's apple-specific script
(`differant setting/prediy7.py`, the only script in the repo with apple-tuned
two-stage training + L2 + a per-epoch LR schedule, as opposed to the other
scripts which are generic multi-crop/multi-backbone templates reused across
tasks). NOTE: the repo has no single unambiguous "canonical" script the way
ConViTX's repo does -- prediy7.py is the closest apple-specific match, but
this is a judgment call, documented here rather than silently assumed.

What the official code actually does, replacing our earlier from-paper-text
guesses:
  - Head: GAP -> Dropout(0.5) -> ReLU -> Dense(4, softmax). NOT a bare
    GAP->Linear as we'd guessed from "GAP connects directly to the output
    layer" -- there IS a dropout+ReLU in between.
  - Input 256x256, NOT 299x299 (InceptionV3's "native" size -- the paper's
    own code uses a smaller size).
  - TWO-STAGE training, not single-stage full fine-tuning from the start:
    stage 1 freezes the whole backbone and trains only the new head
    (RMSprop lr=1e-3, steps_per_epoch=100, 10 epochs); stage 2 unfreezes
    everything and continues (steps_per_epoch=200, 40 more epochs).
  - LR schedule: `--lr_strategy` supports both, since the evidence is
    genuinely split (documented, not silently picked):
      * "clr" (default): true Cyclical LR (triangular2, base_lr=0.001,
        max_lr=0.006 -- exactly what the paper's Figure 5B claims as the
        best-performing strategy, >99.31% acc, beating fixed and decay LR).
        `prediy7.py` itself imports `from clr_callback import *` and has a
        commented-out `CyclicLR(mode='triangular')` + commented-out
        checkpoint filenames `weights_inceptionv3_peachclr_tanh.h5` /
        `..._potatoclr1e-5_tanh.h5` -- concrete evidence CLR-trained models
        genuinely existed for other crops in this same codebase, even
        though `clr_callback.py` itself (an external/local file, likely the
        common bckenstler/CLR gist) is missing from what got pushed to
        GitHub, and apple's `prediy7.py` currently has CLR commented out in
        favor of the step-decay path below. `clr_step_size` (half-cycle
        length in batches) is still unknown -- clr_callback.py's absence
        means we can't recover it, kept as a guessed/exposed flag.
      * "step_decay": what's literally active (uncommented) in the apple
        script currently committed to the official repo: lr=1e-3 for
        epoch<=20 (spanning both stage 1 and part of stage 2), lr=1e-4 for
        epoch>20. (There's also a dead `elif epoch>40: lr*=1e-2` branch,
        unreachable since the preceding `if epoch>20` already catches every
        epoch>40 -- reproduced as the 2-tier schedule actually in effect.)
      Both are real, code-evidenced possibilities pointing at different
      crops' checkpoints in the same repo -- not a case of trusting paper
      prose over code, or vice versa. Both are run and reported side by
      side, matching the same "don't silently pick one" convention used for
      ConViTX's head-count/projection-dim ablations.
  - NO early stopping is actually active. `EarlyStopping(patience=13)` is
    constructed in the source but the resulting `callbacks` list is never
    passed to `fit_generator` (a literal `[tensorboard, lr]` list is passed
    instead) -- both stages just run their full fixed epoch budget. Only
    the best val_accuracy checkpoint is kept (`ModelCheckpoint(monitor=
    'val_accuracy', mode='max')` in stage 2).
  - Augmentation is more aggressive than we'd guessed: rotation 90°,
    width/height shift 0.3, shear 0.3°, zoom range 0.3 (scale 0.7-1.3),
    horizontal AND vertical flip.
  - L2 regularization (`layer.W_regularizer = l2(1e-3)`) is very likely a
    NO-OP in practice: `W_regularizer` is Keras-1-style API; Keras 2.x reads
    `kernel_regularizer`, and setting a bare attribute on an already-built
    layer after the fact does not hook into the loss computation. We can't
    re-run their exact Keras version to confirm, so this is flagged as a
    caveat, not asserted -- `--weight_decay` defaults to 0.0 here (matching
    what the code most likely actually did at runtime), exposed as a flag
    to test 1e-3 as an alternative reading.

Run:
    python train_apple_pitlid.py --data_dir ../data/apple_pitlid_split --output_dir ../runs/apple_seed1
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import Inception_V3_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(
        weights=weights,
        aux_logits=True,      # required to load the pretrained checkpoint
        init_weights=False,
        transform_input=False,  # we do our own (rescale-only) preprocessing
    )
    # Keras' InceptionV3(include_top=False) has no aux classifier.
    model.aux_logits = False
    model.AuxLogits = None

    # Official code's head: GAP (built into Inception3.avgpool) -> Dropout(0.5)
    # -> ReLU -> Dense(num_classes, softmax via CrossEntropyLoss).
    model.dropout = nn.Dropout(p=0.5)
    model.fc = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(model.fc.in_features, num_classes),
    )

    # Stage 1 starts with the backbone frozen; main() unfreezes for stage 2.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    return model.to(device)


def build_transforms(img_size: int, imagenet_norm: bool):
    normalize = (
        [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        if imagenet_norm
        else []
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Keras ImageDataGenerator: rotation_range=90, width/height_shift_range=0.3,
            # shear_range=0.3 (degrees), zoom_range=0.3 (scale 0.7-1.3)
            transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=0.3),
            transforms.ToTensor(),  # scales pixels to [0, 1], i.e. divide by 255
            *normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            *normalize,
        ]
    )
    return train_tf, eval_tf


def make_train_loader(dataset, batch_size, steps_per_epoch):
    # Keras' fit_generator(steps_per_epoch=N) on a 120-image directory cycles
    # through the generator with replacement; reproduce that with a sampler.
    num_samples = batch_size * steps_per_epoch
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, batch_scheduler=None):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
                if batch_scheduler is not None:
                    batch_scheduler.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def lr_for_epoch(epoch: int) -> float:
    # official lr_schedule(epoch): lr=1e-3; if epoch>20: lr*=1e-1 (-> 1e-4).
    # The `elif epoch>40: lr*=1e-2` branch is unreachable dead code (the
    # preceding `if epoch>20` already catches epoch>40), so the real
    # schedule is just this one step at epoch 20.
    return 1e-3 if epoch <= 20 else 1e-4


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        y_pred.extend(out.argmax(1).cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(cm, class_names, out_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (PiTLiD - Apple)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir with train/val/test subfolders (see make_pitlid_apple_split.py)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--stage1_epochs", type=int, default=10, help="frozen-backbone warmup stage")
    ap.add_argument("--stage1_steps_per_epoch", type=int, default=100)
    ap.add_argument("--stage2_epochs", type=int, default=50, help="total epoch count incl. stage1 (matches official initial_epoch=10)")
    ap.add_argument("--stage2_steps_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size_train", type=int, default=32)
    ap.add_argument("--batch_size_val", type=int, default=16)
    ap.add_argument("--lr_strategy", choices=["clr", "step_decay"], default="clr",
                     help="'clr' = true Cyclical LR (matches paper Fig 5B's explicit best-strategy "
                          "claim + evidence CLR checkpoints existed for other crops); 'step_decay' = "
                          "what's literally active in the currently-committed apple script. See "
                          "module docstring for the full evidence trail.")
    ap.add_argument("--clr_base_lr", type=float, default=0.001)
    ap.add_argument("--clr_max_lr", type=float, default=0.006)
    ap.add_argument("--clr_step_size", type=int, default=2000,
                     help="half-cycle length in batches -- unknown, clr_callback.py is missing from "
                          "the official repo, this is a guess")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                     help="L2 strength. Official code sets layer.W_regularizer=l2(1e-3) in "
                          "stage 2, but that's Keras-1-style API that Keras 2.x silently "
                          "ignores on an already-built layer -- almost certainly a no-op in "
                          "practice, so this defaults to 0.0. Pass 1e-3 to test the other "
                          "reading.")
    ap.add_argument("--imagenet_norm", action="store_true",
                     help="paper specifies rescale-only (/255); pass this flag to additionally "
                          "apply ImageNet mean/std normalization if convergence is a problem")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_tf, eval_tf = build_transforms(args.img_size, args.imagenet_norm)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    assert train_ds.classes == val_ds.classes == test_ds.classes
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"classes: {class_names}")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    stage1_loader = make_train_loader(train_ds, args.batch_size_train, args.stage1_steps_per_epoch)
    stage2_loader = make_train_loader(train_ds, args.batch_size_train, args.stage2_steps_per_epoch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_val, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size_val, shuffle=False)

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    # ---- stage 1: backbone frozen, train head only ----
    # Same constant starting lr (1e-3) either way -- CLR's base_lr equals
    # step_decay's pre-epoch-20 value, so the short frozen warmup doesn't
    # need to branch on lr_strategy.
    optimizer = torch.optim.RMSprop(model.fc.parameters(), lr=lr_for_epoch(0))
    for epoch in range(args.stage1_epochs):
        for g in optimizer.param_groups:
            g["lr"] = lr_for_epoch(epoch)
        train_loss, train_acc = run_epoch(model, stage1_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "stage": 1, "lr": lr_for_epoch(epoch),
                         "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"[stage1] epoch {epoch+1}/{args.stage1_epochs}  lr={lr_for_epoch(epoch):.0e}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ---- stage 2: unfreeze everything, continue from global epoch = stage1_epochs ----
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.clr_base_lr if args.lr_strategy == "clr" else lr_for_epoch(args.stage1_epochs),
        weight_decay=args.weight_decay
    )
    clr_scheduler = None
    if args.lr_strategy == "clr":
        clr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.clr_base_lr, max_lr=args.clr_max_lr,
            step_size_up=args.clr_step_size, mode="triangular2", cycle_momentum=False,
        )
    for epoch in range(args.stage1_epochs, args.stage2_epochs):
        if args.lr_strategy == "step_decay":
            for g in optimizer.param_groups:
                g["lr"] = lr_for_epoch(epoch)
        train_loss, train_acc = run_epoch(model, stage2_loader, criterion, optimizer, device, train=True,
                                           batch_scheduler=clr_scheduler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "stage": 2, "lr": cur_lr,
                         "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"[stage2] epoch {epoch+1}/{args.stage2_epochs}  lr={cur_lr:.2e}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / "best_model.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    # official code has no active early stopping (dead-code EarlyStopping,
    # never passed to fit_generator) -- both stages always run their full
    # fixed epoch budget; only the best val_accuracy checkpoint is kept.

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pt")

    x = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(x, [h["train_acc"] for h in history], "o-", label="train")
    axes[0].plot(x, [h["val_acc"] for h in history], "o-", label="val")
    axes[0].set_ylabel("accuracy"); axes[0].legend(); axes[0].set_title("Accuracy vs epoch")
    axes[1].plot(x, [h["train_loss"] for h in history], ".-", label="train")
    axes[1].plot(x, [h["val_loss"] for h in history], ".-", label="val")
    axes[1].set_ylabel("loss"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].set_title("Loss vs epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_loss.png", dpi=150)
    plt.close(fig)

    # ---- test-set evaluation ----
    y_true, y_pred = predict_all(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    print("\n=== Test set results ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision (macro): {precision:.4f}")
    print(f"recall/sensitivity (macro): {recall:.4f}")
    print(f"f1 (macro): {f1:.4f}")
    print(report)

    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_sensitivity_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": report,
        "best_val_acc": best_val_acc,
        "stopped_epoch": len(history),
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nsaved: {out_dir}/best_model.pt, history.json, accuracy_loss.png, confusion_matrix.png, test_metrics.json")


if __name__ == "__main__":
    main()
