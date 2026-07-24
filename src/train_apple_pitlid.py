"""
PiTLiD reproduction — Apple 4-class small-sample experiment (PyTorch).

Reference: "PiTLiD: Identification of Plant Disease From Leaf Images Based on
Convolutional Neural Network" (IEEE, 2022).

Setup implemented per the paper's stated protocol:
  - Backbone: Inception-V3, ImageNet-pretrained, ALL layers fine-tuned (none frozen)
  - Head: GlobalAveragePooling -> Softmax(4)  (top FC of Inception-V3 removed)
  - Input: 299x299, rescaled to [0,1] (divide by 255), no other normalization
  - Augmentation (train only): horizontal/vertical flip, rotation, shift, zoom, shear
  - Optimizer: RMSprop, L2 weight regularization
  - Batch size: train=32, val=16
  - Epochs: 50, steps/epoch: 200 (small train set is resampled with replacement,
    matching a Keras ImageDataGenerator + steps_per_epoch setup)
  - LR schedule: Cyclical LR, mode='triangular2', base_lr=0.001, max_lr=0.006
  - Early stopping: patience=14 on val_loss, best weights restored

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
    # Disable the auxiliary head: Keras' InceptionV3(include_top=False) has no
    # aux classifier, and the paper's head is GAP -> Softmax directly.
    model.aux_logits = False
    model.AuxLogits = None

    # torchvision's Inception3 already ends in AdaptiveAvgPool2d (=GAP) -> Dropout -> fc.
    # Paper: GAP connects directly to the output layer, so drop the dropout.
    model.dropout = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Fine-tune ALL layers (paper: "None_frozen").
    for p in model.parameters():
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
            # combines rotation / shift / zoom / shear in one affine op,
            # matching Keras ImageDataGenerator's random_{rotate,shift,zoom,shear}
            transforms.RandomAffine(degrees=40, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
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
    # Keras' fit_generator(steps_per_epoch=200) on a 120-image directory cycles
    # through the generator with replacement; reproduce that with a sampler.
    num_samples = batch_size * steps_per_epoch
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool):
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
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


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
    ap.add_argument("--img_size", type=int, default=299)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size_train", type=int, default=32)
    ap.add_argument("--batch_size_val", type=int, default=16)
    ap.add_argument("--base_lr", type=float, default=0.001)
    ap.add_argument("--max_lr", type=float, default=0.006)
    ap.add_argument("--clr_step_size", type=int, default=2000, help="half-cycle length in optimizer steps (batches)")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization strength")
    ap.add_argument("--rmsprop_momentum", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=14)
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

    train_loader = make_train_loader(train_ds, args.batch_size_train, args.steps_per_epoch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_val, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size_val, shuffle=False)

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,   # L2 regularization
        momentum=args.rmsprop_momentum,
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        step_size_up=args.clr_step_size,
        mode="triangular2",
        cycle_momentum=False,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, None, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch+1} (no val_loss improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

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
        "best_val_loss": best_val_loss,
        "stopped_epoch": len(history),
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nsaved: {out_dir}/best_model.pt, history.json, accuracy_loss.png, confusion_matrix.png, test_metrics.json")


if __name__ == "__main__":
    main()
