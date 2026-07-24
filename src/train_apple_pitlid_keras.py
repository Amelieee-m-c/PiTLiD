"""
PiTLiD reproduction — Apple 4-class small-sample experiment (Keras/TensorFlow).

This mirrors the original PiTLiD repo's own framework choice
(`training approach/tl.py`: keras.applications InceptionV3 + GlobalAveragePooling2D
+ Dense(softmax), ImageDataGenerator augmentation, `from clr_callback import *`),
using modern (non-deprecated) tf.keras APIs instead of the original's broken/EOL
calls (`keras.optimizers.callbacks`, TF1-style imports, hardcoded paths, etc).

Setup implemented per the paper's stated protocol:
  - Backbone: Inception-V3, ImageNet-pretrained, ALL layers fine-tuned (none frozen)
  - Head: GlobalAveragePooling -> Dense(4, softmax) directly (no extra FC)
  - Input: 299x299, rescaled to [0,1] (divide by 255), no other normalization
  - Augmentation (train only, via ImageDataGenerator): horizontal/vertical flip,
    rotation, width/height shift, zoom, shear
  - Optimizer: RMSprop; L2 weight regularization on every layer with kernel weights
  - Batch size: train=32, val=16. Epochs=50, steps/epoch=200
  - LR schedule: Cyclical LR (clr_callback.py), mode='triangular2',
    base_lr=0.001, max_lr=0.006
  - Early stopping: patience=14 on val_loss, best weights restored

Run:
    python train_apple_pitlid_keras.py --data_dir ../data/apple_pitlid_split --output_dir ../runs/apple_seed1_keras
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from clr_callback import CyclicLR


def build_model(num_classes: int, weight_decay: float) -> keras.Model:
    base_model = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",          # = GlobalAveragePooling2D, built into the base model output
        input_shape=(299, 299, 3),
    )
    # GAP output connects DIRECTLY to the softmax output layer (paper spec, no extra FC).
    predictions = keras.layers.Dense(
        num_classes, activation="softmax", kernel_regularizer=l2(weight_decay)
    )(base_model.output)
    model = keras.Model(inputs=base_model.input, outputs=predictions)

    # Fine-tune ALL layers (paper: "None_frozen") + L2 on every layer that has kernel weights,
    # matching the original repo's own `layer.W_regularizer = l2(...)` intent.
    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, "kernel_regularizer"):
            layer.kernel_regularizer = l2(weight_decay)

    return model


def build_generators(data_dir: Path, batch_size_train: int, batch_size_val: int):
    # class list is auto-detected (alphabetical) from data_dir/train's subfolders,
    # same convention as torchvision.datasets.ImageFolder, so this works for any
    # crop produced by data_prep/make_pitlid_split.py (apple/grape/peach/...).
    classes = sorted(p.name for p in (data_dir / "train").iterdir() if p.is_dir())

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        data_dir / "train",
        target_size=(299, 299),
        batch_size=batch_size_train,
        class_mode="categorical",
        classes=classes,
        shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        data_dir / "val",
        target_size=(299, 299),
        batch_size=batch_size_val,
        class_mode="categorical",
        classes=classes,
        shuffle=True,
    )
    test_gen = eval_datagen.flow_from_directory(
        data_dir / "test",
        target_size=(299, 299),
        batch_size=batch_size_val,
        class_mode="categorical",
        classes=classes,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen, classes


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
    ax.set_title("Confusion matrix (PiTLiD-Keras - Apple)")
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
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size_train", type=int, default=32)
    ap.add_argument("--batch_size_val", type=int, default=16)
    ap.add_argument("--base_lr", type=float, default=0.001)
    ap.add_argument("--max_lr", type=float, default=0.006)
    ap.add_argument("--clr_step_size", type=float, default=2000.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=14)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen, class_names = build_generators(data_dir, args.batch_size_train, args.batch_size_val)
    print(f"classes: {class_names}")
    print(f"train={train_gen.samples} val={val_gen.samples} test={test_gen.samples}")

    # Keras 3's fit() doesn't reliably loop a legacy DirectoryIterator past its own
    # length ("Your input ran out of data" once steps_per_epoch * batch_size exceeds
    # the 120-image train set). Wrap it as an explicitly infinite tf.data.Dataset so
    # steps_per_epoch=200 can resample the small train set with replacement each epoch,
    # matching the classic Keras `fit_generator(steps_per_epoch=...)` behavior the
    # paper's protocol assumes.
    output_signature = (
        tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(class_names)), dtype=tf.float32),
    )
    train_ds = tf.data.Dataset.from_generator(lambda: train_gen, output_signature=output_signature).repeat()

    model = build_model(len(class_names), args.weight_decay)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=args.base_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    clr = CyclicLR(base_lr=args.base_lr, max_lr=args.max_lr, step_size=args.clr_step_size, mode="triangular2")
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
    )
    ckpt_path = out_dir / "best_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=0
    )

    history = model.fit(
        train_ds,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=[clr, early_stop, checkpoint],
        verbose=2,
    )

    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    x = range(len(history.history["loss"]))
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(x, history.history["accuracy"], "o-", label="train")
    axes[0].plot(x, history.history["val_accuracy"], "o-", label="val")
    axes[0].set_ylabel("accuracy"); axes[0].legend(); axes[0].set_title("Accuracy vs epoch")
    axes[1].plot(x, history.history["loss"], ".-", label="train")
    axes[1].plot(x, history.history["val_loss"], ".-", label="val")
    axes[1].set_ylabel("loss"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].set_title("Loss vs epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_loss.png", dpi=150)
    plt.close(fig)

    # ---- test-set evaluation (best weights already restored by EarlyStopping) ----
    test_gen.reset()
    probs = model.predict(test_gen, steps=int(np.ceil(test_gen.samples / args.batch_size_val)), verbose=0)
    y_pred = np.argmax(probs, axis=1)[: test_gen.samples]
    y_true = test_gen.classes

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
        "stopped_epoch": len(history.history["loss"]),
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nsaved: {ckpt_path}, history.json, accuracy_loss.png, confusion_matrix.png, test_metrics.json")


if __name__ == "__main__":
    main()
