"""
Build the PiTLiD paper's small-sample Apple split from the pooled PlantVillage data.

Paper setting:
  - 4 classes: Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy
  - train: 30 images/class picked at random (120 total)
  - remaining images split ~1:1 into val / test (stratified per class)

Source: E:/plant_disease/PlantVillage_full/<class>/*  (pooled, unsplit PlantVillage)
Output: <output_dir>/{train,val,test}/<class>/*  (files copied, source untouched)
"""
import argparse
import random
import shutil
from pathlib import Path

CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default="E:/plant_disease/PlantVillage_full")
    ap.add_argument("--output_dir", default="E:/plant_disease/PiTLiD_repro/data/apple_pitlid_split")
    ap.add_argument("--n_train_per_class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    src_root = Path(args.source_dir)
    out_root = Path(args.output_dir)
    rng = random.Random(args.seed)

    if out_root.exists():
        shutil.rmtree(out_root)

    summary = []
    for cls in CLASSES:
        src_cls = src_root / cls
        files = sorted(p for p in src_cls.iterdir() if p.is_file())
        if len(files) <= args.n_train_per_class:
            raise ValueError(f"{cls} has only {len(files)} images, need > {args.n_train_per_class}")

        rng.shuffle(files)
        train_files = files[: args.n_train_per_class]
        remaining = files[args.n_train_per_class :]

        # paper's Table 1: on an odd remainder, the extra image goes to val (not test)
        n_val = -(-len(remaining) // 2)  # ceil
        val_files = remaining[:n_val]
        test_files = remaining[n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dst_dir = out_root / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst_dir / f.name)

        summary.append((cls, len(train_files), len(val_files), len(test_files)))

    print(f"seed={args.seed}  output={out_root}")
    print(f"{'class':30s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for cls, n_tr, n_val, n_te in summary:
        print(f"{cls:30s} {n_tr:6d} {n_val:6d} {n_te:6d}")


if __name__ == "__main__":
    main()
