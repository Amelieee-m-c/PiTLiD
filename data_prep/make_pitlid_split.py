"""
Build a PiTLiD-paper-style small-sample split from the pooled PlantVillage data,
for any of the crops the paper evaluates (Apple = main task, Grape/Peach =
generalization/robustness tests, all under the same 30-shot protocol).

Paper setting (same for every crop):
  - train: 30 images/class picked at random
  - remaining images split ~1:1 into val / test (stratified per class;
    on an odd remainder, the extra image goes to val, matching the paper's
    Table 1 for Apple)

Source: E:/plant_disease/PlantVillage_full/<class>/*  (pooled, unsplit PlantVillage)
Output: <output_dir>/{train,val,test}/<class>/*  (files copied, source untouched)
"""
import argparse
import random
import shutil
from pathlib import Path

CROP_CLASSES = {
    "apple": [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
    ],
    "grape": [
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
    ],
    "peach": [
        "Peach___Bacterial_spot",
        "Peach___healthy",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True, choices=sorted(CROP_CLASSES))
    ap.add_argument("--source_dir", default="E:/plant_disease/PlantVillage_full")
    ap.add_argument("--output_dir", default=None, help="default: PiTLiD_repro/data/<crop>_pitlid_split")
    ap.add_argument("--n_train_per_class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    classes = CROP_CLASSES[args.crop]
    src_root = Path(args.source_dir)
    out_root = Path(args.output_dir) if args.output_dir else Path(
        f"E:/plant_disease/PiTLiD_repro/data/{args.crop}_pitlid_split"
    )
    rng = random.Random(args.seed)

    if out_root.exists():
        shutil.rmtree(out_root)

    summary = []
    for cls in classes:
        src_cls = src_root / cls
        files = sorted(p for p in src_cls.iterdir() if p.is_file())
        if len(files) <= args.n_train_per_class:
            raise ValueError(f"{cls} has only {len(files)} images, need > {args.n_train_per_class}")

        rng.shuffle(files)
        train_files = files[: args.n_train_per_class]
        remaining = files[args.n_train_per_class :]

        n_val = -(-len(remaining) // 2)  # ceil, extra odd image goes to val
        val_files = remaining[:n_val]
        test_files = remaining[n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dst_dir = out_root / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst_dir / f.name)

        summary.append((cls, len(train_files), len(val_files), len(test_files)))

    print(f"crop={args.crop}  seed={args.seed}  output={out_root}")
    print(f"{'class':45s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for cls, n_tr, n_val, n_te in summary:
        print(f"{cls:45s} {n_tr:6d} {n_val:6d} {n_te:6d}")


if __name__ == "__main__":
    main()
