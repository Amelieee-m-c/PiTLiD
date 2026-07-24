"""
PiTLiD paper protocol: "under a fixed hyperparameter setting, the model is run
independently 10 times on the test set to confirm stability" (paper reports a
mean accuracy of 99.45% on the Apple 4-class task under this protocol).

Each run: (1) draws a fresh random 30-shot train / ~1:1 val:test split from the
pooled PlantVillage data with that run's seed, (2) trains from scratch with that
seed, (3) evaluates on its own held-out test split. Results are aggregated as
mean +/- std over the 10 runs.

Run:
    python run_multi_seed.py --n_runs 10 --output_root ../runs/apple_10run
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_PREP = HERE.parent / "data_prep" / "make_pitlid_apple_split.py"
TRAIN = HERE / "train_apple_pitlid.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default="E:/plant_disease/PlantVillage_full")
    ap.add_argument("--output_root", default="../runs/apple_10run")
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--start_seed", type=int, default=1)
    # forwarded training hyperparameters (defaults match the paper; override if needed)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--extra_train_args", default="", help="extra CLI args appended verbatim to train_apple_pitlid.py")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for i in range(args.n_runs):
        seed = args.start_seed + i
        run_dir = output_root / f"seed_{seed}"
        split_dir = run_dir / "split"
        print(f"\n=== run {i+1}/{args.n_runs}  seed={seed} ===")

        subprocess.run(
            [sys.executable, str(DATA_PREP),
             "--source_dir", args.source_dir,
             "--output_dir", str(split_dir),
             "--seed", str(seed)],
            check=True,
        )

        train_cmd = [
            sys.executable, str(TRAIN),
            "--data_dir", str(split_dir),
            "--output_dir", str(run_dir),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--steps_per_epoch", str(args.steps_per_epoch),
        ]
        if args.extra_train_args:
            train_cmd += args.extra_train_args.split()
        subprocess.run(train_cmd, check=True)

        with open(run_dir / "test_metrics.json") as f:
            m = json.load(f)
        all_metrics.append(m)

    keys = ["accuracy", "precision_macro", "recall_sensitivity_macro", "f1_macro"]
    summary = {}
    print("\n=== summary over {} runs ===".format(args.n_runs))
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
        print(f"{k:28s} mean={vals.mean()*100:.2f}%  std={vals.std()*100:.2f}%")

    with open(output_root / "summary_10run.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary: {output_root / 'summary_10run.json'}")


if __name__ == "__main__":
    main()
