"""
Aggregate test_metrics.json across seed_* run directories into mean +/- std,
reproducing the paper's "independently run 10 times" reporting format.

Run:
    python aggregate_multi_seed.py --runs_root ../runs/apple_10run_keras
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, help="dir containing seed_*/test_metrics.json")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    seed_dirs = sorted(runs_root.glob("seed_*"))
    all_metrics = []
    for d in seed_dirs:
        f = d / "test_metrics.json"
        if f.exists():
            with open(f) as fh:
                all_metrics.append(json.load(fh))
        else:
            print(f"WARNING: missing {f}, skipping")

    if not all_metrics:
        raise SystemExit(f"no test_metrics.json found under {runs_root}")

    keys = ["accuracy", "precision_macro", "recall_sensitivity_macro", "f1_macro"]
    summary = {"n_runs": len(all_metrics)}
    print(f"=== {runs_root.name}: summary over {len(all_metrics)} runs ===")
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
        print(f"{k:28s} mean={vals.mean()*100:.2f}%  std={vals.std()*100:.2f}%")

    out_path = runs_root / "summary_multirun.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
