#!/bin/bash
# Full PiTLiD 10-run stability protocol, for one or more crops, on Keras/TF + GPU (WSL).
#
# Each (crop, seed) draws a FRESH random 30-shot split and trains from scratch,
# matching the paper's "independently run 10 times" evaluation.
#
# Usage:
#   bash run_multi_crop_10run.sh "apple grape peach" 10 1
#   (crops list, n_runs, start_seed -- all optional, shown defaults below)
set -e
source /home/peter/pitlid_venv/bin/activate
NV_LIB_DIRS=$(find /home/peter/pitlid_venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -iname lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH}"

CROPS="${1:-apple grape peach}"
N_RUNS="${2:-10}"
START_SEED="${3:-1}"

REPO=/mnt/e/plant_disease/PiTLiD_repro
cd "$REPO/src"

for crop in $CROPS; do
    runs_root="$REPO/runs/${crop}_10run_keras"
    mkdir -p "$runs_root"
    for i in $(seq 0 $((N_RUNS - 1))); do
        seed=$((START_SEED + i))
        split_dir="$runs_root/seed_${seed}/split"
        run_dir="$runs_root/seed_${seed}"
        echo ""
        echo "##### crop=$crop  run $((i+1))/$N_RUNS  seed=$seed #####"

        python3 "$REPO/data_prep/make_pitlid_split.py" \
            --crop "$crop" --seed "$seed" --output_dir "$split_dir"

        python3 train_apple_pitlid_keras.py \
            --data_dir "$split_dir" \
            --output_dir "$run_dir" \
            --seed "$seed"
    done

    echo ""
    echo "##### crop=$crop aggregate #####"
    python3 aggregate_multi_seed.py --runs_root "$runs_root"
done

echo ""
echo "##### ALL DONE #####"
