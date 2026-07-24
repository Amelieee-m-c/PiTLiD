#!/bin/bash
set -e
source /home/peter/pitlid_venv/bin/activate
NV_LIB_DIRS=$(find /home/peter/pitlid_venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -iname lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH}"

cd /mnt/e/plant_disease/PiTLiD_repro/src

echo "##### GRAPE seed=1 #####"
python3 train_apple_pitlid_keras.py \
  --data_dir /mnt/e/plant_disease/PiTLiD_repro/data/grape_pitlid_split \
  --output_dir /mnt/e/plant_disease/PiTLiD_repro/runs/grape_seed1_keras \
  --seed 1

echo "##### PEACH seed=1 #####"
python3 train_apple_pitlid_keras.py \
  --data_dir /mnt/e/plant_disease/PiTLiD_repro/data/peach_pitlid_split \
  --output_dir /mnt/e/plant_disease/PiTLiD_repro/runs/peach_seed1_keras \
  --seed 1

echo "##### DONE #####"
