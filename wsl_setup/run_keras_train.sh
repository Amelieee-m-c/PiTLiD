#!/bin/bash
# Run train_apple_pitlid_keras.py inside the WSL venv with GPU libraries on LD_LIBRARY_PATH.
# Usage: bash run_keras_train.sh <args forwarded to train_apple_pitlid_keras.py>
set -e
source /home/peter/pitlid_venv/bin/activate

NV_LIB_DIRS=$(find /home/peter/pitlid_venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -iname lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH}"

cd /mnt/e/plant_disease/PiTLiD_repro/src
python3 train_apple_pitlid_keras.py "$@"
