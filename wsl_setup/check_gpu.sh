#!/bin/bash
set -e
source /home/peter/pitlid_venv/bin/activate

NV_LIB_DIRS=$(find /home/peter/pitlid_venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -iname lib -type d | tr '\n' ':')
echo "NV_LIB_DIRS=$NV_LIB_DIRS"
export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH}"

python3 -c "
import tensorflow as tf
print('GPUs:', tf.config.list_physical_devices('GPU'))
print('Built with CUDA:', tf.test.is_built_with_cuda())
"
