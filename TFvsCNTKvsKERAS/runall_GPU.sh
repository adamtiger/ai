#!/bin/bash

file_name="performance.txt"
rm $file_name
touch $file_name

echo "CNTK tests."
CUDA_VISIBLE_DEVICE=0 python cntk_VGG.py --num-runs 10 --file-name $file_name 
CUDA_VISIBLE_DEVICE=0 python cntk_RL.py --num-runs 100 --file-name $file_name
CUDA_VISIBLE_DEVICE=0 python cntk_RNN.py --num-runs 100 --file-name $file_name

echo "TF tests."
CUDA_VISIBLE_DEVICE=0 python tf_VGG.py --num-runs 10 --file-name $file_name
CUDA_VISIBLE_DEVICE=0 python tf_RL.py --num-runs 100 --file-name $file_name
# CUDA_VISIBLE_DEVICE=0 python tf_RNN.py --num-runs 100 --file-name $file_name

echo "KERAS tests." # change the backend outside
CUDA_VISIBLE_DEVICE=0 python keras_VGG.py --num-runs 10 --file-name $file_name
CUDA_VISIBLE_DEVICE=0 python keras_RL.py --num-runs 100 --file-name $file_name
CUDA_VISIBLE_DEVICE=0 python keras_RNN.py --num-runs 100 --file-name $file_name