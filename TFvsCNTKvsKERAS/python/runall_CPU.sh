#!/bin/bash

file_name="performance_CPU.txt"
rm $file_name
touch $file_name

vgg=20
rl=500
rnn=500
conv=500

echo "KERAS tests with TF backend."
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=tensorflow python keras_VGG.py --num-runs $vgg --file-name $file_name
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=tensorflow python keras_RL.py --num-runs $rl --file-name $file_name
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=tensorflow python keras_RNN.py --num-runs $rnn --file-name $file_name
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=tensorflow python keras_CONV.py --num-runs $conv --file-name $file_name

echo "KERAS tests with CNTK backend."
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=cntk python keras_VGG.py --num-runs $vgg --file-name $file_name
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=cntk python keras_RL.py --num-runs $rl --file-name $file_name
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=cntk python keras_RNN.py --num-runs $rnn --file-name $file_name
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=cntk python keras_CONV.py --num-runs $conv --file-name $file_name

echo "CNTK tests."
CUDA_VISIBLE_DEVICES="" python cntk_VGG.py --num-runs $vgg --file-name $file_name 
CUDA_VISIBLE_DEVICES="" python cntk_RL.py --num-runs $rl --file-name $file_name
CUDA_VISIBLE_DEVICES="" python cntk_RNN.py --num-runs $rnn --file-name $file_name
CUDA_VISIBLE_DEVICES="" python cntk_CONV.py --num-runs $conv --file-name $file_name

echo "TF tests."
CUDA_VISIBLE_DEVICES="" python tf_VGG.py --num-runs $vgg --file-name $file_name
CUDA_VISIBLE_DEVICES="" python tf_RL.py --num-runs $rl --file-name $file_name
CUDA_VISIBLE_DEVICES="" python tf_RNN.py --num-runs $rnn --file-name $file_name
CUDA_VISIBLE_DEVICES="" python tf_CONV.py --num-runs $conv --file-name $file_name