#!/bin/bash
#virtualenv test
source ../deepmet/toy/test/bin/activate
#pip install keras
#pip install tensorflow-gpu
#pip install matplotlib
export PATH=$PATH:/usr/local/cuda-8.0/bin/
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/

export OMP_NUM_THREADS=12
export THEANO_FLAGS='device=gpu2'
export CUDA_VISIBLE_DEVICES='0'
export KERAS_BACKEND=tensorflow
