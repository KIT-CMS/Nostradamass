#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory
export CUDA_VISIBLE_DEVICES=$2

channel=$1
#dir="trainings/neutrinos_68_noPU"
dir="trainings/neutrinos_71_5x300_PU"
input="/storage/b/friese/toymass5/m_*_${channel}_*.root /storage/b/friese/toymass6/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_2*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_9*_0_${channel}_*.root"
#input=${dir}/${channel}/cache.pkl


python train_invisibles.py $channel ${dir}/${channel} $input 

