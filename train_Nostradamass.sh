#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory
export CUDA_VISIBLE_DEVICES=$2

channel=$1
metcovstd=$3
metcovmean=$4
metcorrstd=$5
dir="trainings/neutrinos_152_10x500_EarlyStopping10_dPTtau_relu_Fullmass_lossM2_0p1dz"
dir="trainings/my_training"

# full dataset at ETP
#input="/storage/b/friese/toymass13_all/*_${channel}*.root"
# selected subset at ETP
input="/storage/b/friese/toymass13_lowmass/m_*_0_${channel}*.root"

python train_Nostradamass.py $channel ${dir}/${channel} $metcovstd $metcovmean $metcorrstd $input

