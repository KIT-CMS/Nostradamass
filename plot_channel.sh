#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory

export CUDA_VISIBLE_DEVICES=$2

channel=$1
mass=$3
#dir="trainings/neutrinos_68_noPU"
dir="trainings/neutrinos_69"
#input="/storage/b/friese/toymass5/m_*_${channel}_*.root /storage/b/friese/toymass6/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_*_${channel}_*.root"
input=$(ls /storage/b/friese/toymass5/m_${3}_*_${channel}_*.root /storage/b/friese/toymass6/m_${3}_*_${channel}_*.root)
#input=${dir}/${channel}/cache.pkl
model=$(ls ${dir}/${channel}/*.hdf5 | sort | tail -n1)


python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input
python apply_toymass.py $channel $model ${dir}/${channel}/data
