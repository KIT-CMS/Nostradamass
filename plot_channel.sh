#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory

export CUDA_VISIBLE_DEVICES=

channel=$1
mass=$3
dir="trainings/neutrinos_152_10x500_EarlyStopping10_dPTtau_relu_Fullmass_lossM2_0p1dz"
if [ "$3" -gt 1000 ]; then
	input="/storage/b/friese/toymass13_highmass/m_${3}_*_${channel}_*.root"
else
	input="/storage/b/friese/toymass13_10k/m_${3}_*_${channel}_*.root"
fi
echo $input
model=$(ls ${dir}/${channel}/*.hdf5 | sort | tail -n1)
name=$(echo $dir | cut -d / -f2)

python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input # > log_${mass}_${name}.txt
python apply_toymass.py $channel $model ${dir}/${channel}/data
