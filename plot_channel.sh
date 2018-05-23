#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory

#export CUDA_VISIBLE_DEVICES=$2
export CUDA_VISIBLE_DEVICES=

channel=$1
mass=$3
#dir="trainings/neutrinos_91_gehe"
#dir="trainings/neutrinos_91_gehe_oldMC"
dir="trainings/neutrinos_133_10x500_EarlyStopping100_dPtau_dPTtau_relu_fullmass"
#dir="trainings/neutrinos_132_8xrect_EarlyStopping100_dPtau_dPTtau_relu_lowmass/"
#input="/storage/b/friese/toymass13/m_${3}_*_${channel}_*.root"
if [ "$3" -gt 1000 ]; then
	input="/storage/b/friese/toymass13_highmass/m_${3}_*_${channel}_*.root"
else
	input="/storage/b/friese/toymass13_10k/m_${3}_*_${channel}_*.root"
fi
echo $input
#input="$(ls /storage/b/friese/toymass5/m_${3}_*_${channel}_*.root /storage/b/friese/toymass6/m_${3}_*_${channel}_*.root)"
#input="/storage/b/friese/toymass7/m_*_${channel}_*.root /storage/b/friese/toymass8/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_${3}_*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_${3}_*_${channel}_*.root"
#input=$(ls /storage/b/friese/toymass5/m_${3}_*_${channel}_*.root /storage/b/friese/toymass6/m_${3}_*_${channel}_*.root)
#input=${dir}/${channel}/cache.pkl
model=$(ls ${dir}/${channel}/*.hdf5 | sort | tail -n1)
name=$(echo $dir | cut -d / -f2)

python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input # > log_${mass}_${name}.txt
#cat log_${mass}_${name}.txt
#python apply_toymass.py $channel $model ${dir}/${channel}/data
