#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory

export CUDA_VISIBLE_DEVICES=$2

channel=$1
mass=$3
#dir="trainings/neutrinos_91_gehe"
#dir="trainings/neutrinos_91_gehe_oldMC"
#dir="trainings/neutrinos_100_test_withJets"
#dir="trainings/neutrinos_100_test_0Jets"
dir="trainings/neutrinos_101"
#dir="trainings/neutrinos_100_test_2Jets"
input="/storage/b/friese/toymass13/m_${3}_*_${channel}_*.root"
#input="$(ls /storage/b/friese/toymass5/m_${3}_*_${channel}_*.root /storage/b/friese/toymass6/m_${3}_*_${channel}_*.root)"
#input="/storage/b/friese/toymass7/m_*_${channel}_*.root /storage/b/friese/toymass8/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_${3}_*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_${3}_*_${channel}_*.root"
#input=$(ls /storage/b/friese/toymass5/m_${3}_*_${channel}_*.root /storage/b/friese/toymass6/m_${3}_*_${channel}_*.root)
#input=${dir}/${channel}/cache.pkl
model=$(ls ${dir}/${channel}/*.hdf5 | sort | tail -n1)


echo python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input
#python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input::
echo python apply_toymass.py $channel $model ${dir}/${channel}/data
