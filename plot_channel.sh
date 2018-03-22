#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory

export CUDA_VISIBLE_DEVICES=$2

channel=$1
mass=$3
#dir="trainings/neutrinos_90_dPComb_rollback"
#dir="trainings/neutrinos_90_dPComb_dmTau2"
#dir="trainings/neutrinos_90_dPComb_dmTau3sqrt"
#dir="trainings/neutrinos_90_dPComb_dmTau2fullstat"
#dir="trainings/neutrinos_90_dPComb_dmTau2fullstat_consequent"
#dir="trainings/neutrinos_90_dPComb_dmTau2fullstat_consequent"
#dir="trainings/neutrinos_90_dPComb_dmTau2fullstat_consequent_100k_0p01"
dir="trainings/neutrinos_91_gehe"
#input="/storage/b/friese/toymass5/m_*_${channel}_*.root /storage/b/friese/toymass6/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_*_${channel}_*.root /storage/b/friese/toymass8/m_*_${channel}_*.root"
input="/storage/b/friese/toymass7/m_${3}_*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_${3}_*_${channel}_*.root"
#input=$(ls /storage/b/friese/toymass5/m_${3}_*_${channel}_*.root /storage/b/friese/toymass6/m_${3}_*_${channel}_*.root)
#input=${dir}/${channel}/cache.pkl
model=$(ls ${dir}/${channel}/*.hdf5 | sort | tail -n1)


echo  python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input
#python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input
python apply_toymass.py $channel $model ${dir}/${channel}/data
