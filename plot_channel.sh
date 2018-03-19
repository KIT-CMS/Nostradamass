#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory

export CUDA_VISIBLE_DEVICES=$2

channel=$1
mass=$3
#dir="trainings/neutrinos_68_noPU"
#dir="trainings/neutrinos_69"
#dir="trainings/neutrinos_71_5x300_noPU"
#dir="trainings/neutrinos_76_500_400_300_200_100_PU"
#dir="trainings/neutrinos_75_400_300_200_100_PU_24_7_Nodz"
#dir="trainings/neutrinos_79_500_400_300_200_100_toymass7_PU_dmtauNosquareOnly1_v2" #### good!
#dir="trainings/neutrinos_80_dmtau1abs"
#dir="trainings/neutrinos_80_dmtau1abs_noPU"
#dir="trainings/neutrinos_80_dmtau1abs_noPU_dzForL1"
#dir="trainings/neutrinos_81_new_noPU"
##dir="trainings/neutrinos_81_new_noPU_bothTausmassconst"
#dir="trainings/neutrinos_82_new_noPU"
#dir="trainings/neutrinos_83_nmixed_PU"
dir="trainings/neutrinos_85_noPU_Pt_dmBoson"
#input="/storage/b/friese/toymass5/m_*_${channel}_*.root /storage/b/friese/toymass6/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_*_${channel}_*.root /storage/b/friese/toymass8/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_${3}_*_${channel}_*.root"
input="/storage/b/friese/toymass5/m_${3}_*_${channel}_*.root"
#input=$(ls /storage/b/friese/toymass5/m_${3}_*_${channel}_*.root /storage/b/friese/toymass6/m_${3}_*_${channel}_*.root)
#input=${dir}/${channel}/cache.pkl
model=$(ls ${dir}/${channel}/*.hdf5 | sort | tail -n1)


python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input
#python apply_toymass.py $channel $model ${dir}/${channel}/data
