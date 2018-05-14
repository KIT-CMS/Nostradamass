#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory
export CUDA_VISIBLE_DEVICES=$2

channel=$1
#dir="trainings/neutrinos_68_noPU"
#dir="trainings/neutrinos_76_500_400_300_200_100_PU"
#dir="trainings/neutrinos_79_500_400_300_200_100_toymass7_PU_massLin"
#dir="trainings/neutrinos_80_dmtau1abs_noPU"
#dir="trainings/neutrinos_80_dmtau1abs_noPU_dzForL1"
#dir="trainings/neutrinos_83_nmixed_PU_pT"
#dir="trainings/neutrinos_85_PU_Pt_dAllCartesian_0p1dz"
dir="trainings/neutrinos_101"
#input="/storage/b/friese/toymass5/m_*_${channel}_*.root /storage/b/friese/toymass6/m_*_${channel}_*.root"
input="/storage/b/friese/toymass13/m_*_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_*_${channel}_*.root /storage/b/friese/toymass8/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_*_*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_2*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_9*_0_${channel}_*.root"
#input=${dir}/${channel}/cache.pkl


python train_invisibles.py $channel ${dir}/${channel} $input 

