#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory
export CUDA_VISIBLE_DEVICES=$2

channel=$1
#dir="trainings/neutrinos_85_PU_Pt_dAllCartesian_0p1dz"
#dir="trainings/neutrinos_111_5x1000_Dropout0p2_EarlyStopping250"
#dir="trainings/neutrinos_112_5x1000_Dropout0p2_EarlyStopping250_LinLoss"
#dir="trainings/neutrinos_115_6x500_EarlyStopping100_dPttau_dPtR_Dropout0p1"
#dir="trainings/neutrinos_118_8x500_EarlyStopping100_dPttau_norm"
#dir="trainings/neutrinos_121_9xrect_EarlyStopping100_dPttau_norm_GaussianNoise3"
#dir="trainings/neutrinos_122_9xrect_EarlyStopping100_dPttau_norm_GaussianNoise2_relu"
#dir="trainings/neutrinos_127_8xrect_EarlyStopping100_dPttau_relu_dE_nu_Dropout0p05"
#dir="trainings/neutrinos_129_8xrect_EarlyStopping100_dPtau_relu_Dropout0p05Rect"
#dir="trainings/neutrinos_130_8xrect_EarlyStopping100_dPtau_dPTtau_relu_Dropout0p05Rect"
dir="trainings/neutrinos_134_10x1000_EarlyStopping100_dPtau_dPTtau_relu_fullmass"
#input="/storage/b/friese/toymass5/m_*_${channel}_*.root /storage/b/friese/toymass6/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass13_highmass/m_*_*_${channel}_*.root"
input="/storage/b/friese/toymass13_10k/m_*_*_${channel}_*.root /storage/b/friese/toymass13_highmass/m_*_*_${channel}_*.root"
#input="/storage/b/friese/toymass13/m_*_*_${channel}_*.root /storage/b/friese/toymass13_highmass/m_*_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_*_${channel}_*.root /storage/b/friese/toymass8/m_*_${channel}_*.root"
#input="/storage/b/friese/toymass7/m_*_*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_2*_${channel}_*.root"
#input="/storage/b/friese/toymass5/m_9*_0_${channel}_*.root"
#input=${dir}/${channel}/cache.pkl

python train_invisibles.py $channel ${dir}/${channel} $input

