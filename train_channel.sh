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
#dir="trainings/neutrinos_140_10xrect_EarlyStopping20_dPtau_dPTtau_relu_newmassTemp"
#dir="trainings/neutrinos_142_10xrect_EarlyStopping10_dPtau_dPTtau_relu_medmass_DEMO"
#dir="trainings/neutrinos_144_10x500_EarlyStopping10_dPTtau_relu_Fullmass"
#dir="trainings/neutrinos_145_5x1000_EarlyStopping10_dPTtau_relu_Fullmass"
#dir="trainings/neutrinos_148_20x500_EarlyStopping10_dPTtau_relu_Fullmass"
#dir="trainings/neutrinos_149_5x1000_EarlyStopping10_dPTtau_relu_Fullmass_lossM"
#dir="trainings/neutrinos_151_10x500_EarlyStopping10_dPTtau_relu_Fullmass_lossM_0p1dz"
dir="trainings/neutrinos_152_10x500_EarlyStopping10_dPTtau_relu_Fullmass_lossM2_0p1dz"
#dir="trainings/neutrinos_153_3x1000_EarlyStopping10_dPTtau_relu_Fullmass_lossM2_0p1dz_sqrtLoss"

input="/storage/b/friese/toymass13_all/*_${channel}*.root"
#input="/storage/b/friese/toymass13_all/40to150_${channel}*.root"
#input="/storage/b/friese/toymass13_all/medmass_${channel}.root"

python train_invisibles.py $channel ${dir}/${channel} $input

