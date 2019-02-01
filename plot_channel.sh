#!/bin/bash

#perform a single training and produce some basic evaluation plots in the same directory

export CUDA_VISIBLE_DEVICES=

channel=$1
mass=$2
#dir="trainings/neutrinos_1_10x500_EarlyStopping10_dPTtau_relu_Fullmass_lossM2_0p1dz_MetConditions2017_BCD"
#dir="trainings/neutrinos_1_10x500_EarlyStopping10_dPTtau_relu_Fullmass_lossM2_0p1dz_MetConditions2017_EF"
#dir="trainings/neutrinos_1_10x500_EarlyStopping10_dPTtau_relu_Fullmass_lossM2_0p1dz_realisticMetConditions2017"
dir="pre-trained-models/v2"
#dir="trainings/neutrinos_2_3x1000_EarlyStopping25_dPTtau_relu_Fullmass_lossM2_0p1dz_realisticMetConditions2017Sqrt"
dir="trainings/neutrinos_1_3x1000_EarlyStopping50_dPTtau_elu_Fullmass_lossM2_0p1dz_realisticMetConditions2017Sqrt_with_history"
dir="trainings/neutrinos_1_10x500_EarlyStopping50_dPTtau_elu_Fullmass_lossM2_0p1dz_realisticMetConditions2017Sqrt_with_history"
dir="trainings/neutrinos_1_10x500_EarlyStopping300_dPTtau_elu_Fullmass_lossM2_0p1dz_realisticMetConditions2017Sqrt_with_history"
dir="trainings/neutrinos_1_Gau??_10x500_EarlyStopping300_dPTtau_elu_Fullmass_lossM2_0p1dz_realisticMetConditions2017Sqrt_with_history"
dir="trainings/neutrinos_1_Gauß3_10x500_EarlyStopping50_dPTtau_elu_Fullmass_lossM2_0p1dz_realisticMetConditions2017Sqrt_with_history"
#dir="trainings/Raphaels"
if [ "$2" -gt 1000 ]; then
	input="/storage/b/friese/toymass13_highmass/m_${3}_*_${channel}_*.root"
else
	input="/storage/b/friese/toymass13_10k/m_${3}_*_${channel}_*.root"
fi
echo $input
model=$(ls ${dir}/${channel}/*.hdf5| sort | tail -n1)
#model=$(ls ${dir}/${channel}.hdf5 | sort | tail -n1)
name=$(echo $dir | cut -d / -f2)

#python plot_invisibles.py $channel $model ${dir}/${channel}/plotsm_${mass} $input # > log_${mass}_${name}.txt
python apply_toymass.py $channel $model ${dir}/${channel}/data
#python plot_toymass.py $channel $model ${dir}/${channel}/data
#python plot_stability.py $channel $model ${dir}/${channel}/data
