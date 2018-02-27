#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc7-opt/setup.sh
MASS=$1
SEED=$2
CHANNEL=$3
INVERT=$4
OUT_DIR=$5
/usr/users/friese/toymass/pythia_eventgeneration/eventgeneration $MASS $SEED $CHANNEL $INVERT >> $OUT_DIR/logfile_${MASS}_${SEED}_${CHANNEL}_${INVERT}.log
cp *.root $OUT_DIR

