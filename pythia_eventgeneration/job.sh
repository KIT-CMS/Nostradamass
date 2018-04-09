#!/bin/bash
ls -al
ls /cvmfs/sft.cern.ch/
set -e
MASS=$1
SEED=$2
CHANNEL=$3
INVERT=$4
OUT_DIR=$5
echo "testing if the output works"
(
here=$(pwd)
testfile=${MASS}_${SEED}_${CHANNEL}_${INVERT}.txt
echo ${MASS}_${SEED}_${CHANNEL}_${INVERT} > $testfile
source /cvmfs/grid.cern.ch/emi3ui-latest/etc/profile.d/setup-ui-example.sh
echo "in subshell"
which python
echo gfal-copy file:/$here/$testfile srm://cmssrm-kit.gridka.de:8443/srm/managerv2?SFN=/pnfs/gridka.de/cms/disk-only/store/user/rfriese/$OUT_DIR/$testfile
gfal-copy file:/$here/$testfile srm://cmssrm-kit.gridka.de:8443/srm/managerv2?SFN=/pnfs/gridka.de/cms/disk-only/store/user/rfriese/$OUT_DIR/$testfile
)
echo "out of subshell"
which python


(
source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc7-opt/setup.sh
echo "in subshell2"
which python
./eventgeneration $MASS $SEED $CHANNEL $INVERT
)
echo "event generation done"
(
here=$(pwd)
rf=$(ls *.root)
source /cvmfs/grid.cern.ch/emi3ui-latest/etc/profile.d/setup-ui-example.sh
echo "in subshell3"
which python
gfal-copy file:/$here/$rf srm://cmssrm-kit.gridka.de:8443/srm/managerv2?SFN=/pnfs/gridka.de/cms/disk-only/store/user/rfriese/$OUT_DIR/$rf
)

