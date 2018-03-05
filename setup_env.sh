#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-ubuntu1604-gcc54-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc7-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc7-opt/setup.sh

pip install --user tensorflow-gpu
pip install --user root_pandas
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export PATH=/usr/local/cuda-9.0/lib64/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH
