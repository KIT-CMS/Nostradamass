#!/bin/bash
set -e # exit on errors

if ! [[ -z "$1" ]]; then
    source $1;
    pip install --user root_pandas
    pip install --user tensorflow-gpu
fi

git clone  git@github.com:KIT-CMS/Nostradamass.git
cd Nostradamass
git submodule init
git submodile update
