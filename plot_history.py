import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import os
import glob
from decimal import Decimal

if len(sys.argv) <= 1:
    print "Please provide path to the folder with models up to channel folder."
modelfolderpath = str(sys.argv[1])

with open(os.path.join(modelfolderpath,"history.json"), "r") as hist:
    hist_dict = json.loads(hist.read())

modelfilename = os.path.basename(glob.glob(os.path.join(modelfolderpath, "*.hdf5"))[0])

best_epoch = int(modelfilename.split('-')[0].split('.')[-1])

train_metrics = sorted([m for m in hist_dict if "val_" not in m])
validation_metrics = sorted([m for m in hist_dict if "val_" in m])
max_epoch = -1

for tm, vm in zip(train_metrics, validation_metrics):
    epochs = np.array(range(len(hist_dict[tm])))+1.0
    p = plt.subplot()
    plt.plot(epochs[:max_epoch],hist_dict[tm][:max_epoch])
    plt.plot(epochs[:max_epoch],hist_dict[vm][:max_epoch])
    plt.axvline(x=best_epoch, color='red')
    bestvalue = "best model: %.3E"%Decimal(hist_dict[vm][best_epoch-1])
    #p.set_xscale('log')
    p.set_yscale('log')
    plt.legend([tm, vm, bestvalue], loc='upper right')
    outname = os.path.join(modelfolderpath,tm)
    print "Saving",outname
    plt.savefig(outname+'.pdf')
    plt.savefig(outname+'.png')
    p.clear()
