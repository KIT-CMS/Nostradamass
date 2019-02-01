import matplotlib.pyplot as plt
import json
import numpy as np


with open("trainings/test_history_2/tt/history.json", "r") as hist:
    hist_dict = json.loads(hist.read())

train_metrics = sorted([m for m in hist_dict if "val_" not in m])
validation_metrics = sorted([m for m in hist_dict if "val_" in m])
max_epoch = 200

for tm, vm in zip(train_metrics, validation_metrics):
    print tm, vm
    epochs = np.array(range(len(hist_dict[tm])))+1.0
    p = plt.subplot()
    plt.plot(epochs[:max_epoch],hist_dict[tm][:max_epoch])
    plt.plot(epochs[:max_epoch],hist_dict[vm][:max_epoch])
    #p.set_xscale('log')
    p.set_yscale('log')
    plt.legend([tm, vm], loc='upper right')
    plt.savefig(tm+'.pdf')
    plt.savefig(tm+'.png')
    p.clear()
