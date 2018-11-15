import matplotlib.pyplot as plt
import json
import numpy as np


with open("trainings/test_history_2/tt/history.json", "r") as hist:
    hist_dict = json.loads(hist.read())

train_metrics = sorted([m for m in hist_dict if "val_" not in m])
validation_metrics = sorted([m for m in hist_dict if "val_" in m])

for tm, vm in zip(train_metrics, validation_metrics):
    print tm, vm
    epochs = np.array(range(len(hist_dict[tm])))+1.0
    p = plt.subplot()
    plt.plot(epochs,hist_dict[tm])
    plt.plot(epochs,hist_dict[vm])
    #p.set_xscale('log')
    p.set_yscale('log')
    plt.legend([tm, vm], loc='upper right')
    plt.savefig(tm+'.pdf')
    plt.savefig(tm+'.png')
    p.clear()
