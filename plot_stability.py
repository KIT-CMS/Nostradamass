import csv
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_invisibles import colors
import time
import glob
import re
import json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--channel",type=str,default=None)
parser.add_argument("--models",nargs="+",default=None)
parser.add_argument("--modelnames",nargs="+",default=None)
parser.add_argument("--outpath",type=str,default=None)

args = parser.parse_args()

if args.channel is None or args.models is None or args.outpath is None or args.modelnames is None:
    print "Please provide arguments. See --help"
    exit(1)

if len(args.models) != len(args.modelnames):
    print "number of paths to models and number of their names must be equal"
    exit(1)

channel_name = {"tt": r'$\tau_{h} \tau_{h}$', "mt": r'$\mu \tau_{h}$', "em" : r'$e \mu$', "et" : r'$e \tau_{h}$'}

channel = args.channel
modelpaths = args.models
modelnames = args.modelnames
outpath = args.outpath

channel_outpath = os.path.join(args.outpath,args.channel)


if not os.path.exists(channel_outpath):
    os.makedirs(channel_outpath)


signal_patterns = {
    "SMH" : "^(VBFH|GluGluH|WplusH|WminusH|ZH)",
    "SUSYbbH" : "BBHToTauTau",
    "SUSYggH" : "SUSYGluGluToH",

}

colors_nn = {
    "2016" : "red",
    "2016oldPrescription" : "red",
    "2017BCD" : "orange",
    "Raphaels" : "orange",
    "2017EF" : "green",
    "2017realistic" : "green",
}



for s in signal_patterns:
    if "SUSY" in s:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ranges = [70,3500]
        y_min = 0.0001
        yranges = [y_min,1.0]
        titles =  r'fraction of events $\|m_N - m_H\|/m_H \geq 2$'

        ax.set_xlabel(r'Generator mass $m_H$ (GeV)')
        ax.set_ylabel(titles)
        ax.set_title("Method Stability (" + channel_name[channel] + ")")
        ax.set_xlim(ranges)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([100,200,400,1000,2000,3000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(color='gray', linestyle='-', linewidth=1, which='major')
        ax.grid(color='gray', linestyle='-',alpha=0.3, linewidth=1, which='minor')
        ax.set_ylim(yranges)
        svplotted = False

        for model,name in zip(modelpaths,modelnames):

            results = {}
            results_path = os.path.join(model,channel,"data","%s.json"%s.replace("SUSY",""))
            j = glob.glob(results_path)
            if len(j) == 1:
                f = open(j[0],"r")
                j = os.path.basename(j[0])
                s = j.replace(".json","")
                results[s] = json.load(f)
            masses = [info[2] for info in results[s.replace("SUSY","")]["info"]]
            if not svplotted:
                ax.plot(masses, np.maximum(y_min,results[s.replace("SUSY","")]["stability_sv"]), 'k', color=colors["color_svfit"], marker='.', markersize=10, label = 'SVFit')
                svplotted = True
            ax.plot(masses, np.maximum(y_min,results[s.replace("SUSY","")]["stability_nn"]), 'k', color=colors_nn[name], marker='.', markersize=10, label = 'Nostradamass %s'%name)

        plt.legend(loc=2, ncol=1)
        plt.tight_layout()
        plt.savefig(os.path.join(channel_outpath, "stability_%s.pdf"%s.replace("SUSY","")))
        plt.savefig(os.path.join(channel_outpath, "stability_%s.png"%s.replace("SUSY","")))
        plt.close()
    else:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ranges = [0.,5.]
        y_min = 0.0001
        yranges = [y_min,1.0]
        titles =  r'fraction of events $\|m_N - m_H\|/m_H \geq 2$'

        ax.set_xlabel(r'Process')
        ax.set_ylabel(titles)
        ax.set_title("Method Stability (" + channel_name[channel] + ")")
        ax.set_xlim(ranges)
        ax.set_yscale('log')
        tick_numbers = np.array(range(5))+0.5
        ax.set_xticks(tick_numbers)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(color='gray', linestyle='-', linewidth=1, which='major')
        ax.grid(color='gray', linestyle='-',alpha=0.3, linewidth=1, which='minor')
        ax.set_ylim(yranges)
        svplotted = False
        tick_labels = []

        for model,name in zip(modelpaths,modelnames):

            results = {}
            results_path = os.path.join(model,channel,"data","%s.json"%s.replace("SUSY",""))
            j = glob.glob(results_path)
            if len(j) == 1:
                f = open(j[0],"r")
                j = os.path.basename(j[0])
                s = j.replace(".json","")
                results[s] = json.load(f)
            processes = [info[1] for info in results[s.replace("SUSY","")]["info"]]
            if tick_labels == []:
                tick_labels = processes
            
            if not svplotted:
                ax.plot(tick_numbers, np.maximum(y_min,results[s.replace("SUSY","")]["stability_sv"]),'k', color=colors["color_svfit"], marker='.',linewidth=0.0, markersize=10, label = 'SVFit')
                svplotted = True
            ax.plot(tick_numbers, np.maximum(y_min,results[s.replace("SUSY","")]["stability_nn"]),'k', color=colors_nn[name], marker='.', linewidth=0.0, markersize=10, label = 'Nostradamass %s'%name)
        ax.set_xticklabels(tick_labels)
        #ax.tick_params(axis="x")
        plt.legend(loc=2, ncol=1)
        plt.tight_layout()
        plt.savefig(os.path.join(channel_outpath, "stability_%s.pdf"%s.replace("SUSY","")))
        plt.savefig(os.path.join(channel_outpath, "stability_%s.png"%s.replace("SUSY","")))
        plt.close()
