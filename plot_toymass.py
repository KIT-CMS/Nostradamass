import csv
import sys, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
from fourvector import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_functions import transform_fourvector
from common_functions import full_fourvector
from common_functions import predict
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
    "Raphaels" : "violet",
    "2017BCD" : "orange",
    "2017EF" : "green",
    "NN-based" : "red",
    "DDT" : "red",
    "DDT corr" : "orange",
}



for s in signal_patterns:
    titles = [ r'$\left< p_T^N - p_T^H \right>$', r'$\left< \eta_N - \eta_H \right>$',r' $\left< \phi_N - \phi_H \right>$',r' $\left<\frac{ m_{\tau\tau} - m_H}{m_H}\right>$',]
    yranges = [[-40,40],
        [-0.5,0.5],
        [-1.4,1.4],
        [-0.8,0.8]]
    major_yticks = [[-50,-25,0,25,50],
                    [-0.6,-0.3,0,0.3,0.6],
                    [-1.5,-0.75,0,0.75,1.5],
                    [-0.6,-0.45,-0.3,-0.15,0,0.15,0.3,0.45,0.6]]
    minor_yticks = [[-37.5,-12.5,12.5,37.5],
                    [-0.45,-0.15,0.15,0.45],
                    [-1.125,-0.375,0.375,1.125],
                    [-0.525,-0.375,-0.225,-0.075,0.075,0.225,0.375,0.525]]
    if "SUSY" in s:
        ranges = [
            [70,3500],
            [70,3500],
            [70,3500],
            [70,3500]]
        for a in [0,1,2,3]:
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot(111)
            ax.set_xlabel(r'Generator mass $m_H$ (GeV)')
            ax.set_ylabel(titles[a])
            if a==0:
                ax.set_title("transverse momentum resolution (" + channel_name[channel] + ")")
            if a==1:
                ax.set_title("pseudorapidity resolution (" + channel_name[channel] + ")")
            if a==2:
                ax.set_title("angular Resolution (" + channel_name[channel] + ")")
            if a==3:
                ax.set_title("rel. mass Resolution (" + channel_name[channel] + ")")
            else:
                ax.set_title("Resolution (" + channel_name[channel] + ")")
            ax.set_xlim(ranges[a])
            ax.set_xscale('log')
            ax.set_xticks([100,200,400,1000,2000,3000])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.grid(color='gray', linestyle='-', linewidth=1, which='major')
            ax.grid(color='gray', linestyle='-',alpha=0.3, linewidth=1, which='minor')
            ax.set_ylim(yranges[a])

            ax.set_yticks(major_yticks[a])
            ax.set_yticks(minor_yticks[a],minor=True)
            svplotted = False
            ddtplotted = True

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
                    ax.plot(masses, results[s.replace("SUSY","")]["percentiles_50p0_sv"][a], 'k', color=colors["color_svfit"], marker='.', markersize=10, label = 'ClassicSVFit median')
                    ax.fill_between(masses, np.array(results[s.replace("SUSY","")]["percentiles_15p9_sv"][a]), np.array(results[s.replace("SUSY","")]["percentiles_84p1_sv"][a]), alpha=0.2, linewidth=2, edgecolor=colors["color_svfit"], facecolor=colors["color_svfit"], linestyle='-', antialiased=False, label = "ClassicSVFit 68% CL interval")
                    svplotted = True
                if a == 3 and ddtplotted == False:
                    ax.plot(masses, results[s.replace("SUSY","")]["percentiles_50p0_ddt"][0], 'k', color=colors_nn["DDT"], marker='.', markersize=10, label = '%s median'%"DDT")
                    ax.fill_between(masses, np.array(results[s.replace("SUSY","")]["percentiles_15p9_ddt"][0]), np.array(results[s.replace("SUSY","")]["percentiles_84p1_ddt"][0]), alpha=0.2, linewidth=2, edgecolor=colors_nn["DDT"], facecolor=colors_nn["DDT"], linestyle='-', antialiased=False, label = "%s 68%s CL interval"%("DDT","%"))

                    ax.plot(masses, results[s.replace("SUSY","")]["percentiles_50p0_ddt"][1], 'k', color=colors_nn["DDT corr"], marker='.', markersize=10, label = '%s median'%"DDT corr")
                    ax.fill_between(masses, np.array(results[s.replace("SUSY","")]["percentiles_15p9_ddt"][1]), np.array(results[s.replace("SUSY","")]["percentiles_84p1_ddt"][1]), alpha=0.2, linewidth=2, edgecolor=colors_nn["DDT corr"], facecolor=colors_nn["DDT corr"], linestyle='-', antialiased=False, label = "%s 68%s CL interval"%("DDT corr","%"))

                ax.plot(masses, results[s.replace("SUSY","")]["percentiles_50p0_nn"][a], 'k', color=colors_nn[name], marker='.', markersize=10, label = '%s median'%name)
                ax.fill_between(masses, np.array(results[s.replace("SUSY","")]["percentiles_15p9_nn"][a]), np.array(results[s.replace("SUSY","")]["percentiles_84p1_nn"][a]), alpha=0.2, linewidth=2, edgecolor=colors_nn[name], facecolor=colors_nn[name], linestyle='-', antialiased=False, label = "%s 68%s CL interval"%(name,"%"))

            plt.legend(loc=2, ncol=2)
            plt.tight_layout()
            #print os.path.join(channel_outpath, "resolution-"+str(a)+"_%s.pdf"%s.replace("SUSY",""))
            plt.savefig(os.path.join(channel_outpath, "resolution-"+str(a)+"_%s.pdf"%s.replace("SUSY","")))
            plt.savefig(os.path.join(channel_outpath, "resolution-"+str(a)+"_%s.png"%s.replace("SUSY","")))
            plt.close()
    else:
        yranges = [[-40,40],
            [-0.5,0.5],
            [-1.4,1.4],
            [-0.6,1.2]]
        for a in [0,1,2,3]:
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot(111)
            ranges = [0.,5.]
            ax.set_ylabel(titles[a])
            if a==0:
                ax.set_title("transverse momentum resolution (" + channel_name[channel] + ")")
            if a==1:
                ax.set_title("pseudorapidity resolution (" + channel_name[channel] + ")")
            if a==2:
                ax.set_title("angular Resolution (" + channel_name[channel] + ")")
            if a==3:
                ax.set_title("rel. mass Resolution (" + channel_name[channel] + ")")
            else:
                ax.set_title("Resolution (" + channel_name[channel] + ")")

            ax.set_xlabel(r'Process')
            ax.set_xlim(ranges)
            tick_numbers = np.array(range(5))+0.5
            ax.set_xticks(tick_numbers)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.grid(color='gray', linestyle='-', linewidth=1, which='major')
            ax.grid(color='gray', linestyle='-',alpha=0.3, linewidth=1, which='minor')
            ax.set_ylim(yranges[a])
            ax.set_yticks(major_yticks[a])
            ax.set_yticks(minor_yticks[a],minor=True)
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
                    median = np.array(results[s.replace("SUSY","")]["percentiles_50p0_sv"][a])
                    up = np.array(results[s.replace("SUSY","")]["percentiles_84p1_sv"][a])
                    down  = np.array(results[s.replace("SUSY","")]["percentiles_15p9_sv"][a])
                    ax.errorbar(tick_numbers, median, yerr=[median-down,up-median], color=colors["color_svfit"], linewidth=0.0, elinewidth=1.5, capthick = 1.5, capsize = 10, marker='.', markersize=10, label = 'SVFit median with 68% CL')
                    svplotted = True
                median = np.array(results[s.replace("SUSY","")]["percentiles_50p0_nn"][a])
                up = np.array(results[s.replace("SUSY","")]["percentiles_84p1_nn"][a])
                down  = np.array(results[s.replace("SUSY","")]["percentiles_15p9_nn"][a])
                ax.errorbar(tick_numbers, median, yerr=[median-down,up-median], color=colors_nn[name], linewidth=0.0, elinewidth=1.5, capthick = 1.5, capsize = 10, marker='.', markersize=10, label = 'Nostradamass %s median with 68%s CL'%(name,"%"))

            ax.set_xticklabels(tick_labels)
            plt.legend(loc=2, ncol=1)
            plt.tight_layout()
            #print os.path.join(channel_outpath, "resolution-"+str(a)+"_%s.pdf"%s.replace("SUSY",""))
            plt.savefig(os.path.join(channel_outpath, "resolution-"+str(a)+"_%s.pdf"%s.replace("SUSY","")))
            plt.savefig(os.path.join(channel_outpath, "resolution-"+str(a)+"_%s.png"%s.replace("SUSY","")))
            plt.close()
