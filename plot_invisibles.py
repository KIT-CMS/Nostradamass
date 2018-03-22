# generate 2-dim array / event, 0 padded
# variables: pt, eta, phi, m
# target : - sum(2vector)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from fourvector import *
seed = 1234
np.random.seed(seed)
import pickle
from os import environ
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
environ['THEANO_FLAGS'] = 'device=gpu3'
import keras.backend as K
from matplotlib.colors import LogNorm

from common_functions import full_fourvector, transform_fourvector
from common_functions import original_tau
from common_functions import load_from_root, load_model, add_pu_target, load_from_pickle, predict

colors = {
    "color_nn" : 'red',
    "color_svfit" : 'blue',
    "color_visible" : 'yellow',
    "color_true" : 'green' }

def plot(scaled_Y, X, Y, B, L, channel, out_folder=''):

    channels = { "tt": r'$\tau_{h} \tau_{h}$',
                 "mt": r'$\mu \tau_{h}$', 
                 "et": r'$e \tau_{h}$', 
                 "em": r'$e \mu$', 
                 "ee": r'$ee$', 
                 "mm": r'$\mu\mu$'}

    titles = [ 
               r'$e^{{\tau^1_{vis}}}$',
               r'$p_x^{{\tau^1_{vis}}}$',
               r'$p_y^{{\tau^1_{vis}}}$',
               r'$p_z^{{\tau^1_{vis}}}$',
               r'$e^{{\tau^2_{vis}}}$',
               r'$p_x^{{\tau^2_{vis}}}$',
               r'$p_y^{{\tau^2_{vis}}}$',
               r'$p_z^{{\tau^2_{vis}}}$',

               r'$e^{\nu^{\tau}_1}$',
               r'$p_x^{\nu^{\tau}_1}$',
               r'$p_y^{\nu^{\tau}_1}$',
               r'$p_z^{\nu^{\tau}_1}$',
               r'$e^{\nu^{\tau}_2}$',
               r'$p_x^{\nu^{\tau}_2}$',
               r'$p_y^{\nu^{\tau}_2}$',
               r'$p_z^{\nu^{\tau}_2}$',

               r'$e^{\nu^{\mu)}}$',
               r'$p_x^{\nu^{\mu)}}$',
               r'$p_y^{\nu^{\mu)}}$',
               r'$p_z^{\nu^{\mu)}}$',
               r'$e^{\nu^{\tau)}}$',
               r'$p_x^{\nu^{\tau)}}$',
               r'$p_y^{\nu^{\tau)}}$',
               r'$p_z^{\nu^{\tau)}}$',
               r'$e^{\nu^{e)}}$',
               r'$p_x^{\nu^{e)}}$',
               r'$p_y^{\nu^{e)}}$',
               r'$p_z^{\nu^{e)}}$',
               r'PUx',
               r'PUy',
            ]
    from operator import itemgetter 
    title =["smearing $p_x$", "smearing $p_z$", "smeared MET $p_x$", "smeared MET $p_y$"]
    title = title + list(itemgetter(*[0,1,2,3,4,5,6,7])(titles))
    title = title + ["gen mass"] 
    title = title + list(itemgetter(*[8,9,10,11,12,13,14,15])(titles))

    tau_1_orig_phys = original_tau(0, 1, 2, 3, 13, 14, 15, 16, X, scaled_Y)
    gentau_1_orig_phys = original_tau(0, 1, 2, 3, 13, 14, 15, 16, X, Y)

    tau_2_orig_phys = original_tau(4, 5, 6, 7, 17, 18, 19, 20, X, scaled_Y)
    gentau_2_orig_phys = original_tau(4, 5, 6, 7, 17, 18, 19, 20, X, Y)

    #for a in range(13,21,1):
    for a in range(21):
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        if channel == "tt":
            arange = [-200,600]
            if a == 13 or a == 17:
                arange = [-5,600]
        if channel == "em" or channel =="mt" or channel=="et":
            arange = [-250,1000]
            if a == 13 or a == 17:
                arange = [-5,800]

        n, bins, patches = plt.hist(Y[:,a], 50, color=colors["color_true"], histtype='step', range = arange, label='target')
        n, bins, patches = plt.hist(scaled_Y[:,a], 50, color=colors["color_nn"], histtype='step', range = arange, label='regressed')

        #print "target ", a , " resolution: ", np.std(scaled_Y[:,a] - Y[:,a]), ", delta: ", np.mean(scaled_Y[:,a] - Y[:,a]v)
        print title[a], " & ", "{:2.1f}".format(np.mean(scaled_Y[:,a] - Y[:,a])), " & ", "{:2.1f}".format(np.std(scaled_Y[:,a] - Y[:,a])), "\\\\"
#        ax.text(0.65, 0.6, r'$\sigma(target, reg.)$ = ', fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#        ax.text(0.7, 0.45, str(np.std(scaled_Y[:,a] - Y[:,a]))[0:4] + " GeV", fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax.set_xlabel(title[a] + "   (GeV)", fontsize=15)
        ax.set_ylabel("# events")
        ax.set_title("Target vs. Regressed (" + channels[channel] + ")")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "target-vs-regressed"+str(a)+".pdf"))
        plt.savefig(os.path.join(out_folder, "target-vs-regressed"+str(a)+".png"))
        plt.close()

    # transform to plottable systems
    regressed_physfourvectors, regressed_fourvectors = full_fourvector(scaled_Y, L)
    #regressed_met_pt = np.sqrt(np.square(scaled_Y[:,0] + scaled_Y[:,3]) + np.square(scaled_Y[:,1] + scaled_Y[:,4]))
    target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])
    vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])
    
    diff_fourvectors = regressed_fourvectors-target_fourvectors
    diff_physfourvectors = regressed_physfourvectors-target_physfourvectors
    print '-=-=-=-=-=-=-=-=-'
    for a in range(regressed_physfourvectors.shape[1]):
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ranges = [[0,500],
            [-8,8],
            [-4,4],
            [0,1200]]
        titles = [ r'$p_T$ (GeV)', r'$\eta$',r'$\phi$',r'$m$ (GeV)',]
        n, bins, patches = plt.hist(target_physfourvectors[:,a], 50, color='green', range=ranges[a], histtype='step', label='target')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 50, color='red', range=ranges[a], histtype='step', label='regressed')
        n, bins, patches = plt.hist(vis_physfourvectors[:,a], 50, color='orange', range=ranges[a], histtype='step', label='visible', linestyle='dotted')
        print titles[a], " & ", "{:2.2f}".format(np.mean(diff_physfourvectors[:,a])), " & ", "{:2.2f}".format(np.std(diff_physfourvectors[:,a])), "\\\\"
#        print "phys diffvector mean ", a, np.mean(diff_physfourvectors[:,a]), " stddev " , np.std(diff_physfourvectors[:,a])
#        if a == 0:
#            ax.text(0.7, 0.5, r'$\sigma(p_T^{true}, p_T^{reg.})$ = ', fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#            ax.text(0.7, 0.4, str(np.std(diff_physfourvectors[:,a]))[0:4] +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#
#        if a == 3:
#            ax.text(0.7, 0.5, r'$\sigma(m^{gen}, m^{N})$ = ', fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#            ax.text(0.7, 0.4, str(np.std(diff_physfourvectors[:,a]))[0:4] +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#
#            ax.text(0.7, 0.3, r'$\Delta(m^{gen}, m^{N})$ = ', fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#            ax.text(0.7, 0.2, str(np.mean(diff_physfourvectors[:,a]))[0:4] +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        if a == 3:
            ax.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
        if a == 1:
            ax.set_ylim(top=1750)
        ax.set_title("$\\tau\\tau$ system (" + channels[channel] + ")")
        ax.set_xlabel(titles[a])
        ax.set_ylabel("# events")

        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "phys-target-regressed"+str(a)+".pdf"))
        plt.savefig(os.path.join(out_folder, "phys-target-regressed"+str(a)+".png"))
        plt.close()
# tau mass

    labels = ["$p_T$", "$\eta$", "$\phi$", "$mass$"]

    for a in range(4):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        arange = None
        if a == 3:
            arange = [-10,70]
    
        n, bins, patches = plt.hist(tau_1_orig_phys[:,a], 150, normed=1, color="gray", histtype='step', range = arange, label='regressed tau1')
        n, bins, patches = plt.hist(tau_2_orig_phys[:,a], 150, normed=1, color="green", histtype='step', range = arange, label='regressed tau2')
        n, bins, patches = plt.hist(gentau_1_orig_phys[:,a], 150, normed=1, color="black", histtype='step', range = arange, label='target tau1')
        n, bins, patches = plt.hist(gentau_2_orig_phys[:,a], 150, normed=1, color="orange", histtype='step', range = arange, label='target tau2')
        ax.set_xlabel(labels[a])
        ax.set_ylabel("arb. units")
        ax.set_title("Tau mass (" + channels[channel] + ")")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "original-tau"+str(a)+".pdf"))
        plt.savefig(os.path.join(out_folder, "original-tau"+str(a)+".png"))
        plt.close()

    # compare gen met and regressed met
    #regressed_met_pt = np.sqrt(np.square(scaled_Y[:,0] + scaled_Y[:,3]) + np.square(scaled_Y[:,1] + scaled_Y[:,4]))
    #if True:
    #    irange = None
    #    pts = plt.figure()
    #    n, bins, patches = plt.hist(regressed_met_pt[:], 150, normed=1, color=colors["color_nn"], histtype='step', label='target')
    #    n, bins, patches = plt.hist(phys_M[:,0], 150, normed=1, color=colors["color_true"], histtype='step', label='target')
    #    plt.savefig(os.path.join(out_folder, "met_genmet.png"))


# plotting script
if __name__ == '__main__':
    channel = sys.argv[1]
    model_path = sys.argv[2]
    out_folder = sys.argv[3]
    in_filenames = sys.argv[4:]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    f, ext = os.path.splitext(in_filenames[0])
    #if len(in_filenames) and ext == ".pkl":
    #    X, Y, B, L = load_from_pickle(in_filenames[0])
    #else:
    X, Y, B, L = load_from_root(in_filenames, channel)#, out_folder = out_folder)
    #X, Y = add_pu_target(X, Y, 0.,  0, 0.)
    X, Y = add_pu_target(X, Y, 7., 23., 80.)
    from common_functions import i_inv2_py, i_inv2_px, i_inv2_pz, i_inv2_e
    regressed_Y = predict(model_path, X, channel)
    #regressed_Y[:,13] = Y[:,13] 
    #regressed_Y[:,14] = Y[:,14] 
    #regressed_Y[:,15] = Y[:,15] 
    #regressed_Y[:,16] = Y[:,16] 
    #regressed_Y[:,17] = Y[:,17] 
    #regressed_Y[:,18] = Y[:,18] 
    #regressed_Y[:,19] = Y[:,19] 
    #regressed_Y[:,20] = Y[:,20]
    plot(regressed_Y, X, Y, B, L, channel, out_folder)
