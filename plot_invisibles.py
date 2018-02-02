# generate 2-dim array / event, 0 padded
# variables: pt, eta, phi, m
# target : - sum(2vector)
import copy
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import math
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
selected_channel = 'mt'

def transform_fourvector(vin):
    cartesian = np.array([ [a.e, a.px, a.py, a.pz] for a in vin])
    phys = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in vin])
    return phys, cartesian

from train_invisibles import load_from_log, predict, transform_fourvector, load_from_pickle, load_model


def full_fourvector(scaled_Y, L):
    # transformation
    vlen = scaled_Y.shape[1] - 4
    energy = sum([np.sqrt( sum([np.square(scaled_Y[:,i+j]) for i in range(3)])) for j in range(0, vlen, 3)])
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( (L[i,0] + energy[i]),
                                                                                            (L[i,1] + sum([scaled_Y[i,j] for j in range(0, vlen, 3)])),
                                                                                            (L[i,2] + sum([scaled_Y[i,j] for j in range(1, vlen, 3)])),
                                                                                            (L[i,3] + sum([scaled_Y[i,j] for j in range(2, vlen, 3)]))) for i in range(L.shape[0])])
    return regressed_physfourvectors, regressed_fourvectors

def mod_fourvector(scaled_Y, Y, L):
    # transformation
    vlen = scaled_Y.shape[1]
    energy = np.sqrt( np.square(Y[:,0]) + np.square(Y[:,1]) + np.square(Y[:,2])) + np.sqrt( np.square(Y[:,3]) + np.square(Y[:,4]) + np.square(Y[:,5]))
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( (L[i,0] + energy[i]),
                                                                                            (L[i,1] + sum([scaled_Y[i,j] for j in range(0, vlen, 3)])),
                                                                                            (L[i,2] + sum([scaled_Y[i,j] for j in range(1, vlen, 3)])),
                                                                                            (L[i,3] + sum([scaled_Y[i,j] for j in range(2, vlen, 3)]))) for i in range(L.shape[0])])
    return regressed_physfourvectors, regressed_fourvectors

colors = {
    "color_nn" : 'red',
    "color_svfit" : 'blue',
    "color_visible" : 'yellow',
    "color_true" : 'green' }


def plot(scaled_Y, X, Y, B, M, L, phys_M, out_folder=''):
    from train_invisibles import smear_met_relative, add_pu_target
    X, Y = add_pu_target(X, Y, 0)
    X = smear_met_relative(X, magnitude = 0.)
   
    channel = [ r'$\tau_{had} \tau_{had}$',
                r'$\mu \tau_{had}$', 
                r'$e \tau_{had}$', 
                r'$e \mu$', 
                r'$ee$', 
                r'$\mu\mu$']

    titles = [ r'$p_x^{\nu^{\tau}_1}$',
               r'$p_y^{\nu^{\tau}_1}$',
               r'$p_z^{\nu^{\tau}_1}$',
               r'$p_x^{\nu^{\tau}_2}$',
               r'$p_x^{\nu^{\tau}_2}$',
               r'$p_y^{\nu^{\tau}_2}$',
               r'$p_z^{\nu^{\tau}_2}$',
               r'$p_x^{\nu^{\mu)}}$',
               r'$p_y^{\nu^{\mu)}}$',
               r'$p_z^{\nu^{\mu)}}$',
               r'$p_x^{\nu^{\tau)}}$',
               r'$p_y^{\nu^{\tau)}}$',
               r'$p_z^{\nu^{\tau)}}$',
               r'$p_x^{\nu^{e)}}$',
               r'$p_y^{\nu^{e)}}$',
               r'$p_z^{\nu^{e)}}$',
            ]
    from operator import itemgetter 
    if selected_channel == 'tt':
        titles = itemgetter(*[0,1,2,3,4,5])(titles)
        channel = channel[0]
    elif selected_channel == 'mt':
        titles = itemgetter(*[0,1,2,6,7,8,3,4,5])(titles)
        channel = channel[1]
    elif selected_channel == 'et':
        titles = itemgetter(*[13,14,15,10,11,12])(titles)
        channel = channel[2]

    # target/regressed neutrino vectors
    for a in range(scaled_Y[0].shape[0]):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        arange = [-300,300]
        if a%3 == 2: # Z-Componentes
            arange = [-1000,1000]


        n, bins, patches = plt.hist(Y[:,a], 150, normed=1, color=colors["color_true"], histtype='step', range = arange, label='target')
        n, bins, patches = plt.hist(scaled_Y[:,a], 150, normed=1, color=colors["color_nn"], histtype='step', range = arange, label='regressed')

        print "target ", a , " resolution: ", np.std(scaled_Y[:,a] - Y[:,a])
        ax.text(0.2, 0.93, r'$\sigma(p_x^{true}, p_x^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax.text(0.25, 0.88, str(np.std(scaled_Y[:,a] - Y[:,a]))[0:4] + " GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax.set_xlabel(titles[a] + "   (GeV)")
        ax.set_ylabel("arb. units")
        ax.set_title("Regression target vs. Result (" + channel + ")")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "target-vs-regressed"+str(a)+".png"))
        plt.close()

    # transform to plottable systems
    regressed_physfourvectors, regressed_fourvectors = full_fourvector(scaled_Y, L)
    regressed_met_pt = np.sqrt(np.square(scaled_Y[:,0] + scaled_Y[:,3]) + np.square(scaled_Y[:,1] + scaled_Y[:,4]))
    target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])
    vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])
    
    diff_fourvectors = regressed_fourvectors-target_fourvectors
    diff_physfourvectors = regressed_physfourvectors-target_physfourvectors
    for a in range(regressed_physfourvectors.shape[1]):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ranges = [[0,500],
            [-8,8],
            [-4,4],
            [0,600]]
        titles = [ r'$p_T$ (GeV)', r'$\eta$',r'$\phi$',r'$m$ (GeV)',]
        n, bins, patches = plt.hist(target_physfourvectors[:,a], 100, normed=1, color='green', alpha=0.75, range=ranges[a], histtype='step', label='target')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 100, normed=1, color='red', alpha=0.75, range=ranges[a], histtype='step', label='regressed')
        n, bins, patches = plt.hist(vis_physfourvectors[:,a], 100, normed=1, color='orange', alpha=0.5, range=ranges[a], histtype='step', label='visible', linestyle='dotted')
        print "phys diffvector mean ", a, np.mean(diff_physfourvectors[:,a]), " stddev " , np.std(diff_physfourvectors[:,a])
        if a == 0:
            ax.text(0.6, 0.6, r'$\sigma(p_T^{true}, p_T^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.65, 0.55, str(np.std(diff_physfourvectors[:,a]))[0:4] +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        if a == 3:
            ax.text(0.2, 0.6, r'$\sigma(m^{true}, m^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.25, 0.55, str(np.std(diff_physfourvectors[:,a]))[0:4] +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

            ax.text(0.2, 0.45, r'$\Delta(m^{true}, m^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.25, 0.40, str(np.mean(diff_physfourvectors[:,a]))[0:4] +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        ax.set_xlabel(titles[a])
        ax.set_ylabel("arb. units")
        ax.set_title("Gen vs. regressed system (" + channel + ")")

        plt.legend(loc='best')
        plt.savefig(os.path.join(out_folder, "phys-target-regressed"+str(a)+".png"))
        plt.tight_layout()
        plt.close()

    # compare gen met and regressed met
    regressed_met_pt = np.sqrt(np.square(scaled_Y[:,0] + scaled_Y[:,3]) + np.square(scaled_Y[:,1] + scaled_Y[:,4]))
    if True:
        irange = None
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_met_pt[:], 150, normed=1, color=colors["color_nn"], histtype='step', label='target')
        n, bins, patches = plt.hist(phys_M[:,0], 150, normed=1, color=colors["color_true"], histtype='step', label='target')
        plt.savefig(os.path.join(out_folder, "met_genmet.png"))


# plotting script
from train_invisibles import smear_met
if __name__ == '__main__':
    in_filename = sys.argv[1]
    model_path = sys.argv[2]
    out_folder = sys.argv[3]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if (in_filename[-4:] == ".log") or (in_filename[-5:] == ".data"):
        X, Y, B, M, L, phys_M = load_from_log(in_filename, "pickle.pkl", out_folder=out_folder, save_cache=True)
    elif in_filename[-4:] == ".pkl":
        X, Y, B, M, L, phys_M = load_from_pickle(in_filename)
    model = load_model(model_path)
    X = smear_met(X, magnitude = 0.20)
    regressed_Y = predict(model, X)
    plot(regressed_Y, X, Y, B, M, L, phys_M, out_folder)
