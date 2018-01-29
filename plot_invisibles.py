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


def transform_fourvector(vin):
    cartesian = np.array([ [a.e, a.px, a.py, a.pz] for a in vin])
    phys = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in vin])
    return phys, cartesian

from train_invisibles import load_from_log, predict, transform_fourvector, load_from_pickle, load_model


def plot(scaled_Y, X, Y, B, M, L, phys_M, out_folder=''):
    
    for a in range(scaled_Y[0].shape[0]):
        pts = plt.figure()
        arange = [-300,300]
        if a == 2 or a == 5:
            arange = [-1000,1000]
        n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
        n, bins, patches = plt.hist(scaled_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
        plt.savefig(os.path.join(out_folder, "target-regressed"+str(a)+".png"))
        print "target ", a , " resolution: ", np.std(scaled_Y[:,a] - Y[:,a])
        plt.close()

    energy = np.sqrt(np.square(scaled_Y[:,0]) + np.square(scaled_Y[:,1]) +np.square(scaled_Y[:,2])) + np.sqrt(np.square(scaled_Y[:,3]) + np.square(scaled_Y[:,4]) +np.square(scaled_Y[:,5]))
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( L[i,0] + energy[i],
                                                                                            L[i,1] + scaled_Y[i,0] + scaled_Y[i,3],
                                                                                            L[i,2] + scaled_Y[i,1] + scaled_Y[i,4],
                                                                                            L[i,3] + scaled_Y[i,2] + scaled_Y[i,5]) for i in range(L.shape[0])])
    
    regressed_met_pt = np.sqrt(np.square(scaled_Y[:,0] + scaled_Y[:,3]) + np.square(scaled_Y[:,1] + scaled_Y[:,4]))
    
    target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])

 #   print "vis"

#    vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])
#    neutrino_target_physfourvectors, neutrino_target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in M])
#    neutrino_regressed_physfourvectors, neutrino_regressed_fourvectors = transform_fourvector([ FourMomentum(energy[i], M[i,1], M[i,2], pz[i]) for i in range(M.shape[0])])

    
    for a in range(regressed_fourvectors[0].shape[0]):
        irange = None
        if a==0:
            irange = [-100,2100]
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 100, normed=1, color='red', alpha=1, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(target_fourvectors[:,a], 100, normed=1, color='green', alpha=1, range=irange, histtype='step', label='target')
#        n, bins, patches = plt.hist(vis_fourvectors[:,a], 100, normed=1, color='orange', alpha=1, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig(os.path.join(out_folder, "cartesian-target-regressed"+str(a)+".png"))
    
    for a in range(regressed_physfourvectors.shape[1]):
        pts = plt.figure()
        irange = None
        if a==0:
            irange = [0,500]
        if a==1:
            irange = [-8,8]
        if a==2:
            irange = [-4,4]
        if a==3:
            irange = [0,1100]
        n, bins, patches = plt.hist(target_physfourvectors[:,a], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
#        n, bins, patches = plt.hist(vis_physfourvectors[:,a], 100, normed=1, color='orange', alpha=0.5, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig(os.path.join(out_folder, "phys-target-regressed"+str(a)+".png"))

    diff_fourvectors = regressed_fourvectors-target_fourvectors
    
    diff_physfourvectors = regressed_physfourvectors-target_physfourvectors
    diff_physfourvectors = np.array([diff_physfourvectors[i] for i in range(regressed_physfourvectors.shape[0]) if regressed_physfourvectors[i][3]>0])

    for a in range(diff_physfourvectors.shape[1]):
        irange = None
        if a==0:
            irange = [-300,300]
        if a==1:
            irange = [-8,8]
        if a==2:
            irange = [-8,8]
        if a==3:
            irange = [-500,500]
        pts = plt.figure()
        n, bins, patches = plt.hist(diff_physfourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        plt.savefig(os.path.join(out_folder, "diffvector-phys-target-regressed"+str(a)+".png"))
        print "phys diffvector mean ", a, np.mean(diff_physfourvectors[:,a]), " stddev " , np.std(diff_physfourvectors[:,a])

    for a in range(diff_fourvectors.shape[1]):
        irange = None
        pts = plt.figure()
        n, bins, patches = plt.hist(diff_fourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        plt.savefig(os.path.join(out_folder, "diffvector-cartesian-target-regressed"+str(a)+".png"))
        print "cartesian diffvector mean ", a, np.mean(diff_fourvectors[:,a]), " stddev " , np.std(diff_fourvectors[:,a])

    # compare gen met and regressed met
    regressed_met_pt = np.sqrt(np.square(scaled_Y[:,0] + scaled_Y[:,3]) + np.square(scaled_Y[:,1] + scaled_Y[:,4]))
    if True:
        irange = None
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_met_pt[:], 150, normed=1, facecolor='red', alpha=0.75, range=irange)
        n, bins, patches = plt.hist(phys_M[:,0], 150, normed=1, facecolor='red', alpha=0.75, range=irange)
        plt.savefig(os.path.join(out_folder, "met_genmet.png"))


# plotting script
if __name__ == '__main__':
    in_filename = sys.argv[1]
    out_folder = sys.argv[2]
    previous_model = sys.argv[3]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if in_filename[-4:] == ".log":
        X, Y, B, M, L, phys_M = load_from_log(in_filename, "pickle.pkl", out_folder=out_folder, save_cache=True)
    elif in_filename[-4:] == ".pkl":
        X, Y, B, M, L, phys_M = load_from_pickle(in_filename)
    model = load_model(previous_model)
    regressed_Y = predict(model, X)
    plot(regressed_Y, X, Y, B, M, L, phys_M, out_folder)
