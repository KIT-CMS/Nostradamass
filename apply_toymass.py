import csv
import numpy as np
from fourvector import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_neutrino import transform_fourvector
import sys, os
processes = ["ggH", "DY"]
genmasses = [125, 91]
modelpath = sys.argv[1]
for process, genmass in zip(processes, genmasses):
    #process = "DY"
    filename = "data/" + process+".csv"
    #genmass = 91

    dim = 10
    n_events = sum(1 for line in open(filename))
    X = np.zeros([n_events, dim])
    svfit = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    M = np.zeros([n_events, 4])
    line = 0
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            a = [float(i) for i in row]
            lepton_1 = FourMomentum(a[4], a[5], a[6], a[7], cartesian=False)
            lepton_2 = FourMomentum(a[8], a[9], a[10], a[11], cartesian=False)
            met = FourMomentum(0, a[12], 0, a[13], False)
            x = np.array([  lepton_1.e,
                            lepton_1.px,
                            lepton_1.py,
                            lepton_1.pz,
                            #-lepton_1.px,
                            #-lepton_1.py,
                            #-lepton_1.pz,
                            lepton_2.e,
                            lepton_2.px,
                            lepton_2.py,
                            lepton_2.pz,
                            #-lepton_2.px,
                            #-lepton_2.py,
                            #-lepton_2.pz,
                            met.px,
                            met.py#,
                            #-met.px,
                            #-met.py
                            ])
            X[line,:] = x
            s = FourMomentum(a[0], a[1], a[2], a[3], cartesian=False)
            svfit[line,:] = np.array([s.pt, s.eta, s.phi, s.m()])
            l = np.array([lepton_1.e+lepton_2.e, lepton_1.px+lepton_2.px, lepton_1.py+lepton_2.py, lepton_1.pz+lepton_2.pz])
            L[line,:] = l
            m = np.array([0, met.px, met.py, 0])
            M[line,:] = m
            line +=1


    #from sklearn.preprocessing import PolynomialFeatures
    #poly = PolynomialFeatures(2)
    #X=poly.fit_transform(X)

    import pickle
    pkl_file = open(os.path.join(modelpath, 'scaler.pkl'), 'rb')
    scaler = pickle.load(pkl_file)
    scalerTarget = pickle.load(pkl_file)
    #X = scaler.transform(X)

    from keras.models import load_model
    model = load_model(os.path.join(modelpath, "toy_mass.h5"))

    regressed_Y = model.predict(X)
    scaled_Y = scalerTarget.inverse_transform(regressed_Y)


    energy = np.multiply(scaled_Y[:,0], L[:,0])
    ones = np.ones([scaled_Y.shape[0]])
    F = np.subtract( ones, scaled_Y[:,0])
    energy /= F

    pz = np.sinh(scaled_Y[:,1]) * np.sqrt( np.square(M[:,1]) + np.square(M[:,2]))

    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum(energy[i]+L[i,0], L[i,1]+M[i,1], L[i,2]+M[i,2], L[i,3]+pz[i]) for i in range(L.shape[0])])
    
    #target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])

    #vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])
    
    

    print process, " toy mean: ", np.mean(regressed_physfourvectors[:,3]), 'toy median', np.median(regressed_physfourvectors[:,3]), ", toy resolution: ", np.std(regressed_physfourvectors[:,3])
    print process, " svfit mean: ", np.mean(svfit[:,3]), "svfit median", np.median(svfit[:,3]), ", svfit resolution: ", np.mean(svfit[:,3])
    for a in range(4):
        pts = plt.figure()
        irange = None
        if a == 3:
            irange = [0, 300]
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75, range=irange)
        n, bins, patches = plt.hist(svfit[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
        plt.savefig(process+"-regressed"+str(a)+".png")

#    for a in range(4):
#        pts = plt.figure()
#        n, bins, patches = plt.hist(unscaled_pred[:,a], 150, normed=1, facecolor='red', alpha=0.75)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
#        plt.savefig(process+"-unscaled"+str(a)+".pdf")

    for a in range(4):
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
        plt.savefig(process+"-cartesian"+str(a)+".png")
