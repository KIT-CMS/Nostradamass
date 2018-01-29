import csv
import numpy as np
from fourvector import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_neutrino import transform_fourvector
import sys, os
#processes = ["ggH", "DY"]
processes = ["ggHSM", "susy100", "susy200", "susy300", "susy400",  "susy500", "susy600", "vbfSM"]
#genmasses = [125, 91]
modelpath = sys.argv[1]
for process in processes:
    #process = "DY"
    filename = "data/" + process+".csv"
    #genmass = 91

    dim = 10
    n_events = sum(1 for line in open(filename))
    X = np.zeros([n_events, dim])
    svfit = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    M = np.zeros([n_events, 4])
    phys_M = np.zeros([n_events, 4])
    gen = np.zeros([n_events, 4])
    gen_phys = np.zeros([n_events, 4])

    diff_svfit = np.zeros([n_events, 4])
    diff_nn = np.zeros([n_events, 4])

    fake_met_phys = np.zeros([n_events, 4])
    gen_met_phys = np.zeros([n_events, 4])

    line = 0
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            a = [float(i) for i in row]
            lepton_1 = FourMomentum(a[4], a[5], a[6], a[7], cartesian=False)
            lepton_2 = FourMomentum(a[8], a[9], a[10], a[11], cartesian=False)
            met = FourMomentum(0, a[12], 0, a[13], False)
            gen_boson = FourMomentum(a[14], a[15], a[16], a[17], cartesian=False)
            gen_met_phys[line,:] = np.array([a[18], 0, a[19], 0])
            gen_met = FourMomentum( 0, a[18], 0, a[19], cartesian=False)

            gen_lepton_1 = FourMomentum(a[20], a[21], a[22], a[23], cartesian=False)
            gen_lepton_2 = FourMomentum(a[24], a[25], a[26], a[27], cartesian=False)

            fake_met_phys[line,:] = np.array([a[28], 0, a[29], 0])
            #x = np.array([  lepton_1.e,
            #                lepton_1.px,
            #                lepton_1.py,
            #                lepton_1.pz,
            #                lepton_2.e,
            #                lepton_2.px,
            #                lepton_2.py,
            #                lepton_2.pz,
            #                met.px,
            #                met.py
            #                ])
            x = np.array([  gen_lepton_1.e,
                            gen_lepton_1.px,
                            gen_lepton_1.py,
                            gen_lepton_1.pz,
                            gen_lepton_2.e,
                            gen_lepton_2.px,
                            gen_lepton_2.py,
                            gen_lepton_2.pz,
                            gen_met.px,
                            gen_met.py
                            ])
            X[line,:] = x
            s = FourMomentum(a[0], a[1], a[2], a[3], cartesian=False)
            svfit[line,:] = np.array([s.pt, s.eta, s.phi, s.m()])
            l = np.array([lepton_1.e+lepton_2.e, lepton_1.px+lepton_2.px, lepton_1.py+lepton_2.py, lepton_1.pz+lepton_2.pz])
            L[line,:] = l
            m = np.array([0, met.px, met.py, 0])
            M[line,:] = m
            phys_M[line,:] = np.array([met.pt, 0, met.phi, 0])

            gen[line,:] = np.array([gen_boson.pt, gen_boson.eta, gen_boson.phi, gen_boson.m()])
            gen_phys[line,:] = np.array([gen_boson.e, gen_boson.px, gen_boson.py, gen_boson.pz])

            d_svfit = FourMomentum(0, s.px - gen_boson.px, s.py - gen_boson.py, s.pz - gen_boson.pz)
            diff_svfit[line,:] = np.array([d_svfit.pt, d_svfit.eta, d_svfit.phi, s.m() - gen_boson.m()])

            line +=1


    #from sklearn.preprocessing import PolynomialFeatures
    #poly = PolynomialFeatures(2)
    #X=poly.fit_transform(X)

    import pickle
    #pkl_file = open(os.path.join(modelpath, 'scaler.pkl'), 'rb')
    #scaler = pickle.load(pkl_file)
    #scalerTarget = pickle.load(pkl_file)
    #X = scaler.transform(X)

    from keras.models import load_model
    from train_neutrino import mass_loss_start, custom_loss, mass_loss_final, mass_loss_custom, mass_loss_abs
    model = load_model(os.path.join(modelpath, 'toy_mass.h5'),  custom_objects={'mass_loss_start': mass_loss_start, 'custom_loss':custom_loss, 'mass_loss_final':mass_loss_final, 'mass_loss_custom' : mass_loss_custom , 'mass_loss_abs' : mass_loss_abs})

    scaled_Y = model.predict(X)
    energy = np.sqrt(np.square(scaled_Y[:,0]) + np.square(scaled_Y[:,1]) +np.square(scaled_Y[:,2])) + np.sqrt(np.square(scaled_Y[:,3]) + np.square(scaled_Y[:,4]) +np.square(scaled_Y[:,5]))
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( L[i,0] + energy[i],
                                                                                            L[i,1] + scaled_Y[i,0] + scaled_Y[i,3],
                                                                                            L[i,2] + scaled_Y[i,1] + scaled_Y[i,4],
                                                                                            L[i,3] + scaled_Y[i,2] + scaled_Y[i,5]) for i in range(L.shape[0])])
    



    diff_nn_tmp = [ FourMomentum(0,#regressed_physfourvectors[i,3] - gen[i,3], 
                    regressed_fourvectors[i,1] - gen_phys[i,1],
                    regressed_fourvectors[i,2] - gen_phys[i,2],
                    regressed_fourvectors[i,3] - gen_phys[i,3]) for i in range(regressed_fourvectors.shape[0])]
    diff_nn = np.array([[diff_nn_tmp[i].pt, diff_nn_tmp[i].eta, diff_nn_tmp[i].phi, regressed_physfourvectors[i,3] - gen[i,3]] for i in range(gen.shape[0])])

    
    #target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])

    #vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])
    for a in [0]:
        pts = plt.figure()
        irange = None
#        if a == 0:
 #           irange = [-500, 500]
  #      if a == 3:
   #         irange = [-500, 500]
        n, bins, patches = plt.hist(fake_met_phys[:,0], 150, normed=1, facecolor='orange', alpha=0.5, range=irange)
        n, bins, patches = plt.hist(gen_met_phys[:,0], 150, normed=1, facecolor='green', alpha=0.5, range=irange)
        n, bins, patches = plt.hist(phys_M[:,0], 150, normed=1, facecolor='gray', alpha=0.5, range=irange)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
        plt.savefig("plots_apply/"+process+"-fakemet"+str(a)+".png")
    
    

    print process, " toy mean: ", np.mean(regressed_physfourvectors[:,3]), 'toy median', np.median(regressed_physfourvectors[:,3]), ", toy resolution: ", np.std(regressed_physfourvectors[:,3])
    print process, " svfit mean: ", np.mean(svfit[:,3]), "svfit median", np.median(svfit[:,3]), ", svfit resolution: ", np.mean(svfit[:,3])
    for a in range(4):
        pts = plt.figure()
        irange = None
        if a == 3:
            irange = [0, 1000]
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75, range=irange)
        n, bins, patches = plt.hist(svfit[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        n, bins, patches = plt.hist(gen[:,a], 150, normed=1, facecolor='green', alpha=0.75, range=irange)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
        plt.savefig("plots_apply/"+process+"-regressed"+str(a)+".png")

    for a in range(4):
        pts = plt.figure()
        irange = None
        if a == 0:
            irange = [-500, 500]
        if a == 3:
            irange = [-500, 500]
        n, bins, patches = plt.hist(diff_nn[:,a], 150, normed=1, facecolor='red', alpha=0.75, range=irange)
        n, bins, patches = plt.hist(diff_svfit[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
        plt.savefig("plots_apply/"+process+"-diff"+str(a)+".png")


#    for a in range(4):
#        pts = plt.figure()
#        n, bins, patches = plt.hist(unscaled_pred[:,a], 150, normed=1, facecolor='red', alpha=0.75)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
#        plt.savefig(process+"-unscaled"+str(a)+".pdf")

#    for a in range(4):
#        pts = plt.figure()
#        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
#        plt.savefig("plots_apply/"+process+"-cartesian"+str(a)+".png")
