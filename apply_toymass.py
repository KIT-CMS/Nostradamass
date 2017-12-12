import csv
import numpy as np
from fourvector import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

processes = ["ggH", "DY"]
genmasses = [125, 91]
for process, genmass in zip(processes, genmasses):
    #process = "DY"
    filename = "data/" + process+".csv"
    #genmass = 91

    dim = 12
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
            x = np.array([lepton_1.pt+lepton_2.pt, lepton_1.e+lepton_2.e, lepton_1.e/(lepton_1.e+lepton_2.e), lepton_1.pt/(lepton_1.pt+lepton_2.pt), lepton_1.phi, lepton_1.eta, lepton_2.e/(lepton_1.e+lepton_2.e), lepton_2.pt/(lepton_1.pt+lepton_2.pt), lepton_2.phi, lepton_2.eta, met.pt, met.phi])
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
    pkl_file = open('scaler.pkl', 'rb')
    scaler = pickle.load(pkl_file)
    scalerTarget = pickle.load(pkl_file)
    X = scaler.transform(X)

    from keras.models import load_model
    model = load_model('toy_mass.h5')

    unscaled_pred = model.predict(X)
    regressed_phys = scalerTarget.inverse_transform(unscaled_pred)



    energy = np.multiply(regressed_phys[:,0], L[:,0])
    ones = np.ones([regressed_phys.shape[0]])
    F = np.subtract( ones,  regressed_phys[:,0])
    energy /= F

    pz = np.sinh(regressed_phys[:,1]) * np.sqrt( np.square(M[:,1]) + np.square(M[:,2]))
    
    regressed_fourvectors = [ FourMomentum(energy[i]+L[i,0], L[i,1]+M[i,1], L[i,2]+M[i,2], L[i,3]+pz[i]) for i in range(regressed_phys.shape[0])]
    regressed_physfourvectors = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in regressed_fourvectors])
    regressed_fourvectors = np.array([ [a.e, a.px, a.py, a.pz] for a in regressed_fourvectors])
    

    print process, " toy mean: ", np.mean(regressed_physfourvectors[:,3]), ", toy resolution: ", np.mean(np.abs(regressed_physfourvectors[:,3]-genmass))
    print process, " svfit mean: ", np.mean(svfit[:,3]), ", svfit resolution: ", np.mean(np.abs(svfit[:,3]-genmass))
    for a in range(4):
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
        n, bins, patches = plt.hist(svfit[:,a], 150, normed=1, facecolor='blue', alpha=0.75)
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
