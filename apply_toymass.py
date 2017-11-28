import csv
import numpy as np
from fourvector import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

process = "ggH"
filename = process+".csv"
genmass = 125.0

dim = 11
n_events = sum(1 for line in open(filename))
X = np.zeros([n_events, dim])
svfit = np.zeros([n_events, 4])
line = 0
with open(filename, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        x = np.array([float(i) for i in row[0].split(";")[:11]])
    	X[line,:] = x
        s = np.array([float(i) for i in row[0].split(";")[11:]])
    	svfit[line,:] = s
	line +=1


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)

import pickle
pkl_file = open('scaler.pkl', 'rb')
scaler = pickle.load(pkl_file)
scalerTarget = pickle.load(pkl_file)
print X.shape
X = scaler.transform(X)

from keras.models import load_model
model = load_model('toy_mass.h5')

unscaled_pred = model.predict(X)
regressed_phys = scalerTarget.inverse_transform(unscaled_pred)

print regressed_phys


regressed_fourvectors = [ FourMomentum(a[0], a[1], a[2], a[3]) for a in regressed_phys]
regressed_physfourvectors = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in regressed_fourvectors if a.m2() > 0])
#regressed_fourvectors = np.array([ [a.e, a.px, a.py, a.pz ] for a in regressed_fourvectors])

print "toy mean: ", np.mean(regressed_physfourvectors[:,3]), ", toy resolution: ", np.mean(np.abs(regressed_physfourvectors[:,3]-genmass))
print "svfit mean: ", np.mean(svfit[:,3]), ", svfit resolution: ", np.mean(np.abs(svfit[:,3]-genmass))
#target_fourvectors = [ FourMomentum(a[0], a[1], a[2], a[3]) for a in target_phys]
#target_physfourvectors = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in target_fourvectors])
#target_fourvectors = np.array([ [a.e, a.px, a.py, a.pz ] for a in target_fourvectors])
for a in range(4):
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(svfit[:,a], 150, normed=1, facecolor='blue', alpha=0.75)
    #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig(process+"-regressed"+str(a)+".pdf")
