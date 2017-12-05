# generate 2-dim array / event, 0 padded
# variables: pt, eta, phi, m
# target : - sum(2vector)
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
from fourvector import *
np.random.seed(1234)

from os import environ
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
#environ['KERAS_BACKEND'] = 'theano'

def count_neutrinos(in_string):
    if len(in_string)>0:
        return 2
    else:
        return 1

def fake_met():
    while(True):
        phi = np.random.rand() * np.pi * 2 # phi from 0 to 2 pi
        eta = 0 
        pt = 0#np.random.exponential(scale=10)
        pt = 0 
        m = 0.0 # massless
        yield FourMomentum(m, pt, eta, phi, False)

in_filename = "sim.log"

n_events = sum(1 for line in open(in_filename))

dim = 15
targets = 4
X = np.zeros([n_events, dim])
Y = np.zeros([n_events, targets])

fake_met = fake_met()
mets = []
checks = []
with open(in_filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for line, row in enumerate(reader):
        if line%10000==0:
            print line
        try:
            mass = float(row[0])
            posTauVis = create_FourMomentum(row[1])
            posTauInvis = create_FourMomentum(row[2]+ row[3])
            posTauNNeutrinos = count_neutrinos(row[3])
            negTauVis = create_FourMomentum(row[4])
            negTauInvis = create_FourMomentum(row[5]+ row[6])
            negTauNNeutrinos = count_neutrinos(row[6])
            checks.append( [posTauVis, posTauInvis,  negTauVis, negTauInvis] )
        except:
            continue
        if posTauNNeutrinos >= negTauNNeutrinos:
            lepton_1 = posTauVis 
            lepton_2 = negTauVis
            lepton_1_neutrinos = posTauNNeutrinos
            lepton_2_neutrinos = negTauNNeutrinos
        else:
            lepton_1 = negTauVis 
            lepton_2 = posTauVis
            lepton_1_neutrinos = negTauNNeutrinos
            lepton_2_neutrinos = posTauNNeutrinos

        fake = fake_met.next()
        neutrino_sum = posTauInvis + negTauInvis 
        met= neutrino_sum #+ fake

        boson = lepton_1 + lepton_2 + neutrino_sum
        #mets.append(met)
        #x = np.array([lepton_1.e, lepton_1.px, lepton_1.py, lepton_1.pz, lepton_2.e, lepton_2.px, lepton_2.py, lepton_2.pz, met.px, met.py, met.pt2()])
        x = np.array([lepton_1.e, lepton_1.px, lepton_1.py, lepton_1.pz, lepton_2.e, lepton_2.px, lepton_2.py, lepton_2.pz, met.px, met.py, met.pt2(),
                  lepton_1.e**2, lepton_2.e**2, met.px**2, met.py**2
                    #, lepton_1_neutrinos, lepton_2_neutrinos
                    ])
        y = np.array([boson.e, boson.px, boson.py, boson.pz])
        X[line,:] = x
        Y[line,:] = y

accessors = ['e', 'px', 'py', 'pz', 'pt', 'eta', 'phi']
for a in range(4):
    for ac in accessors:
        pts = plt.figure()
        x_measure =[ getattr(q[a], ac) for q in checks ]
        n, bins, patches = plt.hist(np.array(x_measure), 150, normed=1, facecolor='red', alpha=0.75)
    #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
        plt.savefig("inputs-"+ac+"-"+str(a)+".png")

from keras.models import Sequential
from keras.layers import Dense
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import copy
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

#poly = PolynomialFeatures(2)
#X=poly.fit_transform(X)
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(X)
X = scaler.transform(X)



target_phys = copy.deepcopy(Y)
scalerTarget = StandardScaler(with_mean=True)
scalerTarget.fit(Y)
Y = scalerTarget.transform(Y)

# save transformations to pickle file
import pickle
scaler_output = open('scaler.pkl', 'wb')
pickle.dump(scaler, scaler_output)
pickle.dump(scalerTarget, scaler_output)
scaler_output.close()

# model def
model = Sequential()
model.add(Dense(1000, activation='tanh', input_shape=(X.shape[1],)))
model.add(Dense(Y.shape[1], activation='linear'))
#model.compile(loss='logcosh', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(X, Y, # Training data
            batch_size=20000, # Batch size
            nb_epoch=50, # Number of training epochs
            validation_split=0.1)
model.save("toy_mass.h5")
unscaled_pred = model.predict(X)


# transform back
regressed_phys = scalerTarget.inverse_transform(unscaled_pred)


for a in range(Y.shape[1]):
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_phys[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(target_phys[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("target-regressed"+str(a)+".png")

# inputs plots
#for a in range(X.shape[1]):
#    pts = plt.figure()
#    n, bins, patches = plt.hist(X[:,a], 150, normed=1, facecolor='red', alpha=0.75)
#    plt.savefig("input" + str(a) +".png")

regressed_fourvectors = [ FourMomentum(a[0], a[1], a[2], a[3]) for a in regressed_phys]
regressed_physfourvectors = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in regressed_fourvectors])
regressed_fourvectors = np.array([ [a.e, a.px, a.py, a.pz ] for a in regressed_fourvectors])
target_fourvectors = [ FourMomentum(a[0], a[1], a[2], a[3]) for a in target_phys]
target_physfourvectors = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in target_fourvectors])
target_fourvectors = np.array([ [a.e, a.px, a.py, a.pz ] for a in target_fourvectors])
for a in range(4):
    pts = plt.figure()
    n, bins, patches = plt.hist(unscaled_pred[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("transform-target-regressed"+str(a)+".png")

for a in range(4):
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_fourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(target_fourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("tech-target-regressed"+str(a)+".png")

for a in [0, 2, 3]:
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("phys-target-regressed"+str(a)+".png")

#pts = plt.figure()
b = regressed_physfourvectors-target_physfourvectors
#n, bins, patches = plt.hist(b[:,3], 150, normed=1, facecolor='red', alpha=0.75)
#plt.savefig("mass-resolution.png")
#
#pts = plt.figure()
#b = (regressed_physfourvectors-target_physfourvectors)/target_physfourvectors
#n, bins, patches = plt.hist(b[:,3], 150, normed=1, facecolor='red', alpha=0.75, range=(-2,2))
#plt.savefig("rel-mass-resolution.png")


print "mass resolution: ",  np.mean(np.abs(regressed_physfourvectors[:,3]-target_physfourvectors[:,3]))
print "mean: ",  np.mean(np.abs(regressed_physfourvectors[:,3]-target_physfourvectors[:,3]))
print "rel. mass resolution: ",  np.mean(np.abs(b))

# met
#met_physfourvectors = np.array([ [a.pt, a.eta, a.phi ] for a in mets])
#for a in range(3):
#    pts = plt.figure()
#    n, bins, patches = plt.hist(met_physfourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75)
#    plt.savefig("met-"+str(a)+".png")

sys.exit()
