# generate 2-dim array / event, 0 padded
# variables: pt, eta, phi, m
# target : - sum(2vector)
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

def some_particle(particle):
    while(True):
        phi = np.random.rand() * np.pi * 2 # phi from 0 to 2 pi
        eta = -2.0 + np.random.rand() * 4.0
        #eta = 0
        pt = np.absolute(15.0+np.random.normal(0,100))
        if particle == 0:
            m = 0.005 # electron
        elif particle == 1:
            m = 0.1 # muon
        elif particle == 2: # one-prong no pi0
            # 2/1 chance:
            piZeros = np.random.randint(0,3)
            if piZeros == 0:
                m = 0.14
            else:
                m = np.max([0.1, np.random.normal(1.,0.4)])
        yield FourMomentum(m, pt, eta, phi, False)

#def another_particle(original_particle, particle):
#    while(True):
#        phi = original_particle.phi# + np.random.normal(0,0.2)
#        eta = original_particle.eta# + np.random.normal(0,1)
#        #eta = 0
#        pt = np.absolute(np.random.normal(0,100))+10
#        if particle == 0:
#            m = 0.005 # electron
#        elif particle == 1:
#            m = 0.1 # muon
#        elif particle == 2: # one-prong no pi0
#            # 2/1 chance:
#            piZeros = np.random.randint(0,3)
#            if piZeros == 0:
#                m = 0.14
#            else:
#                m = np.max([0.1, np.random.normal(1.,0.4)])
#        return FourMomentum(m, pt, eta, phi, False)

def hard_neutrino():
    while(True):
        phi = np.random.rand() * np.pi * 2 # phi from 0 to 2 pi
       # eta = -3.0 + np.random.rand() * 6.0 # eta from -3 to 3
        eta = 0
        pt = np.absolute(np.random.normal(0,10))
        m = 0.0 # massless
        yield FourMomentum(m, pt, eta, phi, False)

def fake_met():
    while(True):
        phi = np.random.rand() * np.pi * 2 # phi from 0 to 2 pi
        eta = 0 
        pt = np.absolute(np.random.normal(0,15))
        m = 0.0 # massless
        yield FourMomentum(m, pt, eta, phi, False)

n_events = 500000
dim = 11
targets = 4
X = np.zeros([n_events, dim])
Y = np.zeros([n_events, targets])

leptons = [ some_particle(0), some_particle(1), some_particle(2)]
neutrino = hard_neutrino()
fake_met = fake_met()
mets = []

print "throwing ", n_events, " toy events"
for i in range(n_events):
    if i % 100000 == 0:
        print i
#    lepton1, lepton2 = np.random.randint(0,3,2)
    lepton1=1
    lepton2=2
    n_neutrinos = 2 + int(lepton1 < 2) + int(lepton2<2)
# throw toys according to expected four-vectors
    fake = fake_met.next()
    neutrino_sum = FourMomentum(0,0,0,0)
    for n in range(n_neutrinos):
        neutrino_sum += neutrino.next()

    lepton_1 = leptons[lepton1].next()
    lepton_2 = leptons[lepton2].next()
    #lepton_2 = another_particle(lepton_1, lepton2) 
    boson = lepton_1 + lepton_2 + neutrino_sum
    met = neutrino_sum + fake
    mets.append(met)
    #x = np.array([lepton_1.m(), lepton_1.pt, lepton_1.eta, lepton_2.phi, lepton_2.m(), lepton_2.pt, lepton_2.eta, lepton_2.phi, neutrino_sum.phi, neutrino_sum.pt])
    x = np.array([lepton_1.e, lepton_1.px, lepton_1.py, lepton_1.pz, lepton_2.e, lepton_2.px, lepton_2.py, lepton_2.pz, met.px, met.py, met.pt2()])
    X[i,:] = x
    y = np.array([boson.e, boson.px, boson.py, boson.pz])
    Y[i,:] = y


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import copy
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

poly = PolynomialFeatures(2)
X=poly.fit_transform(X)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print "X.shape: ", X.shape


target_phys = copy.deepcopy(Y)
scalerTarget = StandardScaler()
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
model.add(Dense(500, activation='tanh', input_shape=(X.shape[1],)))
model.add(Dense(500, activation='tanh'))
model.add(Dense(500, activation='tanh'))
model.add(Dense(Y.shape[1], activation='linear'))
model.compile(loss='logcosh', optimizer='adam')
model.summary()
model.fit(X, Y, # Training data
            batch_size=100000, # Batch size
            nb_epoch=20, # Number of training epochs
            validation_split=0.1)
model.save("toy_mass.h5")
unscaled_pred = model.predict(X)


# transform back
regressed_phys = scalerTarget.inverse_transform(unscaled_pred)


for a in range(Y.shape[1]):
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_phys[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(target_phys[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("target-regressed"+str(a)+".pdf")

# inputs plots
#for a in range(X.shape[1]):
#    pts = plt.figure()
#    n, bins, patches = plt.hist(X[:,a], 150, normed=1, facecolor='red', alpha=0.75)
#    plt.savefig("input" + str(a) +".pdf")

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
    plt.savefig("transform-target-regressed"+str(a)+".pdf")

for a in range(4):
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_fourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(target_fourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("tech-target-regressed"+str(a)+".pdf")

for a in range(4):
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("phys-target-regressed"+str(a)+".pdf")

pts = plt.figure()
b = regressed_physfourvectors-target_physfourvectors
n, bins, patches = plt.hist(b[:,3], 150, normed=1, facecolor='red', alpha=0.75)
plt.savefig("mass-resolution.pdf")

pts = plt.figure()
b = (regressed_physfourvectors-target_physfourvectors)/target_physfourvectors
n, bins, patches = plt.hist(b[:,3], 150, normed=1, facecolor='red', alpha=0.75, range=(-2,2))
plt.savefig("rel-mass-resolution.pdf")


print "mass resolution: ",  np.mean(np.abs(regressed_physfourvectors[:,3]-target_physfourvectors[:,3]))
print "mean: ",  np.mean(np.abs(regressed_physfourvectors[:,3]-target_physfourvectors[:,3]))
print "rel. mass resolution: ",  np.mean(np.abs(b))

# met
met_physfourvectors = np.array([ [a.pt, a.eta, a.phi ] for a in mets])
for a in range(3):
    pts = plt.figure()
    n, bins, patches = plt.hist(met_physfourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75)
    plt.savefig("met-"+str(a)+".pdf")

sys.exit()
