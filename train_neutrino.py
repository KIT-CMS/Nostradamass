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
import pickle

from os import environ
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
#environ['KERAS_BACKEND'] = 'theano'
def get_decay(in_string):
    neutrino_id = in_string[:-1].split(',')[-1]
    if neutrino_id == '':
        return "t"
    if abs(int(neutrino_id)) == 12:
        return "e"
    elif abs(int(neutrino_id)) == 14:
        return "m"
    raise Exception("one should never end up here")

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

in_filename = sys.argv[1]
if in_filename[-4:] =='.log':
    n_events = sum(1 for line in open(in_filename))
    
    dim = 12
    targets = 1
    X = np.zeros([n_events, dim])
    Y = np.zeros([n_events, targets])
    B = np.zeros([n_events, 4])
    M = np.zeros([n_events, 4])
    DM = n_events * [None]
    
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
                row[6]
            except:
                continue
            posTauVis = create_FourMomentum(row[1])
            posTauInvis = create_FourMomentum(row[2]+ row[3])
            posTauNNeutrinos = count_neutrinos(row[3])
            posTauDecayType = get_decay(row[3])
            negTauVis = create_FourMomentum(row[4])
            negTauInvis = create_FourMomentum(row[5]+ row[6])
            negTauNNeutrinos = count_neutrinos(row[6])
            negTauDecayType = get_decay(row[6])
            checks.append( [posTauVis, posTauInvis,  negTauVis, negTauInvis] )
            if posTauNNeutrinos >= negTauNNeutrinos:
                lepton_1 = posTauVis 
                lepton_2 = negTauVis
                neutrinos_1 = posTauInvis
                neutrinos_2 = negTauInvis
                lepton_1_neutrinos = posTauNNeutrinos
                lepton_2_neutrinos = negTauNNeutrinos
                decay_channel = posTauDecayType + negTauDecayType
            else:
                lepton_1 = negTauVis 
                lepton_2 = posTauVis
                neutrinos_1 = negTauInvis
                neutrinos_2 = posTauInvis
                lepton_1_neutrinos = negTauNNeutrinos
                lepton_2_neutrinos = posTauNNeutrinos
                decay_channel = negTauDecayType + posTauDecayType
    
            fake = fake_met.next()
            neutrino_sum = posTauInvis + negTauInvis 
            met= neutrino_sum #+ fake
    
            boson = lepton_1 + lepton_2 + neutrino_sum
            #mets.append(met)
            #x = np.array([lepton_1.e, lepton_1.px, lepton_1.py, lepton_1.pz, lepton_2.e, lepton_2.px, lepton_2.py, lepton_2.pz, met.px, met.py, met.pt2()])
            #x = np.array([lepton_1.e, lepton_1.px, lepton_1.py, lepton_1.pz, lepton_2.e, lepton_2.px, lepton_2.py, lepton_2.pz, met.px, met.py,#])
                    #    lepton_1.px**2, lepton_1.py**2, lepton_1.pz**2,
                    #    lepton_2.px**2, lepton_2.py**2, lepton_2.pz**2,
                    #  lepton_1.e**2, lepton_2.e**2, met.px**2, met.py**2
                        #, lepton_1_neutrinos, lepton_2_neutrinos
#                        ])
            x = np.array([lepton_1.pt+lepton_2.pt, lepton_1.e+lepton_2.e, lepton_1.e/(lepton_1.e+lepton_2.e), lepton_1.pt/(lepton_1.pt+lepton_2.pt), lepton_1.phi, lepton_1.eta, lepton_2.e/(lepton_1.e+lepton_2.e), lepton_2.pt/(lepton_1.pt+lepton_2.pt), lepton_2.phi, lepton_2.eta, met.pt, met.phi])
            #y = np.array([neutrino_sum.e/boson.e, neutrino_sum.pz, neutrino_sum.pz/boson.pz, neutrino_sum.eta])
            #y = np.array([neutrinos_1.e/boson.e, neutrinos_2.e/boson.e])
            y = np.array([neutrino_sum.e/boson.e])
            X[line,:] = x
            Y[line,:] = y
            b = np.array([boson.e, boson.px, boson.py, boson.pz])
            m = np.array([neutrino_sum.e, neutrino_sum.px, neutrino_sum.py, neutrino_sum.pz])
            B[line,:] = b
            M[line,:] = m
            DM[line] = decay_channel
    
    # filter for selected Decay modes
    selected_events = [a for a in range(len(DM)) if DM[a] == 'tt']
    X = np.array([X[x] for x in selected_events])
    Y = np.array([Y[x] for x in selected_events])
    B = np.array([B[x] for x in selected_events])
    M = np.array([M[x] for x in selected_events])
    cache_output = open('cache.pkl', 'wb')
    pickle.dump(X, cache_output)
    pickle.dump(Y, cache_output)
    pickle.dump(B, cache_output)
    pickle.dump(M, cache_output)
    cache_output.close()
elif in_filename[-4:] == ".pkl":
    
    cache_output = open(in_filename, 'rb')
    X = pickle.load(cache_output)
    Y = pickle.load(cache_output)
    B = pickle.load(cache_output)
    M = pickle.load(cache_output)
    cache_output.close()
else:
    raise Exception("invalid input selected")

accessors = ['e', 'px', 'py', 'pz']
#accessors = ['e', 'px', 'py', 'pz', 'pt', 'eta', 'phi']
#for a in range(4):
#    for ac in accessors:
#        pts = plt.figure()
#        x_measure =[ getattr(q[a], ac) for q in checks ]
#        n, bins, patches = plt.hist(np.array(x_measure), 150, normed=1, facecolor='red', alpha=0.75)
    #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
#        plt.savefig("inputs-"+ac+"-"+str(a)+".png")

from keras.models import Sequential
from keras.layers import Dense, Dropout
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
##Y = scalerTarget.transform(Y)

# save transformations to pickle file
scaler_output = open('scaler.pkl', 'wb')
pickle.dump(scaler, scaler_output)
pickle.dump(scalerTarget, scaler_output)
scaler_output.close()

import keras.backend as K
def custom_loss(y_true, y_pred):
    mean_squared_error = K.mean(K.square(y_pred[:,0] - y_true[:,0]) + K.square(y_pred[:,1] - y_true[:,1]) + K.square(y_pred[:,2] - y_true[:,2]) + K.square(y_pred[:,3] - y_true[:,3]))/4
    return mean_squared_error

def mass_loss(y_true, y_pred):
    zeros =  K.zeros_like(y_pred[:,0])
    #raise Exception(K.greater(zeros, (K.square(y_pred[:,0]) - K.square(y_pred[:,1])  - K.square(y_pred[:,2]) - K.square(y_pred[:,3]))))
    #raise Exception(tf.shape(tf.where(K.greater(zeros, ((K.square(y_pred[:,0]) - K.square(y_pred[:,1])  - K.square(y_pred[:,2]) - K.square(y_pred[:,3]))))))[0])
    mean_squared_error = K.mean(K.square(y_pred - y_true))
    pred_masses = K.square(y_pred[:,0]) - K.square(y_pred[:,1])  - K.square(y_pred[:,2]) - K.square(y_pred[:,3])
    true_masses = K.square(y_true[:,0]) - K.square(y_true[:,1])  - K.square(y_true[:,2]) - K.square(y_true[:,3])
    penalty = K.abs(K.mean(true_masses - pred_masses))
    pred_masses_negative = - pred_masses * tf.to_float(tf.shape(tf.where(K.greater(zeros, pred_masses)))) 
    return penalty + mean_squared_error #mean_squared_error#penalty

def mass_diff(y_true, y_pred):
    #raise Exception(y_pred)
    y_pred_trans = scalerTarget.inverse_transform(tf.transpose(y_pred))
    y_true_trans = scalerTarget.inverse_transform(tf.transpose(y_true))
    pred_masses = K.square(y_pred_trans[:,0]) - K.square(y_pred_trans[:,1])  - K.square(y_pred_trans[:,2]) - K.square(y_pred_trans[:,3])
    true_masses = K.square(y_true_trans[:,0]) - K.square(y_true_trans[:,1])  - K.square(y_true_trans[:,2]) - K.square(y_true_trans[:,3])
    return K.abs(K.mean(true_masses - pred_masses))


# model def
model = Sequential()
model.add(Dense(80, activation='tanh', input_shape=(X.shape[1],)))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(Y.shape[1], activation='linear'))
#model.compile(loss='logcosh', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
#model.compile(loss='mean_squared_error', optimizer='adam', metrics = [mass_loss])
model.summary()
model.fit(X, Y, # Training data
            batch_size=4000, # Batch size
            nb_epoch=200, # Number of training epochs
            validation_split=0.1)
model.save("toy_mass.h5")
unscaled_pred = model.predict(X)


# transform back
#regressed_phys = scalerTarget.inverse_transform(unscaled_pred)
regressed_phys = unscaled_pred

for a in range(Y.shape[1]):
    print "print a", a
    pts = plt.figure()
    arange = None
    if a ==2:
        arange = [-3,3]
    n, bins, patches = plt.hist(unscaled_pred[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
    n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
    plt.savefig("transform-target-regressed"+str(a)+".png")


for a in range(Y.shape[1]):
    print "print a", a
    pts = plt.figure()
    arange = None
    if a ==2:
        arange = [-3,3]
    n, bins, patches = plt.hist(regressed_phys[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
    n, bins, patches = plt.hist(target_phys[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
    plt.savefig("target-regressed"+str(a)+".png")

for a in range(Y.shape[1]):
    pts = plt.figure()
    arange = None
    n, bins, patches = plt.hist(target_phys[:,a] - regressed_phys[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range = arange)
    print "mean ", a, np.mean(target_phys[:,a] - regressed_phys[:,a]), " stddev " , np.std(target_phys[:,a] - regressed_phys[:,a])
    plt.savefig("diff-target-regressed"+str(a)+".png")
sys.exit()
# inputs plots
#for a in range(X.shape[1]):
#    pts = plt.figure()
#    n, bins, patches = plt.hist(X[:,a], 150, normed=1, facecolor='red', alpha=0.75)
#    plt.savefig("input" + str(a) +".png")
regressed_fourvectors = [ FourMomentum(regressed_phys[i,0], M[i,1], M[i,2], regressed_phys[i,1]) for i in range(regressed_phys.shape[0])]
regressed_physfourvectors = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in regressed_fourvectors])
regressed_fourvectors = np.array([ [a.e, a.px, a.py, a.pz ] for a in regressed_fourvectors])

target_fourvectors = [ FourMomentum(a[0], a[1], a[2], a[3]) for a in target_phys]
target_physfourvectors = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in target_fourvectors])
target_fourvectors = np.array([ [a.e, a.px, a.py, a.pz ] for a in target_fourvectors])
for a in range(Y.shape[1]):
    pts = plt.figure()
    n, bins, patches = plt.hist(unscaled_pred[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("transform-target-regressed"+str(a)+".png")

for a in range(regressed_fourvectors.shape[1]):
    pts = plt.figure()
    n, bins, patches = plt.hist(regressed_fourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    n, bins, patches = plt.hist(target_fourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    plt.savefig("tech-target-regressed"+str(a)+".png")
    #print "resolution ", a,  np.mean(np.abs(regressed_fourvectors[:,a] - target_fourvectors[:,a]))

for a in [0, 1, 2, 3]:
    pts = plt.figure()
    n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
    print regressed_physfourvectors[:,a]
    print target_physfourvectors[:,a]
    plt.savefig("phys-target-regressed"+str(a)+".png")

#pts = plt.figure()
print regressed_physfourvectors.shape
print target_physfourvectors.shape
b = regressed_physfourvectors-target_physfourvectors
#n, bins, patches = plt.hist(b[:,3], 150, normed=1, facecolor='red', alpha=0.75)
#plt.savefig("mass-resolution.png")
#
#pts = plt.figure()
#b = (regressed_physfourvectors-target_physfourvectors)/target_physfourvectors
#n, bins, patches = plt.hist(b[:,3], 150, normed=1, facecolor='red', alpha=0.75, range=(-2,2))
#plt.savefig("rel-mass-resolution.png")


#print "mass resolution: ",  np.mean(np.abs(regressed_physfourvectors[:,3]-target_physfourvectors[:,3]))
#print "mean: ",  np.mean(np.abs(regressed_physfourvectors[:,3]-target_physfourvectors[:,3]))
#print "rel. mass resolution: ",  np.mean(np.abs(b))

