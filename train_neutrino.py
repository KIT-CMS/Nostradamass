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
np.random.seed(1234)
import pickle
from os import environ
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
environ['THEANO_FLAGS'] = 'device=gpu2'

def norm_phi(phi):
    if phi < -np.pi:
        return norm_phi(phi+np.pi)
    elif phi > np.pi:
        return norm_phi(phi - np.pi)
    else:
        return phi

def transform_fourvector(vin):
    cartesian = np.array([ [a.e, a.px, a.py, a.pz] for a in vin])
    phys = np.array([ [a.pt, a.eta, a.phi, math.sqrt(a.m2()) if a.m2() > 0 else 0] for a in vin])
    return phys, cartesian

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

def load_from_log(in_filename, out_filename):
    n_events = sum(1 for line in open(in_filename))
    
    dim = 10
    targets = 2
    X = np.zeros([n_events, dim])
    Y = np.zeros([n_events, targets])
    B = np.zeros([n_events, 4])
    M = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    genmass = np.zeros([n_events, 4])
    DM = n_events * [None]
    
    #fake_met = fake_met()
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
    
            #fake = fake_met.next()
            neutrino_sum = posTauInvis + negTauInvis 
            met= neutrino_sum #+ fake
    
            boson = lepton_1 + lepton_2 + neutrino_sum
            dilepton = lepton_1 + lepton_2
            #mets.append(met)
            #x = np.array([lepton_1.e, lepton_1.px, lepton_1.py, lepton_1.pz, lepton_2.e, lepton_2.px, lepton_2.py, lepton_2.pz, met.px, met.py, met.pt2()])
            #x = np.array([lepton_1.e, lepton_1.px, lepton_1.py, lepton_1.pz, lepton_2.e, lepton_2.px, lepton_2.py, lepton_2.pz, met.px, met.py,#])
                    #    lepton_1.px**2, lepton_1.py**2, lepton_1.pz**2,
                    #    lepton_2.px**2, lepton_2.py**2, lepton_2.pz**2,
                    #  lepton_1.e**2, lepton_2.e**2, met.px**2, met.py**2
                        #, lepton_1_neutrinos, lepton_2_neutrinos
#                        ])
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
            #x = np.array([  lepton_1.pt+lepton_2.pt,
            #                dilepton.pt,
            #                dilepton.e,
            #                lepton_1.e/dilepton.pt,
            #                lepton_1.pt/dilepton.pt,
            #                lepton_1.phi-dilepton.phi,
            #                lepton_1.eta,
            #                lepton_2.e/dilepton.e,
            #                lepton_2.pt/dilepton.pt,
            #                lepton_2.phi-dilepton.phi,
            #                lepton_2.eta,
            #                met.pt/dilepton.pt,
            #                met.phi-dilepton.phi])
            y = np.array([neutrino_sum.e/boson.e, neutrino_sum.eta])

            X[line,:] = x
            Y[line,:] = y
            b = np.array([boson.e, boson.px, boson.py, boson.pz])
            l = np.array([lepton_1.e+lepton_2.e, lepton_1.px+lepton_2.px, lepton_1.py+lepton_2.py, lepton_1.pz+lepton_2.pz])
            m = np.array([neutrino_sum.e, neutrino_sum.px, neutrino_sum.py, neutrino_sum.pz])
            B[line,:] = b
            M[line,:] = m
            L[line,:] = l
            DM[line] = decay_channel
            genmass[line] = boson.m()
    
    # filter for selected Decay modes
    #selected_events = [a for a in range(len(DM)) if DM[a] == 'tt' and genmass[a][0] < 300]
    selected_events = [a for a in range(len(DM)) if DM[a] == 'tt']
    X = np.array([X[x] for x in selected_events])
    Y = np.array([Y[x] for x in selected_events])
    B = np.array([B[x] for x in selected_events])
    M = np.array([M[x] for x in selected_events])
    L = np.array([L[x] for x in selected_events])
    cache_output = open('cache.pkl', 'wb')
    pickle.dump(X, cache_output)
    pickle.dump(Y, cache_output)
    pickle.dump(B, cache_output)
    pickle.dump(M, cache_output)
    pickle.dump(L, cache_output)
    cache_output.close()
    return X, Y, B, M, L

def load_from_pickle(in_filename):
    cache_output = open(in_filename, 'rb')
    X = pickle.load(cache_output)
    Y = pickle.load(cache_output)
    B = pickle.load(cache_output)
    M = pickle.load(cache_output)
    L = pickle.load(cache_output)
    cache_output.close()
    return X, Y, B, M, L

def plot_inputs(Y):
    accessors = ['e', 'px', 'py', 'pz']
    #accessors = ['e', 'px', 'py', 'pz', 'pt', 'eta', 'phi']
    #for a in range(4):
    #    for ac in accessors:
    #        pts = plt.figure()
    #        x_measure =[ getattr(q[a], ac) for q in checks ]
    #        n, bins, patches = plt.hist(np.array(x_measure), 150, normed=1, facecolor='red', alpha=0.75)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
    #        plt.savefig("inputs-"+ac+"-"+str(a)+".png")

def get_scaled(raw_X, raw_Y, scaler_filename = "scaler.pkl"):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(raw_X)
    #X = scaler.transform(raw_X)
    X = raw_X
    
    scalerTarget = StandardScaler(with_mean=True)
    scalerTarget.fit(raw_Y)
    Y = scalerTarget.transform(raw_Y)
    
    # save transformations to pickle file
    scaler_output = open(scaler_filename, 'wb')
    pickle.dump(scaler, scaler_output)
    pickle.dump(scalerTarget, scaler_output)
    scaler_output.close()
    return X, Y

def train_model(X, Y, model_filename = "toy_mass.h5"):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    
    
    import keras.backend as K
    
    
    # model def # energy: 10x40, tanh, mean_squared_error
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(X.shape[1],)))
    for a in range(5):
        model.add(Dense(200, activation='relu'))
    for a in range(5):
        model.add(Dense(150, activation='relu'))
    for a in range(10):
        model.add(Dense(100, activation='relu'))
    model.add(Dense(Y.shape[1], activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='nadam')
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics = [mass_loss])
    model.summary()
    model.fit(X, Y, # Training data
                batch_size=40000, # Batch size
                epochs=300, # Number of training epochs
                validation_split=0.1)
    model.save(model_filename)
    return model

def predict(model, X):
    regressed_Y = model.predict(X)
    return regressed_Y

def get_inverse(regressed_Y, scaler_target_filename = "scaler.pkl"):
    # transform back
    pkl_file = open(scaler_target_filename, 'rb')
    scaler = pickle.load(pkl_file)
    scalerTarget = pickle.load(pkl_file)
    scaled_Y = scalerTarget.inverse_transform(regressed_Y)
    return scaled_Y

#def plot(regressed_phys, raw_pred, target_phys, X, Y, B, M, L):
def plot(scaled_Y, regressed_Y, raw_Y, X, Y, B, M, L):
    for a in range(Y.shape[1]):
        print "print a", a
        pts = plt.figure()
        arange = None
        n, bins, patches = plt.hist(regressed_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
        n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
        plt.savefig("transform-target-regressed"+str(a)+".png")
        plt.close()
    
    
    for a in range(Y.shape[1]):
        print "print a", a
        pts = plt.figure()
        arange = None
        if a ==2:
            arange = [-3,3]
        n, bins, patches = plt.hist(scaled_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
        n, bins, patches = plt.hist(raw_Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
        plt.savefig("target-regressed"+str(a)+".png")
        plt.close()
    
 #   for a in range(Y.shape[1]):
 #       pts = plt.figure()
 #       arange = [-3,3]
 #       n, bins, patches = plt.hist(target_phys[:,a] - regressed_phys[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range = arange)
  #      print "mean ", a, np.mean(target_phys[:,a] - regressed_phys[:,a]), " stddev " , np.std(target_phys[:,a] - regressed_phys[:,a])
 #       plt.savefig("diff-target-regressed"+str(a)+".png")
#        plt.close()
    
    # neutrino 4-vectors
    #y = ns / (l1 + l2 + ns)
    #y(l1+l2+ns) = ns
    #y(l1+l2) = ns-yns = ns(1-y)
    #y(l1+l2) / (1-y) = ns
    energy = np.multiply(scaled_Y[:,0], L[:,0])
    ones = np.ones([scaled_Y.shape[0]])
    F = np.subtract( ones, scaled_Y[:,0])
    energy /= F

    pz = np.sinh(scaled_Y[:,1]) * np.sqrt( np.square(M[:,1]) + np.square(M[:,2]))
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum(energy[i]+L[i,0], L[i,1]+M[i,1], L[i,2]+M[i,2], L[i,3]+pz[i]) for i in range(L.shape[0])])
    
    target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])

    vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])

    
    for a in range(regressed_fourvectors[0].shape[0]):
        irange = None
        if a==0:
            irange = [-100,2100]
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 100, normed=1, color='red', alpha=1, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(target_fourvectors[:,a], 100, normed=1, color='green', alpha=1, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(vis_fourvectors[:,a], 100, normed=1, color='orange', alpha=1, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig("cartesian-target-regressed"+str(a)+".png")
    
    for a in range(regressed_physfourvectors.shape[1]):
        pts = plt.figure()
        irange = None
        if a==3:
            irange = [0,1100]
        n, bins, patches = plt.hist(target_physfourvectors[:,a], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(vis_physfourvectors[:,a], 100, normed=1, color='orange', alpha=0.5, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig("phys-target-regressed"+str(a)+".png")

    # neutrino energy
    pts = plt.figure()
    irange = None
    n, bins, patches = plt.hist(M[:,3], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
    n, bins, patches = plt.hist(pz, 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
    plt.legend()
    plt.savefig("neutrino-pz.png")

    pts = plt.figure()
    irange = None
    n, bins, patches = plt.hist(M[:,3], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
    n, bins, patches = plt.hist(energy, 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
    plt.legend()
    plt.savefig("neutrino-pz.png")
    
    diff_fourvectors = regressed_physfourvectors-target_physfourvectors
    for a in range(diff_fourvectors.shape[1]):
        irange = None
        if a==3:
            irange = [-800,800]
        pts = plt.figure()
        n, bins, patches = plt.hist(diff_fourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        plt.savefig("diffvector-target-regressed"+str(a)+".png")
        print "diffvector mean ", a, np.mean(diff_fourvectors[:,a]), " stddev " , np.std(diff_fourvectors[:,a])
    
if __name__ == '__main__':
    in_filename = sys.argv[1]
    if in_filename[-4:] == ".log":
        raw_X, raw_Y, B, M, L = load_from_log(in_filename, "pickle.pkl")
    elif in_filename[-4:] == ".pkl":
        raw_X, raw_Y, B, M, L = load_from_pickle(in_filename)
    X, Y = get_scaled(raw_X, raw_Y)
    model = train_model(X, Y)
    regressed_Y = predict(model, X)
    scaled_Y = get_inverse(regressed_Y)
    plot(scaled_Y, regressed_Y, raw_Y, X, Y, B, M, L)
    # raw_X unscaled
    # raw_Y unscaled
    # X input for DNN
    # Y target for DNN
    # regressed_Y unscaled output
    # scaled_Y scaled output
