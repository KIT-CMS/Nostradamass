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

def load_from_log(in_filename, out_filename, save_cache=False, out_folder=""):
    n_events = sum(1 for line in open(in_filename))
    
    dim = 10
    targets = 6
    X = np.zeros([n_events, dim])
    Y = np.zeros([n_events, targets])
    B = np.zeros([n_events, 4])
    M = None#np.zeros([n_events, 4])
    phys_M = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    DM = n_events * [None]
    
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
            met= neutrino_sum
    
            boson = lepton_1 + lepton_2 + neutrino_sum
            #dilepton = lepton_1 + lepton_2
            x = np.array([  lepton_1.e,
                            lepton_1.px,
                            lepton_1.py,
                            lepton_1.pz,
                            lepton_2.e,
                            lepton_2.px,
                            lepton_2.py,
                            lepton_2.pz,
                            met.px,
                            met.py
                            ])
            y = np.array([  neutrinos_1.px,
                            neutrinos_1.py,
                            neutrinos_1.pz,
                            neutrinos_2.px,
                            neutrinos_2.py,
                            neutrinos_2.pz ]
                            )

            X[line,:] = x
            Y[line,:] = y
            b = np.array([boson.e, boson.px, boson.py, boson.pz])
            l = np.array([lepton_1.e+lepton_2.e, lepton_1.px+lepton_2.px, lepton_1.py+lepton_2.py, lepton_1.pz+lepton_2.pz])
            #m = np.array([neutrino_sum.e, neutrino_sum.px, neutrino_sum.py, neutrino_sum.pz])
            #M[line,:] = m
            phys_m = np.array([neutrino_sum.pt, neutrino_sum.eta, neutrino_sum.phi, neutrino_sum.m()])
            phys_M[line,:] = phys_m


            B[line,:] = b
            L[line,:] = l
            DM[line] = decay_channel
            #genmass[line] = boson.m()
    
    # filter for selected Decay modes
    #selected_events = [a for a in range(len(DM)) if DM[a] == 'tt' and genmass[a][0] < 300]
    selected_events = [a for a in range(len(DM)) if DM[a] == 'tt']
    X = np.array([X[x] for x in selected_events])
    Y = np.array([Y[x] for x in selected_events])
    B = np.array([B[x] for x in selected_events])
    #M = np.array([M[x] for x in selected_events])
    L = np.array([L[x] for x in selected_events])
    if save_cache:
        cache_output = open(os.path.join(out_folder, 'cache.pkl'), 'wb')
        pickle.dump(X, cache_output)
        pickle.dump(Y, cache_output)
        pickle.dump(B, cache_output)
        pickle.dump(M, cache_output)
        pickle.dump(L, cache_output)
        pickle.dump(phys_M, cache_output)
        cache_output.close()
    return X, Y, B, M, L, phys_M

def load_from_pickle(in_filename):
    cache_output = open(in_filename, 'rb')
    X = pickle.load(cache_output)
    Y = pickle.load(cache_output)
    B = pickle.load(cache_output)
    M = pickle.load(cache_output)
    L = pickle.load(cache_output)
    phys_M = pickle.load(cache_output)
    cache_output.close()
    return X, Y, B, M, L, phys_M


def load_model(model_folder):
    from keras.models import load_model

    model = load_model(os.path.join(model_folder, 'toy_mass.h5'))
    return model

def train_model(X, Y, model_filename = "toy_mass.h5", out_folder='', previous_model=None):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    from keras.layers import GaussianNoise
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    sess = tf.Session(config=config)
    set_session(sess)
    kernel_initializer = "random_uniform"
    bias_initializer = "Zeros"
    
    if previous_model == None:    
        model = Sequential()
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(X.shape[1],)))
#        model.add(GaussianNoise(stddev=2))
  #      model.add(Dropout(0.01))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
 #       model.add(Dropout(0.01))
#        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dropout(0.01))
   #     model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(Y.shape[1], activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = None)
    else:
        model = load_model(previous_model)

    model.summary()
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    model_checkpoint = ModelCheckpoint(os.path.join(out_folder, 'model.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stopping = EarlyStopping(patience = 20)
    model.fit(X, Y, # Training data
                batch_size=100000, # Batch size
                epochs=500, # Number of training epochs
                validation_split=0.1,
                callbacks = [model_checkpoint])#early_stopping, 
    model.save(os.path.join(out_folder, model_filename))
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

    
if __name__ == '__main__':
    in_filename = sys.argv[1]
    out_folder = sys.argv[2]
    if len(sys.argv) > 3:
        previous_model = sys.argv[3]
    else:
        previous_model = None
    print "previous: ", previous_model
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if in_filename[-4:] == ".log":
        X, Y, B, M, L, phys_M = load_from_log(in_filename, "pickle.pkl", out_folder=out_folder, save_cache=True)
    elif in_filename[-4:] == ".pkl":
        X, Y, B, M, L, phys_M = load_from_pickle(in_filename)
    model = train_model(X, Y, out_folder=out_folder, previous_model = previous_model)
    model.summary()		
