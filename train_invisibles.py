from os import environ
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ['CUDA_VISIBLE_DEVICES'] = "1"
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

import keras.backend as K
from matplotlib.colors import LogNorm

selected_channel = 'tt'

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

def add_pu_target(X, Y, magnitude):
    tmp_Y = np.zeros([Y.shape[0], Y.shape[1]+4])
    tmp_X = X.copy()
   
    for i in range(tmp_Y.shape[0]):
        smear_x = np.random.normal(loc = 0.0, scale = magnitude)
        smear_y = np.random.normal(loc = 0.0, scale = magnitude)
        tmp_X[i,8] = tmp_X[i,8] + smear_x
        tmp_X[i,9] = tmp_X[i,9] + smear_y

        tmp_Y[i] = np.array([a for a in Y[i]] + [smear_x, smear_y, tmp_X[i,8], tmp_X[i,9]])

    return tmp_X, tmp_Y

def smear_met_relative(X, magnitude):
    new_X = X.copy()
    pT = np.sqrt(np.square(X[:,1]+X[:,5]) + np.square(X[:,2]+X[:,6]))
    for i in range(X.shape[0]):
        new_X[i,8] = 0#X[i,8] + np.random.normal(loc = 0.0, scale = pT[i] * magnitude)
        new_X[i,9] = 0#X[i,9] + np.random.normal(loc = 0.0, scale = pT[i] * magnitude)

    return new_X

def smear_met(X, magnitude):
    new_X = X.copy()
    for i in range(X.shape[0]):
        new_X[i,8] = X[i,8] + np.random.normal(loc = 0.0, scale = magnitude)
        new_X[i,9] = X[i,9] + np.random.normal(loc = 0.0, scale = magnitude)

    return new_X

def load_from_log(in_filename, out_filename, save_cache=False, out_folder=""):
    n_events = sum(1 for line in open(in_filename))
    
    dim = 10
    targets = 12
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
            posTauInvis1 = create_FourMomentum(row[2])
            posTauInvis2 = create_FourMomentum(row[3])
            posTauNNeutrinos = count_neutrinos(row[3])
            posTauDecayType = get_decay(row[3])
            negTauVis = create_FourMomentum(row[4])
            negTauInvis1 = create_FourMomentum(row[5])
            negTauInvis2 = create_FourMomentum(row[6])
            negTauNNeutrinos = count_neutrinos(row[6])
            negTauDecayType = get_decay(row[6])
            if posTauNNeutrinos >= negTauNNeutrinos:
                lepton_1 = posTauVis 
                lepton_2 = negTauVis
                neutrinos_1_1 = posTauInvis1
                neutrinos_1_2 = posTauInvis2
                neutrinos_2_1 = negTauInvis1
                neutrinos_2_2 = negTauInvis2
                lepton_1_neutrinos = posTauNNeutrinos
                lepton_2_neutrinos = negTauNNeutrinos
                decay_channel = posTauDecayType + negTauDecayType
            else:
                lepton_1 = negTauVis 
                lepton_2 = posTauVis
                neutrinos_1_1 = negTauInvis1
                neutrinos_1_2 = negTauInvis2
                neutrinos_2_1 = posTauInvis1
                neutrinos_2_2 = posTauInvis2
                lepton_1_neutrinos = negTauNNeutrinos
                lepton_2_neutrinos = posTauNNeutrinos
                decay_channel = negTauDecayType + posTauDecayType
    
            neutrino_sum = posTauInvis1 + posTauInvis2 + negTauInvis1 + negTauInvis2
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
            y = np.array([  neutrinos_1_1.px,
                            neutrinos_1_1.py,
                            neutrinos_1_1.pz,
                            neutrinos_1_2.px,
                            neutrinos_1_2.py,
                            neutrinos_1_2.pz,
                            neutrinos_2_1.px,
                            neutrinos_2_1.py,
                            neutrinos_2_1.pz,
                            neutrinos_2_2.px,
                            neutrinos_2_2.py,
                            neutrinos_2_2.pz ]
                            )

            X[line,:] = x
            Y[line,:] = y
            b = np.array([boson.e, boson.px, boson.py, boson.pz])
            l = np.array([lepton_1.e+lepton_2.e, lepton_1.px+lepton_2.px, lepton_1.py+lepton_2.py, lepton_1.pz+lepton_2.pz])
            phys_m = np.array([neutrino_sum.pt, neutrino_sum.eta, neutrino_sum.phi, neutrino_sum.m()])
            phys_M[line,:] = phys_m


            B[line,:] = b
            L[line,:] = l
            DM[line] = decay_channel
    
    # filter for selected Decay modes
    #selected_events = [a for a in range(len(DM)) if DM[a] == 'tt' and genmass[a][0] < 300]
    selected_events = [a for a in range(len(DM)) if DM[a] == selected_channel]
    X = np.array([X[x] for x in selected_events])
    Y = np.array([Y[x] for x in selected_events])
    B = np.array([B[x] for x in selected_events])
    #M = np.array([M[x] for x in selected_events])
    L = np.array([L[x] for x in selected_events])
        
    if selected_channel == 'tt':
        for a in [3,3,3,6,6,6]:
            Y = np.delete(Y, a, 1)

    if selected_channel == 'mt' or selected_channel == 'et':
        for a in [9,9,9]:
            Y = np.delete(Y, a, 1)

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


def load_model(model_path):
    from keras.models import load_model

    model = load_model(model_path, custom_objects={'custom_loss':custom_loss })
    return model

def custom_loss(y_true, y_pred):
    dx = K.mean(K.square(y_pred[:,0] - y_true[:,0])) + K.mean(K.square(y_pred[:,3] - y_true[:,3])) + K.mean(K.square(y_pred[:,6] - y_true[:,6]))
    dy = K.mean(K.square(y_pred[:,1] - y_true[:,1])) + K.mean(K.square(y_pred[:,4] - y_true[:,4])) + K.mean(K.square(y_pred[:,7] - y_true[:,7]))
    dz = K.mean(K.square(y_pred[:,2] - y_true[:,2])) + K.mean(K.square(y_pred[:,5] - y_true[:,5]))
    dtrue = K.square( y_true[:,8] - (y_pred[:,0] + y_pred[:,3] +  y_pred[:,6])) + K.square( y_true[:,9] - (y_pred[:,1] + y_pred[:,4] +  y_pred[:,7]))

    mean_squared_error = dx + dy + dz + dtrue
    return mean_squared_error

def train_model(X, Y, model_filename = "toy_mass.h5", out_folder='', previous_model=None):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    from keras.layers import GaussianNoise
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    set_session(sess)
    kernel_initializer = "random_uniform"
    bias_initializer = "Zeros"

    X, Y = add_pu_target(X, Y, 10.0)
    
    if previous_model == None:    
        model = Sequential()
        model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(X.shape[1],)))
        model.add(GaussianNoise(stddev=0.5))
  #      model.add(Dropout(0.01))
        model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
  #      model.add(Dropout(0.01))
        model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dropout(0.01))
        model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(Y.shape[1], activation='linear'))
        model.compile(loss=custom_loss, optimizer='adam', metrics = ['mean_squared_error'])
    else:
        model = load_model(previous_model)

    model.summary()
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience = 20)


    from sklearn.model_selection import train_test_split
    print X
    print Y

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
#    X_test = smear_met_relative(X_test, magnitude = 20.0) # test under hardest conditions; maybe even too strict
#    tmp_X = smear_met_relative(X_train, magnitude = 0.)
    tmp_X = X_train 

    for i in range(1):
        model_checkpoint = ModelCheckpoint( os.path.join(out_folder, 'model.'+str(i)+'.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        print "turn ", i
        #tmp_X = smear_met(X_train, magnitude = 2. + i/2.0)
        model.fit(tmp_X, Y_train, # Training data
                    batch_size=50000, # Batch size
                    epochs=1000, # Number of training epochs
                    validation_data = (X_test, Y_test),
                    callbacks = [model_checkpoint])#early_stopping
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
