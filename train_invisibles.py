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

from plot_invisibles import transform_fourvector
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

def add_pu_target(X, Y, offset, slope, loc):
    tmp_Y = np.zeros([Y.shape[0], Y.shape[1]+17])
    tmp_X = np.zeros([X.shape[0], X.shape[1]+2])
   
    for i in range(tmp_Y.shape[0]):
        for j in range(X.shape[1]):
            tmp_X[i,j] = X[i,j]

        pT = np.sqrt(np.square(tmp_X[i,1] + tmp_X[i,5]) + np.square(tmp_X[i,2] + tmp_X[i, 6]))
        scale = offset + np.sqrt(pT) * slope

        cov_x = np.max([np.random.normal(loc = loc, scale = scale), 0.0])
        cov_y = np.max([np.random.normal(loc = loc, scale = scale), 0.0])
        smear_x = np.random.normal(loc = 0.0, scale = cov_x)
        smear_y = np.random.normal(loc = 0.0, scale = cov_y)
        tmp_X[i,8] = tmp_X[i,8] + smear_x
        tmp_X[i,9] = tmp_X[i,9] + smear_y

        tmp_X[i,10] = np.abs(cov_x)
        tmp_X[i,11] = np.abs(cov_y)

        vis = [X[i,0] + X[i,4], X[i,1]+X[i,5], X[i,2]+X[i,6], X[i,3]+X[i,7]]
        tau_1 = [X[i,0], X[i,1], X[i,2], X[i,3]]
        tau_2 = [X[i,4], X[i,5], X[i,6], X[i,7]]

        tmp_Y[i] = np.array([a for a in Y[i]] + [smear_x, smear_y, tmp_X[i,8], tmp_X[i,9], pT] + vis + tau_1 + tau_2)

    return tmp_X, tmp_Y

def load_from_log(in_filename, out_filename, save_cache=False, out_folder=""):
    n_events = sum(1 for line in open(in_filename))
    
    dim = 10
    targets = 13
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
                            neutrinos_2_2.pz,
                            boson.m() ]
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


    # Y: 0-5 : Neutrino 1/2 x, y, z
    # Y: 6 : gen Mass

    # Y: 7/8: Smear x/y
    # Y: 9/10: smeared met???
    # Y: 11: pt
    # Y: 12-15: 4-vector visible
    # Y: 16-19: 4-vector tau1
    # Y: 20-23: 4-vector tau2
mtau_squared = np.square(np.float64(1.776))
def custom_loss(y_true, y_pred):
    gen_mass = y_true[:,6]
#    dm = K.mean(K.square(y_pred[:,6] - y_true[:,6]) ) / gen_mass
    dx = (K.square(y_pred[:,0] - y_true[:,0])/gen_mass) + (K.square(y_pred[:,3] - y_true[:,3])/gen_mass) + (K.square(y_pred[:,7] - y_true[:,7])/gen_mass)
    dy = (K.square(y_pred[:,1] - y_true[:,1])/gen_mass) + (K.square(y_pred[:,4] - y_true[:,4])/gen_mass) + (K.square(y_pred[:,8] - y_true[:,8])/gen_mass)
#    dz = (K.square(y_pred[:,2] - y_true[:,2])/gen_mass) + (K.square(y_pred[:,5] - y_true[:,5])/gen_mass)

	# difference of final mass
    #e_squared = K.square(y_true[:,12] +
    #                     K.sqrt( K.square(y_pred[:,0]) + K.square(y_pred[:,1]) + K.square(y_pred[:,2])) +
    #                     K.sqrt( K.square(y_pred[:,3]) + K.square(y_pred[:,4]) + K.square(y_pred[:,5])))
	#
    #p_squared = (K.square(y_true[:,13] + y_pred[:,0] + y_pred[:,3]) +
    #             K.square(y_true[:,14] + y_pred[:,1] + y_pred[:,4]) +
    #             K.square(y_true[:,15] + y_pred[:,2] + y_pred[:,5]))
    #m_loss = (K.square((e_squared - p_squared - K.square(gen_mass)) / K.square(gen_mass)))

    # impulserhaltung der met
    dmet_x = (K.square((y_pred[:,0] + y_pred[:,3] + y_pred[:,7]) - y_true[:,9]) / gen_mass)
    dmet_y = (K.square((y_pred[:,1] + y_pred[:,4] + y_pred[:,8]) - y_true[:,10]) / gen_mass)

    # invariante tau-masse
    dm_tau_1 = ((K.square(y_true[:,16] + K.sqrt( K.square(y_pred[:,0]) + K.square(y_pred[:,1]) + K.square(y_pred[:,2]))) -
                     ( K.square(y_true[:,17] + y_pred[:,0]) + K.square(y_true[:,18] + y_pred[:,1]) + K.square(y_true[:,19] + y_pred[:,2])) -
                       mtau_squared)/gen_mass)

    dm_tau_2 = ((K.square(y_true[:,20] + K.sqrt( K.square(y_pred[:,3]) + K.square(y_pred[:,4]) + K.square(y_pred[:,5]))) -
                     ( K.square(y_true[:,21] + y_pred[:,3]) + K.square(y_true[:,22] + y_pred[:,4]) + K.square(y_true[:,23] + y_pred[:,5])) -
                       mtau_squared)/gen_mass)

    return K.mean(dm_tau_1 + dm_tau_2 + dx + dy + dmet_x + dmet_y)

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
    X, Y = add_pu_target(X, Y, 6., 0.0, 24.)
    
    if previous_model == None:    
        model = Sequential()
        model.add(Dense(1000, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(X.shape[1],)))
        model.add(GaussianNoise(stddev=1.0))
        model.add(Dropout(0.1))
        model.add(Dense(1000, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dropout(0.1))
        model.add(Dense(1000, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(Y.shape[1], activation='linear'))
        model.compile(loss=custom_loss, optimizer='adam')
    else:
        model = load_model(previous_model)

    model.summary()
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience = 50)

    from sklearn.model_selection import train_test_split

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
        model.fit(tmp_X, Y_train, # Training data
                    batch_size=50000, # Batch size
                    epochs=1000, # Number of training epochs
                    validation_data = (X_test, Y_test),
                    callbacks = [model_checkpoint, early_stopping])
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
