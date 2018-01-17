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
    mean_squared_error = K.mean(K.pow((y_pred[:,0] - y_true[:,0]), 2)) + K.mean(K.pow((y_pred[:,1] - y_true[:,1]), 2))
    return mean_squared_error

def mass_loss_start(y_true, y_pred):
    m_squared = K.square(y_true[:,10])
    e_vis = y_true[:,2] + y_true[:,6]
    # eta true
    mass_fraction = y_pred[:,0]
    eta = y_true[:,1]
    F = K.ones_like(mass_fraction) - mass_fraction
    e_neutrino = e_vis * mass_fraction * K.pow(F, -1)

    sinh = 0.5 * (K.exp(eta) - K.exp(-eta))
    pz = sinh * K.sqrt( K.square(y_true[:,11]) + K.square(y_true[:,12]))
    e_squared = K.square(e_vis + e_neutrino)
    p_squared = (K.square(y_true[:,3] + y_true[:,7] + y_true[:,11]) +
                K.square(y_true[:,4] + y_true[:,8] + y_true[:,12]) +
                K.square(y_true[:,5] + y_true[:,9] + pz))

    m_loss_fraction = K.abs(e_squared - p_squared - m_squared)/m_squared

    # fraction true
    mass_fraction = y_true[:,0]
    eta = y_pred[:,1]
    F = K.ones_like(mass_fraction) - mass_fraction
    e_neutrino = e_vis * mass_fraction * K.pow(F, -1)

    sinh = 0.5 * (K.exp(eta) - K.exp(-eta))
    pz = sinh * K.sqrt( K.square(y_true[:,11]) + K.square(y_true[:,12]))
    e_squared = K.square(e_vis + e_neutrino)
    p_squared = (K.square(y_true[:,3] + y_true[:,7] + y_true[:,11]) +
                K.square(y_true[:,4] + y_true[:,8] + y_true[:,12]) +
                K.square(y_true[:,5] + y_true[:,9] + pz))
    m_loss_eta = K.abs(e_squared - p_squared - m_squared)/m_squared

    #delta_eta = K.square(y_pred[:,1] - y_true[:,1])

    return K.mean(m_loss_fraction) + K.mean(m_loss_eta)


def mass_loss_custom(y_true, y_pred):
    m_squared = K.square(y_true[:,10])
    e_vis = y_true[:,2] + y_true[:,6]

    # nix true
    mass_fraction = y_pred[:,0]
    eta = y_pred[:,1]
    F = K.ones_like(mass_fraction) - mass_fraction
    e_neutrino = e_vis * mass_fraction * K.pow(F, -1)

    sinh = 0.5 * (K.exp(eta) - K.exp(-eta))
    pz = sinh * K.sqrt( K.square(y_true[:,11]) + K.square(y_true[:,12]))
    e_squared = K.square(e_vis + e_neutrino)
    p_squared = (K.square(y_true[:,3] + y_true[:,7] + y_true[:,11]) +
                K.square(y_true[:,4] + y_true[:,8] + y_true[:,12]) +
                K.square(y_true[:,5] + y_true[:,9] + pz))

    m_loss = K.abs(e_squared - p_squared - m_squared)/m_squared

    dEnergy = K.square(y_pred[:,0] - y_true[:,0])
    dEta = K.square(y_pred[:,1] - y_true[:,1])

    return K.mean(m_loss) + K.mean(dEnergy) + K.mean(dEta)

def mass_loss_final(y_true, y_pred):
    m_squared = K.square(y_true[:,10])
    e_vis = y_true[:,2] + y_true[:,6]

    # nix true
    mass_fraction = y_pred[:,0]
    eta = y_pred[:,1]
    F = K.ones_like(mass_fraction) - mass_fraction
    e_neutrino = e_vis * mass_fraction * K.pow(F, -1)

    sinh = 0.5 * (K.exp(eta) - K.exp(-eta))
    pz = sinh * K.sqrt( K.square(y_true[:,11]) + K.square(y_true[:,12]))
    e_squared = K.square(e_vis + e_neutrino)
    p_squared = (K.square(y_true[:,3] + y_true[:,7] + y_true[:,11]) +
                K.square(y_true[:,4] + y_true[:,8] + y_true[:,12]) +
                K.square(y_true[:,5] + y_true[:,9] + pz))

    m_loss = K.abs(e_squared - p_squared - m_squared)/m_squared

    return K.mean(m_loss)

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

def load_from_log(in_filename, out_filename, save_cache=False, out_folder=""):
    n_events = sum(1 for line in open(in_filename))
    
    dim = 10
    targets = 13
    X = np.zeros([n_events, dim])
    Y = np.zeros([n_events, targets])
    B = np.zeros([n_events, 4])
    M = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    genmass = np.zeros([n_events, 4])
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
            #y = np.array([neutrino_sum.e/boson.e,
            #                neutrino_sum.eta])
            y = np.array([neutrino_sum.e/boson.e,
                            neutrino_sum.eta,
                            lepton_1.e,
                            lepton_1.px,
                            lepton_1.py,
                            lepton_1.pz,
                            lepton_2.e,
                            lepton_2.px,
                            lepton_2.py,
                            lepton_2.pz,
                            mass,
                            met.px,
                            met.py ])

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
    if save_cache:
        cache_output = open(os.path.join(out_folder, 'cache.pkl'), 'wb')
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

def get_scaled(raw_X, raw_Y, scaler_filename = "scaler.pkl",output_folder=''):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(raw_X)
    #X = scaler.transform(raw_X)
    X = raw_X
    
    scalerTarget = StandardScaler(with_mean=True)
    scalerTarget.fit(raw_Y)
    #Y = scalerTarget.transform(raw_Y)
    Y = raw_Y 
    # save transformations to pickle file
    scaler_output = open(os.path.join(out_folder, scaler_filename), 'wb')
    pickle.dump(scaler, scaler_output)
    pickle.dump(scalerTarget, scaler_output)
    scaler_output.close()
    return X, Y

def load_model(model_folder):
    from keras.models import load_model

    model = load_model(os.path.join(model_folder, 'toy_mass.h5'),  custom_objects={'mass_loss_start': mass_loss_start, 'custom_loss':custom_loss, 'mass_loss_final':mass_loss_final })
    return model

def train_model(X, Y, model_filename = "toy_mass.h5", out_folder='', previous_model=None):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    from keras.initializers import RandomNormal
    from keras.layers import GaussianNoise
    #from keras.layers.normalization import BatchNormalization
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    sess = tf.Session(config=config)
    set_session(sess)
    kernel_initializer = "random_uniform"#RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    bias_initializer = "Zeros" #RandomNormal(mean=0.0, stddev=.25, seed=seed)
    
    if previous_model == None:    
        # model def # energy: 10x40, tanh, mean_squared_error
        model = Sequential()
        model.add(Dense(1000, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(X.shape[1],)))
        #model.add(GaussianNoise(stddev=5))
#        model.add(Dense(900, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dense(800, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dense(700, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dense(600, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(500, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        #for a in range(10):
        #    model.add(Dense(1000, activation='relu'))
            #model.add(BatchNormalization())
#        for a in range(5):
#            model.add(Dense(50, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        for a in range(10):
            model.add(Dense(100, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dense(200, activation='linear'))
        #for a in range(2):
        #for a in range(12):
        #    model.add(Dense(130, activation='relu'))
        model.add(Dense(Y.shape[1], activation='linear'))
        model.compile(loss=mass_loss_custom, optimizer='adam', metrics = [custom_loss, mass_loss_final])
        #model.compile(loss=custom_loss, optimizer='nadam', metrics = [mass_loss])
    else:
        model = load_model(previous_model)

    model.summary()
#    from keras.callbacks import EarlyStopping
#    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    # preliminatry fit to distribute eta
    model.fit(X, Y, # Training data
                batch_size=75000, # Batch size
                epochs=200, # Number of training epochs
                validation_split=0.1)
   #             callbacks = [early_stopping])

    # final fit
#    model.compile(loss=mass_loss_final, optimizer='adam', metrics = [custom_loss, mass_loss_start])
#    model.summary()
#    model.fit(X, Y, # Training data
#                batch_size=75000, # Batch size
#                epochs=10, # Number of training epochs
#                validation_split=0.1)
   #             callbacks = [early_stopping])
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

#def plot(regressed_phys, raw_pred, target_phys, X, Y, B, M, L):
def plot(scaled_Y, regressed_Y, raw_Y, X, Y, B, M, L, out_folder=''):
#    for a in range(Y.shape[1]):
#        pts = plt.figure()
#        arange = None
#        n, bins, patches = plt.hist(regressed_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
#        n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
#        plt.savefig(os.path.join(out_folder, "transform-target-regressed"+str(a)+".png"))
#        plt.close()
    
    
    for a in [0,1]:
        pts = plt.figure()
        arange = None
        n, bins, patches = plt.hist(raw_Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
        n, bins, patches = plt.hist(scaled_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
        plt.savefig(os.path.join(out_folder, "target-regressed"+str(a)+".png"))
        print "target ", a , " resolution: ", np.std(scaled_Y[:,a] - raw_Y[:,a])
        plt.close()

    for a in [0,1]:
        if a == 0:
            xedges = [x/100.0 for x in range(0,100,1)]
        if a == 1:
            xedges = [x/100.0 for x in range(-500,500,50)]

        yedges = xedges
        H, xedges, yedges = np.histogram2d(raw_Y[:,a], scaled_Y[:,a], bins=(xedges, yedges))
        H = H.T
        fig = plt.figure(figsize=(7, 7))
#        ax = fig.add_subplot(131, title='imshow: square bins')
        plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=LogNorm())

        plt.colorbar()
        plt.savefig(os.path.join(out_folder, "nn_target_over_regressed"+str(a)+".png"))

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
    neutrino_target_physfourvectors, neutrino_target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in M])
    neutrino_regressed_physfourvectors, neutrino_regressed_fourvectors = transform_fourvector([ FourMomentum(energy[i], M[i,1], M[i,2], pz[i]) for i in range(M.shape[0])])

    
    for a in range(regressed_fourvectors[0].shape[0]):
        irange = None
        if a==0:
            irange = [-100,2100]
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 100, normed=1, color='red', alpha=1, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(target_fourvectors[:,a], 100, normed=1, color='green', alpha=1, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(vis_fourvectors[:,a], 100, normed=1, color='orange', alpha=1, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig(os.path.join(out_folder, "cartesian-target-regressed"+str(a)+".png"))
    
    for a in range(regressed_physfourvectors.shape[1]):
        pts = plt.figure()
        irange = None
        if a==0:
            irange = [0,500]
        if a==1:
            irange = [-8,8]
        if a==2:
            irange = [-4,4]
        if a==3:
            irange = [0,1100]
        n, bins, patches = plt.hist(target_physfourvectors[:,a], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(vis_physfourvectors[:,a], 100, normed=1, color='orange', alpha=0.5, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig(os.path.join(out_folder, "phys-target-regressed"+str(a)+".png"))
    print 'target neutrino'
    print neutrino_regressed_fourvectors
    print neutrino_target_fourvectors
    # neutrino target/result
    for a in range(neutrino_regressed_fourvectors[0].shape[0]):
        irange = None
        if a==0:
            irange = [-100,2100]
        pts = plt.figure()
        n, bins, patches = plt.hist(neutrino_regressed_fourvectors[:,a], 100, normed=1, color='red', alpha=1, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(neutrino_target_fourvectors[:,a], 100, normed=1, color='green', alpha=1, range=irange, histtype='step', label='target')
        print 'neutrino cartesian resolution ', a, ': ', np.std(neutrino_regressed_fourvectors[:,a] - neutrino_target_fourvectors[:,a])
        plt.legend()
        plt.savefig(os.path.join(out_folder, "neutrino-cartesian-target-regressed"+str(a)+".png"))
    
    for a in range(neutrino_regressed_physfourvectors.shape[1]):
        pts = plt.figure()
        irange = None
        if a==0:
            irange = [0,500]
        if a==1:
            irange = [-8,8]
        if a==2:
            irange = [-4,4]
        if a==3:
            irange = [0,1100]
        n, bins, patches = plt.hist(neutrino_target_physfourvectors[:,a], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(neutrino_regressed_physfourvectors[:,a], 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
        print 'neutrino phys resolution ', a, ': ', np.std(neutrino_regressed_physfourvectors[:,a] - neutrino_target_physfourvectors[:,a])
        plt.legend()
        plt.savefig(os.path.join(out_folder, "neutrino-phys-target-regressed"+str(a)+".png"))

    diff_fourvectors = regressed_fourvectors-target_fourvectors
    
    diff_physfourvectors = regressed_physfourvectors-target_physfourvectors
    diff_physfourvectors = np.array([diff_physfourvectors[i] for i in range(regressed_physfourvectors.shape[0]) if regressed_physfourvectors[i][3]>0])

    for a in range(diff_physfourvectors.shape[1]):
        irange = None
        if a==0:
            irange = [-300,300]
        if a==1:
            irange = [-8,8]
        if a==2:
            irange = [-8,8]
        if a==3:
            irange = [-500,500]
        pts = plt.figure()
        n, bins, patches = plt.hist(diff_physfourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        plt.savefig(os.path.join(out_folder, "diffvector-phys-target-regressed"+str(a)+".png"))
        print "phys diffvector mean ", a, np.mean(diff_physfourvectors[:,a]), " stddev " , np.std(diff_physfourvectors[:,a])

    for a in range(diff_fourvectors.shape[1]):
        irange = None
        pts = plt.figure()
        n, bins, patches = plt.hist(diff_fourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        plt.savefig(os.path.join(out_folder, "diffvector-cartesian-target-regressed"+str(a)+".png"))
        print "cartesian diffvector mean ", a, np.mean(diff_fourvectors[:,a]), " stddev " , np.std(diff_fourvectors[:,a])

    #2D distributions
    for a in range(regressed_fourvectors[0].shape[0]):
        xedges = range(-1000,1000,50)
        if a == 0:
            xedges = range(0,2000,50)

        yedges = xedges 
        H, xedges, yedges = np.histogram2d(regressed_fourvectors[:,a], target_fourvectors[:,a], bins=(xedges, yedges))
        H = H.T
        fig = plt.figure(figsize=(7, 7))
#        ax = fig.add_subplot(131, title='imshow: square bins')
        plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=LogNorm())

        plt.colorbar()
        plt.savefig(os.path.join(out_folder, "cartesian_regressed_over_target"+str(a)+".png"))

    for a in range(regressed_physfourvectors[0].shape[0]):
        if a == 0:
            xedges = range(0,1000,50)
        if a == 1:
            xedges = range(-8,8,1)
        if a == 2:
            xedges = [x/10.0 for x in range(-40,40,1)]
        if a == 3:
            xedges = range(0,500,10)

        yedges = xedges
        H, xedges, yedges = np.histogram2d(regressed_physfourvectors[:,a], target_physfourvectors[:,a], bins=(xedges, yedges))
        H = H.T
        fig = plt.figure(figsize=(7, 7))
#        ax = fig.add_subplot(131, title='imshow: square bins')
        plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=LogNorm())

        plt.colorbar()
        plt.savefig(os.path.join(out_folder, "phys_regressed_over_target"+str(a)+".png"))
    
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
        raw_X, raw_Y, B, M, L = load_from_log(in_filename, "pickle.pkl", out_folder=out_folder, save_cache=True)
    elif in_filename[-4:] == ".pkl":
        raw_X, raw_Y, B, M, L = load_from_pickle(in_filename)
    #X, Y = get_scaled(raw_X, raw_Y)
    X, Y = raw_X, raw_Y
    model = train_model(X, Y, out_folder=out_folder, previous_model = previous_model)
    regressed_Y = predict(model, X)
    import pprint
    pprint.pprint(regressed_Y)
    #scaled_Y = get_inverse(regressed_Y, scaler_target_filename = os.path.join(out_folder, 'scaler.pkl'))
    scaled_Y = regressed_Y
    print np.amax(scaled_Y[:,1])
    print np.amin(scaled_Y[:,1])
    plot(scaled_Y, regressed_Y, raw_Y, X, Y, B, M, L, out_folder)
    # raw_X unscaled
    # raw_Y unscaled
    # X input for DNN
    # Y target for DNN
    # regressed_Y unscaled output
    # scaled_Y scaled output
