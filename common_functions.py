import numpy as np
import csv
from fourvector import FourVector, FourMomentum, create_FourMomentum
import pickle
import os
import time
import ROOT as r
from multiprocessing import Pool
from functools import partial
import datetime

def get_random(cov_matrix):
    c= np.array(cov_matrix)
    return np.random.multivariate_normal([0,0], c)

def get_resolution_correlation(mass):
    res = 0.01 * mass + 26.
    res_x = np.random.normal(res, 7)
    res_y = np.random.normal(res, 7)
    correlation = np.random.random() * res
    
    return np.array([res_x, res_y, correlation])

def get_random_metcov_general(index, low_edges, high_edges, nbins, probabilities):
    random_xs = [np.random.uniform(low_edges[index][i],high_edges[index][i]) for i in range(nbins[index])]
    return np.random.choice(random_xs, p=probabilities[index])

def add_pu_target(X, Y, primary_dataset, inputfile, savedsmearing):

    tmp_Y = np.zeros([Y.shape[0], Y.shape[1]+X.shape[1]+2])
    tmp_X = np.zeros([X.shape[0], X.shape[1]+3])
    smear = np.zeros([X.shape[0], 2])
    parameters = np.zeros([X.shape[0], 3])

    for i in range(X.shape[1]):
        tmp_X[:,i] = X[:,i]

    if os.path.exists(savedsmearing):
        npz = np.load(savedsmearing)
        parameters[:,0] = npz["cov00"]
        parameters[:,1] = npz["cov11"]
        parameters[:,2] = npz["cov01"]
        smear = npz["smear"]
    else:
        inp = r.TFile(inputfile,"read")
        hcov = [inp.Get("_".join(["hcov00",primary_dataset])), inp.Get("_".join(["hcov11",primary_dataset])), inp.Get("_".join(["hcov01",primary_dataset]))]
        nbins = [h.GetNbinsX() for h in hcov]
        low_edges = [[h.GetBinLowEdge(i+1) for i in range(nbins[j])] for j,h in enumerate(hcov)]
        high_edges = [[ h.GetBinLowEdge(i+1)+h.GetBinWidth(i+1) for i in range(nbins[j])] for j,h in enumerate(hcov)]
        probabilities = [[ h.GetBinContent(i+1) for i in range(nbins[j])] for j,h in enumerate(hcov)]


        cov = np.zeros([X.shape[0], 2,2])
        # mass-dependent smearing
        #n_processes = 10
        #pool = Pool(processes=n_processes)
        #parameters = np.array(pool.map(get_resolution_correlation, Y[:,0].tolist()))

        n_processes = 20
        pool = Pool(processes=n_processes)
        get_random_metcov=partial(get_random_metcov_general, low_edges=low_edges, high_edges=high_edges, nbins=nbins, probabilities=probabilities)

        print "start",datetime.datetime.now()
        for j in range(3):
            print datetime.datetime.now(),"performing met covariance simulation for",j
            indices = [j]*X.shape[0]
            parameters[:,j] = np.array(pool.map(get_random_metcov, indices))
        print "end",datetime.datetime.now()

        cov = np.zeros([Y.shape[0], 2, 2])
        cov[:,0,0] = parameters[:,0]
        cov[:,1,1] = parameters[:,1]
        cov[:,0,1] = parameters[:,2]
        cov[:,1,0] = parameters[:,2]
        starttime = time.time()

        smear = np.array(pool.map(get_random, cov.tolist()))
        duration = time.time() - starttime
        print "{:4.1f}".format(duration), " seconds passed, ", \
             "{:8.2f}".format(tmp_Y.shape[0]/duration), \
             " events/s"
        np.savez(savedsmearing, cov00=cov[:,0,0], cov11=cov[:,1,1], cov01=cov[:,0,1], smear=smear)

    tmp_X[:,10] = np.sqrt(parameters[:,0])
    tmp_X[:,11] = np.sqrt(parameters[:,1])
    #tmp_X[:,10] = parameters[:,0]
    #tmp_X[:,11] = parameters[:,1]
    tmp_X[:,12] = parameters[:,2]


    tmp_X[:,8] = tmp_X[:,8] + smear[:,0]
    tmp_X[:,9] = tmp_X[:,9] + smear[:,1]

    # fill tmp_Y
    tmp_Y[:,0] = smear[:,0]
    tmp_Y[:,1] = smear[:,1]
    tmp_Y[:,2] = tmp_X[:,8]
    tmp_Y[:,3] = tmp_X[:,9]
    print 'newdim', X.shape[1]-2
    for i in range(X.shape[1]-2):
        tmp_Y[:,i+4] = X[:,i]
    for i in range(Y.shape[1]):
        tmp_Y[:,i+12] = Y[:,i]
    return tmp_X, tmp_Y

def load_from_root(in_filenames, channel, use_jets=False):
    from root_pandas import read_root
    particle_postfix = ["B", "1", "2", "t1n", "l1n", "t2n", "l2n"]
    jet_range = range(1, use_jets+1, 1)
    if use_jets:
        for ji in jet_range:
            particle_postfix.append("j" + str(ji))
        
    particle_prefix = ["e", "px", "py", "pz"]
    branches = []
    dims = 0
    for postfix in particle_postfix:
        for prefix in particle_prefix:
            branches.append(prefix + "_" + postfix)
            dims = dims + 1
    branchesnoexp = ["noexpand:%s"%s for s in branches]
    print "loading tree from ", len(in_filenames), " files..."
    in_array = read_root(in_filenames, "tree", columns = branchesnoexp).values

    starttime = time.time()
    n_events = in_array.shape[0]
    print n_events, "loaded!"
    print "allocating memory" 
    X = np.zeros([n_events, 10 + 4*use_jets])
    Y = np.zeros([n_events, 9])
    B = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    I1 = np.zeros([n_events, 4])
    I2 = np.zeros([n_events, 4])
    Jets = []
    indices = {}
    index_names = ["boson", "lepton_1", "lepton_2", "tau1tn", "tau1ln", "tau2tn", "tau2ln"]
    if use_jets:
        for ji in jet_range:
            Jets.append(np.zeros([n_events,4]))
            index_names.append("j" + str(ji))

    dimrange = range(dims)
    for index, name in enumerate(index_names):
        indices[name] = range(index*4,index*4+4)
    print "Converting to required formats..."
    for i in range(4):
        B[:,i] = in_array[:,indices["boson"][i]]

    Y[:,0] = np.sqrt( np.square(B[:,0]) - np.square(B[:,1]) - np.square(B[:,2]) - np.square(B[:,3])) 
    print "Leptons"

    for i in range(4):
        L[:,i] = in_array[:,indices["lepton_1"][i]] + in_array[:,indices["lepton_2"][i]]
        X[:,i] = in_array[:,indices["lepton_1"][i]]
        X[:,i+4] = in_array[:,indices["lepton_2"][i]]

    # jets
    for jet in jet_range:
        for i in range(4):
            Jets[jet-1][:,i] = in_array[:,indices['j'+str(jet)][i]]
            X[:,10+4*(jet-1)+i] = in_array[:,indices['j'+str(jet)][i]]

    print "Neutrinos"

    if (channel[0]=="t"):
        for i in range(4):
            I1[:,i] = in_array[:,indices["tau1tn"][i]]
    else:
        for i in range(4):
            I1[:,i] = in_array[:,indices["tau1tn"][i]]+in_array[:,indices["tau1ln"][i]]

    if (channel[1]=="t"):
        for i in range(4):
            I2[:,i] = in_array[:,indices["tau2tn"][i]]
    else:
        for i in range(4):
            I2[:,i] = in_array[:,indices["tau2tn"][i]]+in_array[:,indices["tau2ln"][i]]
    X[:,8] = I1[:,1] + I2[:,1]
    X[:,9] = I1[:,2] + I2[:,2]

    for i in range(4):
        Y[:,i+1] = I1[:,i]
        Y[:,i+5] = I2[:,i]

    return X, Y, B, L


def load_from_pickle(in_filename):
    cache_output = open(in_filename, 'rb')
    X = pickle.load(cache_output)
    Y = pickle.load(cache_output)
    B = pickle.load(cache_output)
    L = pickle.load(cache_output)
    cache_output.close()
    return X, Y, B, L



def load_model(model_path):
    from keras.models import load_model
    from losses import loss_fully_hadronic, loss_semi_leptonic, loss_fully_leptonic
    from losses import loss_dM_had, loss_dPTtaus, loss_dMtaus_had, loss_dxyz, loss_dPtaus, loss_dmet
    model = load_model(model_path, custom_objects={'loss_fully_hadronic': loss_fully_hadronic, 'loss_semi_leptonic': loss_semi_leptonic, 'loss_fully_leptonic': loss_fully_leptonic, 'loss_dM_had' : loss_dM_had, 'loss_dPTtaus':loss_dPTtaus, 'loss_dMtaus_had' : loss_dMtaus_had, 'loss_dxyz': loss_dxyz, 'loss_dPtaus' : loss_dPtaus, 'loss_dmet':loss_dmet})
    return model

def transform_fourvector(vin, cartesian_types=np.float64, hc_types=np.float64):
    cartesian = np.array([ a.as_list() for a in vin], dtype=cartesian_types)
    phys = np.array([ a.as_list_hcc() for a in vin], dtype=hc_types)
    return phys, cartesian



def full_fourvector(scaled_Y, L, cartesian_types=np.float64, hc_types=np.float64):
    # transformation
    offset = 13 
    vlen = 8 
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( (L[i,0] + sum([scaled_Y[i,j] for j in range(offset + 0, offset + vlen, 4)])),
                                                                                            (L[i,1] + sum([scaled_Y[i,j] for j in range(offset + 1, offset + vlen, 4)])),
                                                                                            (L[i,2] + sum([scaled_Y[i,j] for j in range(offset + 2, offset + vlen, 4)])),
                                                                                            (L[i,3] + sum([scaled_Y[i,j] for j in range(offset + 3, offset + vlen, 4)]))) for i in range(L.shape[0])], cartesian_types, hc_types)
    return regressed_physfourvectors, regressed_fourvectors


#def original_tauh(te_i, tx_i, ty_i, tz_i, nx_i, ny_i, nz_i, X, Y):
#    tau_orig_cartesian = [ FourMomentum( X[i,te_i] + np.sqrt(np.square(Y[i,nx_i]) + np.square(Y[i,ny_i]) + np.square(Y[i,nz_i])),
#                                 X[i,tx_i] + Y[i,nx_i],
#                                 X[i,ty_i] + Y[i,ny_i],
#                                 X[i,tz_i] + Y[i,nz_i]) for i in range(X.shape[0])]
#    tau_orig_phys = np.array( [ [tau_orig_cartesian[i].pt,
#                                 tau_orig_cartesian[i].eta, 
#                                 tau_orig_cartesian[i].phi,
#                                 tau_orig_cartesian[i].m() if tau_orig_cartesian[i].m2()>0 else 0.0] for i in range(len(tau_orig_cartesian))])
#
#    return tau_orig_phys

def original_tau(te_i, tx_i, ty_i, tz_i, ne_i, nx_i, ny_i, nz_i, X, Y):
    tau_orig_cartesian = [ FourMomentum( X[i,te_i] + Y[i,ne_i],
                                 X[i,tx_i] + Y[i,nx_i],
                                 X[i,ty_i] + Y[i,ny_i],
                                 X[i,tz_i] + Y[i,nz_i]) for i in range(X.shape[0])]
    tau_orig_phys = np.array( [ [tau_orig_cartesian[i].pt,
                                 tau_orig_cartesian[i].eta, 
                                 tau_orig_cartesian[i].phi,
                                 #tau_orig_cartesian[i].m() if tau_orig_cartesian[i].m2()>0 else -1.0] for i in range(len(tau_orig_cartesian))])
                                 np.sqrt(np.abs(tau_orig_cartesian[i].m2()))] for i in range(len(tau_orig_cartesian))])

    return tau_orig_phys

def original_tau_cart(te_i, tx_i, ty_i, tz_i, ne_i, nx_i, ny_i, nz_i, X, Y):
    tau_orig_cartesian = [ FourMomentum( X[i,te_i] + Y[i,ne_i],
                                 X[i,tx_i] + Y[i,nx_i],
                                 X[i,ty_i] + Y[i,ny_i],
                                 X[i,tz_i] + Y[i,nz_i]) for i in range(X.shape[0])]
    tau_orig_np = np.array( [ [tau_orig_cartesian[i].e,
                                 tau_orig_cartesian[i].px, 
                                 tau_orig_cartesian[i].py,
                                 tau_orig_cartesian[i].pz] for i in range(len(tau_orig_cartesian))])

    return tau_orig_np

from losses import i_inv1_e, i_inv1_px, i_inv1_py, i_inv1_pz 
from losses import i_inv2_e, i_inv2_px, i_inv2_py, i_inv2_pz 
from losses import i_tau1_e, i_tau1_px, i_tau1_py, i_tau1_pz
from losses import i_tau2_e, i_tau2_px, i_tau2_py, i_tau2_pz
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def predict(model_path, X, channel):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    set_session(sess)
    model = load_model(model_path)
    Y = model.predict(X)
    mTau_squared = np.full([X.shape[0]], (1.77**2))

    # fill energy if there is no extra target
    if channel[0] == "t":
        Y[:,i_inv1_e] = np.sqrt( np.square(Y[:,i_inv1_px]) + np.square(Y[:,i_inv1_py]) + np.square(Y[:,i_inv1_pz]))
    else:
        P_1_squared = np.square(X[:,1] + Y[:,i_inv1_px]) + \
                      np.square(X[:,2] + Y[:,i_inv1_py]) + \
                      np.square(X[:,3] + Y[:,i_inv1_pz])
        Y[:,i_inv1_e] = np.sqrt(mTau_squared + P_1_squared) - X[:,0]

    if channel[1] == "t":
        Y[:,i_inv2_e] = np.sqrt( np.square(Y[:,i_inv2_px]) + np.square(Y[:,i_inv2_py]) + np.square(Y[:,i_inv2_pz]))
    else:
        P_2_squared = np.square(X[:,5] + Y[:,i_inv2_px]) + \
                      np.square(X[:,6] + Y[:,i_inv2_py]) + \
                      np.square(X[:,7] + Y[:,i_inv2_pz])
        Y[:,i_inv2_e] = np.sqrt(mTau_squared + P_2_squared) - X[:,4]

    return Y

