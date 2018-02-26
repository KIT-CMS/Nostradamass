import numpy as np
import csv
from fourvector import FourVector, FourMomentum, create_FourMomentum
from root_numpy import root2array
import pickle
import os

def get_index(channel, n_neutrino):
    if channel == 'tt':
        if n_neutrino == 0:
            return "nt_1"
        else:
            return "nt_2"
    elif channel == "mt" or channel == "et":
        if n_neutrino == 0:
            return "nt_1"
        elif n_neutrino == 1:
            return "nl_1"
        else:
            return "nt_1"

def add_pu_target(X, Y, offset, loc):
    tmp_Y = np.zeros([Y.shape[0], Y.shape[1]+12])
    tmp_X = np.zeros([X.shape[0], X.shape[1]+2])
   
    for i in range(tmp_Y.shape[0]):
        for j in range(X.shape[1]):
            tmp_X[i,j] = X[i,j]

        cov_x = np.max([np.random.normal(loc = loc, scale = offset), 0.0])
        cov_y = np.max([np.random.normal(loc = loc, scale = offset), 0.0])
        smear_x = np.random.normal(loc = 0.0, scale = cov_x)
        smear_y = np.random.normal(loc = 0.0, scale = cov_y)
        tmp_X[i,8] = tmp_X[i,8] + smear_x
        tmp_X[i,9] = tmp_X[i,9] + smear_y

        tmp_X[i,10] = np.abs(cov_x)
        tmp_X[i,11] = np.abs(cov_y)

        tau_1 = [X[i,0], X[i,1], X[i,2], X[i,3]]
        tau_2 = [X[i,4], X[i,5], X[i,6], X[i,7]]

        tmp_Y[i] = np.array( [smear_x, smear_y, tmp_X[i,8], tmp_X[i,9]] + tau_1 + tau_2 + [a for a in Y[i]])

    return tmp_X, tmp_Y


def load_from_root(in_filenames, channel, out_folder=None):
    particle_postfix = ["B", "1", "2", "t1n", "l1n", "t2n", "l2n"]
    particle_prefix = ["id", "e", "px", "py", "pz"]
    branches = []
    for postfix in particle_postfix:
        for prefix in particle_prefix:
            branches.append(prefix + "_" + postfix)
    in_array = root2array(in_filenames, "tree", branches = branches)
    n_events = in_array.shape[0]
    
    dim = 10
    targets = 13
    X = np.zeros([n_events, dim])
    Y = np.zeros([n_events, targets])
    B = np.zeros([n_events, 4])
    M = None#np.zeros([n_events, 4])
    phys_M = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    boson, lepton_1, lepton_2, tau1tn, tau1ln, tau2tn, tau2ln = None, None, None, None, None, None, None

    for line_number, line in enumerate(in_array):
        four_vectors = [] 
        for index in range(0, 35, 5):
            four_vectors.append(FourMomentum(line[index+1], line[index+2], line[index+3], line[index+4]))
        boson, lepton_1, lepton_2, tau1tn, tau1ln, tau2tn, tau2ln = four_vectors
        met = tau1tn + tau2tn
        if tau1ln != None:
            met = met + tau1ln
        if tau2ln != None:
            met = met + tau2ln

        visible = lepton_1 + lepton_2  
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
        y = np.array([  boson.m(), 
                        tau1tn.px,
                        tau1tn.py,
                        tau1tn.pz,
                        tau1ln.px,
                        tau1ln.py,
                        tau1ln.pz,
                        tau2tn.px,
                        tau2tn.py,
                        tau2tn.pz,
                        tau2ln.px,
                        tau2ln.py,
                        tau2ln.pz ]
                        )
        X[line_number,:] = x
        Y[line_number,:] = y
        b = np.array([boson.e, boson.px, boson.py, boson.pz])
        l = np.array([visible.e, visible.px, visible.py, visible.pz])
        phys_m = np.array([met.pt, 0, met.phi, 0])
        phys_M[line_number,:] = phys_m

        B[line_number,:] = b
        L[line_number,:] = l
        
    if channel == 'tt':
        for a in [4,4,4,7,7,7]:
            Y = np.delete(Y, a, 1)

    if channel == 'mt' or channel == 'et':
        for a in [10,10,10]:
            Y = np.delete(Y, a, 1)
    if out_folder != None:
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
    from losses import loss_fully_hadronic, loss_semi_leptonic, loss_fully_leptonic
    model = load_model(model_path, custom_objects={'loss_fully_hadronic': loss_fully_hadronic, 'loss_semi_leptonic': loss_semi_leptonic, 'loss_fully_leptonic': loss_fully_leptonic})
    return model

def transform_fourvector(vin, cartesian_types=np.float64, hc_types=np.float64):
    cartesian = np.array([ a.as_list() for a in vin], dtype=cartesian_types)
    phys = np.array([ a.as_list_hcc() for a in vin], dtype=hc_types)
    return phys, cartesian



def full_fourvector(scaled_Y, L, vlen=6, cartesian_types=np.float64, hc_types=np.float64):
    # transformation
    offset = 13 
    energy = sum([np.sqrt( sum([np.square(scaled_Y[:,i+j]) for i in range(3)])) for j in range(13, 13+vlen, 3)])
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( (L[i,0] + energy[i]),
                                                                                            (L[i,1] + sum([scaled_Y[i,j] for j in range(offset + 0, offset + vlen, 3)])),
                                                                                            (L[i,2] + sum([scaled_Y[i,j] for j in range(offset + 1, offset + vlen, 3)])),
                                                                                            (L[i,3] + sum([scaled_Y[i,j] for j in range(offset + 2, offset + vlen, 3)]))) for i in range(L.shape[0])], cartesian_types, hc_types)
    return regressed_physfourvectors, regressed_fourvectors


def original_tauh(te_i, tx_i, ty_i, tz_i, nx_i, ny_i, nz_i, X, Y):
    tau_orig_cartesian = [ FourMomentum( X[i,te_i] + np.sqrt(np.square(Y[i,nx_i]) + np.square(Y[i,ny_i]) + np.square(Y[i,nz_i])),
                                 X[i,tx_i] + Y[i,nx_i],
                                 X[i,ty_i] + Y[i,ny_i],
                                 X[i,tz_i] + Y[i,nz_i]) for i in range(X.shape[0])]
    tau_orig_phys = np.array( [ [tau_orig_cartesian[i].pt,
                                 tau_orig_cartesian[i].eta, 
                                 tau_orig_cartesian[i].phi,
                                 tau_orig_cartesian[i].m() if tau_orig_cartesian[i].m2()>0 else 0.0] for i in range(len(tau_orig_cartesian))])

    return tau_orig_phys

def original_taul(te_i, tx_i, ty_i, tz_i, ntx_i, nty_i, ntz_i, nlx_i, nly_i, nlz_i, X, Y):
    tau_orig_cartesian = [ FourMomentum( X[i,te_i] + np.sqrt(np.square(Y[i,ntx_i]) + np.square(Y[i,nty_i]) + np.square(Y[i,ntz_i])) +
                                                     np.sqrt(np.square(Y[i,nlx_i]) + np.square(Y[i,nly_i]) + np.square(Y[i,nlz_i])),
                                 X[i,tx_i] + Y[i,ntx_i] + Y[i,nlx_i],
                                 X[i,ty_i] + Y[i,nty_i] + Y[i,nly_i],
                                 X[i,tz_i] + Y[i,ntz_i] + Y[i,nlz_i]) for i in range(X.shape[0])]
    tau_orig_phys = np.array( [ [tau_orig_cartesian[i].pt,
                                 tau_orig_cartesian[i].eta, 
                                 tau_orig_cartesian[i].phi,
                                 tau_orig_cartesian[i].m() if tau_orig_cartesian[i].m2()>0 else 0.0] for i in range(len(tau_orig_cartesian))])

    return tau_orig_phys

