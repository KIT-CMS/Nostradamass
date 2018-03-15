import numpy as np
import csv
from fourvector import FourVector, FourMomentum, create_FourMomentum
import pickle
import os
import time

def add_pu_target(X, Y, offset, loc):
    tmp_Y = np.zeros([Y.shape[0], Y.shape[1]+12])
    tmp_X = np.zeros([X.shape[0], X.shape[1]+2])
    smear = np.zeros([X.shape[0], 2])
    

    for i in range(X.shape[1]):
        tmp_X[:,i] = X[:,i]

    if(offset == 0.):
        cov = np.zeros([X.shape[0], 2])
    else:
        cov =  np.abs(np.random.normal(loc,offset,tmp_X.shape[0]*2).reshape(tmp_X.shape[0],2))
        print "Smearing Loop"   
        starttime = time.time()
        for i in range(cov.shape[0]):
            if i%1000000 == 1:
                duration = time.time() - starttime
                print "{:3.0f}".format(float(i)/tmp_Y.shape[0]*100), " %, ", \
                     "{:4.1f}".format(duration), " seconds passed, ", \
                     "{:8.2f}".format(float(i)/duration), \
                     " events/s; done in approx", "{:4.4f}".format((tmp_Y.shape[0]-i)/( i/duration)), " s"
            smear[i,0] = np.random.normal(loc = 0.0, scale = cov[i,0])
            smear[i,1] = np.random.normal(loc = 0.0, scale = cov[i,1])
    tmp_X[:,10] = cov[:,0]
    tmp_X[:,11] = cov[:,1]


    tmp_X[:,8] = tmp_X[:,8] + smear[:,0]
    tmp_X[:,9] = tmp_X[:,9] + smear[:,1]

    # fill tmp_Y
    tmp_Y[:,0] = smear[:,0]
    tmp_Y[:,1] = smear[:,1]
    tmp_Y[:,2] = tmp_X[:,8]
    tmp_Y[:,3] = tmp_X[:,9]
    for i in range(8):
        tmp_Y[:,i+4] = X[:,i]
    for i in range(Y.shape[1]):
        tmp_Y[:,i+12] = Y[:,i]

    return tmp_X, tmp_Y

def load_from_root(in_filenames, channel, out_folder=None):
    from root_pandas import read_root
    particle_postfix = ["B", "1", "2", "t1n", "l1n", "t2n", "l2n"]
    particle_prefix = ["e", "px", "py", "pz"]
    branches = []
    dims = 0
    for postfix in particle_postfix:
        for prefix in particle_prefix:
            branches.append(prefix + "_" + postfix)
            dims = dims + 1
    print "loading tree from ", len(in_filenames), " files..."
    in_array = read_root(in_filenames, "tree", columns = branches).as_matrix()

    starttime = time.time()
    n_events = in_array.shape[0]
    print n_events, "loaded!"
    print "allocating memory" 
    X = np.zeros([n_events, 10])
    Y = np.zeros([n_events, 9])
    B = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    I1 = np.zeros([n_events, 4])
    I2 = np.zeros([n_events, 4])

    dimrange = range(dims)
  

    boson, lepton_1, lepton_2, tau1tn, tau1ln, tau2tn, tau2ln = [range(i*4,i*4+4) for i in range(7)]

    print "Converting to required formats..."
    for i in range(4):
        B[:,i] = in_array[:,boson[i]]

    Y[:,0] = np.sqrt( np.square(B[:,0]) - np.square(B[:,1]) - np.square(B[:,2]) - np.square(B[:,3])) 
    print "Leptons"

    for i in range(4):
        L[:,i] = in_array[:,lepton_1[i]] + in_array[:,lepton_2[i]]
        X[:,i] = in_array[:,lepton_1[i]]
        X[:,i+4] = in_array[:,lepton_2[i]]
    print "Neutrinos"

    if (channel[0]=="t"):
        for i in range(4):
            I1[:,i] = in_array[:,tau1tn[i]]
    else:
        for i in range(4):
            I1[:,i] = in_array[:,tau1tn[i]]+in_array[:,tau1ln[i]]

    if (channel[1]=="t"):
        for i in range(4):
            I2[:,i] = in_array[:,tau2tn[i]]
    else:
        for i in range(4):
            I2[:,i] = in_array[:,tau2tn[i]]+in_array[:,tau2ln[i]]
    X[:,8] = I1[:,1] + I2[:,1]
    X[:,9] = I1[:,2] + I2[:,2]

    for i in range(4):
        Y[:,i+1] = I1[:,i]
        Y[:,i+5] = I2[:,i]

    print "Conversion done!"        
    if out_folder != None:
        cache_output = open(os.path.join(out_folder, 'cache.pkl'), 'wb')
        pickle.dump(X, cache_output)
        pickle.dump(Y, cache_output)
        pickle.dump(B, cache_output)
        pickle.dump(L, cache_output)
        cache_output.close()
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
    model = load_model(model_path, custom_objects={'loss_fully_hadronic': loss_fully_hadronic, 'loss_semi_leptonic': loss_semi_leptonic, 'loss_fully_leptonic': loss_fully_leptonic})
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

from losses import i_inv1_e, i_inv1_px, i_inv1_py, i_inv1_pz 
from losses import i_inv2_e, i_inv2_px, i_inv2_py, i_inv2_pz 
from losses import i_tau1_e, i_tau1_px, i_tau1_py, i_tau1_pz
from losses import i_tau2_e, i_tau2_px, i_tau2_py, i_tau2_pz

def predict(model_path, X, channel):
    model = load_model(model_path)
    Y = model.predict(X)
    mTau_squared = np.full([X.shape[0], 1], (1.77**2))

    # fill energy if there is no extra target
    if channel[0] == "t":
        Y[:,i_inv1_e] = np.sqrt( np.square(Y[:,i_inv1_px]) + np.square(Y[:,i_inv1_py]) + np.square(Y[:,i_inv1_pz]))
    else:
        P_1 = mTau_squared[:,0] - np.square(X[:,1] + Y[:,i_inv1_px]) - np.square(X[:,2] + Y[:,i_inv1_py]) - np.square(X[:,3] + Y[:,i_inv1_pz])
        Y[:,i_inv1_e] = (-2*X[:,0] + np.sqrt( (4*np.square(X[:,0]) - 4 * ( np.square(X[:,0]) + P_1 )))) / 2

    if channel[1] == "t":
        Y[:,i_inv2_e] = np.sqrt( np.square(Y[:,i_inv2_px]) + np.square(Y[:,i_inv2_py]) + np.square(Y[:,i_inv2_pz]))
    else:
        P_2 = mTau_squared[:,0] - np.square(X[:,5] + Y[:,i_inv2_px]) - np.square(X[:,6] + Y[:,i_inv2_py]) - np.square(X[:,7] + Y[:,i_inv2_pz])
        Y[:,i_inv2_e] = (-2*X[:,4] + np.sqrt( (4*np.square(X[:,4]) - 4 * ( np.square(X[:,4]) + P_2 )))) / 2

    return Y

