import numpy as np
from fourvector import FourVector, FourMomentum
import pickle

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
    from losses import custom_loss
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

def transform_fourvector(vin, cartesian_types=np.float64, hc_types=np.float64):
    cartesian = np.array([ a.as_list() for a in vin], dtype=cartesian_types)
    phys = np.array([ a.as_list_hcc() for a in vin], dtype=hc_types)
    return phys, cartesian



def full_fourvector(scaled_Y, L, vlen=6, cartesian_types=np.float64, hc_types=np.float64):
    # transformation
    energy = sum([np.sqrt( sum([np.square(scaled_Y[:,i+j]) for i in range(3)])) for j in range(0, vlen, 3)])
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( (L[i,0] + energy[i]),
                                                                                            (L[i,1] + sum([scaled_Y[i,j] for j in range(0, vlen, 3)])),
                                                                                            (L[i,2] + sum([scaled_Y[i,j] for j in range(1, vlen, 3)])),
                                                                                            (L[i,3] + sum([scaled_Y[i,j] for j in range(2, vlen, 3)]))) for i in range(L.shape[0])], cartesian_types, hc_types)
    return regressed_physfourvectors, regressed_fourvectors


def original_tau(te_i, tx_i, ty_i, tz_i, nx_i, ny_i, nz_i, X, Y):
    tau_orig_cartesian = [ FourMomentum( X[i,te_i] + np.sqrt(np.square(Y[i,nx_i]) + np.square(Y[i,ny_i]) + np.square(Y[i,nz_i])),
                                 X[i,tx_i] + Y[i,nx_i],
                                 X[i,ty_i] + Y[i,ny_i],
                                 X[i,tz_i] + Y[i,nz_i]) for i in range(X.shape[0])]
    tau_orig_phys = np.array( [ [tau_orig_cartesian[i].pt,
                                 tau_orig_cartesian[i].eta, 
                                 tau_orig_cartesian[i].phi,
                                 tau_orig_cartesian[i].m() if tau_orig_cartesian[i].m2()>0 else 0.0] for i in range(len(tau_orig_cartesian))])

    return tau_orig_phys

