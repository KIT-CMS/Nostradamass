import numpy as np
import keras.backend as K
mtau_squared = np.square(np.float64(1.776))
mmu_squared = np.square(np.float64(0.105))

i_smear_px = 0
i_smear_py = 1
i_smeared_met_px = 2
i_smeared_met_py = 3
i_tau1_e = 4
i_tau1_px = 5
i_tau1_py = 6
i_tau1_pz = 7
i_tau2_e = 8
i_tau2_px = 9
i_tau2_py = 10
i_tau2_pz = 11
i_gen_mass = 12
i_inv1_e = 13
i_inv1_px = 14
i_inv1_py = 15
i_inv1_pz = 16
i_inv2_e = 17
i_inv2_px = 18
i_inv2_py = 19
i_inv2_pz = 20

## helper functions

def gen_mass(y_true):

    return K.square(y_true[:,i_gen_mass])

def P_squared_tau1(y_true, y_pred):

    return K.square(y_pred[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_pred[:,i_inv1_py]+y_true[:,i_tau1_py]) + K.square(y_pred[:,i_inv1_pz]+y_true[:,i_tau1_pz])

def P_squared_tau2(y_true, y_pred):

    return K.square(y_pred[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_pred[:,i_inv2_py]+y_true[:,i_tau2_py]) + K.square(y_pred[:,i_inv2_pz]+y_true[:,i_tau2_pz])

def P_squared_resonance(y_true, y_pred):

    Px = y_true[:,i_tau1_px] + y_pred[:,i_inv1_px] + y_true[:,i_tau2_px] + y_pred[:,i_inv2_px]
    Py = y_true[:,i_tau1_py] + y_pred[:,i_inv1_py] + y_true[:,i_tau2_py] + y_pred[:,i_inv2_py]
    Pz = y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz] + y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz]

    return K.square(Px) + K.square(Py) + K.square(Pz)

def E_tau1_had(y_true, y_pred):

    E_tau1_inv = K.sqrt(K.square(y_pred[:,i_inv1_px]) + K.square(y_pred[:,i_inv1_py]) + K.square(y_pred[:,i_inv1_pz])) # asuming only 1 neutrino in tau decay -> mass_inv = 0

    return  y_true[:,i_tau1_e] + E_tau1_inv

def E_tau2_had(y_true, y_pred):

    E_tau2_inv = K.sqrt(K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz])) # asuming only 1 neutrino in tau decay -> mass_inv = 0

    return  y_true[:,i_tau2_e] + E_tau2_inv

## metrics (intermediate components of the full loss functions)

def loss_dxyz(y_true, y_pred):

    # construct difference for the 8 target components (invisible 3-momenta & detector pt)
    target_components_indices = [i_inv1_px, i_inv2_px, i_smear_px, i_inv1_py, i_inv2_py, i_smear_py, i_inv1_pz, i_inv2_pz]
    target_components_diff_list = [ K.square(y_pred[:,i] - y_true[:,i])/gen_mass(y_true) for i in target_components_indices]

    dxyz = 0
    for d in target_components_diff_list: dxyz+=d

    return dxyz

def loss_dmet(y_true, y_pred):

    # construct difference for the MET components
    met_x = y_pred[:,i_inv1_px] + y_pred[:,i_inv2_px] + y_pred[:,i_smear_px]
    met_y = y_pred[:,i_inv1_py] + y_pred[:,i_inv2_py] + y_pred[:,i_smear_py]

    dmet_x = K.square(met_x - y_true[:,i_smeared_met_px])/gen_mass(y_true)
    dmet_y = K.square(met_y - y_true[:,i_smeared_met_py])/gen_mass(y_true)

    return dmet_x + dmet_y

def loss_dPTtaus(y_true, y_pred):

    # construct difference in absolute value of tau PTs
    PT_tau1 = K.sqrt(K.square(y_pred[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_pred[:,i_inv1_py]+y_true[:,i_tau1_py]))
    PT_tau2 = K.sqrt(K.square(y_pred[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_pred[:,i_inv2_py]+y_true[:,i_tau2_py]))

    true_PT_tau1 = K.sqrt(K.square(y_true[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_true[:,i_inv1_py]+y_true[:,i_tau1_py]))
    true_PT_tau2 = K.sqrt(K.square(y_true[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_true[:,i_inv2_py]+y_true[:,i_tau2_py]))

    return (K.square(PT_tau1 - true_PT_tau1) + K.square(PT_tau2 - true_PT_tau2))/gen_mass(y_true)

def loss_dPtaus(y_true, y_pred):

    # construct difference in absolute value of tau momemta
    P_tau1 =  K.sqrt(P_squared_tau1(y_true, y_pred))
    P_tau2 =  K.sqrt(P_squared_tau2(y_true, y_pred))

    true_P_tau1 = K.sqrt(K.square(y_true[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_true[:,i_inv1_py]+y_true[:,i_tau1_py]) + K.square(y_true[:,i_inv1_pz]+y_true[:,i_tau1_pz]))
    true_P_tau2 = K.sqrt(K.square(y_true[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_true[:,i_inv2_py]+y_true[:,i_tau2_py]) + K.square(y_true[:,i_inv2_pz]+y_true[:,i_tau2_pz]))

    return (K.square(P_tau1 - true_P_tau1) + K.square(P_tau2 - true_P_tau2))/gen_mass(y_true)

def loss_dM_had(y_true, y_pred):

    # construct constraint on resonance mass in case of fully hadronic decay
    E_squared_resonance = K.square(E_tau1_had(y_true, y_pred) + E_tau2_had(y_true, y_pred))

    return K.abs(E_squared_resonance - P_squared_resonance(y_true, y_pred) - gen_mass(y_true)) / gen_mass(y_true)

def loss_dMtaus_had(y_true, y_pred):

    # construct constraint on tau masses in case of fully hadronic decay
    M_squared_tau1 = K.square(E_tau1_had(y_true, y_pred)) - P_squared_tau1(y_true, y_pred)
    M_squared_tau2 = K.square(E_tau2_had(y_true, y_pred)) - P_squared_tau2(y_true, y_pred)

    return (K.abs(M_squared_tau1 - mtau_squared) + K.abs(M_squared_tau2 - mtau_squared)) / gen_mass(y_true)

## loss functions

def loss_fully_hadronic(y_true, y_pred):

    dxyz = loss_dxyz(y_true, y_pred)
    dmet = loss_dmet(y_true, y_pred)
    dPTtaus = loss_dPTtaus(y_true, y_pred)
    dPtaus = loss_dPtaus(y_true, y_pred)
    dm_taus = loss_dMtaus_had(y_true, y_pred)
    dM = loss_dM_had(y_true, y_pred)

    return dxyz + dm_taus + dmet + dPTtaus + dM + dPtaus

def loss_semi_leptonic(y_true, y_pred):
    mTau_squared = (1.77**2)
    gen_mass = K.square(y_true[:,i_gen_mass])
    gen_mass_sqrt = y_true[:,i_gen_mass]

    dx = (K.square(y_pred[:,i_inv1_px] - y_true[:,i_inv1_px])/gen_mass) + \
         (K.square(y_pred[:,i_inv2_px] - y_true[:,i_inv2_px])/gen_mass) + \
         (K.square(y_pred[:,i_smear_px] - y_true[:,i_smear_px])/gen_mass)

    dy = (K.square(y_pred[:,i_inv1_py] - y_true[:,i_inv1_py])/gen_mass) + \
         (K.square(y_pred[:,i_inv2_py] - y_true[:,i_inv2_py])/gen_mass) + \
         (K.square(y_pred[:,i_smear_py] - y_true[:,i_smear_py])/gen_mass)

    dz = (K.square(y_pred[:,i_inv1_pz] - y_true[:,i_inv1_pz])/gen_mass) + \
         (K.square(y_pred[:,i_inv2_pz] - y_true[:,i_inv2_pz])/gen_mass)

    dmet_x = (K.square((y_pred[:,i_inv1_px] +
                        y_pred[:,i_inv2_px] +
                        y_pred[:,i_smear_px]) -
                        y_true[:,i_smeared_met_px]) / gen_mass)

    dmet_y = (K.square((y_pred[:,i_inv1_py] +
                        y_pred[:,i_inv2_py] +
                        y_pred[:,i_smear_py]) -
                        y_true[:,i_smeared_met_py]) / gen_mass)

    dm_tau_2 = K.abs((
                         K.square(y_true[:,i_tau2_e] + \
                                  K.sqrt( K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz]))) - \
                       ( K.square(y_true[:,i_tau2_px] + y_pred[:,i_inv2_px]) + \
                         K.square(y_true[:,i_tau2_py] + y_pred[:,i_inv2_py]) + \
                         K.square(y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz])) -
                       mtau_squared)/gen_mass_sqrt)


    dPT_tau = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_pred[:,i_inv1_py]+y_true[:,i_tau1_py])) - \
                   K.sqrt(K.square(y_true[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_true[:,i_inv1_py]+y_true[:,i_tau1_py])) ) / gen_mass) + \
              (K.square(K.sqrt(K.square(y_pred[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_pred[:,i_inv2_py]+y_true[:,i_tau2_py])) - \
                   K.sqrt(K.square(y_true[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_true[:,i_inv2_py]+y_true[:,i_tau2_py])) ) / gen_mass)

    P_1_squared = K.square(y_true[:,i_tau1_px] + y_pred[:,i_inv1_px]) + \
                  K.square(y_true[:,i_tau1_py] + y_pred[:,i_inv1_py]) + \
                  K.square(y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz])
    E_1 = K.sqrt(mTau_squared + P_1_squared)

    E_2_inv =  K.sqrt(K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz]))
    E_2 = y_true[:,i_tau2_e] + E_2_inv

    P_total_squared = K.square(y_true[:,i_tau1_px] + y_pred[:,i_inv1_px] + y_true[:,i_tau2_px] + y_pred[:,i_inv2_px] ) + \
                      K.square(y_true[:,i_tau1_py] + y_pred[:,i_inv1_py] + y_true[:,i_tau2_py] + y_pred[:,i_inv2_py] ) + \
                      K.square(y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz] + y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz] )

    dM = K.square((K.square(E_1 + E_2) - P_total_squared - gen_mass) / gen_mass)

    return K.mean(dx + dy + 0.1 * dz + dm_tau_2 + dmet_x + dmet_y + dPT_tau + dM)

def loss_fully_leptonic(y_true, y_pred):
    mTau_squared = (1.77**2)
    gen_mass = K.square(y_true[:,i_gen_mass])

    dx = (K.square(y_pred[:,i_inv1_px] - y_true[:,i_inv1_px])/gen_mass) + \
         (K.square(y_pred[:,i_inv2_px] - y_true[:,i_inv2_px])/gen_mass) + \
         (K.square(y_pred[:,i_smear_px] - y_true[:,i_smear_px])/gen_mass)


    dy = (K.square(y_pred[:,i_inv1_py] - y_true[:,i_inv1_py])/gen_mass) + \
         (K.square(y_pred[:,i_inv2_py] - y_true[:,i_inv2_py])/gen_mass) + \
         (K.square(y_pred[:,i_smear_py] - y_true[:,i_smear_py])/gen_mass)

    dz = (K.square(y_pred[:,i_inv1_pz] - y_true[:,i_inv1_pz])/gen_mass) + \
         (K.square(y_pred[:,i_inv2_pz] - y_true[:,i_inv2_pz])/gen_mass)
    dPT_tau = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_pred[:,i_inv1_py]+y_true[:,i_tau1_py])) - \
                   K.sqrt(K.square(y_true[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_true[:,i_inv1_py]+y_true[:,i_tau1_py])) ) / gen_mass) + \
              (K.square(K.sqrt(K.square(y_pred[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_pred[:,i_inv2_py]+y_true[:,i_tau2_py])) - \
                   K.sqrt(K.square(y_true[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_true[:,i_inv2_py]+y_true[:,i_tau2_py])) ) / gen_mass)

    dP_tau = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px]+y_true[:,i_tau1_px]) + \
                              K.square(y_pred[:,i_inv1_py]+y_true[:,i_tau1_py]) + \
                              K.square(y_pred[:,i_inv1_pz]+y_true[:,i_tau1_pz])) - \
                       K.sqrt(K.square(y_true[:,i_inv1_px]+y_true[:,i_tau1_px]) + \
                              K.square(y_true[:,i_inv1_py]+y_true[:,i_tau1_py]) + \
                              K.square(y_true[:,i_inv1_pz]+y_true[:,i_tau1_pz]))) / gen_mass) + \
             (K.square(K.sqrt(K.square(y_pred[:,i_inv2_px]+y_true[:,i_tau2_px]) + \
                              K.square(y_pred[:,i_inv2_py]+y_true[:,i_tau2_py]) + \
                              K.square(y_pred[:,i_inv2_pz]+y_true[:,i_tau2_pz])) - \
                       K.sqrt(K.square(y_true[:,i_inv2_px]+y_true[:,i_tau2_px]) + \
                              K.square(y_true[:,i_inv2_py]+y_true[:,i_tau2_py]) + \
                              K.square(y_true[:,i_inv2_pz]+y_true[:,i_tau2_pz]))) / gen_mass)

    dmet_x = (K.square((y_pred[:,i_inv1_px] +
                        y_pred[:,i_inv2_px] +
                        y_pred[:,i_smear_px]) -
                        y_true[:,i_smeared_met_px]) / gen_mass)

    dmet_y = (K.square((y_pred[:,i_inv1_py] +
                        y_pred[:,i_inv2_py] +
                        y_pred[:,i_smear_py]) -
                        y_true[:,i_smeared_met_py]) / gen_mass)

    P_1_squared = K.square(y_true[:,i_tau1_px] + y_pred[:,i_inv1_px]) + \
                  K.square(y_true[:,i_tau1_py] + y_pred[:,i_inv1_py]) + \
                  K.square(y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz])
    E_1 = K.sqrt(mTau_squared + P_1_squared)

    P_2_squared = K.square(y_true[:,i_tau2_px] + y_pred[:,i_inv2_px]) + \
                  K.square(y_true[:,i_tau2_py] + y_pred[:,i_inv2_py]) + \
                  K.square(y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz])
    E_2 = K.sqrt(mTau_squared + P_2_squared)


    P_total_squared = K.square(y_true[:,i_tau1_px] + y_pred[:,i_inv1_px] + y_true[:,i_tau2_px] + y_pred[:,i_inv2_px] ) + \
                      K.square(y_true[:,i_tau1_py] + y_pred[:,i_inv1_py] + y_true[:,i_tau2_py] + y_pred[:,i_inv2_py] ) + \
                      K.square(y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz] + y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz] )

    dM = K.square((K.square(E_1 + E_2) - P_total_squared - gen_mass) / gen_mass)

    return K.mean(dx + dy + dz + dmet_x + dmet_y + dPT_tau + dP_tau + dM)
