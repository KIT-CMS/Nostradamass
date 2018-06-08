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


def loss_fully_hadronic(y_true, y_pred):
    #mTau_squared = (1.77**2)
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


    dmet_x = (K.square((y_pred[:,i_inv1_px] + y_pred[:,i_inv2_px] + y_pred[:,i_smear_px]) - y_true[:,i_smeared_met_px]) / gen_mass)
    dmet_y = (K.square((y_pred[:,i_inv1_py] + y_pred[:,i_inv2_py] + y_pred[:,i_smear_py]) - y_true[:,i_smeared_met_py]) / gen_mass)

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

    dm_tau_1 = K.abs((
                         K.square(y_true[:,i_tau1_e] + \
                                  K.sqrt( K.square(y_pred[:,i_inv1_px]) + K.square(y_pred[:,i_inv1_py]) + K.square(y_pred[:,i_inv1_pz]))) - \
                       ( K.square(y_true[:,i_tau1_px] + y_pred[:,i_inv1_px]) + \
                         K.square(y_true[:,i_tau1_py] + y_pred[:,i_inv1_py]) + \
                         K.square(y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz])) -
                       mtau_squared)/gen_mass_sqrt)

    dm_tau_2 = K.abs((
                         K.square(y_true[:,i_tau2_e] + \
                                  K.sqrt( K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz]))) - \
                       ( K.square(y_true[:,i_tau2_px] + y_pred[:,i_inv2_px]) + \
                         K.square(y_true[:,i_tau2_py] + y_pred[:,i_inv2_py]) + \
                         K.square(y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz])) -
                       mtau_squared)/gen_mass_sqrt)

    E_1_inv =  K.sqrt(K.square(y_pred[:,i_inv1_px]) + K.square(y_pred[:,i_inv1_py]) + K.square(y_pred[:,i_inv1_pz]))
    E_1 = y_true[:,i_tau1_e] + E_1_inv
    E_2_inv =  K.sqrt(K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz]))
    E_2 = y_true[:,i_tau2_e] + E_2_inv
    
    P_total_squared = K.square(y_true[:,i_tau1_px] + y_pred[:,i_inv1_px] + y_true[:,i_tau2_px] + y_pred[:,i_inv2_px] ) + \
                      K.square(y_true[:,i_tau1_py] + y_pred[:,i_inv1_py] + y_true[:,i_tau2_py] + y_pred[:,i_inv2_py] ) + \
                      K.square(y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz] + y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz] ) 

    dM = K.square((K.square(E_1 + E_2) - P_total_squared - gen_mass) / gen_mass)

    return K.mean(dx + dy + 0.1 * dz + dm_tau_1 + dm_tau_2 + dmet_x + dmet_y + dPT_tau + dM)

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

def loss_M(y_true, y_pred):
    mTau_squared = (1.77**2)
    gen_mass = K.square(y_true[:,i_gen_mass])
    gen_mass_sqrt = y_true[:,i_gen_mass]
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
    return K.mean(dM)

def loss_PT(y_true, y_pred):
    mTau_squared = (1.77**2)
    gen_mass = K.square(y_true[:,i_gen_mass])
    gen_mass_sqrt = y_true[:,i_gen_mass]
    dPT_tau = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_pred[:,i_inv1_py]+y_true[:,i_tau1_py])) - \
                   K.sqrt(K.square(y_true[:,i_inv1_px]+y_true[:,i_tau1_px]) + K.square(y_true[:,i_inv1_py]+y_true[:,i_tau1_py])) ) / gen_mass) + \
              (K.square(K.sqrt(K.square(y_pred[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_pred[:,i_inv2_py]+y_true[:,i_tau2_py])) - \
                   K.sqrt(K.square(y_true[:,i_inv2_px]+y_true[:,i_tau2_px]) + K.square(y_true[:,i_inv2_py]+y_true[:,i_tau2_py])) ) / gen_mass)
    return K.mean(dPT_tau)

def loss_dx(y_true, y_pred):
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
    return K.mean(dx + dy + dz)

def loss_dmTau(y_true, y_pred):
    mTau_squared = (1.77**2)
    gen_mass = K.square(y_true[:,i_gen_mass])
    gen_mass_sqrt = y_true[:,i_gen_mass]
    dm_tau_2 = K.abs((
                         K.square(y_true[:,i_tau2_e] + \
                                  K.sqrt( K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz]))) - \
                       ( K.square(y_true[:,i_tau2_px] + y_pred[:,i_inv2_px]) + \
                         K.square(y_true[:,i_tau2_py] + y_pred[:,i_inv2_py]) + \
                         K.square(y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz])) -
                       mtau_squared)/gen_mass_sqrt)
    return K.mean(dm_tau_2)


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
