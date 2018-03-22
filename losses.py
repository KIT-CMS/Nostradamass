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
    gen_mass = K.square(y_true[:,i_gen_mass])

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

    return K.mean(dx + dy + dz + dmet_x + dmet_y + dm_tau_1 + dm_tau_2)

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

#    dPT = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px]) + K.square(y_pred[:,i_inv1_py])) - \
#                   K.sqrt(K.square(y_true[:,i_inv1_px]) + K.square(y_true[:,i_inv1_py])) ) / gen_mass) + \
#          (K.square(K.sqrt(K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py])) - \
#                   K.sqrt(K.square(y_true[:,i_inv2_px]) + K.square(y_true[:,i_inv2_py])) ) / gen_mass)
#
#    dP = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px]) + K.square(y_pred[:,i_inv1_py]) + K.square(y_pred[:,i_inv1_pz])) - \
#                   K.sqrt(K.square(y_true[:,i_inv1_px]) + K.square(y_true[:,i_inv1_py]) + K.square(y_true[:,i_inv1_pz])) ) / gen_mass) + \
#         (K.square(K.sqrt(K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz])) - \
#                   K.sqrt(K.square(y_true[:,i_inv2_px]) + K.square(y_true[:,i_inv2_py]) + K.square(y_true[:,i_inv2_pz])) ) / gen_mass) 

#    dPcomb = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px] + y_pred[:,i_inv2_px]) + \
#                              K.square(y_pred[:,i_inv1_py] + y_pred[:,i_inv2_py]) + \
#                              K.square(y_pred[:,i_inv1_pz] + y_pred[:,i_inv2_pz])) - \
#                       K.sqrt(K.square(y_true[:,i_inv1_px] + y_true[:,i_inv2_px]) + \
#                              K.square(y_true[:,i_inv1_py] + y_true[:,i_inv2_py]) + \
#                              K.square(y_true[:,i_inv1_pz] + y_true[:,i_inv2_pz])) ) / gen_mass)

#    dPTcomb = (K.square(K.sqrt(K.square(y_pred[:,i_inv1_px] + y_pred[:,i_inv2_px]) + \
#                              K.square(y_pred[:,i_inv1_py] + y_pred[:,i_inv2_py])) - \
#                       K.sqrt(K.square(y_true[:,i_inv1_px] + y_true[:,i_inv2_px]) + \
#                              K.square(y_true[:,i_inv1_py] + y_true[:,i_inv2_py])) ) / gen_mass)

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

 #   P_1 = ( - np.square(y_true[:,i_tau1_px] + y_pred[:,i_inv1_px]) - \
 #             np.square(y_true[:,i_tau1_py] + y_pred[:,i_inv1_py]) - \
 #             np.square(y_true[:,i_tau1_pz] + y_pred[:,i_inv1_pz])) + mtau_squared

#    de = (K.square(((-2*y_true[:,i_tau1_e] + K.sqrt( (4*np.square(y_true[:,i_tau1_e]) - 4 * ( np.square(y_true[:,i_tau1_e]) + P_1 )))) / 2) - y_true[:,i_tau1_e])/gen_mass)  + \
#         (K.square(K.sqrt( K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz])) - y_true[:,i_tau2_e])/gen_mass)

#    de2 = (K.square(y_pred[:,i_inv1_e] - y_true[:,i_inv1_e])/gen_mass) + (K.square(y_pred[:,i_inv2_e] - y_true[:,i_inv2_e])/gen_mass)

    dm_tau_2 = K.abs((
                         K.square(y_true[:,i_tau2_e] + \
                                  K.sqrt( K.square(y_pred[:,i_inv2_px]) + K.square(y_pred[:,i_inv2_py]) + K.square(y_pred[:,i_inv2_pz]))) - \
                       ( K.square(y_true[:,i_tau2_px] + y_pred[:,i_inv2_px]) + \
                         K.square(y_true[:,i_tau2_py] + y_pred[:,i_inv2_py]) + \
                         K.square(y_true[:,i_tau2_pz] + y_pred[:,i_inv2_pz])) -
                       mtau_squared)/gen_mass_sqrt)

    #return K.mean(dx + dy + dz + dmet_x + dmet_y + de + dPT +  dP + dPcomb + dm_tau_2)#+ de2)
    #return K.mean(dx + dy + dz + dmet_x + dmet_y + de + dPT + dm_tau_2)#+ de2)
    return K.mean(dx + dy + dz + dmet_x + dmet_y + dm_tau_2)

def loss_fully_leptonic(y_true, y_pred):
    gen_mass = K.square(y_true[:,i_gen_mass])

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

    return K.mean(dx + dy + dz + dmet_x + dmet_y)
