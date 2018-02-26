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
i_tt_t1n_px = 13
i_tt_t1n_py = 14
i_tt_t1n_pz = 15
i_tt_t2n_px = 16
i_tt_t2n_py = 17
i_tt_t2n_pz = 18


def loss_fully_hadronic(y_true, y_pred):
    gen_mass = y_true[:,i_gen_mass]
    dx = (K.square(y_pred[:,i_tt_t1n_px] - y_true[:,i_tt_t1n_px])/gen_mass) + (K.square(y_pred[:,i_tt_t2n_px] - y_true[:,i_tt_t2n_px])/gen_mass) + (K.square(y_pred[:,i_smear_px] - y_true[:,i_smear_px])/gen_mass)
    dy = (K.square(y_pred[:,i_tt_t1n_py] - y_true[:,i_tt_t1n_py])/gen_mass) + (K.square(y_pred[:,i_tt_t2n_py] - y_true[:,i_tt_t2n_py])/gen_mass) + (K.square(y_pred[:,i_smear_py] - y_true[:,i_smear_py])/gen_mass)

    dmet_x = (K.square((y_pred[:,i_tt_t1n_px] + y_pred[:,i_tt_t2n_px] + y_pred[:,i_smear_px]) - y_true[:,i_smeared_met_px]) / gen_mass)
    dmet_y = (K.square((y_pred[:,i_tt_t1n_py] + y_pred[:,i_tt_t2n_py] + y_pred[:,i_smear_py]) - y_true[:,i_smeared_met_py]) / gen_mass)

    dm_tau_1 = K.square((K.square(y_true[:,i_tau1_e] + K.sqrt( K.square(y_pred[:,i_tt_t1n_px]) + K.square(y_pred[:,i_tt_t1n_py]) + K.square(y_pred[:,i_tt_t1n_pz]))) -
                     ( K.square(y_true[:,i_tau1_px] + y_pred[:,i_tt_t1n_px]) + K.square(y_true[:,i_tau1_py] + y_pred[:,i_tt_t1n_py]) + K.square(y_true[:,i_tau1_pz] + y_pred[:,i_tt_t1n_pz])) -
                       mtau_squared)/gen_mass)

    dm_tau_2 = K.square((K.square(y_true[:,i_tau2_e] + K.sqrt( K.square(y_pred[:,i_tt_t2n_px]) + K.square(y_pred[:,i_tt_t2n_py]) + K.square(y_pred[:,i_tt_t2n_pz]))) -
                     ( K.square(y_true[:,i_tau2_px] + y_pred[:,i_tt_t2n_px]) + K.square(y_true[:,i_tau2_py] + y_pred[:,i_tt_t2n_py]) + K.square(y_true[:,i_tau2_pz] + y_pred[:,i_tt_t2n_pz])) -
                       mtau_squared)/gen_mass)

    return K.mean(dm_tau_1 + dm_tau_2 + dx + dy + dmet_x + dmet_y)

i_lt_t1n_px = 13
i_lt_t1n_py = 14
i_lt_t1n_pz = 15
i_lt_l1n_px = 16
i_lt_l1n_py = 17
i_lt_l1n_pz = 18
i_lt_t2n_px = 19
i_lt_t2n_py = 20
i_lt_t2n_pz = 21

def loss_semi_leptonic(y_true, y_pred):
    gen_mass = y_true[:,i_gen_mass]

    dx = (K.square(y_pred[:,i_lt_t1n_px] - y_true[:,i_lt_t1n_px])/gen_mass) +
         (K.square(y_pred[:,i_lt_l1n_px] - y_true[:,i_lt_l1n_px])/gen_mass) +
         (K.square(y_pred[:,i_lt_t2n_px] - y_true[:,i_lt_t2n_px])/gen_mass) +
         (K.square(y_pred[:,i_smear_px] - y_true[:,i_smear_px])/gen_mass)

    dy = (K.square(y_pred[:,i_lt_t1n_py] - y_true[:,i_lt_t1n_py])/gen_mass) +
         (K.square(y_pred[:,i_lt_l1n_py] - y_true[:,i_lt_l1n_py])/gen_mass) +
         (K.square(y_pred[:,i_lt_t2n_py] - y_true[:,i_lt_t2n_py])/gen_mass) +
         (K.square(y_pred[:,i_smear_py] - y_true[:,i_smear_py])/gen_mass)

    dmet_x = (K.square((y_pred[:,i_lt_t1n_px] +
                        y_pred[:,i_lt_l1n_px] + 
                        y_pred[:,i_lt_t2n_px] +
                        y_pred[:,i_smear_px]) -
                        y_true[:,i_smeared_met_px]) / gen_mass)

    dmet_y = (K.square((y_pred[:,i_lt_t1n_py] +
                        y_pred[:,i_lt_l1n_py] +
                        y_pred[:,i_lt_t2n_py] +
                        y_pred[:,i_smear_py]) -
                        y_true[:,i_smeared_met_py]) / gen_mass)

    dm_tau_1 = K.square(
                        (K.square(y_true[:,i_tau1_e] + # tau_vis energy
                         K.sqrt( K.square(y_pred[:,i_lt_t1n_px]) + K.square(y_pred[:,i_lt_t1n_py]) + K.square(y_pred[:,i_lt_t1n_pz])) + # tau1 tau neutrino energy
                         K.sqrt( K.square(y_pred[:,i_lt_l1n_px]) + K.square(y_pred[:,i_lt_l1n_py]) + K.square(y_pred[:,i_lt_l1n_pz]))) - # tau1 lepton neutrino energy
                        (K.square(y_true[:,i_tau1_px] + y_pred[:,i_lt_t1n_px] + y_pred[:,i_lt_l1n_px]) + # tau plus neutrino momenta
                         K.square(y_true[:,i_tau1_py] + y_pred[:,i_lt_t1n_py] + y_pred[:,i_lt_l1n_py]) + 
                         K.square(y_true[:,i_tau1_pz] + y_pred[:,i_lt_t1n_pz] + y_pred[:,i_lt_l1n_pz]))- 
                         mtau_squared)/ # nominal tau mass 
                         gen_mass) # regularization



    dm_tau_2 = K.square(
                        (K.square(y_true[:,i_tau2_e] + # tau vis energy
                         K.sqrt( K.square(y_pred[:,i_lt_t2n_px]) + K.square(y_pred[:,i_lt_t2n_py]) + K.square(y_pred[:,i_lt_t2n_pz]))) - # tau2 tau neutrino energy
                       ( K.square(y_true[:,i_tau2_px] + y_pred[:,i_lt_t2n_px]) + # tau plus neutrino momenta
                         K.square(y_true[:,i_tau2_py] + y_pred[:,i_lt_t2n_py]) +
                         K.square(y_true[:,i_tau2_pz] + y_pred[:,i_lt_t2n_pz])) -
                         mtau_squared)/ # nominal tau mass
                         gen_mass) # regularization

    return K.mean(dm_tau_1 + dm_tau_2 + dx + dy + dmet_x + dmet_y)

def loss_fully_leptonic(y_true, y_pred):
    pass
