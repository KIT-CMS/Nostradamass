import numpy as np
import keras.backend as K
mtau_squared = np.square(np.float64(1.776))
    # Y: 0-5 : Neutrino 1/2 x, y, z
    # Y: 6 : gen Mass

    # Y: 7/8: Smear x/y
    # Y: 9/10: smeared met???
    # Y: 11: pt
    # Y: 12-15: 4-vector visible
    # Y: 16-19: 4-vector tau1
    # Y: 20-23: 4-vector tau2
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
