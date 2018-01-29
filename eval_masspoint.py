import sys, os
from fourvector import *
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_scaler(model_folder):
    import pickle
    pkl_file = open(os.path.join(model_folder, 'scaler.pkl'), 'rb')
    scaler = pickle.load(pkl_file)
    scalerTarget = pickle.load(pkl_file)
    return scaler, scalerTarget

def scale_input(X, scaler):
    X = scaler.transform(X)
    return X
    

def load_model(model_folder):
    from keras.models import load_model
    from train_neutrino import mass_loss_start, custom_loss, mass_loss_final, mass_loss_custom, mass_loss_abs
    model = load_model(os.path.join(model_folder, 'toy_mass.h5'),  custom_objects={'mass_loss_start': mass_loss_start, 'custom_loss':custom_loss, 'mass_loss_final':mass_loss_final, 'mass_loss_custom' : mass_loss_custom , 'mass_loss_abs' : mass_loss_abs})
    return model

from train_invisibles import load_from_log, get_inverse, predict, transform_fourvector


def plot(scaled_Y, X, Y, B, M, L, mass):
    output_folder = "plots/m"+str(mass)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    for a in range(Y.shape[1]):
        print "print a", a
        pts = plt.figure()
        arange = None
        if a ==2:
            arange = [-3,3]
        n, bins, patches = plt.hist(scaled_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
        n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
        plt.savefig(output_folder+"/target-regressed"+str(a)+".png")
        plt.close()
   
    
    energy = np.sqrt(np.square(scaled_Y[:,0]) + np.square(scaled_Y[:,1]) +np.square(scaled_Y[:,2])) + np.sqrt(np.square(scaled_Y[:,3]) + np.square(scaled_Y[:,4]) +np.square(scaled_Y[:,5]))
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum( L[i,0] + energy[i],
                                                                                            L[i,1] + scaled_Y[i,0] + scaled_Y[i,3],
                                                                                            L[i,2] + scaled_Y[i,1] + scaled_Y[i,4],
                                                                                            L[i,3] + scaled_Y[i,2] + scaled_Y[i,5]) for i in range(L.shape[0])])
    
    regressed_met_pt = np.sqrt(np.square(scaled_Y[:,0] + scaled_Y[:,3]) + np.square(scaled_Y[:,1] + scaled_Y[:,4]))
    target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])

#    target_physfourvectors = np.array([target_physfourvectors[i] for i in range(regressed_physfourvectors.shape[0]) if regressed_physfourvectors[i][3] > 0])
#    regressed_physfourvectors = np.array([a for a in regressed_physfourvectors if a[3] > 0])
#    print "0 Entries: ", regressed_fourvectors.shape[0] - regressed_physfourvectors.shape[0]

 #   vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])

    
    for a in range(regressed_fourvectors[0].shape[0]):
        irange = None
        if a==0:
            irange = [-100,2100]
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 100, normed=1, color='red', alpha=1, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(target_fourvectors[:,a], 100, normed=1, color='green', alpha=1, range=irange, histtype='step', label='target')
 #       n, bins, patches = plt.hist(vis_fourvectors[:,a], 100, normed=1, color='orange', alpha=1, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig(output_folder+"/cartesian-target-regressed"+str(a)+".png")
    
    for a in range(regressed_physfourvectors.shape[1]):
        pts = plt.figure()
        irange = None
        if a==3:
            irange = [0,2*float(mass)]
        n, bins, patches = plt.hist(target_physfourvectors[:,a], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
 #       n, bins, patches = plt.hist(vis_physfourvectors[:,a], 100, normed=1, color='orange', alpha=0.5, range=irange, histtype='step', label='visible', linestyle='dotted')
        print "result, mean: ", a, np.mean(regressed_physfourvectors[:,a]), "median: ", np.median(regressed_physfourvectors[:,a]), " stddev " , np.std(regressed_physfourvectors[:,a], ddof=1)
        plt.legend()
        plt.savefig(output_folder+"/phys-target-regressed"+str(a)+".png")

    # neutrino energy
#    pts = plt.figure()
#    irange = None
#    n, bins, patches = plt.hist(M[:,3], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
#    n, bins, patches = plt.hist(pz, 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
#    plt.legend()
#    plt.savefig(output_folder+"/neutrino-pz.png")

#    pts = plt.figure()
#    irange = None
#    n, bins, patches = plt.hist(M[:,3], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
#    n, bins, patches = plt.hist(energy, 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
#    plt.legend()
#    plt.savefig(output_folder+"/neutrino-pz.png")
    
    diff_fourvectors = regressed_physfourvectors-target_physfourvectors
    print "diff: "
    for a in range(diff_fourvectors.shape[1]):
        irange = None
        if a==3:
            irange = [-250,250]
        pts = plt.figure()
        n, bins, patches = plt.hist(diff_fourvectors[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
        print "result, mean: ", a, np.mean(diff_fourvectors[:,a]), "median: ", np.median(diff_fourvectors[:,a]), " stddev " , np.std(diff_fourvectors[:,a], ddof=1)
        plt.savefig(output_folder+"/diffvector-target-regressed"+str(a)+".png")

if __name__ == '__main__':
    mass = sys.argv[1]
    folder = sys.argv[2]
    model_folder = sys.argv[3]
    X, Y, B, M, L, phys_M = load_from_log(folder + "/m"+mass+".log", "pickle.pkl", save_cache=False)
    #scaler, scalerTarget = get_scaler(model_folder)
    model = load_model(model_folder)
    regressed_Y = predict(model, X)
    plot(regressed_Y, X, Y, B, M, L, mass)
    # raw_X unscaled
    # raw_Y unscaled
    # X input for DNN
    # Y target for DNN
    # regressed_Y unscaled output
    # scaled_Y scaled output
