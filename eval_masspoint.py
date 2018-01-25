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

from train_neutrino import load_from_log, get_inverse, predict, transform_fourvector


def plot(scaled_Y, regressed_Y, raw_Y, X, Y, B, M, L, mass):
    output_folder = "plots/m"+str(mass)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for a in range(Y.shape[1]):
        pts = plt.figure()
        arange = None
        n, bins, patches = plt.hist(regressed_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
        n, bins, patches = plt.hist(Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
        plt.savefig(output_folder+"/transform-target-regressed"+str(a)+".png")
        plt.close()
    
    
    for a in range(Y.shape[1]):
        print "print a", a
        pts = plt.figure()
        arange = None
        if a ==2:
            arange = [-3,3]
        n, bins, patches = plt.hist(scaled_Y[:,a], 150, normed=1, facecolor='red', alpha=0.75, range = arange)
        n, bins, patches = plt.hist(raw_Y[:,a], 150, normed=1, facecolor='green', alpha=0.75, range = arange)
        plt.savefig(output_folder+"/target-regressed"+str(a)+".png")
        plt.close()
    
 #   for a in range(Y.shape[1]):
 #       pts = plt.figure()
 #       arange = [-3,3]
 #       n, bins, patches = plt.hist(target_phys[:,a] - regressed_phys[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range = arange)
  #      print "mean ", a, np.mean(target_phys[:,a] - regressed_phys[:,a]), " stddev " , np.std(target_phys[:,a] - regressed_phys[:,a])
 #       plt.savefig("diff-target-regressed"+str(a)+".png")
#        plt.close()
    
    # neutrino 4-vectors
    #y = ns / (l1 + l2 + ns)
    #y(l1+l2+ns) = ns
    #y(l1+l2) = ns-yns = ns(1-y)
    #y(l1+l2) / (1-y) = ns
    energy = np.multiply(scaled_Y[:,0], L[:,0])
    ones = np.ones([scaled_Y.shape[0]])
    F = np.subtract( ones, scaled_Y[:,0])
    energy /= F

    pz = np.sinh(scaled_Y[:,1]) * np.sqrt( np.square(M[:,1]) + np.square(M[:,2]))
    
    regressed_physfourvectors, regressed_fourvectors = transform_fourvector([ FourMomentum(energy[i]+L[i,0], L[i,1]+M[i,1], L[i,2]+M[i,2], L[i,3]+pz[i]) for i in range(L.shape[0])])
    # clean 0 mass entries
    target_physfourvectors, target_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in B])

    target_physfourvectors = np.array([target_physfourvectors[i] for i in range(regressed_physfourvectors.shape[0]) if regressed_physfourvectors[i][3] > 0])
    regressed_physfourvectors = np.array([a for a in regressed_physfourvectors if a[3] > 0])
    print "0 Entries: ", regressed_fourvectors.shape[0] - regressed_physfourvectors.shape[0]

    vis_physfourvectors, vis_fourvectors = transform_fourvector([ FourMomentum(a[0], a[1], a[2], a[3]) for a in L])

    
    for a in range(regressed_fourvectors[0].shape[0]):
        irange = None
        if a==0:
            irange = [-100,2100]
        pts = plt.figure()
        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 100, normed=1, color='red', alpha=1, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(target_fourvectors[:,a], 100, normed=1, color='green', alpha=1, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(vis_fourvectors[:,a], 100, normed=1, color='orange', alpha=1, range=irange, histtype='step', label='visible', linestyle='dotted')
        plt.legend()
        plt.savefig(output_folder+"/cartesian-target-regressed"+str(a)+".png")
    
    for a in range(regressed_physfourvectors.shape[1]):
        pts = plt.figure()
        irange = None
        if a==3:
            irange = [0,2*float(mass)]
        n, bins, patches = plt.hist(target_physfourvectors[:,a], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
        n, bins, patches = plt.hist(vis_physfourvectors[:,a], 100, normed=1, color='orange', alpha=0.5, range=irange, histtype='step', label='visible', linestyle='dotted')
        print "result, mean: ", a, np.mean(regressed_physfourvectors[:,a]), "median: ", np.median(regressed_physfourvectors[:,a]), " stddev " , np.std(regressed_physfourvectors[:,a], ddof=1)
        plt.legend()
        plt.savefig(output_folder+"/phys-target-regressed"+str(a)+".png")

    # neutrino energy
    pts = plt.figure()
    irange = None
    n, bins, patches = plt.hist(M[:,3], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
    n, bins, patches = plt.hist(pz, 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
    plt.legend()
    plt.savefig(output_folder+"/neutrino-pz.png")

    pts = plt.figure()
    irange = None
    n, bins, patches = plt.hist(M[:,3], 100, normed=1, color='green', alpha=0.75, range=irange, histtype='step', label='target')
    n, bins, patches = plt.hist(energy, 100, normed=1, color='red', alpha=0.75, range=irange, histtype='step', label='regressed')
    plt.legend()
    plt.savefig(output_folder+"/neutrino-pz.png")
    
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
    raw_X, raw_Y, B, M, L = load_from_log(folder + "/m"+mass+".log", "pickle.pkl", save_cache=False)
    #scaler, scalerTarget = get_scaler(model_folder)
    X = raw_X#scale_input(raw_X, scaler)
    Y = raw_Y#scale_input(raw_Y, scalerTarget)
    model = load_model(model_folder)
    regressed_Y = predict(model, X)
    scaled_Y = regressed_Y#get_inverse(regressed_Y, scaler_target_filename = os.path.join(model_folder, 'scaler.pkl') )
    plot(scaled_Y, regressed_Y, raw_Y, X, Y, B, M, L, mass)
    # raw_X unscaled
    # raw_Y unscaled
    # X input for DNN
    # Y target for DNN
    # regressed_Y unscaled output
    # scaled_Y scaled output
