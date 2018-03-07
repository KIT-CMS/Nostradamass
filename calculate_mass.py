# -*- coding: utf-8 -*-
from root_numpy import root2array, array2tree, array2root, list_trees
from ROOT import TTree, TFile
import numpy as np

from common_functions import load_model
from fourvector import FourMomentum
from common_functions import get_index
from common_functions import predict
import hashlib
import time

# load the input file
channel = 'tt'

n_neutrinos = 2

branches=[
        "m_1", "pt_1", "eta_1", "phi_1", 
        "m_2", "pt_2", "eta_2", "phi_2", 
        "met", "metphi",
        "metcov00", "metcov11"]

def calculate_arrays(l, args):
        starttime = time.time()
        input_file = args[0]
        treename = args[1]
        foldername = args[2]
        output_file = args[3]
        model_path = args[4]
        full_output = args[5]
        print os.getpid(), " file", os.path.basename(output_file)

        arr = root2array(input_file, foldername+"/"+treename, branches = branches)

        # pre-allocate the input vector to Keras

        X = np.zeros([arr.shape[0], len(branches)])
        L = np.zeros([arr.shape[0], 4])

        # convert inputs to cartesian coordinates
        for index, a in enumerate(arr):
            tau_1 = FourMomentum(a[0], a[1], a[2], a[3], False)
            tau_2 = FourMomentum(a[4], a[5], a[6], a[7], False)
            met   = FourMomentum(0, a[8], 0, a[9], False)
            metcovxx = np.sqrt(a[10])
            metcovyy = np.sqrt(a[11])

            X[index,:] = np.array([  tau_1.e, tau_1.px, tau_1.py, tau_1.pz,
                            tau_2.e, tau_2.px, tau_2.py, tau_2.pz,
                            met.px, met.py,
                            metcovxx, metcovyy ])

            visible = tau_1 + tau_2
            L[index,:] = visible.as_numpy_array()
    	Y = predict(model_path, X, channel)
        # convert Y to usual hadron-collider coordinates

        from common_functions import full_fourvector, transform_fourvector

        fullvector_hc, fullvector_cartesian = full_fourvector(Y, L,
                                                       cartesian_types = [("e_nn",np.float64),  ("px_nn", np.float64),  ("py_nn", np.float64),  ("pz_nn", np.float64)],
                                                       hc_types =        [("pt_nn",np.float64), ("eta_nn", np.float64), ("phi_nn", np.float64), ("m_nn", np.float64)])

        outputs = [fullvector_hc, fullvector_cartesian]
        if full_output:
            for n_neutrino in range(n_neutrinos):
                four_momenta = []
                for line in range(Y.shape[0]):
                    four_momenta.append(FourMomentum(None, Y[line,0+n_neutrino], Y[line,1+n_neutrino], Y[line,2+n_neutrino], cartesian=True, massless=True))
                s = "_" + get_index(channel, n_neutrino)
                neutrino_hc, neutrino_cartesian = transform_fourvector(four_momenta,
                                                           cartesian_types = [("e"+s,np.float64),  ("px"+s, np.float64),  ("py"+s, np.float64),  ("pz"+s, np.float64)],
                                                           hc_types =        [("pt"+s,np.float64), ("eta"+s, np.float64), ("phi"+s, np.float64), ("m"+s, np.float64)])
                outputs.append(neutrino_hc)
                outputs.append(neutrino_cartesian)
        l.acquire()
        print os.getpid(), ":lock hold by process creating", output_file

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        f = TFile(output_file, "RECREATE")
        f.mkdir(foldername)
        getattr(f, foldername).cd()
        tree = None
        for output in outputs:
            tree = array2tree(output, name = treename, tree = tree)
        f.Write()
        f.Close()
        l.release()
        runtime = time.time() - starttime
        t = open(str('logs/'+str(os.getpid())), 'a')
        t.write(str(runtime)+"; " + str(arr.shape[0])+'\n')
        t.close()
        print os.getpid(), ": lock released by process "

def get_output_filename(input_file):
    #return os.path.join(os.path.dirname(input_file), os.path.basename(input_file).replace(".root", "-m_nn.root"))
    filename = os.path.basename(input_file)
    dirname = os.path.join("/storage/b/friese/m_nn/Artus_2017-12-02/all_1/", os.path.dirname(input_file).split("/")[-1], filename)
    return dirname 

from multiprocessing import Pool, Manager
from functools import partial
import os, io, sys, yaml
if __name__ == '__main__':
    # first argument: config file
    config_file = sys.argv[1]
    print config_file

    # Read YAML file
    with open(config_file, 'r') as stream:
        data_loaded = yaml.load(stream)

    import pprint
    models = data_loaded["models"]
    files = data_loaded["files"]
    full_output = data_loaded["full output"]

    args = []
    for f in files:
        trees = list_trees(f)
        for tree in trees:
            if tree in models:
                model_path = models[tree]
            else:
                continue
            foldername, treename = tree.split("/")
            output_filename = get_output_filename(f)
            args.append([f, treename, foldername, output_filename, model_path, full_output])
    pprint.pprint(args)
#    sys.exit()
    # todo: do not write friend trees but modify the original ones with the new entries
    pool = Pool(processes=3)
    m = Manager()
    l = m.Lock()
    func = partial(calculate_arrays, l)
    pool.map(func, args)
