# -*- coding: utf-8 -*-
from os import environ

environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] ='2'
from root_numpy import root2array, array2tree, array2root, list_trees, list_directories
from ROOT import TTree, TFile
import numpy as np

from common_functions import load_model
from fourvector import FourMomentum
from common_functions import predict
import hashlib
import time
from shutil import copyfile
# load the input file

branches=[
        "m_1", "pt_1", "eta_1", "phi_1", 
        "m_2", "pt_2", "eta_2", "phi_2", 
        "met", "metphi",
        "metcov00", "metcov11", "metcov01"]

def calculate_arrays(args):
        input_file = args[0]
        treename = args[1]
        foldername = args[2]
        output_file = args[3]
        model_path = args[4]
        full_output = args[5]
        channel = args[6]
        mode = args[7]
        lock = args[8]
        try:
            arr = root2array(input_file, foldername+"/"+treename, branches = branches)
        except:
            # The input file has not been found or the tree size is 0
            return False

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
            metcovxy = a[12]

            X[index,:] = np.array([  tau_1.e, tau_1.px, tau_1.py, tau_1.pz,
                            tau_2.e, tau_2.px, tau_2.py, tau_2.pz,
                            met.px, met.py,
                            metcovxx, metcovyy, metcovxy ])

            visible = tau_1 + tau_2
            L[index,:] = visible.as_numpy_array()
        Y = predict(model_path, X, channel)
        # convert Y to usual hadron-collider coordinates

        from common_functions import full_fourvector, transform_fourvector

        fullvector_hc, fullvector_cartesian = full_fourvector(Y, L,
                                                       cartesian_types = [("e_N",np.float64),  ("px_N", np.float64),  ("py_N", np.float64),  ("pz_N", np.float64)],
                                                       hc_types =        [("pt_N",np.float64), ("eta_N", np.float64), ("phi_N", np.float64), ("m_N", np.float64)])

        outputs = [fullvector_hc, fullvector_cartesian]
#        if mode == 'copy':
#            outputs.append(arr_all)
#        if full_output:
#            for i in range(2):
#                neutrino_four_momenta = []
#                for line in range(Y.shape[0]):
#                    neutrino_four_momenta.append(FourMomentum(Y[line,13+4*i], Y[line,14+4*i], Y[line,15+4*i], Y[line,16+4*i], cartesian=True))
#                    s = "_n" + str(i+1) 
#                    neutrino_hc, neutrino_cartesian = transform_fourvector(neutrino_four_momenta,
#                                                           cartesian_types = [("e"+s,np.float64),  ("px"+s, np.float64),  ("py"+s, np.float64),  ("pz"+s, np.float64)],
#                                                           hc_types =        [("pt"+s,np.float64), ("eta"+s, np.float64), ("phi"+s, np.float64), ("m"+s, np.float64)])
#                    outputs.append(neutrino_hc)
#                    outputs.append(neutrino_cartesian)
        lock.acquire()
        print os.getpid(), ": lock hold by process creating", output_file, " lock: ", lock

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        if not os.path.exists(output_file):
            if mode=='copy':
                copyfile(input_file, output_file)
        f = TFile(output_file, "UPDATE")

        if not foldername in list_directories(output_file):
            f.mkdir(foldername)

        if mode =='friend':
            tree = None 
        elif mode == 'copy':
            tree = f.Get(foldername+"/"+treename)
        getattr(f, foldername).cd()
        for output in outputs:
            tree = array2tree(output, name = treename, tree = tree)
        f.Write()
        f.Close()
        lock.release()
        print os.getpid(), ": lock released by process "
        return True

def get_output_filename(input_file, output_folder):
    filename = os.path.basename(input_file)
    dirname = os.path.join(output_folder, os.path.dirname(input_file).split("/")[-1], filename)
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
    channels_models = data_loaded["models"]
    files = data_loaded["files"]
    full_output = data_loaded["full output"]
    output_folder = data_loaded["output_folder"]
    n_processes = data_loaded["n_processes"]
    mode = data_loaded["mode"]

    args = []
    locks = {}
    m = Manager()
    for index, f in enumerate(files):
        trees = list_trees(f)
        for tree in trees:
            if tree in channels_models:
                model_path = channels_models[tree][1]
                channel = channels_models[tree][0]
            else:
                continue
            foldername, treename = tree.split("/")
            output_filename = get_output_filename(f, output_folder)
            if not output_filename in locks:
                locks[output_filename] = m.Lock()
            args.append([f, treename, foldername, output_filename, model_path, full_output, channel, mode, locks[output_filename]])
    if len(args) == 0:
        raise RuntimeError("None of the specified input trees have been found in the input files.")
    pool = Pool(processes=n_processes)
    results = pool.map(calculate_arrays, args)
    pool.close()
    pool.join()
    if all(results):
        print "Nostradamass successfully applied on ", len(results), " trees"
    else:
        raise RuntimeError("Some threads finished with errors")

