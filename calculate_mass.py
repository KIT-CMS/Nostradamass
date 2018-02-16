# -*- coding: utf-8 -*-
from root_numpy import root2array, array2tree, array2root
from ROOT import TTree, TFile
import numpy as np

from common_functions import load_model
from fourvector import FourMomentum
from common_functions import get_index
import hashlib

# load the input file

branches=[
        "m_1", "pt_1", "eta_1", "phi_1", 
        "m_2", "pt_2", "eta_2", "phi_2", 
        "met", "metphi",
        "metcov00", "metcov11"]

def calculate_arrays(l, args):
        input_file = args[0]
        treename = args[1]
        foldername = args[2]
        output_file = args[3]
        model_path = args[4]
        full_output = args[5]

        arr = root2array(input_file, foldername+"/"+treename, branches = branches)

        # pre-allocate the input vector to Keras

        X = np.zeros([arr.shape[0], len(branches)])
        L = np.zeros([arr.shape[0], 4])
        model = load_model(model_path)

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
        Y = model.predict(X)
        # convert Y to usual hadron-collider coordinates

        from common_functions import full_fourvector, transform_fourvector

        fullvector_hc, fullvector_cartesian = full_fourvector(Y, L, vlen=3*n_neutrinos,
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

        f = TFile(output_file, "RECREATE")
        f.mkdir(foldername)
        getattr(f, foldername).cd()
        tree = None
        for output in outputs:
            tree = array2tree(output, name = treename, tree = tree)
        f.Write()
        f.Close()
        l.release()

from multiprocessing import Pool, Manager
from functools import partial
import os
def get_output_filename(input_file):
    return os.path.join(os.path.dirname(input_file), os.path.basename(input_file).replace(".root", "-m_nn.root"))

if __name__ == '__main__':

import yaml
import io

# Define data
data = {'a list': [1, 42, 3.141, 1337, 'help', u'€'],
        'a string': 'bla',
        'another dict': {'foo': 'bar',
                         'key': 'value',
                         'the answer': 42}}

# Write YAML file
with io.open('data.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# Read YAML file
with open("data.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

print(data == data_loaded)

    # parser
    channel = 'tt'
    full_output = True

    n_neutrinos = 2

    model_path = "/storage/b/friese/trainings/neutrinos_57_medimnet5x100/model.0.298-7.44.hdf5"
    input_file = "/storage/b/friese/htautau/artus/2018-01-23_13-20_analysis/workdir/se_output/merged/SUSYGluGluToHToTauTauM300_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8/SUSYGluGluToHToTauTauM300_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8.root"
    treename = "ntuple"
    foldername = "tt_nominal"
    args = []
    args.append([input_file, treename, foldername, get_output_filename(input_file), model_path, full_output])


    # do the actual work in the calculate_arrays functions and write the results to files that can be friended with the input trees
    # todo: do not write friend trees but modify the original ones with the new entries
    pool = Pool()
    m = Manager()
    l = m.Lock()
    func = partial(calculate_arrays, l)
    pool.map(func, args)
