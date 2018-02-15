from root_numpy import root2array, array2root
import numpy as np

from common_functions import load_model
from fourvector import FourMomentum

channel = 'tt'
full_output = True

def get_index(channel, n_neutrino):
    if channel == 'tt':
        if n_neutrino == 0:
            return "nt_1"
        else:
            return "nt_2"
    elif channel == "mt" or channel == "et":
        if n_neutrino == 0:
            return "nt_1"
        elif n_neutrino == 1:
            return "nl_1"
        else:
            return "nt_1"
        

n_neutrinos = 2

model_path = "/storage/b/friese/trainings/neutrinos_57_medimnet5x100/model.0.298-7.44.hdf5"
input_file = "/storage/b/friese/htautau/artus/2018-01-23_13-20_analysis/workdir/se_output/merged/SUSYGluGluToHToTauTauM300_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8/SUSYGluGluToHToTauTauM300_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8.root"
treename = "tt_nominal/ntuple"

# load the input file

branches=[
        "m_1", "pt_1", "eta_1", "phi_1", 
        "m_2", "pt_2", "eta_2", "phi_2", 
        "met", "metphi",
        "metcov00", "metcov11"]

arr = root2array(input_file, treename, branches = branches)

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
# convert Y in usual hadron-collider coordinates

from common_functions import full_fourvector, transform_fourvector

fullvector_hc, fullvector_cartesian = full_fourvector(Y, L, vlen=3*n_neutrinos,
                                               cartesian_types = [("e_nn",np.float64),  ("px_nn", np.float64),  ("py_nn", np.float64),  ("pz_nn", np.float64)],
                                               hc_types =        [("pt_nn",np.float64), ("eta_nn", np.float64), ("phi_nn", np.float64), ("m_nn", np.float64)])
# save output
array2root(fullvector_hc, filename='m_nn.root')
array2root(fullvector_cartesian, filename='m_nn.root')

if full_output:
    for n_neutrino in range(n_neutrinos):
        four_momenta = []
        for line in range(Y.shape[0]):
            four_momenta.append(FourMomentum(None, Y[line,0+n_neutrino], Y[line,1+n_neutrino], Y[line,2+n_neutrino], cartesian=True, massless=True))
        s = "_" + get_index(channel, n_neutrino)
        neutrino_hc, neutrino_cartesian = transform_fourvector(four_momenta,
                                                   cartesian_types = [("e"+s,np.float64),  ("px"+s, np.float64),  ("py"+s, np.float64),  ("pz"+s, np.float64)],
                                                   hc_types =        [("pt"+s,np.float64), ("eta"+s, np.float64), ("phi"+s, np.float64), ("m"+s, np.float64)])
        array2root(neutrino_hc, filename='m_nn.root')
        array2root(neutrino_cartesian, filename='m_nn.root')
