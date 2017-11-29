from root_numpy import root2array, tree2array
from root_numpy import testdata
from fourvector import *
import csv
filenames = ['ggH', 'DY']

for filename in filenames:
    # Convert a TTree in a ROOT file into a NumPy structured array
    arr = root2array("data/"+filename + '.root', 'mt_jecUncNom_tauEsNom/ntuple', branches=[
            "m_sv", "pt_sv", "eta_sv", "phi_sv",
            "m_1", "pt_1", "eta_1", "phi_1",
            "m_2", "pt_2", "eta_2", "phi_2",
            "met", "metphi"])
    print arr

    csvfile = open("data/"+filename+'.csv', 'wb')
    writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for a in arr:
        svfit = FourMomentum(a[0], a[1], a[2], a[3], False)
        lepton1 = FourMomentum(a[4], a[5], a[6], a[7], False)
        lepton2 = FourMomentum(a[8], a[9], a[10], a[11], False)
        met = FourMomentum(0, a[12], 0, a[13], False)

        writer.writerow([lepton1.e, lepton1.px, lepton1.py, lepton1.pz, lepton2.e, lepton2.px, lepton2.py, lepton2.pz, met.px, met.py, met.pt2(), svfit.pt, svfit.eta, svfit.phi, svfit.m()])
    csvfile.close()

