from root_numpy import root2array, tree2array
from root_numpy import testdata
from fourvector import *

import numpy as np
import csv
filenames = ["GluGluHToTauTauM125_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_powheg-pythia8",
            "SUSYGluGluToHToTauTauM100_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8",
            "SUSYGluGluToHToTauTauM200_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8",
            "SUSYGluGluToHToTauTauM300_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8",
            "SUSYGluGluToHToTauTauM400_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8",
            "SUSYGluGluToHToTauTauM500_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8",
            "SUSYGluGluToHToTauTauM600_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_pythia8",
            "VBFHToTauTauM125_RunIISummer16MiniAODv2_PUMoriond17_13TeV_MINIAOD_powheg-pythia8"]
folder = "/storage/b/friese/htautau/artus/2018-01-23_13-20_analysis/workdir/se_output/merged/"

new_filenames = ["ggHSM",
                "susy100",
                "susy200",
                "susy300",
                "susy400",
                "susy500",
                "susy600",
                "vbfSM"]

for filename, new_filename in zip(filenames, new_filenames):
    # Convert a TTree in a ROOT file into a NumPy structured array
    arr = root2array(folder + filename + "/" + filename + '.root', 'tt_nominal/ntuple', branches=[
            "m_sv", "pt_sv", "eta_sv", "phi_sv",
            "m_1", "pt_1", "eta_1", "phi_1",
            "m_2", "pt_2", "eta_2", "phi_2",
            "met", "metphi",
            "genBosonMass", "genBosonPt", "genBosonEta", "genBosonPhi",
            "genMetPt", "genMetPhi",
            "genMatchedLep1LV.fCoordinates.fM", "genMatchedLep1LV.fCoordinates.fPt", "genMatchedLep1LV.fCoordinates.fEta", "genMatchedLep1LV.fCoordinates.fPhi", 
            "genMatchedLep2LV.fCoordinates.fM", "genMatchedLep2LV.fCoordinates.fPt", "genMatchedLep2LV.fCoordinates.fEta", "genMatchedLep2LV.fCoordinates.fPhi",
            "metcov00", "metcov01", "metcov10", "metcov11"
            
])

    csvfile = open("data/"+new_filename+'.csv', 'wb')
    writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for a in arr:
        met = FourMomentum(0, a[12], 0, a[13], False)
        genmet = FourMomentum(0, a[18], 0, a[19], False)
        fake_met = FourMomentum(0, met.px - genmet.px, met.py - genmet.py, 0)
        writer.writerow([b for b in a] + [fake_met.px, fake_met.py])
    csvfile.close()

