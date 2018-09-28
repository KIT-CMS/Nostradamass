import ROOT as r
import numpy as np
import glob
from root_numpy import tree2array
import os
import copy


r.gROOT.SetBatch()

# For the canvas:
r.gStyle.SetCanvasBorderMode(0)
r.gStyle.SetCanvasColor(r.kWhite)
r.gStyle.SetCanvasDefH(600)  # Height of canvas
r.gStyle.SetCanvasDefW(600)  # Width of canvas
r.gStyle.SetCanvasDefX(0)    # POsition on screen
r.gStyle.SetCanvasDefY(0)

# For the Pad:
r.gStyle.SetPadBorderMode(0)
r.gStyle.SetPadColor(r.kWhite)
r.gStyle.SetPadGridX(False)
r.gStyle.SetPadGridY(False)
r.gStyle.SetGridColor(0)
r.gStyle.SetGridStyle(3)
r.gStyle.SetGridWidth(1)

# For the frame:
r.gStyle.SetFrameBorderMode(0)
r.gStyle.SetFrameBorderSize(1)
r.gStyle.SetFrameFillColor(0)
r.gStyle.SetFrameFillStyle(0)
r.gStyle.SetFrameLineColor(1)
r.gStyle.SetFrameLineStyle(1)
r.gStyle.SetFrameLineWidth(1)

# For the histo:
r.gStyle.SetHistLineColor(2)
r.gStyle.SetHistLineStyle(0)
r.gStyle.SetHistLineWidth(3)
r.gStyle.SetEndErrorSize(2)
r.gStyle.SetMarkerStyle(20)

# For the fit/function:
r.gStyle.SetOptFit(1)
r.gStyle.SetFitFormat('5.4g')
r.gStyle.SetFuncColor(2)
r.gStyle.SetFuncStyle(1)
r.gStyle.SetFuncWidth(1)

# For the date:
r.gStyle.SetOptDate(0)
# For the statistics box:
r.gStyle.SetOptFile(0)
r.gStyle.SetOptStat(0)
# To display the mean and r.S:   SetOptStat('mr')
r.gStyle.SetStatColor(r.kWhite)
r.gStyle.SetStatFont(42)
r.gStyle.SetStatFontSize(0.025)
r.gStyle.SetStatTextColor(1)
r.gStyle.SetStatFormat('6.4g')
r.gStyle.SetStatBorderSize(1)
r.gStyle.SetStatH(0.1)
r.gStyle.SetStatW(0.15)

# Margins:
r.gStyle.SetPadTopMargin(0.05)
r.gStyle.SetPadBottomMargin(0.13)
r.gStyle.SetPadLeftMargin(0.16)
r.gStyle.SetPadRightMargin(0.07)

# For the Global title:
r.gStyle.SetOptTitle(0)
r.gStyle.SetTitleFont(42)
r.gStyle.SetTitleColor(1)
r.gStyle.SetTitleTextColor(1)
r.gStyle.SetTitleFillColor(10)
r.gStyle.SetTitleFontSize(0.05)

# For the axis titles:
r.gStyle.SetTitleColor(1, 'XYZ')
r.gStyle.SetTitleFont(42, 'XYZ')
r.gStyle.SetTitleSize(0.06, 'XYZ')
r.gStyle.SetTitleXOffset(0.9)
r.gStyle.SetTitleYOffset(1.38)

# For the axis labels:
r.gStyle.SetLabelColor(1, 'XYZ')
r.gStyle.SetLabelFont(42, 'XYZ')
r.gStyle.SetLabelOffset(0.007, 'XYZ')
r.gStyle.SetLabelSize(0.04, 'XYZ')

# For the axis:
r.gStyle.SetAxisColor(1, 'XYZ')
r.gStyle.SetStripDecimals(True)
r.gStyle.SetTickLength(0.03, 'XYZ')
r.gStyle.SetNdivisions(5, 'XYZ')
r.gStyle.SetPadTickX(1)
r.gStyle.SetPadTickY(1)

# Change for log plots:
r.gStyle.SetOptLogx(0)
r.gStyle.SetOptLogy(0)
r.gStyle.SetOptLogz(0)

# Postscript options:
r.gStyle.SetPaperSize(20., 20.)
r.gStyle.SetHatchesLineWidth(5)
r.gStyle.SetHatchesSpacing(0.05)


data_list = glob.glob("/storage/b/akhmet/merged_files_from_naf/01_09_2018_correctedZptWeights_newBTaggingEffs_postsync/*Run2017*/*.root")
channel_dict = {"SingleMuon": "mt_nominal", "SingleElectron": "et_nominal", "Tau": "tt_nominal", "MuonEG": "em_nominal"}

array_dict_lowpu = {
    "mt_nominal" : np.empty(shape=[0,2]),
    "et_nominal" : np.empty(shape=[0,2]),
    "tt_nominal" : np.empty(shape=[0,2]),
    "em_nominal" : np.empty(shape=[0,2]),
}

array_dict_highpu= {
    "mt_nominal" : np.empty(shape=[0,2]),
    "et_nominal" : np.empty(shape=[0,2]),
    "tt_nominal" : np.empty(shape=[0,2]),
    "em_nominal" : np.empty(shape=[0,2]),
}

c = r.TCanvas()

for f in sorted(data_list):
    for d,ch in channel_dict.items():
        if not d in f:
            continue
        F = r.TFile(f,"read")
        t = F.Get(ch).Get("ntuple")
        c.cd()
        hcov00 = r.TH1D("hcov00","hcov00", 100, 0.0, 2000.0)
        hcov00.GetXaxis().SetTitle("MET #sigma_{x}^{2} (GeV^{2})")
        hcov00.GetYaxis().SetTitle("arb. units")
        hcov00.SetLineColor(r.kRed+2)
        hcov11 = r.TH1D("hcov11","hcov11", 100, 0.0, 2000.0)
        hcov11.GetXaxis().SetTitle("MET #sigma_{y}^{2} (GeV^{2})")
        hcov11.GetYaxis().SetTitle("arb. units")
        hcov11.SetLineColor(r.kOrange+2)
        hcov01 = r.TH1D("hcov01","hcov01", 100, -400.0, 400.0)
        hcov01.GetXaxis().SetTitle("MET cov_{xy} (GeV^{2})")
        hcov01.GetYaxis().SetTitle("arb. units")
        hcov01.SetLineColor(r.kBlue+2)

        t.Draw("metcov00>>hcov00","metcov00 <= 2000.0","HIST")
        c.SaveAs("_".join([f,"cov00"])+".pdf")
        c.SaveAs("_".join([f,"cov00"])+".png")
        t.Draw("metcov11>>hcov11","metcov11 <= 2000.0","HIST")
        c.SaveAs("_".join([f,"cov11"])+".pdf")
        c.SaveAs("_".join([f,"cov11"])+".png")
        t.Draw("metcov01>>hcov01","metcov01 <= 400.0 && metcov01 >= -400.0","HIST")
        c.SaveAs("_".join([f,"cov01"])+".png")
        c.SaveAs("_".join([f,"cov01"])+".pdf")

        array = tree2array(t, branches=["sqrt(metcov00)","1.0*metcov01"])
        array = array.view(np.float64).reshape(array.shape + (-1,))

        if "Run2017B" in f or "Run2017C" in f or "Run2017D" in f:
            array_dict_lowpu[ch] = np.concatenate((array_dict_lowpu[ch], array), axis=0)
        elif "Run2017E" in f or "Run2017F" in f:
            array_dict_highpu[ch] = np.concatenate((array_dict_highpu[ch], array), axis=0)
        else:
            print "----------------NO MATCH----------",f
        print os.path.basename(f).replace(".root",""),ch,"[mean sqrt(metcov00), std sqrt(metcov00), std  metcov01] =",np.mean(array,axis=0)[0],np.std(array,axis=0)[0],np.std(array,axis=0)[1]


print "Low PU runs B,C,D"
for ch, a in array_dict_lowpu.items():
    print "\t",ch,"mean cov:",np.mean(a,axis=0)[0],"std cov:",np.std(a,axis=0)[0],"std corr:",np.std(a,axis=0)[1]
print "High PU runs E,F"
for ch, a in array_dict_highpu.items():
    print "\t",ch,"mean cov:",np.mean(a,axis=0)[0],"std cov:",np.std(a,axis=0)[0],"std corr:",np.std(a,axis=0)[1]
