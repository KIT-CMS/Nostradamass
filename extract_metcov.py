import ROOT as r
import numpy as np
import glob
from root_numpy import tree2array
import os
import copy


def DrawTitle(pad, text, align):
    pad_backup = r.gPad
    pad.cd()
    t = pad.GetTopMargin()
    l = pad.GetLeftMargin()
    R = pad.GetRightMargin()

    pad_ratio = (float(pad.GetWh()) * pad.GetAbsHNDC()) / \
        (float(pad.GetWw()) * pad.GetAbsWNDC())
    if pad_ratio < 1.:
        pad_ratio = 1.

    textSize = 0.6
    textOffset = 0.2

    latex = r.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(r.kBlack)
    latex.SetTextFont(42)
    latex.SetTextSize(textSize * t * pad_ratio)

    y_off = 1 - t + textOffset * t
    if align == 1:
        latex.SetTextAlign(11)
    if align == 1:
        latex.DrawLatex(l, y_off, text)
    if align == 2:
        latex.SetTextAlign(21)
    if align == 2:
        latex.DrawLatex(l + (1 - l - R) * 0.5, y_off, text)
    if align == 3:
        latex.SetTextAlign(31)
    if align == 3:
        latex.DrawLatex(1 - R, y_off, text)
    pad_backup.cd()

def DrawCMSLogo(pad, cmsText, extraText, iPosX, relPosX, relPosY, relExtraDY, extraText2='', cmsTextSize=1.2):
    """Blah
    
    Args:
        pad (TYPE): Description
        cmsText (TYPE): Description
        extraText (TYPE): Description
        iPosX (TYPE): Description
        relPosX (TYPE): Description
        relPosY (TYPE): Description
        relExtraDY (TYPE): Description
        extraText2 (str): Description
        cmsTextSize (float): Description
    
    Returns:
        TYPE: Description
    """
    pad.cd()
    cmsTextFont = 62  # default is helvetic-bold

    writeExtraText = len(extraText) > 0
    writeExtraText2 = len(extraText2) > 0
    extraTextFont = 52

    # text sizes and text offsets with respect to the top frame
    # in unit of the top margin size
    lumiTextOffset = 0.2
    # cmsTextSize = 0.8
    # float cmsTextOffset    = 0.1;  // only used in outOfFrame version

    # ratio of 'CMS' and extra text size
    extraOverCmsTextSize = 0.76

    outOfFrame = False
    if iPosX / 10 == 0:
        outOfFrame = True

    alignY_ = 3
    alignX_ = 2
    if (iPosX / 10 == 0):
        alignX_ = 1
    if (iPosX == 0):
        alignX_ = 1
    if (iPosX == 0):
        alignY_ = 1
    if (iPosX / 10 == 1):
        alignX_ = 1
    if (iPosX / 10 == 2):
        alignX_ = 2
    if (iPosX / 10 == 3):
        alignX_ = 3
    # if (iPosX == 0): relPosX = 0.14
    align_ = 10 * alignX_ + alignY_

    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    R = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = r.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(r.kBlack)

    extraTextSize = extraOverCmsTextSize * cmsTextSize
    pad_ratio = (float(pad.GetWh()) * pad.GetAbsHNDC()) / \
        (float(pad.GetWw()) * pad.GetAbsWNDC())
    if (pad_ratio < 1.):
        pad_ratio = 1.

    if outOfFrame:
        latex.SetTextFont(cmsTextFont)
        latex.SetTextAlign(11)
        latex.SetTextSize(cmsTextSize * t * pad_ratio)
        latex.DrawLatex(l, 1 - t + lumiTextOffset * t, cmsText)

    posX_ = 0
    if iPosX % 10 <= 1:
        posX_ = l + relPosX * (1 - l - R)
    elif (iPosX % 10 == 2):
        posX_ = l + 0.5 * (1 - l - R)
    elif (iPosX % 10 == 3):
        posX_ = 1 - r - relPosX * (1 - l - R)

    posY_ = 1 - t - relPosY * (1 - t - b)
    if not outOfFrame:
        latex.SetTextFont(cmsTextFont)
        latex.SetTextSize(cmsTextSize * t * pad_ratio)
        latex.SetTextAlign(align_)
        latex.DrawLatex(posX_, posY_, cmsText)
        if writeExtraText:
            latex.SetTextFont(extraTextFont)
            latex.SetTextAlign(align_)
            latex.SetTextSize(extraTextSize * t * pad_ratio)
            latex.DrawLatex(
                posX_, posY_ - relExtraDY * cmsTextSize * t, extraText)
            if writeExtraText2:
                latex.DrawLatex(
                    posX_, posY_ - 1.8 * relExtraDY * cmsTextSize * t, extraText2)
    elif writeExtraText:
        if iPosX == 0:
            posX_ = l + relPosX * (1 - l - R)
            posY_ = 1 - t + lumiTextOffset * t
        latex.SetTextFont(extraTextFont)
        latex.SetTextSize(extraTextSize * t * pad_ratio)
        latex.SetTextAlign(align_)
        latex.DrawLatex(posX_, posY_, extraText)
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


cov00_hists = {
    "SingleMuon" : r.TH1D("hcov00_SingleMuon","hcov00_SingleMuon", 100, 0.0, 2000.0),
    "SingleElectron" : r.TH1D("hcov00_SingleElectron","hcov00_SingleElectron", 100, 0.0, 2000.0),
    "Tau" : r.TH1D("hcov00_Tau","hcov00_Tau", 100, 0.0, 2000.0),
    "MuonEG" : r.TH1D("hcov00_MuonEG","hcov00_MuonEG", 100, 0.0, 2000.0),
}

cov11_hists = {
    "SingleMuon" : r.TH1D("hcov11_SingleMuon","hcov11_SingleMuon", 100, 0.0, 2000.0),
    "SingleElectron" : r.TH1D("hcov11_SingleElectron","hcov11_SingleElectron", 100, 0.0, 2000.0),
    "Tau" : r.TH1D("hcov11_Tau","hcov11_Tau", 100, 0.0, 2000.0),
    "MuonEG" : r.TH1D("hcov11_MuonEG","hcov11_MuonEG", 100, 0.0, 2000.0),
}

cov01_hists = {
    "SingleMuon" : r.TH1D("hcov01_SingleMuon","hcov01_SingleMuon", 100, -400.0, 400.0),
    "SingleElectron" : r.TH1D("hcov01_SingleElectron","hcov01_SingleElectron", 100, -400.0, 400.0),
    "Tau" : r.TH1D("hcov01_Tau","hcov01_Tau", 100, -400.0, 400.0),
    "MuonEG" : r.TH1D("hcov01_MuonEG","hcov01_MuonEG", 100, -400.0, 400.0),
}

std00_hists = {
    "SingleMuon" : r.TH1D("hstd00_SingleMuon","hstd00_SingleMuon", 100, 0.0, 2000.0),
    "SingleElectron" : r.TH1D("hstd00_SingleElectron","hstd00_SingleElectron", 100, 0.0, 2000.0),
    "Tau" : r.TH1D("hstd00_Tau","hstd00_Tau", 100, 0.0, 2000.0),
    "MuonEG" : r.TH1D("hstd00_MuonEG","hstd00_MuonEG", 100, 0.0, 2000.0),
}

std11_hists = {
    "SingleMuon" : r.TH1D("hstd11_SingleMuon","hstd11_SingleMuon", 100, 0.0, 2000.0),
    "SingleElectron" : r.TH1D("hstd11_SingleElectron","hstd11_SingleElectron", 100, 0.0, 2000.0),
    "Tau" : r.TH1D("hstd11_Tau","hstd11_Tau", 100, 0.0, 2000.0),
    "MuonEG" : r.TH1D("hstd11_MuonEG","hstd11_MuonEG", 100, 0.0, 2000.0),
}

histlist = []

for f in sorted(data_list):
    for d,ch in channel_dict.items():
        if not d in f:
            continue
        F = r.TFile(f,"read")
        t = F.Get(ch).Get("ntuple")
        primary_dataset =  os.path.basename(f).replace(".root","").split("_")[0]
        era  =  os.path.basename(f).replace(".root","").split("_")[1]

        cov00name = "_".join(["hcov00",primary_dataset,era])
        cov11name = "_".join(["hcov11",primary_dataset,era])
        cov01name = "_".join(["hcov01",primary_dataset,era])
        std00name = "_".join(["hstd00",primary_dataset,era])
        std11name = "_".join(["hstd11",primary_dataset,era])

        c.cd()
        hcov00 = r.TH1D(cov00name,cov00name, 100, 0.0, 2000.0)
        hcov00.GetXaxis().SetTitle("MET #sigma_{x}^{2} (GeV^{2})")
        hcov00.GetYaxis().SetTitle("arb. units")
        hcov00.SetLineColor(r.kRed+2)
        
        hcov11 = r.TH1D(cov11name,cov11name, 100, 0.0, 2000.0)
        hcov11.GetXaxis().SetTitle("MET #sigma_{y}^{2} (GeV^{2})")
        hcov11.GetYaxis().SetTitle("arb. units")
        hcov11.SetLineColor(r.kOrange+2)

        hcov01 = r.TH1D(cov01name,cov01name, 100, -400.0, 400.0)
        hcov01.GetXaxis().SetTitle("MET cov_{xy} (GeV^{2})")
        hcov01.GetYaxis().SetTitle("arb. units")
        hcov01.SetLineColor(r.kBlue+2)
        
        hstd00 = r.TH1D(std00name,std00name, 100, 0.0, 50.0)
        hstd00.GetXaxis().SetTitle("MET #sigma_{x} (GeV)")
        hstd00.GetYaxis().SetTitle("arb. units")
        hstd00.SetLineColor(r.kRed+2)
        
        hstd11 = r.TH1D(std11name,std11name, 100, 0.0, 50.0)
        hstd11.GetXaxis().SetTitle("MET #sigma_{y} (GeV)")
        hstd11.GetYaxis().SetTitle("arb. units")
        hstd11.SetLineColor(r.kOrange+2)

        drawstringcov00 = "metcov00>>%s"%cov00name
        t.Draw(drawstringcov00,"metcov00 <= 2000.0","HIST")
        DrawCMSLogo(c, 'CMS', '', 11, 0.025, 0.05, 1.0, '', 1.2)
        DrawTitle(c, '41.5 fb^{-1} (13 TeV)', 3)
        c.SaveAs("_".join([f,"cov00"])+".pdf")
        c.SaveAs("_".join([f,"cov00"])+".png")
        histlist.append(copy.deepcopy(hcov00))
        cov00_hists[primary_dataset].Add(histlist[-1])

        c.cd()
        drawstringcov11 = "metcov11>>%s"%cov11name
        t.Draw(drawstringcov11,"metcov11 <= 2000.0","HIST")
        DrawCMSLogo(c, 'CMS', '', 11, 0.025, 0.05, 1.0, '', 1.2)
        DrawTitle(c, '41.5 fb^{-1} (13 TeV)', 3)
        c.SaveAs("_".join([f,"cov11"])+".pdf")
        c.SaveAs("_".join([f,"cov11"])+".png")
        histlist.append(copy.deepcopy(hcov11))
        cov11_hists[primary_dataset].Add(histlist[-1])

        r.gStyle.SetOptFit()

        c.cd()
        drawstringcov01 = "metcov01>>%s"%cov01name
        t.Draw(drawstringcov01,"metcov01 <= 2000.0","HIST")
        hcov01.Fit("gaus")
        hcov01.Draw()
        DrawCMSLogo(c, 'CMS', '', 11, 0.025, 0.05, 1.0, '', 1.2)
        DrawTitle(c, '41.5 fb^{-1} (13 TeV)', 3)
        c.SaveAs("_".join([f,"cov01"])+".png")
        c.SaveAs("_".join([f,"cov01"])+".pdf")
        histlist.append(copy.deepcopy(hcov01))
        cov01_hists[primary_dataset].Add(histlist[-1])

        c.cd()
        drawstringstd00 = "sqrt(metcov00)>>%s"%std00name
        t.Draw(drawstringstd00,"sqrt(metcov00) <= 50.0","HIST")
        hstd00.Fit("gaus")
        hstd00.Draw()
        DrawCMSLogo(c, 'CMS', '', 11, 0.025, 0.05, 1.0, '', 1.2)
        DrawTitle(c, '41.5 fb^{-1} (13 TeV)', 3)
        c.SaveAs("_".join([f,"std00"])+".pdf")
        c.SaveAs("_".join([f,"std00"])+".png")
        histlist.append(copy.deepcopy(hstd00))
        std00_hists[primary_dataset].Add(histlist[-1])

        c.cd()
        drawstringstd11 = "sqrt(metcov11)>>%s"%std11name
        t.Draw(drawstringstd11,"sqrt(metcov11) <= 50.0","HIST")
        hstd11.Fit("gaus")
        hstd11.Draw()
        DrawCMSLogo(c, 'CMS', '', 11, 0.025, 0.05, 1.0, '', 1.2)
        DrawTitle(c, '41.5 fb^{-1} (13 TeV)', 3)
        c.SaveAs("_".join([f,"std11"])+".pdf")
        c.SaveAs("_".join([f,"std11"])+".png")
        histlist.append(copy.deepcopy(hstd11))
        std11_hists[primary_dataset].Add(histlist[-1])

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


out = r.TFile("metdata/metcovariance.root","recreate")
for h in cov00_hists.values()+cov11_hists.values()+cov01_hists.values():
    h.Scale(1.0/h.Integral())
    print h.Integral()
    h.Write()

out.Close()
