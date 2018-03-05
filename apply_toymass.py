import csv
import sys, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
from fourvector import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_functions import transform_fourvector
from common_functions import full_fourvector
from common_functions import predict
from common_functions import original_tauh, original_taul
from plot_invisibles import colors


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
masses = [100, 200, 300, 400, 500, 600]
masses_sv = [105, 205, 305, 405, 505, 605]



channel_name = {"tt": r'$\tau_{had} \tau_{had}$', "mt": r'$\mu \tau_{had}$'}
binning = [50, 50, 50, 50, 50, 50, 100, 100]
channel = sys.argv[1]
model_path = sys.argv[2]
outpath = sys.argv[3]
if not os.path.exists(outpath):
    os.makedirs(outpath)

means_nn = [[],[], [], []]
widths_nn = [[],[], [], []]
means_sv = [[],[], [], []]
widths_sv = [[],[], [], []]

met_uncs = {}

def fix_between(number, minimum, maximum):
    return min(max(number, minimum), maximum)

for index, filename in enumerate(filenames):
    from root_pandas import read_root
    process = new_filenames[index]
    branches = ["genBosonMass", "genBosonPt", "genBosonEta", "genBosonPhi",
            "m_sv", "pt_sv", "eta_sv", "phi_sv",
            "m_1", "pt_1", "eta_1", "phi_1",
            "m_2", "pt_2", "eta_2", "phi_2",
            "met", "metphi",
            "metcov00", "metcov11"]
    in_array = read_root(os.path.join(folder,filename,filename+".root"), channel+"_nominal/ntuple", columns = branches).as_matrix()

    dim = 12
    n_events = in_array.shape[0]
    X = np.zeros([n_events, dim])
    svfit = np.zeros([n_events, 4])
    Boson = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
#    M = np.zeros([n_events, 4])
#    phys_M = np.zeros([n_events, 4])
    gen = np.zeros([n_events, 4])
#    gen_phys = np.zeros([n_events, 4])

 #   diff_svfit = np.zeros([n_events, 4])
#    diff_nn = np.zeros([n_events, 4])

    fake_met_cart = np.zeros([n_events, 4])
    gen_met_phys = np.zeros([n_events, 4])
    #met_cov = np.zeros([n_events, 2])
    met_unc = np.zeros([n_events, 2])

    print in_array.size
    for i in range(n_events):
        a = in_array[i,:]
        gen_boson = FourMomentum(a[0], a[1], a[2], a[3], cartesian=False)
        s = FourMomentum(a[4], a[5], a[6], a[7], cartesian=False)
        lepton_1 = FourMomentum(a[8], a[9], a[10], a[11], cartesian=False)
        lepton_2 = FourMomentum(a[12], a[13], a[14], a[15], cartesian=False)
        met = FourMomentum(0, a[16], 0, a[17], False)


        met_unc[i,:] = np.array([np.sqrt(a[18]), np.sqrt(a[19])])
        met_resx = np.sqrt(a[18])
        met_resy = np.sqrt(a[19])

        #fake_met_cart[line,:] = np.array([a[-2], a[-1], 0, 0])
        x = np.array([  lepton_1.e,
                        lepton_1.px,
                        lepton_1.py,
                        lepton_1.pz,
                        lepton_2.e,
                        lepton_2.px,
                        lepton_2.py,
                        lepton_2.pz,
                        met.px,
                        met.py,
                        met_resx,
                        met_resy
                        ])
        X[i,:] = x
        svfit[i,:] = np.array([s.pt, s.eta, s.phi, s.m()])
        l = np.array([lepton_1.e+lepton_2.e, lepton_1.px+lepton_2.px, lepton_1.py+lepton_2.py, lepton_1.pz+lepton_2.pz])
        L[i,:] = l
        #m = np.array([0, met.px, met.py, 0])
        #M[line,:] = m
        #phys_M[line,:] = np.array([met.pt, 0, met.phi, 0])

        gen[i,:] = np.array([gen_boson.pt, gen_boson.eta, gen_boson.phi, gen_boson.m()])
        #gen_phys[line,:] = np.array([gen_boson.e, gen_boson.px, gen_boson.py, gen_boson.pz])

       # d_svfit = FourMomentum(0, s.px - gen_boson.px, s.py - gen_boson.py, s.pz - gen_boson.pz)
       # diff_svfit[line,:] = np.array([d_svfit.pt, d_svfit.eta, d_svfit.phi, s.m() - gen_boson.m()])

        
#line +=1




    scaled_Y = predict(model_path, X, channel)
    regressed_physfourvectors, regressed_fourvectors = full_fourvector(scaled_Y, L)
    diff_nn = np.array([ [   regressed_physfourvectors[i,0] - gen[i, 0],
                             regressed_physfourvectors[i,1] - gen[i, 1],
                             regressed_physfourvectors[i,2] - gen[i, 2],
                             regressed_physfourvectors[i,3] - gen[i, 3], ] for i in range(gen.shape[0]) if abs(regressed_physfourvectors[i,3] - gen[i, 3])<200  ])

    diff_svfit = np.array([  [svfit[i,0] - gen[i, 0],
                              svfit[i,1] - gen[i, 1],
                              svfit[i,2] - gen[i, 2],
                              svfit[i,3] - gen[i, 3],] for i in range(gen.shape[0]) if abs(svfit[i,3] - gen[i, 3])<200])

    met_uncs[process] = met_unc
    # pt-dependency
    for a in [0]:
        pt = np.sqrt(np.square(L[:,1]) + np.square(L[:,2]))
        unc = np.sqrt(np.square(met_unc[:,0]) + met_unc[:,1])
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        print pt
        print unc
        pts = plt.figure()
        irange = [0,80]
#        n, bins, patches = plt.hist(fake_met_cart[:,a], 150, normed=1, facecolor=colors["color_true"], alpha=0.5, range=irange, histtype='step', label="fake met")
        ax.hist2d(pt, unc, range = [[0, 500], [0, 150]] )
 #       n, bins, patches = plt.hist(scaled_Y[:,a+1], 150, normed=1, facecolor="black", alpha=0.5, range=irange, histtype='step', label="smear")
#        plt.legend(loc='best')
        plt.savefig(os.path.join(outpath, process+"-metunc-over-pt.png"))
#        print process, " fake met: ", np.mean(fake_met_cart[:,a]), ' median', np.median(fake_met_cart[:,a]), ", resolution: ", np.std(fake_met_cart[:,a])
#        print process, " met cov: ", np.mean(v[:,a]), ' median', np.median(v[:,a]), ", toy resolution: ", np.std(met_unc[:,a])

    for a in range(regressed_physfourvectors.shape[1]):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ranges = [[0,500],
            [-8,8],
            [-4,4],
            [0,600]]
        titles = [ r'$p_T$ (GeV)', r'$\eta$',r'$\phi$',r'$m$ (GeV)',]
        n, bins, patches = plt.hist(gen[:,a], bins=binning[index], normed=1, color=colors["color_true"], alpha=0.75, range=ranges[a], histtype='step', label='True')
        n, bins, patches = plt.hist(regressed_physfourvectors[:,a], bins=binning[index], normed=1, color=colors["color_nn"], alpha=0.75, range=ranges[a], histtype='step', label='Regressed')
        n, bins, patches = plt.hist(svfit[:,a], bins=binning[index], normed=1, color=colors["color_svfit"], alpha=0.5, range=ranges[a], histtype='step', label='SVFit', linestyle='dotted')
#        n, bins, patches = plt.hist(diff_nn[:,a], bins=binning[index], normed=1, color=colors["color_nn"], alpha=0.75, range=ranges[a], histtype='step', label='Regressed')
#        n, bins, patches = plt.hist(diff_svfit[:,a], bins=binning[index], normed=1, color=colors["color_svfit"], alpha=0.5, range=ranges[a], histtype='step', label='SVFit', linestyle='dotted')
        #print "phys diffvector mean ", a, np.mean(diff_physfourvectors[:,a]), " stddev " , np.std(diff_physfourvectors[:,a])

        if a == 0:
            ax.text(0.7, 0.75, r'$\sigma(p_T^{true}, p_T^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.75, 0.7, "{:10.1f}".format(np.std(diff_nn[:,a])) +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

            ax.text(0.7, 0.6, r'$\sigma(p_T^{true}, p_T^{SVFit})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.75, 0.55, "{:10.1f}".format(np.std(diff_svfit[:,a])) +" GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        if a == 3:
            ax.text(0.6, 0.6, r'$\sigma / \Delta (m^{true}, m^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.65, 0.55, "{:10.1f}".format(np.std(diff_nn[:,a])) +" GeV / " + "{:10.1f}".format(np.mean(diff_nn[:,a])) + " GeV",  fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.6, 0.4, r'$\sigma / \Delta (m^{true}, m^{SVFit})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
            ax.text(0.65, 0.35, "{:10.1f}".format(np.std(diff_svfit[:,a])) +" GeV / " + "{:10.1f}".format(np.mean(diff_svfit[:,a])) + " GeV",  fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        if index < 6:
            means_nn[a].append(np.mean(diff_nn[:,a]))
            widths_nn[a].append(np.std(diff_nn[:,a]))
            means_sv[a].append(np.mean(diff_svfit[:,a]))
            widths_sv[a].append(np.std(diff_svfit[:,a]))

        ax.set_xlabel(titles[a])
        ax.set_ylabel("arb. units")
        ax.set_title("Gen vs. reconstruction (" + channel_name[channel] + ", " + process + ")")

        plt.legend(loc='best')
        plt.savefig(os.path.join(outpath, process+"-regressed"+str(a)+".png"))
        plt.tight_layout()
        plt.close()


#    for a in range(4):
#        pts = plt.figure()
#        irange = None
#        if a == 0:
#            irange = [-500, 500]
#        if a == 3:
#            irange = [-500, 500]
#        n, bins, patches = plt.hist(diff_nn[:,a], 150, normed=1, facecolor='red', alpha=0.75, range=irange)
#        n, bins, patches = plt.hist(diff_svfit[:,a], 150, normed=1, facecolor='blue', alpha=0.75, range=irange)
#        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
#        plt.savefig("plots_apply/"+process+"-diff"+str(a)+".png")


#    for a in range(4):
#        pts = plt.figure()
#        n, bins, patches = plt.hist(unscaled_pred[:,a], 150, normed=1, facecolor='red', alpha=0.75)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
#        plt.savefig(process+"-unscaled"+str(a)+".pdf")

#    for a in range(4):
#        pts = plt.figure()
#        n, bins, patches = plt.hist(regressed_fourvectors[:,a], 150, normed=1, facecolor='red', alpha=0.75)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
#        plt.savefig("plots_apply/"+process+"-cartesian"+str(a)+".png")
##    for a in [6]:
##        fig = plt.figure(figsize=(5,5))
##        ax = fig.add_subplot(111)
##        arange = [-0,700]
##    
##        n, bins, patches = plt.hist(scaled_Y[:,a], 150, normed=1, color=colors["color_nn"], histtype='step', range = arange, label='regressed')
##        n, bins, patches = plt.hist(gen[:,3], 150, normed=1, color=colors["color_true"], histtype='step', range = arange, label='target')
##        print "mass target ", a , " resolution: ", np.std(scaled_Y[:,a] - gen[:,3])
##        ax.text(0.2, 0.93, r'$\sigma(p_x^{true}, p_x^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
##        ax.text(0.25, 0.88, str(np.std(scaled_Y[:,a] - gen[:,3]))[0:4] + " GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
##        ax.set_xlabel("mass  (GeV)")
##        ax.set_ylabel("arb. units")
##        ax.set_title("Regression target vs. Result (" + channel + ")")
##        plt.legend()
##        plt.tight_layout()
##        plt.savefig(os.path.join(outpath, process+"-target-vs-regressed"+str(a)+".png"))
##        plt.close()

# tau mass
    tau_1_orig_phys = original_tauh(0, 1, 2, 3, 14, 15, 16, X, scaled_Y)
    tau_2_orig_phys = original_tauh(4, 5, 6, 7, 18, 19, 20, X, scaled_Y)
#    gentau_1_orig_phys = original_tau(0, 1, 2, 3, 0, 1, 2, X, Y)
#    gentau_2_orig_phys = original_tau(4, 5, 6, 7, 3, 4, 5, X, Y)

    for a in range(4):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
#        arange = [-0,700]
        arange = None
        if a == 3:
            arange = [-2, 20]
    
        n, bins, patches = plt.hist(tau_1_orig_phys[:,a], 150, normed=1, color=colors["color_nn"], histtype='step', range = arange, label='regressed tau1')
        n, bins, patches = plt.hist(tau_2_orig_phys[:,a], 150, normed=1, color="blue", histtype='step', range = arange, label='regressed tau2')
#        n, bins, patches = plt.hist(gen[:,3], 150, normed=1, color=colors["color_true"], histtype='step', range = arange, label='target')
#        print "mass target ", a , " resolution: ", np.std(scaled_Y[:,a] - gen[:,3])
#        ax.text(0.2, 0.93, r'$\sigma(p_x^{true}, p_x^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#        ax.text(0.25, 0.88, str(np.std(scaled_Y[:,a] - gen[:,3]))[0:4] + " GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        if a == 3:
            ax.set_xlabel("mass  (GeV)")
            ax.set_ylabel("arb. units")
        ax.set_title("Tau property (" + channel + ")")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, process+"-taumass"+str(a)+".png"))
        plt.close()




for a in range(4):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ranges = [[0,500],
        [-8,8],
        [-4,4],
        [0,600]]
    titles = [ r'$p_T$ (GeV)', r'$\eta$',r'$\phi$',r'$m$ (GeV)',]
    ax.errorbar(masses, means_nn[a], yerr=widths_nn[a], fmt='o', color = colors["color_nn"], label = "Regressed")
    ax.errorbar(masses_sv, means_sv[a], yerr=widths_sv[a], fmt='o', color = colors["color_svfit"], label = "SVFit")

    ax.set_xlabel(r'$m$ (GeV)')
    ax.set_ylabel(titles[a])
    ax.set_title("Resolution (" + channel + ")")

    plt.legend(loc='best')
    plt.savefig(os.path.join(outpath, "resolution-"+str(a)+".png"))
    plt.tight_layout()
    plt.close()

for k, v in met_uncs.iteritems():
    # fake met / vgl mit cov matrix
    for a in [0,1]:
        pts = plt.figure()
        irange = [0,80]
#        n, bins, patches = plt.hist(fake_met_cart[:,a], 150, normed=1, facecolor=colors["color_true"], alpha=0.5, range=irange, histtype='step', label="fake met")
        n, bins, patches = plt.hist(v[:,a], 100, normed=1, alpha=0.5, range=irange, histtype='step', label=k)
 #       n, bins, patches = plt.hist(scaled_Y[:,a+1], 150, normed=1, facecolor="black", alpha=0.5, range=irange, histtype='step', label="smear")
        plt.legend(loc='best')
        plt.savefig(os.path.join(outpath, "met_unc"+str(a)+".png"))
#        print process, " fake met: ", np.mean(fake_met_cart[:,a]), ' median', np.median(fake_met_cart[:,a]), ", resolution: ", np.std(fake_met_cart[:,a])
        print process, " met cov: ", np.mean(v[:,a]), ' median', np.median(v[:,a]), ", toy resolution: ", np.std(met_unc[:,a])
