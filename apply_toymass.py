import csv
import sys, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import numpy as np
from fourvector import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_neutrino import transform_fourvector
from plot_invisibles import colors
channel = r'$\tau_{had} \tau_{had}$'
processes = ["susy100", "susy200", "susy300", "susy400",  "susy500", "susy600", "vbfSM", "ggHSM"]
masses = [100, 200, 300, 400, 500, 600]
masses_sv = [105, 205, 305, 405, 505, 605]
binning = [50, 50, 50, 50, 50, 50, 100, 100]
modelpath = sys.argv[1]
outpath = sys.argv[2]
if not os.path.exists(outpath):
    os.makedirs(outpath)

means_nn = [[],[], [], []]
widths_nn = [[],[], [], []]
means_sv = [[],[], [], []]
widths_sv = [[],[], [], []]

def fix_between(number, minimum, maximum):
    return min(max(number, minimum), maximum)

for index, process in enumerate(processes):
    filename = "data/" + process+".csv"

    dim = 10
    n_events = sum(1 for line in open(filename))
    X = np.zeros([n_events, dim])
    svfit = np.zeros([n_events, 4])
    L = np.zeros([n_events, 4])
    M = np.zeros([n_events, 4])
    phys_M = np.zeros([n_events, 4])
    gen = np.zeros([n_events, 4])
    gen_phys = np.zeros([n_events, 4])

 #   diff_svfit = np.zeros([n_events, 4])
#    diff_nn = np.zeros([n_events, 4])

    fake_met_phys = np.zeros([n_events, 4])
    gen_met_phys = np.zeros([n_events, 4])

    line = 0
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            a = [float(i) for i in row]
            lepton_1 = FourMomentum(a[4], a[5], a[6], a[7], cartesian=False)
            lepton_2 = FourMomentum(a[8], a[9], a[10], a[11], cartesian=False)
            met = FourMomentum(0, a[12], 0, a[13], False)
            gen_boson = FourMomentum(a[14], a[15], a[16], a[17], cartesian=False)
            gen_met_phys[line,:] = np.array([a[18], 0, a[19], 0])
            gen_met = FourMomentum( 0, a[18], 0, a[19], cartesian=False)

            gen_lepton_1 = FourMomentum(a[20], a[21], a[22], a[23], cartesian=False)
            gen_lepton_2 = FourMomentum(a[24], a[25], a[26], a[27], cartesian=False)

            fake_met_phys[line,:] = np.array([a[28], 0, a[29], 0])
            x = np.array([  lepton_1.e,
                            lepton_1.px,
                            lepton_1.py,
                            lepton_1.pz,
                            lepton_2.e,
                            lepton_2.px,
                            lepton_2.py,
                            lepton_2.pz,
                            met.px,
                            met.py
                            ])
            X[line,:] = x
            s = FourMomentum(a[0], a[1], a[2], a[3], cartesian=False)
            svfit[line,:] = np.array([s.pt, s.eta, s.phi, s.m()])
            l = np.array([lepton_1.e+lepton_2.e, lepton_1.px+lepton_2.px, lepton_1.py+lepton_2.py, lepton_1.pz+lepton_2.pz])
            L[line,:] = l
            m = np.array([0, met.px, met.py, 0])
            M[line,:] = m
            phys_M[line,:] = np.array([met.pt, 0, met.phi, 0])

            gen[line,:] = np.array([gen_boson.pt, gen_boson.eta, gen_boson.phi, gen_boson.m()])
            gen_phys[line,:] = np.array([gen_boson.e, gen_boson.px, gen_boson.py, gen_boson.pz])

           # d_svfit = FourMomentum(0, s.px - gen_boson.px, s.py - gen_boson.py, s.pz - gen_boson.pz)
           # diff_svfit[line,:] = np.array([d_svfit.pt, d_svfit.eta, d_svfit.phi, s.m() - gen_boson.m()])

            line +=1


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    from keras.models import load_model
    from plot_invisibles import full_fourvector, get_mass_constrained_ys
    from train_invisibles import custom_loss

    model = load_model(os.path.join(modelpath), custom_objects={'custom_loss':custom_loss } )

    scaled_Y = model.predict(X)
    #scaled_Y = get_mass_constrained_ys(X, scaled_Y)
    regressed_physfourvectors, regressed_fourvectors = full_fourvector(scaled_Y, L)
    diff_nn = np.array([ [   regressed_physfourvectors[i,0] - gen[i, 0],
                             regressed_physfourvectors[i,1] - gen[i, 1],
                             regressed_physfourvectors[i,2] - gen[i, 2],
                             regressed_physfourvectors[i,3] - gen[i, 3], ] for i in range(gen.shape[0]) if abs(regressed_physfourvectors[i,3] - gen[i, 3])<200  ])

    diff_svfit = np.array([  [svfit[i,0] - gen[i, 0],
                              svfit[i,1] - gen[i, 1],
                              svfit[i,2] - gen[i, 2],
                              svfit[i,3] - gen[i, 3],] for i in range(gen.shape[0]) if abs(svfit[i,3] - gen[i, 3])<200])

    """
    for a in [0]:
        pts = plt.figure()
        irange = None
#        if a == 0:
 #           irange = [-500, 500]
  #      if a == 3:
   #         irange = [-500, 500]
        n, bins, patches = plt.hist(fake_met_phys[:,0], 150, normed=1, facecolor='orange', alpha=0.5, range=irange)
        n, bins, patches = plt.hist(gen_met_phys[:,0], 150, normed=1, facecolor='green', alpha=0.5, range=irange)
        n, bins, patches = plt.hist(phys_M[:,0], 150, normed=1, facecolor='gray', alpha=0.5, range=irange)
        #n, bins, patches = plt.hist(target_physfourvectors[:,a], 150, normed=1, facecolor='green', alpha=0.75)
        plt.savefig("plots_apply/"+process+"-fakemet"+str(a)+".png")
    """
    

#    print process, " toy mean: ", np.mean(regressed_physfourvectors[:,3]), 'toy median', np.median(regressed_physfourvectors[:,3]), ", toy resolution: ", np.std(regressed_physfourvectors[:,3])
#    print process, " svfit mean: ", np.mean(svfit[:,3]), "svfit median", np.median(svfit[:,3]), ", svfit resolution: ", np.mean(svfit[:,3])





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
        ax.set_title("Gen vs. reconstruction (" + channel + ", " + process + ")")

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
    tau_1_orig_cartesian = [ FourMomentum( X[i,0] + np.sqrt(np.square(scaled_Y[i,0]) + np.square(scaled_Y[i,1]) + np.square(scaled_Y[i,2])),
                                 X[i,1] + scaled_Y[i,0],
                                 X[i,2] + scaled_Y[i,1],
                                 X[i,3] + scaled_Y[i,2]) for i in range(X.shape[0])]

    tau_1_orig_phys = np.array( [ [tau_1_orig_cartesian[i].pt,
                                   tau_1_orig_cartesian[i].eta, 
                                   tau_1_orig_cartesian[i].phi,
                                   tau_1_orig_cartesian[i].m()] for i in range(len(tau_1_orig_cartesian))])
    tau_2_orig_cartesian = [ FourMomentum( X[i,4] + np.sqrt(np.square(scaled_Y[i,3]) + np.square(scaled_Y[i,4]) + np.square(scaled_Y[i,5])),
                                 X[i,5] + scaled_Y[i,3],
                                 X[i,6] + scaled_Y[i,4],
                                 X[i,7] + scaled_Y[i,5]) for i in range(X.shape[0])]
    tau_2_orig_phys = np.array( [ [tau_2_orig_cartesian[i].pt,
                                   tau_2_orig_cartesian[i].eta, 
                                   tau_2_orig_cartesian[i].phi,
                                   tau_2_orig_cartesian[i].m()] for i in range(len(tau_1_orig_cartesian))])

    for a in range(4):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
#        arange = [-0,700]
        arange = None
    
        n, bins, patches = plt.hist(tau_1_orig_phys[:,a], 150, normed=1, color=colors["color_nn"], histtype='step', range = arange, label='regressed tau1')
        n, bins, patches = plt.hist(tau_2_orig_phys[:,a], 150, normed=1, color="blue", histtype='step', range = arange, label='regressed tau2')
#        n, bins, patches = plt.hist(gen[:,3], 150, normed=1, color=colors["color_true"], histtype='step', range = arange, label='target')
#        print "mass target ", a , " resolution: ", np.std(scaled_Y[:,a] - gen[:,3])
#        ax.text(0.2, 0.93, r'$\sigma(p_x^{true}, p_x^{regressed})$ = ', fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#        ax.text(0.25, 0.88, str(np.std(scaled_Y[:,a] - gen[:,3]))[0:4] + " GeV", fontsize=10, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax.set_xlabel("mass  (GeV)")
        ax.set_ylabel("arb. units")
        ax.set_title("Tau mass (" + channel + ")")
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
