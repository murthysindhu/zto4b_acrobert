import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, TreeMakerSchema, BaseSchema
from coffea.nanoevents.methods import vector
from coffea.analysis_tools import PackedSelection
from coffea import processor
from coffea.processor import IterativeExecutor
import numpy as np
import glob
from utils import *
import uproot
import warnings
import fnmatch
#import ROOT as r
from hist import Hist
import hist
import matplotlib.pyplot as plt
import mplhep
import matplotlib.ticker as ticker
import matplotlib

plt.rcParams['text.usetex'] = True

#key = 'ZZTo2Q2L_mc2017UL'

class PlotsMap:

    def __init__(self, key='ZZTo4B01j_mc2017UL', bkg=False):
        
        self.key = key
        self.nbins = 100
        self.bkg = bkg
        if self.bkg:
            self.ntypes = 1
            self.types = ['All']
        else:
            self.ntypes = 4
            self.types = ['Z-4B', 'Z-2Q2B', 'Z-4Q', 'Z-2B']

        #self.hist_matchedset = [[Hist(hist.axis.Regular(12, -0.5, 11.5, name='type')) for i in range(3)] for k in range(self.ntypes)]

        self.n_nbjets = [0, 0, 0, 0]

    
    def _process(self, tagnano, index):
        
        #print(tagnano.fields)
        zto4b_evts = tagnano.Zs
        Zsto4b = zto4b_evts[:,0]
        Bs_4B4V = ak.zip({"pt": Zsto4b.final_Bs_pT, "eta": Zsto4b.final_Bs_eta, "phi": Zsto4b.final_Bs_phi, "energy": Zsto4b.final_Bs_E}, with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)

        jets = tagnano.Jet[tagnano.Jet.puId > 4]
        bjets = jets[jets.btagDeepFlavB > 0.6]
        for i in range(min(len(tagnano), 500)):

            nbjets = len(bjets[i])
            if nbjets > 3:
                nbjets = 3
            if self.n_nbjets[nbjets] > 30:
                continue
            self.n_nbjets[nbjets] += 1

            #print(i, 'sums', tagnano.map.nPFTracks[i], np.sum(tagnano.map.PFTracks_pT[i]), tagnano.map.nRecHits[i], np.sum(tagnano.map.RecHits_E[i]))
            
            for k in range(1):
                
                maphist = Hist(hist.axis.Regular(46, -4.002, 4.002, name='eta', label='eta'), hist.axis.Regular(36, -np.pi, np.pi, name='phi', label='phi'))

                custom_obj = []
                custom_label = []
                plt.clf()
                plt.axes()
                plt.gca().set_aspect('equal')

                if k == 1: # fill pftracks
                    maphist.fill(tagnano.map.PFTracks_eta[i], tagnano.map.PFTracks_phi[i],
                                 weight=tagnano.map.PFTracks_pT[i])                    
                    
                if k == 2: # fill rechits
                    maphist.fill(tagnano.map.RecHits_eta[i], tagnano.map.RecHits_phi[i],
                                 weight = tagnano.map.RecHits_E[i])

                if k > 0:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    w, x, y = maphist.to_numpy()
                    mesh = ax.pcolormesh(x, y, w.T, cmap="RdYlBu", norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=300.))
                    fig.colorbar(mesh)

                for j in range(4):
                    leta = Zsto4b.final_Bs_eta[i][j]
                    lphi = Zsto4b.final_Bs_phi[i][j]
                    circle = plt.Circle((leta, lphi), 0.15, fc='green',ec='green', fill=True)
                    plt.gca().add_patch(circle)
                    custom_obj.append(circle)
                    custom_label.append(r"$b{}$ $p_T={}$".format(j, round(Zsto4b.final_Bs_pT[i][j], 1)))
                    plt.text(leta-0.15, lphi-0.1, r"$b{}$".format(j))

                circle = plt.Circle((Zsto4b.eta[i], Zsto4b.phi[i]), 0.15, fc='black',ec='black')
                plt.gca().add_patch(circle)
                custom_obj.append(circle)
                custom_label.append(r"$Z^0\rightarrow 4b$ $p_T={}$".format(round(Zsto4b.pT[i], 1)))
                plt.text(Zsto4b.eta[i]+0.15, Zsto4b.phi[i]-0.1, r"$Z0$")
                
                if len(zto4b_evts[i]) > 1:
                    circle = plt.Circle((zto4b_evts.eta[i][1], zto4b_evts.phi[i][1]), 0.15, fc=None,ec='black', fill=False)
                    plt.gca().add_patch(circle)
                    custom_obj.append(circle)
                    custom_label.append(r"$Z^0$ $p_T={}$".format(round(zto4b_evts.pT[i][1], 1)))
                    plt.text(zto4b_evts.eta[i][1]+0.12, zto4b_evts.phi[i][1]-0.1, r"$Z1$")
                
                for j in range(len(jets[i])):
                    thisjet = jets[i][j]
                    color = 'green' if thisjet.btagDeepFlavB > 0.6 else 'blue'
                    circle = plt.Circle((thisjet.eta, thisjet.phi), 0.4, fc=None,ec=color, fill=False)
                    plt.gca().add_patch(circle)
                    custom_obj.append(circle)
                    custom_label.append(r"Jet{} $p_T={}$".format(j, round(thisjet.pt, 1)))
                    plt.text(thisjet.eta+0.4, thisjet.phi-0.1, r"J{}".format(j))
                
                for j in range(len(tagnano.Electron[i])):
                    thislep = tagnano.Electron[i][j]
                    circle = plt.Circle((thislep.eta, thislep.phi), 0.1, fc=None,ec='red', fill=False)
                    plt.gca().add_patch(circle)
                    custom_obj.append(circle)
                    custom_label.append(r"$e{}$ $p_T={}$".format(j, round(thislep.pt, 1)))
                    plt.text(thislep.eta+0.1, thislep.phi-0.07, r"$e{}$".format(j))
                for j in range(len(tagnano.Muon[i])):
                    thislep = tagnano.Muon[i][j]
                    circle = plt.Circle((thislep.eta, thislep.phi), 0.1, fc=None,ec='red', fill=False)
                    plt.gca().add_patch(circle)
                    custom_obj.append(circle)
                    custom_label.append(r"$\mu{}$ $p_T={}$".format(j, round(thislep.pt, 1)))
                    plt.text(thislep.eta+0.1, thislep.phi-0.07, r"$\mu{}$".format(j))
                for j in range(len(tagnano.SV[i])):
                    thislep = tagnano.SV[i][j]
                    #circle = plt.Circle((thislep.eta, thislep.phi), 0.1, fc=None,ec='red', fill=False)
                    #plt.gca().add_patch(circle)
                    plt.plot(thislep.eta, thislep.phi, "kx")
                    kx = matplotlib.lines.Line2D([], [], color='black', marker='x', linestyle='None',
                                  markersize=10, label='Blue stars')
                    custom_obj.append(kx)
                    custom_label.append(r"SV{} $p_T={}$".format(j, round(thislep.pt, 1)))
                    plt.text(thislep.eta+0.1, thislep.phi-0.07, r"SV{}".format(j))
                print(len(tagnano.SoftActivityJet[i]), tagnano.SoftActivityJet[i].pt)
                for j in range(len(tagnano.SoftActivityJet[i])):
                    thislep = tagnano.SoftActivityJet[i][j]
                    sajdR = tagnano.Jet[i].delta_r(thislep)
                    if ak.min(sajdR) < 0.2:
                        continue
                    circle = plt.Circle((thislep.eta, thislep.phi), 0.4, fc=None,ec='black', fill=False, linestyle='--')
                    plt.gca().add_patch(circle)
                    custom_obj.append(circle)
                    custom_label.append(r"SAJ{} $p_T={}$".format(j, round(thislep.pt, 1)))
                    plt.text(thislep.eta+0.4, thislep.phi-0.1, r"SAJ{}".format(j))



                
                plt.legend(custom_obj, custom_label, prop={'size': 5})
                plt.axis('scaled')
                plt.title(f'Map: Z-4B Event {i} ({nbjets} B-Jets)')
                plt.ylabel(r'$\phi$')
                plt.xlabel(r'$\eta$')
                plt.ylim([-np.pi, np.pi])
                plt.xlim([-4, 4])
                figname = f'plots/map_{self.key}_{nbjets}bjets_{i}.pdf'
                if k == 1:
                    figname = f'plots/map_{self.key}_{nbjets}bjets_PFT_{i}.pdf'
                if k == 2:
                    figname = f'plots/map_{self.key}_{nbjets}bjets_RH_{i}.pdf'
                plt.savefig(figname)
                plt.close()

    def processAll(self, tagnano):
        pass

    def process4B(self, tagnano):
        self._process(tagnano, 0)

    def process2B(self, tagnano):
        pass

    def plot(self):
        pass
