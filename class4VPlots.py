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

plt.rcParams['text.usetex'] = True

class Plots4V:
    
    def __init__(self, key='ZZTo4B01j_mc2017UL', nbins=100):

        self.key = key
        ptthresh = 500 if 'ZJets' in key else 150
        self.hist_Z4B_pt = Hist(hist.axis.Regular(nbins, 0, 2*ptthresh, name='pt'))
        self.hist_Z4B_eta = Hist(hist.axis.Regular(nbins, -5, 5, name='eta'))
        self.hist_Z4B_phi = Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi'))
        self.hist_Z4B_mass = Hist(hist.axis.Regular(nbins, 70, 110, name='mass'))
        self.hist_4B_eta = Hist(hist.axis.Regular(nbins, -5, 5, name='eta'))
        self.hist_4B_eta_all = Hist(hist.axis.Regular(nbins, -5, 5, name='eta'))
        self.hist_4B_eta_indiv = [Hist(hist.axis.Regular(nbins, -5, 5, name='eta')) for i in range(4)]
        self.hist_4B_phi = Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi'))
        self.hist_4B_phi_all = Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi'))
        self.hist_4B_phi_indiv = [Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi')) for i in range(4)]
        self.hist_4B_mass = Hist(hist.axis.Regular(nbins, 0, 120, name='mass'))
        self.hist_4B_pt = Hist(hist.axis.Regular(nbins, 0, 2*ptthresh, name='pt'))
        self.hist_4B_pt_all = Hist(hist.axis.Regular(nbins, 0, ptthresh, name='pt'))
        self.hist_4B_pt_indiv = [Hist(hist.axis.Regular(nbins, 0, ptthresh, name='pt')) for i in range(4)]

        self.hist_Z4B_eta_phi = Hist(hist.axis.Regular(nbins, -5, 5, name='eta', label=r'\eta'), hist.axis.Regular(nbins, -np.pi, np.pi, name='phi', label=r'\phi'))
        self.hist_4B_eta_phi = Hist(hist.axis.Regular(nbins, -5, 5, name='eta', label=r'\eta'), hist.axis.Regular(nbins, -np.pi, np.pi, name='phi', label=r'\phi'))
        self.hist_4Bgap_eta_phi = Hist(hist.axis.Regular(nbins, -0.5, 0.5, name='eta', label=r'\eta'), hist.axis.Regular(nbins, -0.5, 0.5, name='phi', label=r'\phi'))

        self.hist_Z2B_pt = Hist(hist.axis.Regular(nbins, 0, 2*ptthresh, name='pt'))
        self.hist_Z2B_eta = Hist(hist.axis.Regular(nbins, -5, 5, name='eta'))
        self.hist_Z2B_phi = Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi'))
        self.hist_Z2B_mass = Hist(hist.axis.Regular(nbins, 70, 110, name='mass'))
        self.hist_2B_orig_pt = Hist(hist.axis.Regular(nbins, 0, 2*ptthresh, name='pt'))
        self.hist_2B_orig_eta = Hist(hist.axis.Regular(nbins, -5, 5, name='eta'))
        self.hist_2B_orig_phi = Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi'))
        self.hist_2B_orig_pt_indiv = [Hist(hist.axis.Regular(nbins, 0, ptthresh, name='pt')) for i in range(2)]
        self.hist_2B_orig_eta_indiv = [Hist(hist.axis.Regular(nbins, -5, 5, name='eta')) for i in range(2)]
        self.hist_2B_orig_phi_indiv = [Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi')) for i in range(2)]
        self.hist_2B_eta = Hist(hist.axis.Regular(nbins, -5, 5, name='eta'))
        self.hist_2B_eta_all = Hist(hist.axis.Regular(nbins, -5, 5, name='eta'))
        self.hist_2B_eta_indiv = [Hist(hist.axis.Regular(nbins, -5, 5, name='eta')) for i in range(2)]
        self.hist_2B_phi = Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi'))
        self.hist_2B_phi_all = Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi'))
        self.hist_2B_phi_indiv = [Hist(hist.axis.Regular(nbins, -np.pi, np.pi, name='phi')) for i in range(2)]
        self.hist_2B_mass = Hist(hist.axis.Regular(nbins, 0, 120, name='mass'))
        self.hist_2B_pt = Hist(hist.axis.Regular(nbins, 0, 2*ptthresh, name='pt'))
        self.hist_2B_pt_all = Hist(hist.axis.Regular(nbins, 0, ptthresh, name='pt'))
        self.hist_2B_pt_indiv = [Hist(hist.axis.Regular(nbins, 0, ptthresh, name='pt')) for i in range(2)]

        self.hist_Z2B_eta_phi = Hist(hist.axis.Regular(nbins, -5, 5, name='eta', label=r'\eta'), hist.axis.Regular(nbins, -np.pi, np.pi, name='phi', label=r'\phi'))
        self.hist_2B_eta_phi = Hist(hist.axis.Regular(nbins, -5, 5, name='eta', label=r'\eta'), hist.axis.Regular(nbins, -np.pi, np.pi, name='phi', label=r'\phi'))
        self.hist_2Bgap_eta_phi = Hist(hist.axis.Regular(nbins, -0.5, 0.5, name='eta', label=r'\eta'), hist.axis.Regular(nbins, -0.5, 0.5, name='phi', label=r'\phi'))

    def process4B(self, tagnano):

        zto4b_evts = tagnano.Zs#[hasZ4B]#np.logical_and(hasZ4B, np.logical_not(twoZ4B))]
        
        Zsto4b = zto4b_evts[:,0]
        
        self.hist_Z4B_pt.fill(Zsto4b.pT)
        self.hist_Z4B_eta.fill(Zsto4b.eta)
        self.hist_Z4B_phi.fill(Zsto4b.phi)
        self.hist_Z4B_mass.fill(Zsto4b.mass)
        
        self.hist_Z4B_eta_phi.fill(Zsto4b.eta, Zsto4b.phi)
        
        Bs_4B4V = ak.zip({"pt": Zsto4b.final_Bs_pT, "eta": Zsto4b.final_Bs_eta, "phi": Zsto4b.final_Bs_phi, "energy": Zsto4b.final_Bs_E}, with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        ptsort = ak.argsort(Bs_4B4V.pt, ascending=False)
        Bs_4B4V = Bs_4B4V[ptsort]        
        Bs_all4V = Bs_4B4V[:,0] + Bs_4B4V[:,1] + Bs_4B4V[:,2] + Bs_4B4V[:,3]
        
        Bs_allmass = np.sqrt(Bs_all4V.energy**2 - (Bs_all4V.pt*np.cosh(Bs_all4V.eta))**2)
        self.hist_4B_mass.fill(Bs_allmass)
        self.hist_4B_eta.fill(Bs_all4V.eta)
        self.hist_4B_eta_all.fill(ak.flatten(Bs_4B4V.eta))
        self.hist_4B_phi.fill(Bs_all4V.phi)
        self.hist_4B_phi_all.fill(ak.flatten(Bs_4B4V.phi))
        self.hist_4B_pt.fill(Bs_all4V.pt)
        self.hist_4B_pt_all.fill(ak.flatten(Bs_4B4V.pt))
        for i in range(4):
            self.hist_4B_eta_indiv[i].fill(Bs_4B4V.eta[:,i])
            self.hist_4B_phi_indiv[i].fill(Bs_4B4V.phi[:,i])
            self.hist_4B_pt_indiv[i].fill(Bs_4B4V.pt[:,i])
        
        self.hist_4B_eta_phi.fill(Bs_all4V.eta, Bs_all4V.phi)
        self.hist_4Bgap_eta_phi.fill(Bs_all4V.eta - Zsto4b.eta, Bs_all4V.phi - Zsto4b.phi)


    def process2B(self, tagnano):

        zto2b_evts = tagnano.Zs#[hasZ2B]

        Z1s = zto2b_evts[:,0]
        #Z2s = zto2b_evts[:,1]
        
        self.hist_Z2B_pt.fill(Z1s.pT)
        self.hist_Z2B_eta.fill(Z1s.eta)
        self.hist_Z2B_phi.fill(Z1s.phi)
        self.hist_Z2B_mass.fill(Z1s.mass)
        
        #hist_Z2B_pt.fill(Z2s.pT)
        #hist_Z2B_eta.fill(Z2s.eta)
        #hist_Z2B_phi.fill(Z2s.phi)
        #hist_Z2B_mass.fill(Z2s.mass)
        
        self.hist_Z2B_eta_phi.fill(Z1s.eta, Z1s.phi)
        #hist_Z2B_eta_phi.fill(Z2s.eta, Z2s.phi)
        
        Bs_2B4V = ak.zip({"pt": Z1s.final_Bs_pT, "eta": Z1s.final_Bs_eta, "phi": Z1s.final_Bs_phi, "mass": Z1s.final_Bs_mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        ptsort = ak.argsort(Bs_2B4V.pt, ascending=False)
        Bs_2B4V = Bs_2B4V[ptsort]

        Bs_all4V = Bs_2B4V[:,0] + Bs_2B4V[:,1]
        
        self.hist_2B_mass.fill(Bs_all4V.mass)
        self.hist_2B_eta.fill(Bs_all4V.eta)
        self.hist_2B_eta_all.fill(ak.flatten(Bs_2B4V.eta))
        self.hist_2B_phi.fill(Bs_all4V.phi)
        self.hist_2B_phi_all.fill(ak.flatten(Bs_2B4V.phi))
        self.hist_2B_pt.fill(Bs_all4V.pt)
        self.hist_2B_pt_all.fill(ak.flatten(Bs_2B4V.pt))
        for i in range(2):
            self.hist_2B_eta_indiv[i].fill(Bs_2B4V.eta[:,i])
            self.hist_2B_phi_indiv[i].fill(Bs_2B4V.phi[:,i])
            self.hist_2B_pt_indiv[i].fill(Bs_2B4V.pt[:,i])
        
        self.hist_2B_eta_phi.fill(Bs_all4V.eta, Bs_all4V.phi)
        self.hist_2Bgap_eta_phi.fill(Bs_all4V.eta - Z1s.eta, Bs_all4V.phi - Z1s.phi)
        
        has2origB = (ak.num(Z1s.orig_Bs_pT) == 2)
        Bs_2B4V = ak.zip({"pt": Z1s.orig_Bs_pT[has2origB], "eta": Z1s.orig_Bs_eta[has2origB], "phi": Z1s.orig_Bs_phi[has2origB], "mass": Z1s.orig_Bs_mass[has2origB]}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        ptsort = ak.argsort(Bs_2B4V.pt, ascending=False)
        Bs_2B4V = Bs_2B4V[ptsort]
        
        Bs_all4V = Bs_2B4V[:,0] + Bs_2B4V[:,1]
        
        self.hist_2B_orig_eta.fill(ak.flatten(Bs_2B4V.eta))
        self.hist_2B_orig_phi.fill(ak.flatten(Bs_2B4V.phi))
        self.hist_2B_orig_pt.fill(ak.flatten(Bs_2B4V.pt))
        
        for i in range(2):
            self.hist_2B_orig_eta_indiv[i].fill(Bs_2B4V.eta[:,i])
            self.hist_2B_orig_phi_indiv[i].fill(Bs_2B4V.phi[:,i])
            self.hist_2B_orig_pt_indiv[i].fill(Bs_2B4V.pt[:,i])

    def plot(self):
            
        name_list_eta_phi = ['hist_Z4B_eta_phi', 'hist_4B_eta_phi', 'hist_4Bgap_eta_phi', 'hist_Z2B_eta_phi','hist_2B_eta_phi', 'hist_2Bgap_eta_phi']
        hist_list_eta_phi = [self.hist_Z4B_eta_phi, self.hist_4B_eta_phi, self.hist_4Bgap_eta_phi, self.hist_Z2B_eta_phi, self.hist_2B_eta_phi, self.hist_2Bgap_eta_phi]
        titles = [r"Z-4B $\eta$ vs $\phi$", r"4B $\eta$ vs $\phi$", r"Z vs 4B $\eta$ vs $\phi$", r"Z-2B $\eta$ vs $\phi$", r"2B $\eta$ vs $\phi$", r"Z vs 2B $\eta$ vs $\phi$"]
        for hist, name, title in zip(hist_list_eta_phi, name_list_eta_phi, titles):
            fig, ax = plt.subplots(figsize=(8, 5))
            w, y, x = hist.to_numpy()
            #exec('w, y, x = self.{}.to_numpy()'.format(hist), d, locals())
            mesh = ax.pcolormesh(x, y, np.log10(w), cmap="RdYlBu")
            ax.set_xlabel(r"$\phi$")
            ax.set_ylabel(r"$\eta$")
            ax.set_title(title)
            fig.colorbar(mesh)
            plt.savefig(f'plots/{name}_{self.key}.pdf')
            plt.close(fig)
        
        name_list4 = ['hist_Z4B_pt', 'hist_Z4B_eta', 'hist_Z4B_phi', 'hist_Z4B_mass', 'hist_4B_eta', 'hist_4B_phi', 'hist_4B_mass', 'hist_4B_eta_all', 'hist_4B_phi_all', 'hist_4B_pt', 'hist_4B_pt_all']
        name_list2 = ['hist_Z2B_pt', 'hist_Z2B_eta', 'hist_Z2B_phi', 'hist_Z2B_mass', 'hist_2B_eta', 'hist_2B_phi', 'hist_2B_mass', 'hist_2B_eta_all', 'hist_2B_phi_all', 'hist_2B_pt', 'hist_2B_pt_all']
        hist_list4 = [self.hist_Z4B_pt, self.hist_Z4B_eta, self.hist_Z4B_phi, self.hist_Z4B_mass, self.hist_4B_eta, self.hist_4B_phi, self.hist_4B_mass, self.hist_4B_eta_all, self.hist_4B_phi_all, self.hist_4B_pt, self.hist_4B_pt_all]
        hist_list2 = [self.hist_Z2B_pt, self.hist_Z2B_eta, self.hist_Z2B_phi, self.hist_Z2B_mass, self.hist_2B_eta, self.hist_2B_phi, self.hist_2B_mass, self.hist_2B_eta_all, self.hist_2B_phi_all, self.hist_2B_pt, self.hist_2B_pt_all]
        labels = ['pT', 'eta', 'phi', 'mass', 'eta', 'phi', 'mass', 'eta', 'phi', 'pt', 'pt']
        titles = ['Z pT', 'Z eta', 'Z phi', 'Z mass', 'Gen B Group Eta', 'Gen B Group Phi', 'Gen B Group Mass', 'Gen B Eta', 'Gen B Phi', 'Gen B Group Pt', 'Gen B Pt']
        
        for n4, n2, h2, h4, xlabel, title in zip(name_list4, name_list2, hist_list2, hist_list4, labels, titles):
            fig, ax = plt.subplots(figsize=(8, 5))
            mplhep.histplot(h4, density=True, label="Z to 4B", ax=ax)
            mplhep.histplot(h2, density=True, label="Z to 2B", ax=ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Normalized Events')
            ax.set_title(title)
            ax.legend()
            #print(f'plots/{n4}_{self.key}.pdf')
            plt.savefig(f'plots/{n4}_{self.key}.pdf')
            plt.close(fig)
        
        name_list4 = ['hist_4B_pt_indiv', 'hist_4B_eta_indiv', 'hist_4B_phi_indiv', 'hist_4B_pt_indiv', 'hist_4B_eta_indiv', 'hist_4B_phi_indiv', 'hist_2B_pt_indiv', 'hist_2B_eta_indiv', 
                      'hist_2B_phi_indiv']
        name_list2 = ['hist_2B_pt_indiv', 'hist_2B_eta_indiv', 'hist_2B_phi_indiv', 'hist_2B_orig_pt_indiv', 'hist_2B_orig_eta_indiv', 'hist_2B_orig_phi_indiv', 'hist_2B_orig_pt_indiv', 
                      'hist_2B_orig_eta_indiv', 'hist_2B_orig_phi_indiv']
        hist_list4 = [self.hist_4B_pt_indiv, self.hist_4B_eta_indiv, self.hist_4B_phi_indiv, self.hist_4B_pt_indiv, self.hist_4B_eta_indiv, self.hist_4B_phi_indiv, self.hist_2B_pt_indiv, 
                      self.hist_2B_eta_indiv, self.hist_2B_phi_indiv]
        hist_list2 = [self.hist_2B_pt_indiv, self.hist_2B_eta_indiv, self.hist_2B_phi_indiv, self.hist_2B_orig_pt_indiv, self.hist_2B_orig_eta_indiv, self.hist_2B_orig_phi_indiv, 
                      self.hist_2B_orig_pt_indiv, self.hist_2B_orig_eta_indiv, self.hist_2B_orig_phi_indiv]
        labels = ['pt', 'eta', 'phi', 'pt', 'eta', 'phi', 'pt', 'eta', 'phi']
        titles = ['Gen B Indiv Pt', 'Gen B Indiv Eta', 'Gen B Indiv Phi', 'Gen B pt vs Initial State Bs', 'Gen B eta vs Initial State Bs', 'Gen B phi vs Initial State Bs', 'Gen B pt vs Initial State Bs', 'Gen B eta vs Initial State Bs', 'Gen B phi vs Initial State Bs']
        
        for n4, n2, h2, h4, xlabel, title in zip(name_list4, name_list2, hist_list2, hist_list4, labels, titles):
            l4 = len(h4)
            l2 = len(h2)
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(l4):
                if '2B' in n4:
                    mplhep.histplot(h4[i], density=True, label="Z to 2B final: "+str(i), ax=ax)
                else:
                    mplhep.histplot(h4[i], density=True, label="Z to 4B: "+str(i), ax=ax)
            for i in range(l2):
                if 'orig' in n2:
                    mplhep.histplot(h2[i], density=True, label="Z to 2B Init: "+str(i), ax=ax)
                else:
                    mplhep.histplot(h2[i], density=True, label="Z to 2B: "+str(i), ax=ax)
            ax.set_ylabel('Normalized Events')
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.legend()
            if 'orig' in n2:
                plt.savefig(f'plots/{n4}_orig_{self.key}.pdf')
            else:
                plt.savefig(f'plots/{n4}_{self.key}.pdf')
            plt.close(fig)


