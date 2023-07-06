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

tmr = timer()

class PlotsDR:

    def __init__(self, key = 'ZZTo4B01j_mc2017UL'):

        self.nbins = 100
        self.key = key
        self.hist_4B_dRcone = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_4B_dRmax = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_4B_dR_all = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_4B_dR_indiv = [Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR')) for i in range(6)]
        self.hist_4B_Z_dR_all = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_4B_Z_dR_indiv = [Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR')) for i in range(4)]

        self.hist_Z_4B_pt_dR = Hist(hist.axis.Regular(self.nbins, 0, 300, name='pt', label=r'p_{T}'), hist.axis.Regular(self.nbins, 0, 4, name='dR', label=r'\Delta R'))
        self.hist_4B_pt_ZdR = Hist(hist.axis.Regular(self.nbins, 0, 300, name='pt', label=r'p_{T}'), hist.axis.Regular(self.nbins, 0, 4, name='dR', label=r'\Delta R'))

        self.hist_2B_dRcone = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_2B_dRmax = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_2B_dR_all = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_2B_dR_indiv = [Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))]
        self.hist_2B_Z_dR_all = Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR'))
        self.hist_2B_Z_dR_indiv = [Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR')) for i in range(2)]
        self.hist_2B_Z_dR_orig_indiv = [Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR')) for i in range(2)]

        self.hist_Z_2B_pt_dR = Hist(hist.axis.Regular(self.nbins, 0, 300, name='pt', label=r'p_{T}'), hist.axis.Regular(self.nbins, 0, 4, name='dR', label=r'\Delta R'))
        self.hist_2B_pt_ZdR = Hist(hist.axis.Regular(self.nbins, 0, 300, name='pt', label=r'p_{T}'), hist.axis.Regular(self.nbins, 0, 4, name='dR', label=r'\Delta R'))

    def process4B(self, tagnano):
        
        zto4b_evts = tagnano.Zs#[hasZ4B]#np.logical_and(hasZ4B, np.logical_not(twoZ4B))]

        Zsto4b = zto4b_evts[:,0]

        Bs_4B4V = ak.zip({"pt": Zsto4b.final_Bs_pT, "eta": Zsto4b.final_Bs_eta, "phi": Zsto4b.final_Bs_phi, "energy": Zsto4b.final_Bs_E}, with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        Z_4V = ak.zip({"pt": Zsto4b.pT, "eta": Zsto4b.eta, "phi": Zsto4b.phi, "energy": Zsto4b.E}, with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        ZdR_arr = Z_4V.delta_r(Bs_4B4V)
        self.hist_4B_pt_ZdR.fill(ak.flatten(Zsto4b.final_Bs_pT), ak.flatten(ZdR_arr))

        ptsort = ak.argsort(Bs_4B4V.pt, ascending=False)
        Bs_4B4V = Bs_4B4V[ptsort]
        
        Bs_all4V = Bs_4B4V[:,0] + Bs_4B4V[:,1] + Bs_4B4V[:,2] + Bs_4B4V[:,3]
        
        group_radius = ak.max(Bs_all4V.delta_r(Bs_4B4V), axis=-1)
        self.hist_4B_dRcone.fill(group_radius)
        
        pairing = [[0,0,0,1,1,2],[1,2,3,2,3,3]]
        
        dR_arr = Bs_4B4V[:,pairing[0]].delta_r(Bs_4B4V[:,pairing[1]])
        max_dR = ak.max(dR_arr, axis=-1)
        self.hist_4B_dRmax.fill(max_dR)
        self.hist_4B_dR_all.fill(ak.flatten(dR_arr))
        
        dRsort = ak.argsort(dR_arr)
        dR_arr = dR_arr[dRsort]
        for i in range(6):
            self.hist_4B_dR_indiv[i].fill(dR_arr[:,i])
        self.hist_Z_4B_pt_dR.fill(Zsto4b.pT, max_dR)
        
        dRsort = ak.argsort(ZdR_arr)
        ZdR_arr = ZdR_arr[dRsort]
        for i in range(4):
            self.hist_4B_Z_dR_indiv[i].fill(ZdR_arr[:,i])
        self.hist_4B_Z_dR_all.fill(ak.flatten(ZdR_arr))

    def process2B(self, tagnano):
    
        zto2b_evts = tagnano.Zs#[hasZ2B]
        
        Z1s = zto2b_evts[:,0]
        #Z2s = zto2b_evts[:,1]
        
        Bs_2B4V = ak.zip({"pt": Z1s.final_Bs_pT, "eta": Z1s.final_Bs_eta, "phi": Z1s.final_Bs_phi, "mass": Z1s.final_Bs_mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        Z_4V = ak.zip({"pt": Z1s.pT, "eta": Z1s.eta, "phi": Z1s.phi, "energy": Z1s.E}, with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        ZdR_arr = Z_4V.delta_r(Bs_2B4V)
        self.hist_2B_pt_ZdR.fill(ak.flatten(Z1s.final_Bs_pT), ak.flatten(ZdR_arr))

        ptsort = ak.argsort(Bs_2B4V.pt, ascending=False)
        Bs_2B4V = Bs_2B4V[ptsort]
        Bs_all4V = Bs_2B4V[:,0] + Bs_2B4V[:,1]
        
        group_radius = ak.max(Bs_all4V.delta_r(Bs_2B4V), axis=-1)
        self.hist_2B_dRcone.fill(group_radius)
        self.hist_2B_dR_all.fill(Bs_2B4V[:,0].delta_r(Bs_2B4V[:,1]))
        self.hist_2B_dRmax.fill(Bs_2B4V[:,0].delta_r(Bs_2B4V[:,1]))
        self.hist_2B_dR_indiv[0].fill(Bs_2B4V[:,0].delta_r(Bs_2B4V[:,1]))
        
        max_dR = Bs_2B4V[:,0].delta_r(Bs_2B4V[:,1])
        
        self.hist_Z_2B_pt_dR.fill(Z1s.pT, max_dR)
        
        dRsort = ak.argsort(ZdR_arr)
        ZdR_arr = ZdR_arr[dRsort]
        for i in range(2):
            self.hist_2B_Z_dR_indiv[i].fill(ZdR_arr[:,i])
        self.hist_2B_Z_dR_all.fill(ak.flatten(ZdR_arr))
        
        has2origB = (ak.num(Z1s.orig_Bs_pT) == 2)
        Bs_2B4V = ak.zip({"pt": Z1s.orig_Bs_pT[has2origB], "eta": Z1s.orig_Bs_eta[has2origB], "phi": Z1s.orig_Bs_phi[has2origB], "mass": Z1s.orig_Bs_mass[has2origB]}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        Z_4V = ak.zip({"pt": Z1s.pT[has2origB], "eta": Z1s.eta[has2origB], "phi": Z1s.phi[has2origB], "energy": Z1s.E[has2origB]}, with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        ZdR_arr = Z_4V.delta_r(Bs_2B4V)
        
        dRsort = ak.argsort(ZdR_arr)
        ZdR_arr = ZdR_arr[dRsort]
        #print(ZdR_arr.type)
        for i in range(2):
            self.hist_2B_Z_dR_orig_indiv[i].fill(ZdR_arr[:,i])
    
    def plot(self):

        name_list_eta_phi = ['hist_Z_4B_pt_dR', 'hist_Z_2B_pt_dR', 'hist_4B_pt_ZdR', 'hist_2B_pt_ZdR']
        hist_list_eta_phi = [self.hist_Z_4B_pt_dR, self.hist_Z_2B_pt_dR, self.hist_4B_pt_ZdR, self.hist_2B_pt_ZdR]
        titles = ['Z pt vs group dR [Z-4B]', 'Z pt vs group dR [Z-2B]', 'B pt vs dR(Z, B) [Z-4B]', 'B pt vs dR(Z, B) [Z-2B]']
        for name, hist, title in zip(name_list_eta_phi, hist_list_eta_phi, titles):
            fig, ax = plt.subplots(figsize=(8, 5))
            w, x, y = hist.to_numpy()
            mesh = ax.pcolormesh(x, y, w.T, cmap="RdYlBu")
            ax.set_xlabel(r"$p_{T}$ [GeV]")
            ax.set_ylabel(r"$\Delta R$")
            ax.set_title(title)
            fig.colorbar(mesh)
            plt.savefig(f'plots/{name}_{self.key}.pdf')
            plt.close(fig)
        
        name_list4 = ['hist_4B_dRcone', 'hist_4B_dR_all', 'hist_4B_dRmax', 'hist_4B_Z_dR_all']
        name_list2 = ['hist_2B_dRcone', 'hist_2B_dR_all', 'hist_2B_dRmax', 'hist_2B_Z_dR_all']
        hist_list4 = [self.hist_4B_dRcone, self.hist_4B_dR_all, self.hist_4B_dRmax, self.hist_4B_Z_dR_all]
        hist_list2 = [self.hist_2B_dRcone, self.hist_2B_dR_all, self.hist_2B_dRmax, self.hist_2B_Z_dR_all]
        labels = ['dR', 'dR', 'dR', 'dR']
        titles = ['Gen B dR Cone', 'Gen B Pair dR', 'Gen B dR Max', 'Gen Z to B dR']
        
        for n4, n2, h2, h4, xlabel, title in zip(name_list4, name_list2, hist_list2, hist_list4, labels, titles):
            fig, ax = plt.subplots(figsize=(8, 5))
            mplhep.histplot(h4, density=True, label="Z to 4B", ax=ax)
            mplhep.histplot(h2, density=True, label="Z to 2B", ax=ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Normalized Events')
            ax.set_title(title)
            ax.legend()
            plt.savefig(f'plots/{n4}_{self.key}.pdf')
            plt.close(fig)
        
        name_list4 = ['hist_4B_dR_indiv', 'hist_4B_Z_dR_indiv', 'hist_2B_Z_dR_orig_indiv']
        name_list2 = ['hist_2B_dR_indiv', 'hist_2B_Z_dR_indiv', 'hist_2B_Z_dR_indiv']
        hist_list4 = [self.hist_4B_dR_indiv, self.hist_4B_Z_dR_indiv, self.hist_2B_Z_dR_orig_indiv]
        hist_list2 = [self.hist_2B_dR_indiv, self.hist_2B_Z_dR_indiv, self.hist_2B_Z_dR_indiv]
        labels = ['dR', 'dR', 'dR']
        titles = ['Gen B Pairwise dR', 'Gen Z to B dR', 'Gen Z to B dR (initial vs final)']
        
        for n2, n4, h2, h4, xlabel, title in zip(name_list2, name_list4, hist_list2, hist_list4, labels, titles):
            l4 = len(h4)
            l2 = len(h2)
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(l4):
                if 'orig' in n4:
                    mplhep.histplot(h4[i], density=True, label="Z to 2B initial: "+str(i), ax=ax)
                else:
                    mplhep.histplot(h4[i], density=True, label="Z to 4B: "+str(i), ax=ax)
            for i in range(l2):
                mplhep.histplot(h2[i], density=True, label="Z to 2B: "+str(i), ax=ax)
            ax.set_ylabel('Normalized Events')
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.legend()
            plt.savefig(f'plots/{n4}_{self.key}.pdf')
            plt.close(fig)

