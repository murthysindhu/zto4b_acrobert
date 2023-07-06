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

plt.rcParams['text.usetex'] = True

#key = 'ZZTo2Q2L_mc2017UL'

def match_with_jets(jets, Bs4V, jetradii=[0.4, 0.8, 1.5], radbuffer=0.2, exclusive=False):

    nB = len(Bs4V[0])
    nj = len(jets)

    has_jet = [(ak.num(jets[i]) >= 1) for i in range(nj)]
        
    jetdR = [ak.Array([Bs4V[has_jet[i]][:,j].delta_r(jets[i][has_jet[i]]) for j in range(nB)]) for i in range(nj)]
    jetmin = [ak.min(jetdR[i], axis=-1) for i in range(nj)]
    jetarg = [ak.argmin(jetdR[i], axis=-1) for i in range(nj)]

    evtbl = [(jetdR[i] < jetradii[i]+radbuffer) & (jetdR[i] == ak.min(jetdR[i], axis=-1)) for i in range(nj)]
    bs_per_jet = [ak.sum(evtbl[i], axis=0) for i in range(nj)]

    return evtbl, bs_per_jet


class PlotsSel:

    def __init__(self, key='ZZTo4B01j_mc2017UL', bkg=False):
        
        self.key = key
        self.nbins = 50
        self.bkg = bkg
        if self.bkg:
            self.ntypes = 1
            self.types = ['All']
        else:
            self.ntypes = 4
            self.types = ['Z-4B', 'Z-2Q2B', 'Z-4Q', 'Z-2B']

        self.hist_seljets_pt = [[Hist(hist.axis.Regular(self.nbins, 0, 250, name='jetpt')) for i in range(4)] for j in range(self.ntypes)]
        self.hist_seljets_eta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='jeteta')) for i in range(4)] for j in range(self.ntypes)]
        self.hist_seljets_btags = [[Hist(hist.axis.Regular(self.nbins, 0.5, 1, name='jetbtags')) for i in range(4)] for j in range(self.ntypes)]
        self.hist_seljets_dRmax = [[Hist(hist.axis.Regular(self.nbins, 0, 5, name='jetbbtags')) for i in range(4)] for j in range(self.ntypes)]
        self.hist_seljets_mass = [[Hist(hist.axis.Regular(self.nbins, 0, 500, name='jetbbtags')) for i in range(4)] for j in range(self.ntypes)]

        self.hist_2j_selleps_pt = [[Hist(hist.axis.Regular(self.nbins, 0, 250, name='leppt')) for i in range(2)] for j in range(self.ntypes)]
        self.hist_2j_selleps_eta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='phoeta')) for i in range(2)] for j in range(self.ntypes)]
        self.hist_2j_selphos_pt = [[Hist(hist.axis.Regular(self.nbins, 0, 250, name='phopt')) for i in range(2)] for j in range(self.ntypes)]
        self.hist_2j_selphos_eta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='phoeta')) for i in range(2)] for j in range(self.ntypes)]
        
        self.hist_3j_selleps_pt = [[Hist(hist.axis.Regular(self.nbins, 0, 250, name='leppt')) for i in range(2)] for j in range(self.ntypes)]
        self.hist_3j_selleps_eta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='phoeta')) for i in range(2)] for j in range(self.ntypes)]
        self.hist_3j_selphos_pt = [[Hist(hist.axis.Regular(self.nbins, 0, 250, name='phopt')) for i in range(2)] for j in range(self.ntypes)]
        self.hist_3j_selphos_eta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='phoeta')) for i in range(2)] for j in range(self.ntypes)]
        

        #self.hist_selleps_bdt
        self.lepptthresh = 30
        self.btagthresh = 0.6

    def _process(self, tagnano, index):

        # select 1L HLT
        tagnano = tagnano[tagnano.HLT.IsoMu24 | tagnano.HLT.Mu50 | tagnano.HLT.Ele27_WPTight_Gsf]

        #print(tagnano.SV.fields)
        jets = tagnano.Jet
        bjets_t = jets[jets.btagDeepFlavB > self.btagthresh]
        bjets = bjets_t.mask[ak.num(bjets_t) >= 1]

        SV = tagnano.SV.mask[ak.num(tagnano.SV) >= 1]
        SV4V = ak.zip({"pt": SV.pt, "eta": SV.eta, "phi": SV.phi, "mass": SV.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        SVjetdR = SV4V[:,0].delta_r(bjets) #bjets.delta_r(SV4V)
        isoSV = SV.mask[ak.min(SVjetdR, axis=-1) > 0.6]
        #print(SV.type, isoSV.type)

        # fill [1, 4] jets
        for i in range(4):

            #bjets_r = bjets[ak.where(ak.num(bjets) != None)[0]]
            selected = tagnano[ak.num(tagnano.Jet[tagnano.Jet.btagDeepFlavB > self.btagthresh]) >= i+1]
            if len(selected) == 0:
                continue
            thesebjets = selected.Jet[selected.Jet.btagDeepFlavB > self.btagthresh]
            thesebjets = thesebjets[ak.argsort(thesebjets.pt, ascending=False)]
            seljets = thesebjets[:, 0:i+1]

            self.hist_seljets_pt[index][i].fill(ak.flatten(seljets.pt))
            self.hist_seljets_eta[index][i].fill(ak.flatten(seljets.eta))
            self.hist_seljets_btags[index][i].fill(ak.flatten(seljets.btagDeepFlavB))

            if i != 0:

                if i == 1:
                    pairing = [[0], [1]]
                elif i == 2:
                    pairing = [[0,0,1], [1,2,2]]
                elif i == 3:
                    pairing = [[0,0,0,1,1,2], [1,2,3,2,3,3]]
                    
                dR_arr = seljets[:,pairing[0]].delta_r(seljets[:,pairing[1]])
                max_dR = ak.max(dR_arr, axis=-1)
                self.hist_seljets_dRmax[index][i].fill(max_dR[np.where(max_dR != None)[0]])
              
            if i == 0:
                group4V = seljets[:,0]
            elif i == 1:
                group4V = seljets[:,0] + seljets[:,1] 
            elif i == 2:
                group4V = seljets[:,0] + seljets[:,1] + seljets[:,2]
            elif i == 3:
                group4V = seljets[:,0] + seljets[:,1] + seljets[:,2] + seljets[:,3]

            self.hist_seljets_mass[index][i].fill(group4V.mass[np.where(group4V.mass != None)[0]])
                                                  
        # 2 jets fill pho, lep
        selected = tagnano[ak.num(bjets) >= 2]
        electrons = selected.Electron[selected.Electron.pt > self.lepptthresh]
        muons = selected.Muon[selected.Muon.pt > self.lepptthresh]
        photons = selected.Photon[selected.Photon.pt > self.lepptthresh]
            
        for i in range(2):
            
            self.hist_2j_selleps_pt[index][i].fill(ak.flatten(muons[ak.num(muons) == i+1].pt))
            self.hist_2j_selleps_eta[index][i].fill(ak.flatten(muons[ak.num(muons) == i+1].eta))
            self.hist_2j_selleps_pt[index][i].fill(ak.flatten(electrons[ak.num(electrons) == i+1].pt))
            self.hist_2j_selleps_eta[index][i].fill(ak.flatten(electrons[ak.num(electrons) == i+1].eta))
            self.hist_2j_selphos_pt[index][i].fill(ak.flatten(photons[ak.num(photons) == i+1].pt))
            self.hist_2j_selphos_eta[index][i].fill(ak.flatten(photons[ak.num(photons) == i+1].eta))

        # 3 jets fill pho, lep
        selected = tagnano[ak.num(bjets) >= 3]
        print('3j',len(selected))
        electrons = selected.Electron[selected.Electron.pt > self.lepptthresh]
        muons = selected.Muon[selected.Muon.pt > self.lepptthresh]
        photons = selected.Photon[selected.Photon.pt > self.lepptthresh]
            
        for i in range(2):
            
            self.hist_3j_selleps_pt[index][i].fill(ak.flatten(muons[ak.num(muons) == i+1].pt))
            self.hist_3j_selleps_eta[index][i].fill(ak.flatten(muons[ak.num(muons) == i+1].eta))
            self.hist_3j_selleps_pt[index][i].fill(ak.flatten(electrons[ak.num(electrons) == i+1].pt))
            self.hist_3j_selleps_eta[index][i].fill(ak.flatten(electrons[ak.num(electrons) == i+1].eta))
            self.hist_3j_selphos_pt[index][i].fill(ak.flatten(photons[ak.num(photons) == i+1].pt))
            self.hist_3j_selphos_eta[index][i].fill(ak.flatten(photons[ak.num(photons) == i+1].eta))


    def processAll(self, tagnano):
        
        assert self.bkg
        self._process(tagnano, 0)

    def process4B(self, tagnano):
        
        assert not self.bkg
        self._process(tagnano, 0)

    def process2Q2B(self, tagnano):

        assert not self.bkg
        self._process(tagnano, 1)

    def process4Q(self, tagnano):

        assert not self.bkg
        self._process(tagnano, 2)

    def process2B(self, tagnano):

        assert not self.bkg
        self._process(tagnano, 3)


    def plot(self):

        types_to_plot = [0,1,2,3] if not self.bkg else [0]

        name_list = ['hist_seljets_pt', 'hist_seljets_eta', 'hist_seljets_btags', 'hist_seljets_dRmax', 'hist_seljets_mass']
        hist_list = [self.hist_seljets_pt, self.hist_seljets_eta, self.hist_seljets_btags, self.hist_seljets_dRmax, self.hist_seljets_mass]
        labels = ['pt', 'eta', 'btag', 'dR', 'mass']
        titles = ['Selected AK4 Jet Pt', 'Selected AK4 Jet Eta', 'Selected AK4 Jet B-Tags', 'Selected AK4 Multijet deltaR', 'Selected AK4 Multijet Mass']

        for name, hist, xlabel, title in zip(name_list, hist_list, labels, titles):
            for i in types_to_plot:
                fig, ax = plt.subplots(figsize=(8, 5))
                for j in range(4):
                    mplhep.histplot(hist[i][j], density=True, label="njets="+str(j+1), ax=ax)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Normalized Events')
                ax.set_title(title+" "+self.types[i])
                ax.legend()
                plt.savefig(f'plots/{name}_HLT1L_{self.types[i]}_{self.key}.pdf')
                plt.close(fig)


        name_list = ['hist_selleps_pt', 'hist_selleps_eta', 'hist_selphos_pt', 'hist_selphos_eta']
        hist_list_2j = [self.hist_2j_selleps_pt, self.hist_2j_selleps_eta, self.hist_2j_selphos_pt, self.hist_2j_selphos_eta,]
        hist_list_3j = [self.hist_3j_selleps_pt, self.hist_3j_selleps_eta, self.hist_3j_selphos_pt, self.hist_3j_selphos_eta,]
        labels = ['pt', 'eta', 'pt', 'eta']
        titles = ['Selected Lepton Pt', 'Selected Lepton Eta', 'Selected Photon Pt', 'Selected Photon Eta']

        for name, hist2, hist3, xlabel, title in zip(name_list, hist_list_2j, hist_list_3j, labels, titles):
            for i in types_to_plot:
                fig, ax = plt.subplots(figsize=(8, 5))
                for j in range(2):
                    mplhep.histplot(hist2[i][j], density=True, label="2j n="+str(j+1), ax=ax)
                    mplhep.histplot(hist3[i][j], density=True, label="3j n="+str(j+1), ax=ax)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Normalized Events')
                ax.set_title(title+" "+self.types[i])
                ax.legend()
                plt.savefig(f'plots/{name}_HLT1L_{self.types[i]}_{self.key}.pdf')
                plt.close(fig)

