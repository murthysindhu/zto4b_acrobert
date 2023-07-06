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

def object_combiner(obj_list, redundant_rad = 0.4):

    for obj in obj_list:
        if 'mass' not in obj.fields:
            obj['mass'] = ak.zeros_like(obj)

    list4V = [ak.zip({"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior) for obj in obj_list]
    listidx = [ak.local_index(i4V, axis=1) for i4V in list4V]

    finlist4V = [list4V[0]]
    finlistidx = [listidx[0]]
    finlisttype = [ak.zeros_like(list4V[0].pt)]

    for i in range(1, len(list4V)):
        before = finlist4V[0] if i == 1 else ak.concatenate(finlist4V, axis=1)
        objmat = ak.cartesian([list4V[i], before], axis=1, nested=True)
        matdR = objmat['0'].delta_r(objmat['1'])
        redundant_obj = ak.min(matdR, axis=2) < 0.4

        finlist4V.append(list4V[i][~redundant_obj])
        finlistidx.append(listidx[i][~redundant_obj])
        finlisttype.append(ak.full_like(finlist4V[i].pt, i))

    obj_4V = ak.concatenate(finlist4V, axis=1)
    obj_idx = ak.concatenate(finlistidx, axis=1)
    obj_type = ak.concatenate(finlisttype, axis=1)

    return obj_4V, obj_idx, obj_type

def match_with_4V(obj_4V, Bs4V, jetrad=0.4, radbuffer=0.2):

    has_obj = ak.num(obj_4V) >= 1
    bobjmat = ak.cartesian([Bs4V, obj_4V], axis=1, nested=True)
    bobjdR = bobjmat['0'].delta_r(bobjmat['1'])
    evtbl = (bobjdR < jetrad+radbuffer) & (bobjdR == ak.min(bobjdR, axis=-1))
    bs_per_obj = ak.sum(evtbl, axis=1)

    return evtbl, bs_per_obj



class PlotsTrig:

    def __init__(self, key = 'ZZTo4B01j_mc2017UL', bkg = False):

        self.nbins = 100
        self.key = key
        self.triggers = ["IsoMu24", "Mu50", "Ele27_WPTight_Gsf", "Photon200", "Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15", "AK8PFJet400_TrimMass30", "AK8PFJet500", "AK8PFJetFwd400", "PFJet500", "PFHT1050", "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", "PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0", "QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1", "DoublePFJets116MaxDeta1p6_DoubleCaloBTagCSV_p33", "DoubleEle33_CaloIdL_MW", "DiEle27_WPTightCaloOnly_L1DoubleEG", "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", "DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350", "ECALHT800", "DoubleL2Mu50", "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8", "Mu37_TkMu27", "DoubleMu20_7_Mass0to30_Photon23", "DoubleMu8_Mass8_PFHT350", "CaloMET350_HBHECleaned", "DiJet110_35_Mjj650_PFMET110", "MET105_IsoTrk50", "PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1", "PFMET250_HBHECleaned", "TripleJet110_35_35_Mjj650_PFMET110"]
        self.comb_triggers = {'single_lepton': self.triggers[0:3], 'double_lepton': self.triggers[14:17] + self.triggers[19:22], 'jet': self.triggers[5:9], 'HT': self.triggers[9:10], 
                              'multibtag': self.triggers[10:14], 'photon': self.triggers[3:5], 'met': self.triggers[24:31]}
        self.best_triggers = {'WZ_mc2017UL': 'single_lepton', 'ZZTo4B01j_mc2017UL': 'multibtag', 'ZZTo2Q2L_mc2017UL': 'single_lepton', 
                              'ZGammaToJJ_mc2017UL': 'photon', 'ZJets_HT_800toInf_mc2017UL': 'jet', 'ZJets_HT_600to800_mc2017UL': 'multibtag', 
                              'TTJets_mc2017UL': 'jet', 'WJetsToLNu_mc2017UL': 'jet'}
        # HLT_CaloMET350_HBHECleaned_v HLT_DiJet110_35_Mjj650_PFMET110_v HLT_MET105_IsoTrk50_v HLT_PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_v HLT_PFMET250_HBHECleaned_v HLT_TripleJet110_35_35_Mjj650_PFMET110_v
        self.bkg = bkg
        if self.bkg:
            self.ntypes = 1
        else:
            self.ntypes = 4
        self.hist_triggers = [Hist(hist.axis.Regular(len(self.triggers)+1, 0, len(self.triggers)+1, name='trig')) for i in range(self.ntypes)]
        self.hist_triggers_comb = [Hist(hist.axis.Regular(len(self.comb_triggers)+1, 0, len(self.comb_triggers)+1, name='trig_comb')) for i in range(self.ntypes)]

        self.wgts_all = np.zeros((self.ntypes))
        self.wgts_best_trig = np.zeros((self.ntypes))
        self.objs = ['nbjets','jets','leps','phos','sajs','SVs']

        self.hist_trig_nobjs = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name=f'n{obj}')) for obj in objs] for j in range(self.ntypes)]
        self.hist_trig_nbjets_nobjs = [[[Hist(hist.axis.Regular(10, -0.5, 9.5, name=f'n{obj}')) for obj in objs] for i in range(5)] for j in range(self.ntypes)]
        self.hist_trigsel_nbjets_nobjs = [[[Hist(hist.axis.Regular(10, -0.5, 9.5, name=f'n{obj}')) for obj in objs] for i in range(5)] for j in range(self.ntypes)]

        self.hist_trig_nmatched = [Hist(hist.axis.Regular(5, -0.5, 4.5, name='nmatched')) for i in range(self.ntypes)]
        
        self.hist_trigsel_nbjets = [Hist(hist.axis.Regular(10, -0.5, 9.5, name='nbjets')) for i in range(self.ntypes)]
        self.hist_trigsel_2bjets_nleps = [Hist(hist.axis.Regular(5, -0.5, 4.5, name='nleps')) for i in range(self.ntypes)] 
        self.hist_trigsel_3bjets_nleps = [Hist(hist.axis.Regular(5, -0.5, 4.5, name='nleps')) for i in range(self.ntypes)]
        self.hist_trigsel_4bjets_nleps = [Hist(hist.axis.Regular(5, -0.5, 4.5, name='nleps')) for i in range(self.ntypes)]
        self.hist_trigsel_2bjets_nphos = [Hist(hist.axis.Regular(5, -0.5, 4.5, name='nphos')) for i in range(self.ntypes)] 
        self.hist_trigsel_3bjets_nphos = [Hist(hist.axis.Regular(5, -0.5, 4.5, name='nphos')) for i in range(self.ntypes)]
        self.hist_trigsel_4bjets_nphos = [Hist(hist.axis.Regular(5, -0.5, 4.5, name='nphos')) for i in range(self.ntypes)]

        self.btagthresh = 0.6
        self.lepptthresh = 30

    def _process(self, tagnano, index):
        HLT = tagnano.HLT
        wgt = tagnano.genWeight
        self.wgts_all[index] += ak.sum(wgt)

        self.hist_triggers[index].fill(0, weight=ak.sum(wgt))
        for i in range(len(self.triggers)):
            trig = self.triggers[i]
            self.hist_triggers[index].fill(i+1, weight=ak.sum(wgt[HLT[trig]]))

        self.hist_triggers_comb[index].fill(0, weight=ak.sum(wgt))
        for i in range(len(list(self.comb_triggers.keys()))):
            key = list(self.comb_triggers.keys())[i]
            passes = np.zeros((len(tagnano)), dtype=bool)
            for trig in self.comb_triggers[key]:
                passes = passes | HLT[trig]
            self.hist_triggers_comb[index].fill(i+1, weight=ak.sum(wgt[passes]))

        passes_best = np.zeros((len(tagnano)), dtype=bool)
        for trig in self.comb_triggers[self.best_triggers[self.key]]:
            passes_best = passes_best | HLT[trig]

        passnano = tagnano[passes_best]
        self.wgts_best_trig[index] += ak.sum(passnano.genWeight)

        bjets = passnano.Jet[passnano.Jet.btagDeepFlavB > self.btagthresh]
        jets = passnano.Jet
        photons = passnano.Photon
        electrons = passnano.Electron
        muons = passnano.Muon
        saj = passnano.SoftActivityJet
        SV = passnano.SV
        
        nums = [ak.num(bjets), ak.num(jets), ak.num(electrons) + ak.num(muons), ak.num(photons), ak.num(saj), ak.num(SV)]
        for i in range(6):
            self.hist_trig_nobjs[index][i].fill(nums[i])
        for i in range(5):
            for j in range(6):
                self.hist_trig_nbjets_nobjs[index][i][j].fill(ak.num(nums[j][ak.num(bjets) == i]))
        

        bjets_trig_2bjets = evt_trig_2bjets.Jet[evt_trig_2bjets.Jet.btagDeepFlavB > self.btagthresh]
        mass_trig_2bjets = (bjets_trig_2bjets[:,0] + bjets_trig_2bjets[:,1]).mass
        evt_sel_2bjets = evt_trig_2bjets[mass_trig_2bjets < 120.]

        bjets_trig_3bjets = evt_trig_3bjets.Jet[evt_trig_3bjets.Jet.btagDeepFlavB > self.btagthresh]
        mass_trig_3bjets = (bjets_trig_3bjets[:,0] + bjets_trig_3bjets[:,1] + bjets_trig_3bjets[:,2]).mass
        evt_sel_3bjets = evt_trig_3bjets[mass_trig_3bjets < 120.]

        bjets_trig_4bjets = evt_trig_4bjets.Jet[evt_trig_4bjets.Jet.btagDeepFlavB > self.btagthresh]
        mass_trig_4bjets = (bjets_trig_4bjets[:,0] + bjets_trig_4bjets[:,1] + bjets_trig_4bjets[:,2] + bjets_trig_4bjets[:,3]).mass
        evt_sel_4bjets = evt_trig_4bjets[mass_trig_4bjets < 120.]

        self.hist_trigsel_nbjets[index].fill(2, weight=ak.sum(evt_sel_2bjets.genWeight))
        self.hist_trigsel_nbjets[index].fill(3, weight=ak.sum(evt_sel_3bjets.genWeight))
        self.hist_trigsel_nbjets[index].fill(4, weight=ak.sum(evt_sel_4bjets.genWeight))

        self.hist_trigsel_2bjets_nleps[index].fill(ak.num(evt_sel_2bjets.Electron[evt_sel_2bjets.Electron.pt > self.lepptthresh]) +
                                                   ak.num(evt_sel_2bjets.Muon[evt_sel_2bjets.Muon.pt > self.lepptthresh]), weight=evt_sel_2bjets.genWeight)
        self.hist_trigsel_3bjets_nleps[index].fill(ak.num(evt_sel_3bjets.Electron[evt_sel_3bjets.Electron.pt > self.lepptthresh]) +
                                                   ak.num(evt_sel_3bjets.Muon[evt_sel_3bjets.Muon.pt > self.lepptthresh]), weight=evt_sel_3bjets.genWeight)
        self.hist_trigsel_4bjets_nleps[index].fill(ak.num(evt_sel_4bjets.Electron[evt_sel_4bjets.Electron.pt > self.lepptthresh]) +
                                                   ak.num(evt_sel_4bjets.Muon[evt_sel_4bjets.Muon.pt > self.lepptthresh]), weight=evt_sel_4bjets.genWeight)
        self.hist_trigsel_2bjets_nphos[index].fill(ak.num(evt_sel_2bjets.Photon[evt_sel_2bjets.Photon.pt > self.lepptthresh]), weight=evt_sel_2bjets.genWeight)
        self.hist_trigsel_3bjets_nphos[index].fill(ak.num(evt_sel_3bjets.Photon[evt_sel_3bjets.Photon.pt > self.lepptthresh]), weight=evt_sel_3bjets.genWeight)
        self.hist_trigsel_4bjets_nphos[index].fill(ak.num(evt_sel_4bjets.Photon[evt_sel_4bjets.Photon.pt > self.lepptthresh]), weight=evt_sel_4bjets.genWeight)

        
    def processAll(self, tagnano):
        # trigger rates for all relevant triggers
        # few different b-jet selections
        # compare b-jets with matched jets - find excess, deficits
        # ratio with bkg 4Q events
        # compare to bkg 4Q events
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

        bns = ['4B', '2B2Q', '4Q', '2B']
        if self.bkg:
            bns = ['bkg']

        print(self.ntypes, self.bkg)
        for j in range(self.ntypes):
            counts, bins = self.hist_triggers[j].to_numpy()
            s = bns[j]
            for i in range(len(counts)):
                if i == 0:
                    print(counts[0], 'all '+bns[j])
                    s += f"{counts[0]} all_{bns[j]}\n"
                else:
                    print(counts[i], counts[i]/self.wgts_all[j], self.triggers[i-1])
                    s += f"{counts[i]} {self.triggers[i-1]}\n"
            counts, bins = self.hist_triggers_comb[j].to_numpy()
            for i in range(len(counts)):
                if i > 0:
                    print(counts[i], counts[i]/self.wgts_all[j], list(self.comb_triggers.keys())[i-1])
                    s += f"{counts[i]} {list(self.comb_triggers.keys())[i-1]}\n"
            
            if j == 0:
                with open (f'trigs/{self.key}_trigs.txt', 'w') as fl:
                    fl.write(s)

        for i in range(self.ntypes):
            print(i, bns[i])
            counts, bins = self.hist_trig_nbjets[i].to_numpy()
            print('nbjets', counts/self.wgts_all[i])
            counts, bins = self.hist_trig_nleps[i].to_numpy()
            print('nleps', counts/self.wgts_all[i])        
            counts, bins = self.hist_trig_nphos[i].to_numpy()
            print('nphos', counts/self.wgts_all[i])
            
            counts, bins = self.hist_trig_2bjets_nphos[i].to_numpy()
            print('2bjet nphos', counts/self.wgts_all[i])
            counts, bins = self.hist_trig_3bjets_nphos[i].to_numpy()
            print('3bjet nphos', counts/self.wgts_all[i])
            counts, bins = self.hist_trig_4bjets_nphos[i].to_numpy()
            print('4bjet nphos', counts/self.wgts_all[i])
            counts, bins = self.hist_trig_2bjets_nleps[i].to_numpy()
            print('2bjet nleps', counts/self.wgts_all[i])
            counts, bins = self.hist_trig_3bjets_nleps[i].to_numpy()
            print('3bjet nleps', counts/self.wgts_all[i])
            counts, bins = self.hist_trig_4bjets_nleps[i].to_numpy()
            print('4bjet nleps', counts/self.wgts_all[i])
        
            counts, bins = self.hist_trigsel_nbjets[i].to_numpy()
            print('sel nbjets', counts/self.wgts_all[i])
            counts, bins = self.hist_trigsel_2bjets_nleps[i].to_numpy()
            print('2bjet sel nleps', counts/self.wgts_all[i])
            counts, bins = self.hist_trigsel_3bjets_nleps[i].to_numpy()
            print('3bjet sel nleps', counts/self.wgts_all[i])
            counts, bins = self.hist_trigsel_4bjets_nleps[i].to_numpy()
            print('4bjet sel nleps', counts/self.wgts_all[i])
            counts, bins = self.hist_trigsel_2bjets_nphos[i].to_numpy()
            print('2bjet sel nphos', counts/self.wgts_all[i])
            counts, bins = self.hist_trigsel_3bjets_nphos[i].to_numpy()
            print('3bjet sel nphos', counts/self.wgts_all[i])

