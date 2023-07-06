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

    nj = len(jets)
    has_jet = [(ak.num(jets[i]) >= 1) for i in range(nj)]
    bjetmat = [ak.cartesian([Bs4V, jets[i]], axis=1, nested=True) for i in range(nj)]
    jetdR = [bjetmat[i]['0'].delta_r(bjetmat[i]['1']) for i in range(nj)] 
    evtbl = [(jetdR[i] < jetradii[i]+radbuffer) & (jetdR[i] == ak.min(jetdR[i], axis=-1)) for i in range(nj)]
    bs_per_jet = [ak.sum(evtbl[i], axis=1) for i in range(nj)]

    #print(evtbl[0].type, bs_per_jet[0].type)
    return evtbl, bs_per_jet


def match_with_SVjets(jets, SV, Bs4V, jetradii=[0.4, 0.8, 1.5], radbuffer=0.2, exclusive=False):

    nB = len(Bs4V[0])
    nj = len(jets)
    
    has_jet = [(ak.num(jets[i]) >= 1) for i  in range(nj)]
    has_SV = ak.num(SV) >= 1

    jetdR = [ak.Array([Bs4V.mask[has_jet[i]][:,j].delta_r(jets[i].mask[has_jet[i]]) for j in range(nB)]) for i in range(nj)]
    evtbl = [(jetdR[i] < jetradii[i]+radbuffer) & (jetdR[i] == ak.min(jetdR[i], axis=-1)) for i in range(nj)]
    bs_per_jet = [ak.sum(evtbl[i], axis=0) for i in range(nj)]

    SVdR = ak.Array([Bs4V.mask[has_SV][:,j].delta_r(SV.mask[has_SV]) for j in range(nB)])
    evtblSV = (SVdR < jetradii[0]+radbuffer) & (SVdR == ak.min(SVdR, axis=-1))
    bs_per_SV = ak.sum(evtblSV, axis=0)
    
    return evtbl, bs_per_jet, evtblSV, bs_per_SV

def match_with_any(jets, SV, saj, Bs4V, jetradii=[0.4, 0.8, 1.5], radbuffer=0.2, exclusive=False):

    nB = len(Bs4V[0])
    nj = len(jets)

    has_jet = [(ak.num(jets[i]) >= 1) for i in range(nj)]
    has_SV = ak.num(SV) >= 1
    has_saj = ak.num(saj) > 1

    jetdR = [ak.Array([Bs4V.mask[has_jet[i]][:,j].delta_r(jets[i].mask[has_jet[i]]) for j in range(nB)]) for i in range(nj)]
    evtbl = [(jetdR[i] < jetradii[i]+radbuffer) & (jetdR[i] == ak.min(jetdR[i], axis=-1)) for i in range(nj)]
    bs_per_jet = [ak.sum(evtbl[i], axis=0) for i in range(nj)]

    SVdR = ak.Array([Bs4V.mask[has_SV][:,j].delta_r(SV.mask[has_SV]) for j in range(nB)])
    evtblSV = (SVdR < jetradii[0]+radbuffer) & (SVdR == ak.min(SVdR, axis=-1))
    bs_per_SV = ak.sum(evtblSV, axis=0)

    sajdR = ak.Array([Bs4V.mask[has_saj][:,j].delta_r(saj.mask[has_saj]) for j in range(nB)])
    evtblsaj = (sajdR < jetradii[0]+radbuffer) & (sajdR == ak.min(sajdR, axis=-1))
    bs_per_saj = ak.sum(evtblsaj, axis=0)

    return evtbl, bs_per_jet, evtblSV, bs_per_SV, evtblsaj, bs_per_saj

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

class PlotsMatch:

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

        self.hist_nbs_matched = [[Hist(hist.axis.Regular(5, -0.5, 4.5, name='nbs')) for i in range(3)] for k in range(self.ntypes)]
        self.hist_njets_matched = [[Hist(hist.axis.Regular(5, -0.5, 4.5, name='njets')) for i in range(3)] for k in range(self.ntypes)]
        self.hist_matchedbs_pt = [[Hist(hist.axis.Regular(self.nbins, 0, 250, name='matchedbpt')) for i in range(3)] for k in range(self.ntypes)]
        self.hist_matchedbs_eta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='matchedbeta')) for i in range(3)] for k in range(self.ntypes)]
        self.hist_matchedjets_pt = [[[Hist(hist.axis.Regular(self.nbins, 0, 250*(j+1), name='matchedjetpt')) for i in range(5)] for j in range(3)] for k in range(self.ntypes)]
        self.hist_matchedjets_eta = [[[Hist(hist.axis.Regular(self.nbins, -5, 5, name='matchedjeteta')) for i in range(5)] for j in range(3)] for k in range(self.ntypes)]
        self.hist_matchedjets_btags = [[[Hist(hist.axis.Regular(self.nbins, 0, 1, name='matchedjetbtags')) for i in range(5)] for j in range(3)] for k in range(self.ntypes)]
        self.hist_matchedjets_bbtags = [[[Hist(hist.axis.Regular(self.nbins, 0, 1, name='matchedjetbbtags')) for i in range(5)] for j in range(3)] for k in range(self.ntypes)]
        self.hist_matchedjets_dRmax = [[[Hist(hist.axis.Regular(self.nbins, 0, 1, name='matchedjetbbtags')) for i in range(5)] for j in range(3)] for k in range(self.ntypes)]
        self.hist_matchedjets_mass = [[[Hist(hist.axis.Regular(self.nbins, 0, 1, name='matchedjetbbtags')) for i in range(5)] for j in range(3)] for k in range(self.ntypes)]
        self.hist_mostbs_2ndbs = [[Hist(hist.axis.Regular(5, -0.5, 4.5, name='nbs', label='nbs'), 
                                        hist.axis.Regular(5, -0.5, 4.5, name='2nbs', label='2nbs')) for i in range(3)] for k in range(self.ntypes)]
        self.hist_matchedset = [[Hist(hist.axis.Regular(12, -0.5, 11.5, name='type')) for i in range(3)] for k in range(self.ntypes)]
        self.hist_matchedsetobj = [Hist(hist.axis.Regular(12, -0.5, 11.5, name='type')) for k in range(self.ntypes)]

    def _process(self, tagnano, index):
        
        jets = tagnano.Jet
        AK8jets = tagnano.FatJet
        AK15jets = tagnano.AK15Puppi

        btagfields = ['btagDeepFlavB', 'btagDeepB', 'btagDeepB']
        bbtagfields = ['', 'deepTagMD_probQCDbb', 'ParticleNetMD_probQCDbb']

        nqs = 4 if index in [0,1,2] else 2
        if index == 0:
            nbs = 4
        elif index == 1 or index == 3:
            nbs = 2
        else:
            nbs = 0
        Zevts = tagnano.Zs[:,0]
        
        if index == 0 or index == 3:
            bpdg = abs(Zevts.final_Qs_pdgId) == 5
            Qs_indiv4V = ak.zip({"pt": Zevts.final_Qs_pT[bpdg], "eta": Zevts.final_Qs_eta[bpdg], "phi": Zevts.final_Qs_phi[bpdg], "energy": Zevts.final_Qs_E[bpdg]},
                                with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        else:
            Qs_indiv4V = ak.zip({"pt": Zevts.final_Qs_pT, "eta": Zevts.final_Qs_eta, "phi": Zevts.final_Qs_phi, "energy": Zevts.final_Qs_E},
                                with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        ptsort = ak.argsort(Qs_indiv4V.pt, ascending=False)
        Qs_indiv4V = Qs_indiv4V[ptsort]

        if nqs == 4:
            Qs_all4V = Qs_indiv4V[:,0] + Qs_indiv4V[:,1] + Qs_indiv4V[:,2] + Qs_indiv4V[:,3]
        elif nqs == 2:
            Qs_all4V = Qs_indiv4V[:,0] + Qs_indiv4V[:,1]
                
        if nqs == 4:
            pairing = [[0,0,0,1,1,2],[1,2,3,2,3,3]]
        else:
            pairing = [[0], [1]]

        dR_arr = Qs_indiv4V[:,pairing[0]].delta_r(Qs_indiv4V[:,pairing[1]])
        max_dR = ak.max(dR_arr, axis=-1)

        jetlist = []
        idxs = []
        if ak.sum(ak.num(jets) >= 1) > 0:
            jetlist.append(jets)
            idxs.append(0)
        if ak.sum(ak.num(AK8jets) >= 1) > 0:
            jetlist.append(AK8jets)
            idxs.append(1)
        #if ak.sum(ak.num(AK15jets) >= 1) > 0:
        #    jetlist.append(AK15jets)
        #    idxs.append(2)
        
        #print('combining', index)
        obj_4V, obj_idx, obj_type = object_combiner([jetlist[0], tagnano.SoftActivityJet, tagnano.SV])
        b_to_obj_matches, bs_per_obj = match_with_4V(obj_4V, Qs_indiv4V)
        
        b_to_jet_matches, bs_per_jet = match_with_jets(jetlist, Qs_indiv4V)
        bpjsort = [ak.argsort(bs_per_jet[idxs[i]], ascending=False) for i in range(len(jetlist))]
        sorted_bs_per_jet = [bs_per_jet[idxs[i]][bpjsort[idxs[i]]] for i in range(len(jetlist))]

        #bs_per_jet = ak.sum(evtbl, axis=0)
        b_is_matched = [ak.any(b_to_jet_matches[idxs[i]], axis=-1) for i in range(len(jetlist))]

        sorted_bs_per_obj = bs_per_obj[ak.argsort(bs_per_obj, ascending=False)]

        if len(sorted_bs_per_obj) > 0:
            
            n1 = ak.sum(sorted_bs_per_obj == 1, axis=-1)
            n2 = ak.sum(sorted_bs_per_obj == 2, axis=-1)
            n3 = ak.sum(sorted_bs_per_obj == 3, axis=-1)
            n4 = ak.sum(sorted_bs_per_obj == 4, axis=-1)

            self.hist_matchedsetobj[index].fill(0, weight=ak.sum((n1==0) & (n2==0) & (n3==0) & (n4 == 0)))
            self.hist_matchedsetobj[index].fill(1, weight=ak.sum((n1==1) & (n2==0) & (n3==0)))
            self.hist_matchedsetobj[index].fill(2, weight=ak.sum((n2==1) & (n1==0)))
            self.hist_matchedsetobj[index].fill(3, weight=ak.sum((n1==2) & (n2==0)))
            self.hist_matchedsetobj[index].fill(4, weight=ak.sum((n3==1) & (n1==0)))
            self.hist_matchedsetobj[index].fill(5, weight=ak.sum((n2==1) & (n1==1)))
            self.hist_matchedsetobj[index].fill(6, weight=ak.sum((n1==3)))
            self.hist_matchedsetobj[index].fill(7, weight=ak.sum((n4==1)))
            self.hist_matchedsetobj[index].fill(8, weight=ak.sum((n3==1) & (n1==1)))
            self.hist_matchedsetobj[index].fill(9, weight=ak.sum((n2==2)))
            self.hist_matchedsetobj[index].fill(10, weight=ak.sum((n2==1) & (n1==2)))
            self.hist_matchedsetobj[index].fill(11, weight=ak.sum((n1==4)))
            
        for i in range(len(jetlist)):
            
            has_jet = (ak.num(jetlist[idxs[i]]) >= 1)
            nojets = np.zeros((len(jetlist[idxs[i]])))[~has_jet]
            self.hist_nbs_matched[index][idxs[i]].fill(nojets)
            self.hist_mostbs_2ndbs[index][idxs[i]].fill(nojets, nojets)
            self.hist_njets_matched[index][idxs[i]].fill(nojets)

            if len(sorted_bs_per_jet[idxs[i]]) > 0:
                njets = ak.num(sorted_bs_per_jet[idxs[i]])
                mostbs = np.zeros(len(sorted_bs_per_jet[idxs[i]]))
                mostbs[njets > 1] = sorted_bs_per_jet[idxs[i]][njets > 1][:,0]
                secondbs = np.zeros(len(sorted_bs_per_jet[idxs[i]]))
                secondbs[njets > 1] = sorted_bs_per_jet[idxs[i]][njets > 1][:,1]
                self.hist_mostbs_2ndbs[index][idxs[i]].fill(mostbs, secondbs)

                n1 = ak.sum(sorted_bs_per_jet[idxs[i]] == 1, axis=-1)
                n2 = ak.sum(sorted_bs_per_jet[idxs[i]] == 2, axis=-1)
                n3 = ak.sum(sorted_bs_per_jet[idxs[i]] == 3, axis=-1)
                n4 = ak.sum(sorted_bs_per_jet[idxs[i]] == 4, axis=-1)

                self.hist_matchedset[index][idxs[i]].fill(0, weight=ak.sum((n1==0) & (n2==0) & (n3==0) & (n4==0)))
                self.hist_matchedset[index][idxs[i]].fill(0, weight=len(nojets))
                self.hist_matchedset[index][idxs[i]].fill(1, weight=ak.sum((n1==1) & (n2==0) & (n3==0)))
                self.hist_matchedset[index][idxs[i]].fill(2, weight=ak.sum((n2==1) & (n1==0)))
                self.hist_matchedset[index][idxs[i]].fill(3, weight=ak.sum((n1==2) & (n2==0)))
                self.hist_matchedset[index][idxs[i]].fill(4, weight=ak.sum((n3==1) & (n1==0)))
                self.hist_matchedset[index][idxs[i]].fill(5, weight=ak.sum((n2==1) & (n1==1)))
                self.hist_matchedset[index][idxs[i]].fill(6, weight=ak.sum((n1==3)))
                self.hist_matchedset[index][idxs[i]].fill(7, weight=ak.sum((n4==1)))
                self.hist_matchedset[index][idxs[i]].fill(8, weight=ak.sum((n3==1) & (n1==1)))
                self.hist_matchedset[index][idxs[i]].fill(9, weight=ak.sum((n2==2)))
                self.hist_matchedset[index][idxs[i]].fill(10, weight=ak.sum((n2==1) & (n1==2)))
                self.hist_matchedset[index][idxs[i]].fill(11, weight=ak.sum((n1==4)))

            self.hist_nbs_matched[index][idxs[i]].fill(ak.sum(bs_per_jet[idxs[i]], axis=-1))
            self.hist_njets_matched[index][idxs[i]].fill(ak.sum(bs_per_jet[idxs[i]] > 0, axis=-1))

            nm_1 = ak.sum(bs_per_jet[idxs[i]] > 0, axis=-1) == 1
            one_match = jetlist[idxs[i]][nm_1][bs_per_jet[idxs[i]][nm_1] > 0]
            if len(one_match) > 0:
                self.hist_matchedjets_dRmax[index][idxs[i]][0].fill(one_match[:,0].pt*0.)
                self.hist_matchedjets_dRmax[index][idxs[i]][4].fill(one_match[:,0].pt*0.)
                self.hist_matchedjets_mass[index][idxs[i]][0].fill(one_match[:,0].mass)
                self.hist_matchedjets_mass[index][idxs[i]][4].fill(one_match[:,0].mass)

            nm_2 = ak.sum(bs_per_jet[idxs[i]] > 0, axis=-1) == 2
            two_matches = jetlist[idxs[i]][nm_2][bs_per_jet[idxs[i]][nm_2] > 0]
            if len(two_matches) > 0:
                self.hist_matchedjets_dRmax[index][idxs[i]][1].fill(two_matches[:,0].delta_r(two_matches[:,1]))
                self.hist_matchedjets_dRmax[index][idxs[i]][4].fill(two_matches[:,0].delta_r(two_matches[:,1]))
                self.hist_matchedjets_mass[index][idxs[i]][1].fill((two_matches[:,0] + two_matches[:,1]).mass)
                self.hist_matchedjets_mass[index][idxs[i]][4].fill((two_matches[:,0] + two_matches[:,1]).mass)

            nm_3 = ak.sum(bs_per_jet[idxs[i]] > 0, axis=-1) == 3
            three_matches = jetlist[idxs[i]][nm_3][bs_per_jet[idxs[i]][nm_3] > 0]
            pairing = [[0,0,1],[1,2,2]]
            if len(three_matches) > 0:
                dR_arr = three_matches[:,pairing[0]].delta_r(three_matches[:,pairing[1]])
                self.hist_matchedjets_dRmax[index][idxs[i]][2].fill(ak.max(dR_arr, axis=-1))
                self.hist_matchedjets_dRmax[index][idxs[i]][4].fill(ak.max(dR_arr, axis=-1))
                self.hist_matchedjets_mass[index][idxs[i]][2].fill((three_matches[:,0] + three_matches[:,1] + three_matches[:,2]).mass)
                self.hist_matchedjets_mass[index][idxs[i]][4].fill((three_matches[:,0] + three_matches[:,1] + three_matches[:,2]).mass)

            nm_4 = ak.sum(bs_per_jet[idxs[i]] > 0, axis=-1) == 4
            four_matches = jetlist[idxs[i]][nm_4][bs_per_jet[idxs[i]][nm_4] > 0]
            pairing = [[0,0,0,1,1,2],[1,2,3,2,3,3]]
            if len(four_matches) > 0:
                dR_arr = four_matches[:,pairing[0]].delta_r(four_matches[:,pairing[1]])
                self.hist_matchedjets_dRmax[index][idxs[i]][3].fill(ak.max(dR_arr, axis=-1))
                self.hist_matchedjets_dRmax[index][idxs[i]][4].fill(ak.max(dR_arr, axis=-1))
                self.hist_matchedjets_mass[index][idxs[i]][3].fill((four_matches[:,0] + four_matches[:,1] + four_matches[:,2] + four_matches[:,3]).mass)
                self.hist_matchedjets_mass[index][idxs[i]][4].fill((four_matches[:,0] + four_matches[:,1] + four_matches[:,2] + four_matches[:,3]).mass)

            for j in range(5):
                has_jet = ak.num(jetlist[idxs[i]]) > 0
                self.hist_matchedjets_pt[index][idxs[i]][j].fill(ak.flatten(jetlist[idxs[i]][has_jet][bs_per_jet[idxs[i]][has_jet] == j].pt))
                self.hist_matchedjets_eta[index][idxs[i]][j].fill(ak.flatten(jetlist[idxs[i]][has_jet][bs_per_jet[idxs[i]][has_jet] == j].eta))
                self.hist_matchedjets_btags[index][idxs[i]][j].fill(ak.flatten(jetlist[idxs[i]][has_jet][bs_per_jet[idxs[i]][has_jet] == j][btagfields[idxs[i]]]))
                if i != 0:
                    self.hist_matchedjets_bbtags[index][idxs[i]][j].fill(ak.flatten(jetlist[idxs[i]][has_jet][bs_per_jet[idxs[i]][has_jet] == j][bbtagfields[idxs[i]]]))

        b_is_matched = [ak.any(b_to_jet_matches[idxs[i]], axis=-1) for i in range(len(jetlist))]
        for i in range(2 if index == 3 else 4):
            for j in range(len(jetlist)):
                self.hist_matchedbs_pt[index][idxs[j]].fill(Qs_indiv4V[ak.num(jetlist[j]) > 0][:,i][b_is_matched[j][:,i][ak.num(jetlist[j]) > 0]].pt)
                self.hist_matchedbs_eta[index][idxs[j]].fill(Qs_indiv4V[ak.num(jetlist[j]) > 0][:,i][b_is_matched[j][:,i][ak.num(jetlist[j]) > 0]].eta)



    def processAll(self, tagnano):
        pass

    def process4B(self, tagnano): #, Zfriend):
        self._process(tagnano, 0)

    def process2B(self, tagnano):
        self._process(tagnano, 3)

    def process2Q2B(self, tagnano):
        self._process(tagnano,1) 

    def process4Q(self, tagnano):
        self._process(tagnano,2) 

    def plot(self):

        name_list_matches = ['hist_matchedjets_pt', 'hist_matchedjets_eta', 'hist_matchedjets_btags', 'hist_matchedjets_bbtags']
        hist_list_matches = [self.hist_matchedjets_pt, self.hist_matchedjets_eta, self.hist_matchedjets_btags, self.hist_matchedjets_bbtags]
        quants = ['pt', 'eta', 'btag', 'bbtag', 'pt', 'eta', 'btag', 'bbtag']
        jetlabels = ['AK4 Jets', 'AK8 Jets', 'AK15 Jets']
        
        for name, hist, quant in zip(name_list_matches, hist_list_matches, quants):
            for k in range(self.ntypes):
                for i in range(3):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    nj = 5 if '4B' in name else 3
                    for j in range(nj):
                        counts, bins = hist[k][i][j].to_numpy()
                        plt.hist(bins[:-1], bins, weights=counts, label="nb = "+str(j), density=True, histtype='step')
                    ax.set_title('Matched Jet '+quant+' '+' '+self.types[k]+' '+jetlabels[i])
                    ax.set_xlabel(quant)
                    ax.set_ylabel('Normalized Events')
                    ax.legend()
                    plt.savefig(f'plots/{name}_{self.key}_{self.types[k]}_{jetlabels[i].split(" ")[0]}.pdf')
                    plt.close(fig)


        
        name4 = ['hist_matchedbs_pt', 'hist_matchedbs_eta', 'hist_nbs_matched', 'hist_njets_matched']
        hist4 = [self.hist_matchedbs_pt, self.hist_matchedbs_eta, self.hist_nbs_matched, self.hist_njets_matched]
        quants = ['pt', 'eta', 'nbs', 'njets']
        jetlabels = ['AK4 Jets', 'AK8 Jets', 'AK15 Jets']
        jetcolors = ['r','b','g']
        
        for n4, h4, quant in zip(name4, hist4, quants):
            if 'matchedbs' in n4:
                fig, ax = plt.subplots(figsize=(8, 5))
                for i in range(3):
                    for j in range(self.ntypes):
                        counts, bins = h4[j][i].to_numpy()
                        plt.hist(bins[:-1], bins, weights=counts, label='Z-4B: '+self.types[j]+' '+jetlabels[i], color=jetcolors[i], density=True, histtype='step', lw=1)
                ax.set_title('Matched B '+quant)
                ax.set_xlabel(quant)
                ax.set_ylabel('Normalized Events')
                ax.legend()
                plt.savefig(f'plots/{n4}_{self.key}.pdf')
                plt.close(fig)
            elif 'nbs_matched' in n4 or 'njets_matched' in n4:
                fig, ax = plt.subplots(figsize=(8, 5))
                for i in range(3):
                    for j in range(self.ntypes):
                        counts, bins = h4[j][i].to_numpy()
                        plt.hist(bins[:-1], bins, weights=counts, label='Z-4B: '+self.types[j]+' '+jetlabels[i], density=True, color=jetcolors[i], histtype='step', lw=1)
                ax.set_title(quant+' matched')
                ax.set_xlabel(quant)
                ax.set_ylabel('Normalized Events')
                ax.legend()
                plt.savefig(f'plots/{n4}_{self.key}.pdf')
                plt.close(fig)
        
        jetlabels = ['AK4 Jets', 'AK8 Jets', 'AK15 Jets']
        for j in range(self.ntypes):
            for i in range(3):
                fig, ax = plt.subplots(figsize=(8, 5))
                w, x, y = self.hist_mostbs_2ndbs[j][i].to_numpy()
                mesh = ax.pcolormesh(x, y, w.T, cmap="RdYlBu")
                ax.set_title("Jets to Matched "+self.types[j]+" B Quarks: "+jetlabels[i])
                ax.set_xlabel("Leading N Matches")
                ax.set_ylabel("Subleading N Matches")
                fig.colorbar(mesh)
                plt.savefig(f'plots/hist_mostbs_2ndbs_{self.key}_{self.types[j]}_{jetlabels[i].split(" ")[0]}.pdf')
                plt.close(fig)

        shortkey = self.key.split("_")[0]                
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['[]', '[1]', '[2]', '[1,1]', '[3]', '[2,1]', '[1,1,1]', '[4]', '[3,1]', '[2,2]', '[2,1,1]', '[1,1,1,1]']
        for i in range(4):
            counts, bins = self.hist_matchedset[i][0].to_numpy()
            plt.hist(bins[:-1], bins, weights=counts, density=True, label = self.types[i], histtype='step')
        ax.set_title(f'Types of Jet-Matched Z Events: {shortkey}')
        ax.set_ylabel('Normalized Events')
        ax.xaxis.set_major_locator(ticker.NullLocator())
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x, label in zip(counts, bin_centers, labels):
            ax.annotate(label, xy=(x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -4), textcoords='offset points', va='top', ha='center')
        ax.legend()
        plt.savefig(f'plots/hist_jetmatchedset_{self.key}.pdf')

        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(4):
            counts, bins = self.hist_matchedsetobj[i].to_numpy()
            plt.hist(bins[:-1], bins, weights=counts, density=True, label = self.types[i], histtype='step')
        ax.set_title(f'Types of Obj-Matched Z Events: {shortkey}')
        ax.set_ylabel('Normalized Events')
        ax.xaxis.set_major_locator(ticker.NullLocator())
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x, label in zip(counts, bin_centers, labels):
            ax.annotate(label, xy=(x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -4), textcoords='offset points', va='top', ha='center')
        ax.legend()
        plt.savefig(f'plots/hist_objmatchedset_{self.key}.pdf')

        fig, ax = plt.subplots(figsize=(8, 5))
        counts, bins = self.hist_matchedsetobj[0].to_numpy()
        plt.hist(bins[:-1], bins, weights=counts, label='Jets, SAJ, SV', histtype='step')
        counts, bins = self.hist_matchedset[i][0].to_numpy()
        plt.hist(bins[:-1], bins, weights=counts, label='Jets only', histtype='step')
        ax.set_title(f'Types of Matched Z-4B Events: {shortkey}')
        ax.set_ylabel('Events')
        ax.xaxis.set_major_locator(ticker.NullLocator())
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x, label in zip(counts, bin_centers, labels):
            ax.annotate(label, xy=(x, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -4), textcoords='offset points', va='top', ha='center')
            ax.legend()
        plt.savefig(f'plots/hist_matchedsetcomp_{self.key}.pdf')


        name_list_matchesnjets = ['hist_matchedjets_dRmax', 'hist_matchedjets_mass']
        hist_list_matchesnjets = [self.hist_matchedjets_dRmax, self.hist_matchedjets_mass]
        labels = ['2 Matched Jets', '3 Matched Jets', '4 Matched Jets', 'All Events']
        titles = ['Matched Jets dRmax', 'Matched Jets Mass']
        xaxes = ['dRmax', 'jet group mass [GeV]']
        for hist, name, title, xax in zip(hist_list_matchesnjets, name_list_matchesnjets, titles, xaxes):
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(4):
                mplhep.histplot(hist[0][0][i], density=True, label=labels[i], ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xax)
            ax.set_ylabel('Normalized Events')
            plt.savefig(f'plots/{name}.pdf')
