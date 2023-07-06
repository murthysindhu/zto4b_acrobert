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
import os, psutil

plt.rcParams['text.usetex'] = True

def find_jets_in_group(jets, Bs4V, radbuffer=0.2):
    nB = len(Bs4V[0])
    if nB == 4:
        pairing = [[0,0,0,1,1,2],[1,2,3,2,3,3]]
    elif nB == 2:
        pairing = [[0],[1]]
    else:
        print(nB)
        raise ValueError 
    dR_arr = Bs4V[:,pairing[0]].delta_r(Bs4V[:,pairing[1]])
    max_dR = ak.max(dR_arr, axis=-1)

    has_jet = (ak.num(jets) >= 1)
    n_jets_in_group = ak.num(jets)
    jet_in_max = ak.all(ak.Array([Bs4V[has_jet][:,i].delta_r(jets[has_jet]) < (max_dR[has_jet]+radbuffer) for i in range(nB)]), axis=0)
    jets_in_group = jets[has_jet][jet_in_max]
    np.asarray(n_jets_in_group)[has_jet] = ak.num(jets_in_group)

    return jets_in_group, n_jets_in_group

class PlotsJet:

    def __init__(self, nbins=100, key='ZZTo4B01j_mc2017UL', bkg=False):
        self.key = key
        self.nbins = nbins
        self.bkg = bkg
        if self.bkg:
            self.ntypes = 1
            self.types = ['All']
        else:
            self.ntypes = 4
            self.types = ['Z-4B', 'Z-2Q2B', 'Z-4Q', 'Z-2B']
            
        self.hist_njets = [[Hist(hist.axis.Regular(20, -0.5, 19.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_nbjets = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_njets_pu0 = [[Hist(hist.axis.Regular(20, -0.5, 19.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_nbjets_pu0 = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]
        if not self.bkg:
            self.hist_njets_group = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_nbjets_group = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_nbbjets_group = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]

        self.hist_jetpt = [[Hist(hist.axis.Regular(self.nbins, 0, 250*(i+1), name='pt')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_jeteta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='eta')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_jetmass = [[Hist(hist.axis.Regular(self.nbins, 0, 150*(i+1), name='mass')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_jetbtag = [[Hist(hist.axis.Regular(self.nbins, 0, 1, name='btag')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_jetpuId = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='puId')) for i in range(3)] for j in range(self.ntypes)]
        if not self.bkg:
            self.hist_jetpt_group = [[Hist(hist.axis.Regular(self.nbins, 0, 250*(i+1), name='pt')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_jeteta_group = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='eta')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_jetmass_group = [[Hist(hist.axis.Regular(self.nbins, 0, 150*(i+1), name='mass')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_jetbtag_group = [[Hist(hist.axis.Regular(self.nbins, 0, 1, name='btag')) for i in range(3)] for j in range(self.ntypes)]

        self.hist_bjetpt = [[Hist(hist.axis.Regular(self.nbins, 0, 250*(i+1), name='pt')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_bjeteta = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='eta')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_bjetmass = [[Hist(hist.axis.Regular(self.nbins, 0, 150*(i+1), name='mass')) for i in range(3)] for j in range(self.ntypes)]
        self.hist_bjetpuId = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='puId')) for i in range(3)] for j in range(self.ntypes)]
        if not self.bkg:
            self.hist_bjetpt_group = [[Hist(hist.axis.Regular(self.nbins, 0, 250*(i+1), name='pt')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_bjeteta_group = [[Hist(hist.axis.Regular(self.nbins, -5, 5, name='eta')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_bjetmass_group = [[Hist(hist.axis.Regular(self.nbins, 0, 150*(i+1), name='mass')) for i in range(3)] for j in range(self.ntypes)]

        self.hist_m2bj = [Hist(hist.axis.Regular(self.nbins, 0, 150, name='mass')) for i in range(self.ntypes)]
        self.hist_m3bj = [Hist(hist.axis.Regular(self.nbins, 0, 150, name='mass')) for i in range(self.ntypes)]
        self.hist_m4bj = [Hist(hist.axis.Regular(self.nbins, 0, 150, name='mass')) for i in range(self.ntypes)]

        if not self.bkg:
            self.hist_njets_nbjets_group = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='njets', label='njets'), 
                                                  hist.axis.Regular(10, -0.5, 9.5, name='nbjets', label='nbjets')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_nbjets_nbbjets_group = [[Hist(hist.axis.Regular(10, -0.5, 9.5, name='nbjets', label='nbjets'), 
                                                    hist.axis.Regular(10, -0.5, 9.5, name='nbbjets', label='nbbjets')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_Z_GenHT_RecoHT = [Hist(hist.axis.Regular(self.nbins, 0, 400, name='genht', label=r'Gen H_{T}'), 
                                             hist.axis.Regular(self.nbins, 0, 400, name='recoht', label=r'Reco H_{T}')) for j in range(self.ntypes)]
            self.hist_dRmax_njets_group = [[Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR', label=r'\Delta R'), 
                                                 hist.axis.Regular(10, -0.5, 9.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]
            self.hist_dRmax_nbjets_group = [[Hist(hist.axis.Regular(self.nbins, 0, 4, name='dR', label=r'\Delta R'), 
                                                  hist.axis.Regular(10, -0.5, 9.5, name='njets')) for i in range(3)] for j in range(self.ntypes)]

        self.btagthresh = 0.6
        
    def _process(self, tagnano, index):
        
        jets = tagnano.Jet
        AK8jets = tagnano.FatJet
        AK15jets = tagnano.AK15Puppi

        jetlist = [jets, AK8jets, AK15jets]
        bjetlist = [jets[jets.btagDeepFlavB > self.btagthresh],
                    AK8jets[AK8jets.btagDeepB > self.btagthresh],
                    AK15jets[AK15jets.btagDeepB > self.btagthresh]]
        bbjetlist = [jets[(jets.btagDeepFlavB) > self.btagthresh],
                     AK8jets[(AK8jets.deepTagMD_probHbb + AK8jets.deepTagMD_probQCDbb + AK8jets.deepTagMD_probZbb) > self.btagthresh],
                     AK15jets[(AK15jets.ParticleNetMD_probQCDbb + AK15jets.ParticleNetMD_probXbb) > self.btagthresh]]
        btags = ['btagDeepFlavB', 'btagDeepB', 'btagDeepB']

        self.hist_m2bj[index].fill((bjetlist[0][ak.num(bjetlist[0]) >= 2][:,0] + bjetlist[0][ak.num(bjetlist[0]) >= 2][:,1]).mass)
        self.hist_m3bj[index].fill((bjetlist[0][ak.num(bjetlist[0]) >= 3][:,0] + bjetlist[0][ak.num(bjetlist[0]) >= 3][:,1] +
                                   bjetlist[0][ak.num(bjetlist[0]) >= 3][:,2]).mass)
        self.hist_m4bj[index].fill((bjetlist[0][ak.num(bjetlist[0]) >= 4][:,0] + bjetlist[0][ak.num(bjetlist[0]) >= 4][:,1] + 
                                    bjetlist[0][ak.num(bjetlist[0]) >= 4][:,2] + bjetlist[0][ak.num(bjetlist[0]) >= 4][:,3]).mass)

        for i in range(3):
            self.hist_njets[index][i].fill(ak.num(jetlist[i]))
            self.hist_nbjets[index][i].fill(ak.num(bjetlist[i]))

            self.hist_jetpt[index][i].fill(ak.flatten(jetlist[i].pt))
            self.hist_jeteta[index][i].fill(ak.flatten(jetlist[i].eta))
            self.hist_jetmass[index][i].fill(ak.flatten(jetlist[i].mass))
            self.hist_jetbtag[index][i].fill(ak.flatten(jetlist[i][btags[i]]))
            self.hist_bjetpt[index][i].fill(ak.flatten(bjetlist[i].pt))
            self.hist_bjeteta[index][i].fill(ak.flatten(bjetlist[i].eta))
            self.hist_bjetmass[index][i].fill(ak.flatten(bjetlist[i].mass))
            if i==0:
                self.hist_jetpuId[index][i].fill(ak.flatten(jetlist[i].puId))
                self.hist_bjetpuId[index][i].fill(ak.flatten(bjetlist[i].puId))
           
            if not self.bkg:
                nqs = 4 if index in [0,1,2] else 2
                if index == 0:
                    nbs = 4
                elif index == 1 or index == 3:
                    nbs = 2
                else:
                    nbs = 0
                Zevts = tagnano.Zs[:,0]
                #print(self.types[index], Zevts.final_Qs_pT[ak.num(Zevts.final_Qs_pT, axis=-1) > 4][0:5])
                if index == 0:
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
                #Qs_all4V = Qs_indiv4V[:,0]
                #for i in range(nqs - 1):
                #    Qs_all4V += Qs_indiv4V[:,i+1]

                if nqs == 4:
                    pairing = [[0,0,0,1,1,2],[1,2,3,2,3,3]]
                else:
                    pairing = [[0], [1]]

                dR_arr = Qs_indiv4V[:,pairing[0]].delta_r(Qs_indiv4V[:,pairing[1]])
                max_dR = ak.max(dR_arr, axis=-1)

                jets_in_group, n_jets_in_group = find_jets_in_group(jetlist[i], Qs_indiv4V)
                self.hist_njets_group[index][i].fill(n_jets_in_group)
                
                bjets_in_group, n_bjets_in_group = find_jets_in_group(bjetlist[i], Qs_indiv4V)
                self.hist_nbjets_group[index][i].fill(n_bjets_in_group)

                self.hist_jetpt_group[index][i].fill(ak.flatten(jets_in_group.pt))
                self.hist_jeteta_group[index][i].fill(ak.flatten(jets_in_group.eta))
                self.hist_jetmass_group[index][i].fill(ak.flatten(jets_in_group.mass))
                self.hist_jetbtag_group[index][i].fill(ak.flatten(jets_in_group[btags[i]]))
                
                self.hist_bjetpt_group[index][i].fill(ak.flatten(bjets_in_group.pt))
                self.hist_bjeteta_group[index][i].fill(ak.flatten(bjets_in_group.eta))
                self.hist_bjetmass_group[index][i].fill(ak.flatten(bjets_in_group.mass))

                self.hist_njets_nbjets_group[index][i].fill((n_jets_in_group - n_bjets_in_group), n_bjets_in_group)

                if i != 0:
                    bbjets_in_group, n_bbjets_in_group = find_jets_in_group(bbjetlist[i], Qs_indiv4V)
                    self.hist_nbbjets_group[index][i].fill(n_bbjets_in_group)

                    self.hist_nbjets_nbbjets_group[index][i].fill(n_bjets_in_group, n_bbjets_in_group)

                self.hist_dRmax_njets_group[index][i].fill(max_dR, n_jets_in_group)
                self.hist_dRmax_nbjets_group[index][i].fill(max_dR, n_bjets_in_group)

                if i == 0:
                    genht = ak.sum(Zevts.final_Qs_pT, axis=-1)
                    recoht = np.zeros((len(Zevts)))
                    recoht[ak.num(jets) >= 1] = ak.sum(jets_in_group.pt, axis=-1)
                    recoht = ak.Array(recoht)
                    self.hist_Z_GenHT_RecoHT[index].fill(genht, recoht)


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

        types_to_plot = [0,3] if not self.bkg else [0]

        name_list = ['hist_jetpt', 'hist_jeteta', 'hist_jetmass', 'hist_jetbtag', 'hist_bjetpt', 'hist_bjeteta', 'hist_bjetmass', 'hist_njets', 'hist_nbjets', 'hist_jetpuId', 'hist_bjetpuId', 'hist_njets_pu0', 'hist_nbjets_pu0']
        hist_list = [self.hist_jetpt, self.hist_jeteta, self.hist_jetmass, self.hist_jetbtag, self.hist_bjetpt, self.hist_bjeteta, self.hist_bjetmass, self.hist_njets, self.hist_nbjets, self.hist_jetpuId, self.hist_bjetpuId, self.hist_njets_pu0, self. hist_nbjets_pu0]
        labels = ['pt', 'eta', 'mass', 'btag', 'pt', 'eta', 'mass', 'njets', 'nbjets', 'puId', 'puId', 'njets', 'njets']
        titles = ['All Jet pt', 'All Jet eta', 'All Jet mass', 'All Jet b-tags', 'All B-Jet pt', 'All B-Jet eta', 'All B-Jet mass', 'NJets', 'NBJets', 'All Jet puId', 'All B-Jet puId', 'NJets puId=0, NBJets puId=0']
        jetlabels = ['AK4 Jet', 'AK8 Jet', 'AK15 Jet']
        
        for name, hist, xlabel, title in zip(name_list, hist_list, labels, titles):
            for i in range(3):
                fig, ax = plt.subplots(figsize=(8, 5))
                for j in types_to_plot:
                    mplhep.histplot(hist[j][i], density=True, label=self.types[j], ax=ax)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Normalized Events')
                ax.set_title(title+' '+jetlabels[i])
                ax.legend()
                plt.savefig(f'plots/{name}_{jetlabels[i].split(" ")[0]}_{self.key}.pdf')
                plt.close(fig)

        if self.bkg:
            return

        name_list = ['hist_jetpt_group', 'hist_jeteta_group', 'hist_jetmass_group', 'hist_jetbtag_group', 'hist_bjetpt_group', 'hist_bjeteta_group', 
                     'hist_bjetmass_group', 'hist_njets_group', 'hist_nbjets_group']
        hist_list = [self.hist_jetpt_group, self.hist_jeteta_group, self.hist_jetmass_group, self.hist_jetbtag_group, self.hist_bjetpt_group, 
                     self.hist_bjeteta_group, self.hist_bjetmass_group, self.hist_njets_group, self.hist_nbjets_group]
        labels = ['pt', 'eta', 'mass', 'btag', 'pt', 'eta', 'mass', 'bjets', 'njets', 'nbjets']
        titles = ['Group Jet pt', 'Group Jet eta', 'Group Jet mass', 'Group Jet b-tags', 'Group B-Jet pt', 'Group B-Jet eta', 'Group B-Jet mass', 'Group NJets', 'Group NBJets']
        jetlabels = ['AK4 Jet', 'AK8 Jet', 'AK15 Jet']
        
        for name, hist, xlabel, title in zip(name_list, hist_list, labels, titles):
            for i in range(3):
                fig, ax = plt.subplots(figsize=(8, 5))
                for j in types_to_plot:
                    mplhep.histplot(hist[j][i], density=True, label=self.types[j], ax=ax)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Normalized Events')
                ax.set_title(title+' '+jetlabels[i])
                ax.legend()
                plt.savefig(f'plots/{name}_{jetlabels[i].split(" ")[0]}_{self.key}.pdf')
                plt.close(fig)

        name_list = ['hist_m2bj', 'hist_m3bj', 'hist_m4bj']
        hist_list = [self.hist_m2bj, self.hist_m3bj, self.hist_m4bj]
        titles = ['2 B-Jet Mass', '3 B-Jet Mass', '4 B-Jet Mass']
        for name, hist, title in zip(name_list, hist_list, titles):
            fig, ax = plt.subplots(figsize=(8, 5))
            for j in types_to_plot:
                mplhep.histplot(hist[j], density=True, label=self.types[j], ax=ax)
            ax.set_xlabel('Mass [GeV]')
            ax.set_ylabel('Normalized Events')
            ax.set_title(title)
            ax.legend()
            plt.savefig(f'plots/{name}_AK4_{self.key}.pdf')
            plt.close(fig)

        name_list_eta_phi = ['hist_Z_GenHT_RecoHT', 'hist_Z_GenHT_RecoHT']
        hist_list_eta_phi = [self.hist_Z_GenHT_RecoHT, self.hist_Z_GenHT_RecoHT]
    
        for name,hist in zip(name_list_eta_phi, hist_list_eta_phi):
            for i in types_to_plot:
                fig, ax = plt.subplots(figsize=(8, 5))
                w, x, y = hist[i].to_numpy()
                mesh = ax.pcolormesh(x, y, w.T, cmap="RdYlBu")
                ax.set_xlabel(r"Gen $H_T$ [GeV]")
                ax.set_ylabel(r"Reco $H_T$ [GeV]")
                if '2B' in name:
                    ax.set_title(r"Reco $H_T$ vs Gen $H_T$ [{}]".format(self.types[i]))
                else:
                    ax.set_title(r"Reco $H_T$ vs Gen $H_T$ [{}]".format(self.types[i]))
                fig.colorbar(mesh)
                plt.savefig(f'plots/{name}_{self.key}.pdf')
                plt.close(fig)
        
        name_list_jets_mesh = ['hist_njets_nbjets_group', 'hist_nbjets_nbbjets_group', 'hist_dRmax_njets_group', 'hist_dRmax_nbjets_group']        
        hist_list_jets_mesh = [self.hist_njets_nbjets_group, self.hist_nbjets_nbbjets_group, self.hist_dRmax_njets_group, self.hist_dRmax_nbjets_group]
        jetlabels = ['AK4 Jets', 'AK8 Jets', 'AK15 Jets']
        for name, hist in zip(name_list_jets_mesh, hist_list_jets_mesh):
            for j in types_to_plot:
                for i in range(3):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if 'dRmax' in name:
                        w, x, y = hist[j][i].to_numpy()
                        mesh = ax.pcolormesh(x, y, w.T, cmap="RdYlBu")
                        ax.set_xlabel(r"$\Delta R_{max}$")
                        if 'njets' in name:
                            ax.set_title("$\Delta R_{max}$ vs Jets in Group: "+jetlabels[i])
                            ax.set_ylabel(r"N jets")
                        elif 'nbjets' in name:
                            ax.set_title(r"$\Delta R_{max}$ vs B-Jets in Group: "+jetlabels[i])
                            ax.set_ylabel(r"N b-jets")
                    elif 'nbjets' in name: 
                        w, x, y = hist[j][i].to_numpy()
                        mesh = ax.pcolormesh(x, y, w.T, cmap="RdYlBu")
                        if 'njets' in name:
                            ax.set_title("Jets vs B-Jets: "+jetlabels[i])
                            ax.set_xlabel(r"N jets (non-b)")
                            ax.set_ylabel(r"N b-jets")
                        elif 'nbbjets' in name:
                            ax.set_title("B-Jets vs BB-Jets: "+jetlabels[i])
                            ax.set_xlabel(r"N b-jets")
                            ax.set_ylabel(r"N bb-jets (QCD or X->bb)")
                    fig.colorbar(mesh)
                    plt.savefig(f'plots/{name}_{self.key}_{jetlabels[i].split(" ")[0]}.pdf')
                    plt.close(fig)
                
        
