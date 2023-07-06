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
import matplotlib.colors as colors

plt.rcParams['text.usetex'] = True

#key = 'ZZTo2Q2L_mc2017UL'

def match_with_jets(jets, Bs4V, jetrad=0.4, radbuffer=0.2, exclusive=False):

    nB = len(Bs4V[0])
    nj = len(jets)

    has_jet = (ak.num(jets) >= 1)
        
    jetdR = ak.Array([Bs4V[has_jet][:,j].delta_r(jets[has_jet]) for j in range(nB)])
    jetmin = ak.min(jetdR, axis=-1)
    jetarg = ak.argmin(jetdR, axis=-1)

    evtbl = (jetdR < jetrad+radbuffer) & (jetdR == ak.min(jetdR, axis=-1))
    bs_per_jet = ak.sum(evtbl, axis=0) 

    return evtbl, bs_per_jet

def match_with_any(jets, SV, saj, Bs4V, jetrad=0.4, radbuffer=0.2, exclusive=False):

    nB = len(Bs4V[0])
    nj = len(jets)

    has_jet = (ak.num(jets) >= 1)
    has_SV = ak.num(SV) >= 1
    has_saj = ak.num(saj) > 1

    jetdR = ak.Array([Bs4V.mask[has_jet][:,j].delta_r(jets.mask[has_jet]) for j in range(nB)])
    evtbl = (jetdR < jetrad+radbuffer) & (jetdR == ak.min(jetdR, axis=-1))
    bs_per_jet = ak.sum(evtbl, axis=0)

    SVdR = ak.Array([Bs4V.mask[has_SV][:,j].delta_r(SV.mask[has_SV]) for j in range(nB)])
    evtblSV = (SVdR < jetrad+radbuffer) & (SVdR == ak.min(SVdR, axis=-1))
    bs_per_SV = ak.sum(evtblSV, axis=0)

    sajdR = ak.Array([Bs4V.mask[has_saj][:,j].delta_r(saj.mask[has_saj]) for j in range(nB)])
    evtblsaj = (sajdR < jetrad+radbuffer) & (sajdR == ak.min(sajdR, axis=-1))
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
    

def newwindow():
    wsize = 100
    pixdim = 0.01745
    return Hist(hist.axis.Regular(wsize, -wsize/2*pixdim, wsize/2*pixdim, name='phi', label='phi'), hist.axis.Regular(wsize, -wsize/2*pixdim, wsize/2*pixdim, name='eta', label='eta'))

class PlotsWindow:

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

        wsize = 100
        pixdim = 0.01745
        npx = 6
        self.npx = 6
        self.siderad = wsize/2.*pixdim
        self.window_allb =                        [newwindow() for i in range(npx)]
        self.window_sajmatchedb =                  [newwindow() for i in range(npx)]
        self.window_SVmatchedb =                  [newwindow() for i in range(npx)]
        self.window_unmatchedb =                  [newwindow() for i in range(npx)]
        self.window_jetmatchedb =                 [newwindow() for i in range(npx)]
        self.window_bpt0to20 =                    [newwindow() for i in range(npx)]
        self.window_bpt20toInf =                  [newwindow() for i in range(npx)]
        self.window_isob_pt0to20 =                [newwindow() for i in range(npx)]
        self.window_isob_pt20toInf =              [newwindow() for i in range(npx)]
                                        
        self.hist_pixvals_dists_all =             [Hist(hist.axis.Regular(50, 0., 50.), hist.axis.Regular(50, 0., 50.*pixdim)) for i in range(npx)]
        self.hist_pixvals_dists_jetmatched =      [Hist(hist.axis.Regular(50, 0., 50.), hist.axis.Regular(50, 0., 50.*pixdim)) for i in range(npx)]
        self.hist_pixvals_dists_SVmatched =       [Hist(hist.axis.Regular(50, 0., 50.), hist.axis.Regular(50, 0., 50.*pixdim)) for i in range(npx)]
        self.hist_pixvals_dists_sajmatched =       [Hist(hist.axis.Regular(50, 0., 50.), hist.axis.Regular(50, 0., 50.*pixdim)) for i in range(npx)]
        self.hist_pixvals_dists_unmatched =       [Hist(hist.axis.Regular(50, 0., 50.), hist.axis.Regular(50, 0., 50.*pixdim)) for i in range(npx)]
        self.hist_pixvals_dists_bpt0to20 =        [Hist(hist.axis.Regular(50, 0., 50.), hist.axis.Regular(50, 0., 50.*pixdim)) for i in range(npx)]
        self.hist_pixvals_dists_bpt20toInf =      [Hist(hist.axis.Regular(50, 0., 50.), hist.axis.Regular(50, 0., 50.*pixdim)) for i in range(npx)]
        self.hist_pixvals_all =                   [Hist(hist.axis.Regular(50, 0., 50. if i in list(range(2*npx)) else 1., name='pixval')) for i in range(npx*3)]
        self.hist_pixvals_jetmatched =            [Hist(hist.axis.Regular(50, 0., 50. if i in list(range(2*npx)) else 1., name='pixval')) for i in range(npx*3)]
        self.hist_pixvals_sajmatched =             [Hist(hist.axis.Regular(50, 0., 50. if i in list(range(2*npx)) else 1., name='pixval')) for i in range(npx*3)]
        self.hist_pixvals_SVmatched =             [Hist(hist.axis.Regular(50, 0., 50. if i in list(range(2*npx)) else 1., name='pixval')) for i in range(npx*3)]
        self.hist_pixvals_unmatched =             [Hist(hist.axis.Regular(50, 0., 50. if i in list(range(2*npx)) else 1., name='pixval')) for i in range(npx*3)]
        self.hist_pixvals_bpt0to20 =              [Hist(hist.axis.Regular(50, 0., 50. if i in list(range(2*npx)) else 1., name='pixval')) for i in range(npx*3)]
        self.hist_pixvals_bpt20toInf =            [Hist(hist.axis.Regular(50, 0., 50. if i in list(range(2*npx)) else 1., name='pixval')) for i in range(npx*3)]
        self.keys = ['all', 'bpt0to20', 'bpt20toInf', 'jetmatchedb', 'sajmatchedb', 'SVmatchedb', 'unmatchedb', 'isob_pt0to20', 'isob_pt20toInf']
        self.fields = ['pt', 'pt', 'pt', 'd0', 'z0', 'd0']
        self.nevts = {}
        for key in self.keys:
            self.nevts[key] = 0
        self.nevts_all = 0
        self.nevts_bpt0to20 = 0
        self.nevts_bpt20toInf = 0
    
    def _process(self, tagnano, index):

        jets = tagnano.Jet
        AK8jets = tagnano.FatJet
        AK15jets = tagnano.AK15Puppi

        btagfields = ['btagDeepFlavB', 'btagDeepB', 'btagDeepB']
        bbtagfields = ['', 'deepTagMD_probQCDbb', 'ParticleNetMD_probQCDbb']

        Zevts = tagnano.Zs[:,0]

        bpdg = abs(Zevts.final_Qs_pdgId) == 5
        Qs_indiv4V = ak.zip({"pt": Zevts.final_Qs_pT[bpdg], "eta": Zevts.final_Qs_eta[bpdg], "phi": Zevts.final_Qs_phi[bpdg], "energy": Zevts.final_Qs_E[bpdg]},
                            with_name="PtEtaPhiELorentzVector", behavior=vector.behavior)
        ptsort = ak.argsort(Qs_indiv4V.pt, ascending=False)
        Qs_indiv4V = Qs_indiv4V[ptsort]

        Qs_all4V = Qs_indiv4V[:,0] + Qs_indiv4V[:,1] + Qs_indiv4V[:,2] + Qs_indiv4V[:,3]
                        
        pairing = [[0,0,0,1,1,2],[1,2,3,2,3,3]]
        
        dR_arr = Qs_indiv4V[:,pairing[0]].delta_r(Qs_indiv4V[:,pairing[1]])
        max_dR = ak.max(dR_arr, axis=-1)

        pft_4V = ak.zip({"pt": tagnano.map.PFTracks_pT, "eta": tagnano.map.PFTracks_eta, "phi": tagnano.map.PFTracks_phi, "mass": ak.zeros_like(tagnano.map.PFTracks_pT),
                         "d0": tagnano.map.PFTracks_d0, "z0": tagnano.map.PFTracks_z0}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        rh_4V = ak.zip({"pt": tagnano.map.RecHits_E, "eta": tagnano.map.RecHits_eta, "phi": tagnano.map.RecHits_phi, "mass": ak.zeros_like(tagnano.map.RecHits_E)},
                        with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        #pfc_4V = ak.zip({"pt": tagnano.map.PFCands_pT, "eta": tagnano.map.PFCands_eta, "phi": tagnano.map.PFCands_phi, "mass": tagnano.map.PFCands_m},
        #                with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        pfc_4V = pft_4V

        haspft = ak.num(pft_4V) > 0
        hasrh = ak.num(rh_4V) > 0
        haspfc = ak.num(pfc_4V) > 0

        self.nevts['all'] += ak.sum(ak.num(Qs_indiv4V))
        self.nevts['bpt0to20'] += ak.sum(ak.num(Qs_indiv4V[Qs_indiv4V.pt < 20]))
        self.nevts['bpt20toInf'] += ak.sum(ak.num(Qs_indiv4V[Qs_indiv4V.pt > 20]))
        self.nevts_all += ak.sum(ak.num(Qs_indiv4V))
        self.nevts_bpt0to20 += ak.sum(ak.num(Qs_indiv4V[Qs_indiv4V.pt < 20]))
        self.nevts_bpt20toInf += ak.sum(ak.num(Qs_indiv4V[Qs_indiv4V.pt > 20]))
        
        b_to_jet_matches, bs_per_jet, b_to_SV_matches, bs_per_SV, b_to_saj_matches, bs_per_saj = match_with_any(jets, tagnano.SV, tagnano.SoftActivityJet, Qs_indiv4V)
        b_is_jetmatched = ak.any(b_to_jet_matches, axis=-1) 
        b_is_sajmatched = ak.any(b_to_saj_matches, axis=-1) & ~b_is_jetmatched
        b_is_SVmatched = ak.any(b_to_SV_matches, axis=-1) & ~b_is_jetmatched & ~b_is_sajmatched
        b_is_matched = b_is_jetmatched | b_is_SVmatched | b_is_sajmatched

        obj_4V, obj_idx, obj_type = object_combiner([jets, tagnano.SoftActivityJet, tagnano.SV])
        b_to_obj_matches, bs_per_obj = match_with_4V(obj_4V, Qs_indiv4V)
        b_is_jetmatched = ak.any(b_to_jet_matches, axis=-1) & obj_type[]
        


        self.nevts['jetmatchedb'] += ak.sum(b_is_jetmatched)
        self.nevts['sajmatchedb'] += ak.sum(b_is_sajmatched)
        self.nevts['SVmatchedb'] += ak.sum(b_is_SVmatched)
        self.nevts['unmatchedb'] += ak.sum(~b_is_matched)
        has_jet = ak.num(jets) >= 1
        
        px = [pft_4V[has_jet], rh_4V[has_jet], pfc_4V[has_jet], pft_4V[has_jet], pft_4V[has_jet], pft_4V[has_jet]]
        haspx = [haspft[has_jet], hasrh[has_jet], haspfc[has_jet], haspft[has_jet], haspft[has_jet], haspft[has_jet]]
        windows = [self.window_allb, self.window_bpt0to20, self.window_bpt20toInf, self.window_jetmatchedb, self.window_sajmatchedb, self.window_SVmatchedb, self.window_unmatchedb]
        hists = [self.hist_pixvals_all, self.hist_pixvals_bpt0to20, self.hist_pixvals_bpt20toInf, self.hist_pixvals_jetmatched, self.hist_pixvals_sajmatched, self.hist_pixvals_SVmatched, self.hist_pixvals_unmatched]
        hists2D = [self.hist_pixvals_dists_all, self.hist_pixvals_dists_bpt0to20, self.hist_pixvals_dists_bpt20toInf, self.hist_pixvals_dists_jetmatched, self.hist_pixvals_dists_sajmatched, self.hist_pixvals_dists_SVmatched, self.hist_pixvals_dists_unmatched]

        for i in range(4):
            
            theseQs = Qs_indiv4V[:,i][has_jet]
            conditions = [ak.full_like(theseQs.pt, True, dtype=bool), theseQs.pt < 20., theseQs.pt > 20., b_is_jetmatched[i], b_is_sajmatched[i], b_is_SVmatched[i], ~b_is_matched[i]]
           
            for j in range(len(conditions)):
                for k in range(self.npx):
                    thesebs = theseQs[haspx[k]][conditions[j][haspx[k]]]
                    thesepx = px[k][haspx[k]][conditions[j][haspx[k]]]

                    dphi = thesebs.phi - thesepx.phi
                    deta = thesebs.eta - thesepx.eta

                    dphi_fixed = ak.concatenate((dphi[abs(dphi) <= np.pi], dphi[dphi > np.pi] - 2*np.pi, dphi[dphi < -np.pi] + 2*np.pi), axis=1)
                    deta_fixed = ak.concatenate((deta[abs(dphi) <= np.pi], deta[dphi > np.pi], deta[dphi < -np.pi]), axis=1)
                    pxpt_fixed = ak.concatenate((thesepx[self.fields[k]][abs(dphi) <= np.pi], thesepx[self.fields[k]][dphi > np.pi], thesepx[self.fields[k]][dphi < -np.pi]), axis=1)
                    
                    #print(pxpt_fixed.type)
                    #print(len(ak.flatten(pxpt_fixed)))
                    
                    if k == 5:
                        pxpt_fixed = ak.concatenate((np.sqrt(thesepx['d0'][abs(dphi) <= np.pi]**2 + thesepx['z0'][abs(dphi) <= np.pi]**2),
                                                     np.sqrt(thesepx['d0'][dphi > np.pi]**2 + thesepx['z0'][dphi > np.pi]**2),
                                                     np.sqrt(thesepx['d0'][dphi < -np.pi]**2 + thesepx['z0'][dphi < -np.pi]**2)), axis=1)

                    windows[j][k].fill(ak.flatten(dphi_fixed), ak.flatten(deta_fixed), weight=ak.flatten(pxpt_fixed))
                    
                    inwindow = (abs(dphi_fixed) < 0.01745*50) & (abs(deta_fixed) < 0.01745*50)
                    pxdR = np.sqrt(dphi_fixed**2 + deta_fixed**2)

                    hists[j][k].fill(ak.flatten(pxpt_fixed[inwindow]))
                    hists[j][k+self.npx].fill(ak.flatten(pxpt_fixed[inwindow]/pxdR[inwindow]))
                    hists[j][k+2*self.npx].fill(ak.flatten(pxdR[inwindow]))

                    hists2D[j][k].fill(ak.flatten(pxpt_fixed[inwindow]), ak.flatten(pxdR[inwindow]))
                    
    def processAll(self, tagnano):

        pass

    def process4B(self, tagnano): #, Zfriend):
        
        self._process(tagnano, 0)
        
    def process2B(self, tagnano): #, Zfriend)

        pass

    def plot(self):

        #plt.rcParams['text.usetex'] = True
        btype = ['all', 'bpt0to20', 'bpt20toInf', 'jetmatchedb', 'sajmatchedb', 'SVmatchedb', 'unmatchedb']
        btitle = ['All', '$p_T < 20$', '$p_T > 20$', 'Jet Matched', 'SAJ Matched', 'SV Matched', 'Unmatched']
        pixtype = ['PFTracks', 'RecHits', 'PFCands - $p_T$', 'PFTracks - $d_{xy}$', 'PFTracks - $d_z$', 'PFTracks - IP']
        pixname = ['PFTracks', 'RecHits', 'PFCands-pt', 'PFTracks-dxy', 'PFTracks-dz', 'PFTracks-ip']
        unittype = ['$p_T$','E', '$p_T$', '$d_{xy}$', '$d_z$', 'IP']
        realunit = ['[GeV]', '[GeV]', '[GeV]', 'cm', 'cm', 'cm']
        hists = [self.window_allb, self.window_bpt0to20, self.window_bpt20toInf, self.window_jetmatchedb, self.window_sajmatchedb, self.window_SVmatchedb, self.window_unmatchedb]
        for i in range(len(hists)):
            for j in range(len(pixtype)):
                fig, ax = plt.subplots(figsize=(8, 5))
                w, x, y = hists[i][j].to_numpy()
                z = w.T/self.nevts[btype[i]]
                #print(z.min(), z.max())
                if j < 3:
                    mesh = ax.pcolormesh(x, y, z, cmap="RdYlBu",
                                         norm=colors.LogNorm(vmin=z.min()+0.001, vmax=z.max()+0.002))
                else:
                    mesh = ax.pcolormesh(x, y, z, cmap="RdYlBu")
                #ax.set_zscale("log")
                ax.set_xlabel(r"Relative $\phi$")
                ax.set_ylabel(r"Relative $\eta$")
                #ax.set_zlabel(f"{unittype[j]} [GeV] per event")
                title = f"B Window {pixtype[j]}: {btitle[i]} (N={self.nevts[btype[i]]})"
                ax.set_title(title)
                bar = plt.colorbar(mesh)
                bar.ax.set_ylabel(f"{unittype[j]} {realunit[j]} per event", rotation=270)
                
                plt.savefig(f'plots/window_{btype[i]}_{pixname[j]}_{self.key}.pdf')
                plt.close(fig)
                        
        hists = [self.hist_pixvals_all, self.hist_pixvals_bpt0to20, self.hist_pixvals_bpt20toInf, self.hist_pixvals_jetmatched, self.hist_pixvals_sajmatched, self.hist_pixvals_SVmatched, self.hist_pixvals_unmatched]
        for j in range(len(pixtype)):
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(len(hists)):
                mplhep.histplot(hists[i][j], density=True, label=btitle[i], ax=ax)
                
            ax.set_title(f'B Data Distributions: {pixtype[j]}')
            ax.set_xlabel(f'{unittype[j]} {realunit[j]}')
            ax.set_ylabel('Arbitrary Unit')
            plt.gca().set_ylim(bottom=0.00001)
            ax.set_yscale('log')
            ax.legend()
            plt.savefig(f'plots/hist_pixvals_{pixname[j]}_{self.key}.pdf')
            plt.close(fig)

        for j in range(len(pixtype)):
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(len(hists)):
                mplhep.histplot(hists[i][j+2], density=True, label=btitle[i], ax=ax)

            ax.set_title(fr'B Data/$\Delta R$ Distributions: {pixtype[j]}')
            ax.set_xlabel(fr'{unittype[j]} {realunit[j]}/$\Delta R$')
            ax.set_ylabel('Arbitrary Unit')
            plt.gca().set_ylim(bottom=0.00001)
            ax.set_yscale('log')
            ax.legend()
            plt.savefig(f'plots/hist_pixvaldists_{pixname[j]}_{self.key}.pdf')
            plt.close(fig)
            
        for j in range(len(pixtype)):
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(len(hists)):
                mplhep.histplot(hists[i][j+4], density=True, label=btitle[i], ax=ax)

            ax.set_title(fr'B Data $\Delta R$ Distributions: {pixtype[j]}')
            ax.set_xlabel(fr'$\Delta R$')
            ax.set_ylabel('Arbitrary Unit')
            plt.gca().set_ylim(bottom=0.00001)
            ax.set_yscale('log')
            ax.legend()
            plt.savefig(f'plots/hist_pixdists_{pixname[j]}_{self.key}.pdf')
            plt.close(fig)
        
        hists = [self.hist_pixvals_dists_all, self.hist_pixvals_dists_bpt0to20, self.hist_pixvals_dists_bpt20toInf, self.hist_pixvals_dists_jetmatched, self.hist_pixvals_dists_sajmatched, self.hist_pixvals_dists_SVmatched, self.hist_pixvals_dists_unmatched]
        for i in range(len(hists)):
            for j in range(len(pixtype)):
                fig, ax = plt.subplots(figsize=(8, 5))
                w, x, y = hists[i][j].to_numpy()
                z = w.T
                mesh = ax.pcolormesh(x, y, z, cmap="RdYlBu",
                                     norm=colors.LogNorm(vmin=1., vmax=z.max()))
                ax.set_xlabel(fr"{unittype[j]} {realunit[j]}")
                ax.set_ylabel(r"$\Delta R$")
                title = f"B Window {pixtype[j]}: {btitle[i]} (N={self.nevts[btype[i]]})"
                ax.set_title(title)
                try:
                    plt.colorbar(mesh)
                except ValueError:
                    print("Skipping empty hist")
                    continue
                
                plt.savefig(f'plots/hist_pixvals_dists_{btype[i]}_{pixname[j]}_{self.key}.pdf')
                plt.close(fig)
