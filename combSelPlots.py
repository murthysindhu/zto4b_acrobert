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
from argparse import ArgumentParser
from classRunCoffea import RunCoffea
import os, psutil
import tracemalloc
import gc


#tracemalloc.start()
plt.rcParams['text.usetex'] = True

parser = ArgumentParser(prog = 'Friend Plotter',
                        description = 'Preliminary plots for nanotuples + friends')

parser.add_argument('-n', '--nfiles', default=1000, help='Number of file pairs to read', type=int)
parser.add_argument('-d', '--debug', action='store_true', default=False, help='Run the plots at the end or not')

tmr = timer()
args = parser.parse_args()
limit_nfiles = args.nfiles

keylist = ['ZZTo2Q2L', 'WZ', 'ZJets_HT_800toInf', 'ZGammaToJJ', 'TTJets', 'WJetsToLNu']
xs = [3.691, 27.64, 12.95, 4.145, 753.3, 66890.]
hlteffs = np.zeros((6))
seleffs = np.zeros((6))
hist_seljets_dRmax = [[Hist(hist.axis.Regular(50, 0, 5, name='jetbbtags')) for i in range(3)] for j in range(len(keylist))]
hist_seljets_mass = [[Hist(hist.axis.Regular(50, 0, 500, name='mass')) for i in range(3)] for j in range(len(keylist))]
hist_selevents_mass = [Hist(hist.axis.Regular(50, 0, 120, name='mass')) for i in range(len(keylist))]
hist_hltevents_nbjets = [Hist(hist.axis.Regular(6, -0.5, 5.5, name='nbjets')) for i in range(len(keylist))]

for j in range(len(keylist)):
    
    print('Dataset: '+keylist[j]+'_mc2017UL')
    #plotters = RunCoffea.initplotters(keylist[j]+'_mc2017UL', 'Sel')
    filepairs = RunCoffea.initnanos(keylist[j]+'_mc2017UL')
    nfilled = 0

    effhlt = np.zeros((min(limit_nfiles, len(filepairs))))
    effsel = np.zeros((min(limit_nfiles, len(filepairs))))

    for i in range(len(filepairs)):
 
        nanofile, friendfile = filepairs[i]
        tagnano = RunCoffea.loadpair(nanofile, friendfile)
        
        if tagnano is None:
            continue
        if 'Zs' not in tagnano.fields and j in list(range(4)):
            continue

        totlen = ak.sum(tagnano.genWeight)
        
        if j < 4:
            tagnano = tagnano[tagnano.Zs.nB[:,0] == 4]
            
        tagnano = tagnano[tagnano.HLT.IsoMu24 | tagnano.HLT.Mu50 | tagnano.HLT.Ele27_WPTight_Gsf]


        if len(tagnano) == 0:
            continue
        #print(tagnano.Jet.fields)
        jets = tagnano.Jet
        jets = jets[(jets.pt >= 30.) & (abs(jets.eta) <= 2.5) & (jets.puId > 4)]
        bjets = jets[jets.btagDeepFlavB > 0.6]
        thesebjets = bjets[ak.argsort(bjets.pt, ascending=False)]

        print(len(tagnano[ak.num(jets) > 0]))

        hist_hltevents_nbjets[j].fill(ak.num(thesebjets))

        nfilled += 1
        sellen = ak.sum(tagnano.genWeight)
        effhlt[nfilled-1] = sellen/totlen
        print(f'Running file pair {i}; elapsed time {round(tmr.time()/60., 3)}')
        
        #seljets = thesebjets[ak.num(thesebjets) == 2][:, 0:2]
        #
        #pairing = [[0], [1]]
        #dR_arr = seljets[:,pairing[0]].delta_r(seljets[:,pairing[1]])
        #max_dR = ak.max(dR_arr, axis=-1)
        #hist_seljets_dRmax[j][0].fill(max_dR)
        #
        #group4V = seljets[:,0] + seljets[:,1]
        #hist_seljets_mass[j][0].fill(group4V.mass)
        #
        #seljets = thesebjets[ak.num(thesebjets) == 3][:, 0:3]
        #
        #pairing = [[0,0,1], [1,2,2]]
        #dR_arr = seljets[:,pairing[0]].delta_r(seljets[:,pairing[1]])
        #max_dR = ak.max(dR_arr, axis=-1)
        #hist_seljets_dRmax[j][1].fill(max_dR)
        #
        #group4V = seljets[:,0] + seljets[:,1] + seljets[:,2]
        #hist_seljets_mass[j][1].fill(group4V.mass)
        #
        #seljets = thesebjets[ak.num(thesebjets) >= 4][:, 0:4]
        #
        #pairing = [[0,0,0,1,1,2], [1,2,3,2,3,3]]
        #dR_arr = seljets[:,pairing[0]].delta_r(seljets[:,pairing[1]])
        #max_dR = ak.max(dR_arr, axis=-1)
        #hist_seljets_dRmax[j][2].fill(max_dR)
        #
        #group4V = seljets[:,0] + seljets[:,1] + seljets[:,2] + seljets[:,3]
        #hist_seljets_mass[j][2].fill(group4V.mass)

        #RunCoffea.runZs(tagnano, plotters, args.key, totwgt=totwgt)

        tagnano = tagnano[(ak.num(thesebjets) == 2) & (ak.num(jets) == 2)]
        jets = tagnano.Jet
        bjets = jets[jets.btagDeepFlavB > 0.6]

        print(len(tagnano))
        
        if len(tagnano[ak.num(bjets) > 0]) != 0:
            thesebjets = bjets[ak.argsort(bjets.pt, ascending=False)]
            
            seljets = thesebjets

            group4V = seljets[:,0] + seljets[:,1]
            hist_selevents_mass[j].fill(group4V.mass[group4V.mass <= 120.])
            
            sellen = ak.sum(tagnano[group4V.mass <= 120.].genWeight)
            effsel[nfilled-1] = sellen/totlen

            print(keylist[j], i, len(tagnano[group4V.mass <= 120.]))
        else:
            effsel[nfilled-1] = 0.

        del tagnano
        print('Current memory:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        if nfilled == limit_nfiles:
            break
        gc.collect()
        
    print(keylist[j], np.mean(effhlt), np.mean(effsel))
    hlteffs[j] = np.mean(effhlt)
    seleffs[j] = np.mean(effsel)

if args.debug:
    exit()
print(f'Plotting; elapsed time {round(tmr.time()/60., 3)}')
#print(f'Total wgts {totwgt}')

#for i in range(3):
#    fig, ax = plt.subplots(figsize=(8, 5))
#    for j in range(len(keylist)):
#        if j > 4:
#            label = keylist[j].replace('_', ' ') 
#        else:
#            label = keylist[j].replace('_', ' ')+' Z-4B'
#        mplhep.histplot(hist_seljets_dRmax[j][i], density=True, label=label, ax=ax)
#    ax.set_xlabel('Multijet dR Diameter')
#    ax.set_ylabel('Normalized Events')
#    ax.set_title(f'1L HLT and {i+2} B-Jets dR')
#    ax.legend()
#    plt.savefig(f'plots/hist_seljets_dRmax_HLT1L_{i+2}bjets_comp.pdf')
#    plt.close(fig)
#
#for i in range(3):
#    fig, ax = plt.subplots(figsize=(8, 5))
#    for j in range(len(keylist)):
#        if j > 4:
#            label = keylist[j].replace('_', ' ')
#        else:
#            label = keylist[j].replace('_', ' ')+' Z-4B'
#        mplhep.histplot(hist_seljets_mass[j][i], density=True, label=label, ax=ax)
#    ax.set_xlabel('Multijet Mass')
#    ax.set_ylabel('Normalized Events')
#    ax.set_title(f'1L HLT and {i+2} B-Jets Mass')
#    ax.legend()
#    plt.savefig(f'plots/hist_seljets_mass_HLT1L_{i+2}bjets_comp.pdf')
#    plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 5))
for j in range(len(keylist)):
    if j > 3:
        label = keylist[j].replace('_', ' ')
    else:
        label = keylist[j].replace('_', ' ')+' Z-4B'
    mplhep.histplot(hist_selevents_mass[j], density=True, label=label, ax=ax)
plt.gca().set_ylim(bottom=0.00001)
ax.set_yscale('log')
ax.set_xlabel('Multijet Mass')
ax.set_ylabel('Normalized Events')
ax.set_title(r'1L HLT, 2 B-Jets Mass ($m_{jj} < 120$)')
ax.legend()
plt.savefig(f'plots/hist_selevents_HLT1L_ex2bjets_mass_comp_norm.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for j in range(len(keylist)):
    if j > 3:
        label = keylist[j].replace('_', ' ')
    else:
        label = keylist[j].replace('_', ' ')+' Z-4B'
    print('yield', keylist[j], 137000*xs[j]*seleffs[j])
    counts, bins = hist_selevents_mass[j].to_numpy()
    print(keylist[j], counts, bins, 137000*xs[j]*seleffs[j]/np.sum(counts))
    hist_selevents_mass[j] = hist_selevents_mass[j]*137000*xs[j]*seleffs[j]/np.sum(counts)
    counts, bins = hist_selevents_mass[j].to_numpy()
    print(counts)
    mplhep.histplot(hist_selevents_mass[j], label=label, ax=ax)
plt.gca().set_ylim(bottom=0.0001)
ax.set_yscale('log')
ax.set_xlabel('Multijet Mass')
ax.set_ylabel(r'$\mathcal{L}*\sigma*\varepsilon$')
ax.set_title(r'1L HLT, 2 B-Jets Mass ($m_{jj} < 120$)')
ax.legend()
plt.savefig(f'plots/hist_selevents_HLT1L_ex2bjets_mass_comp_yield.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for j in range(len(keylist)):
    if j > 3:
        label = keylist[j].replace('_', ' ')
    else:
        label = keylist[j].replace('_', ' ')+' Z-4B'
    mplhep.histplot(hist_hltevents_nbjets[j], density=True, label=label, ax=ax)
ax.set_xlabel('N B-Tagged Jets')
ax.set_ylabel('Normalized Events')
ax.set_title(f'1L HLT, N B-Tagged Jets')
ax.legend()
plt.savefig(f'plots/hist_hltevents_HLT1L_nbjets_comp_norm.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for j in range(len(keylist)):
    if j > 3:
        label = keylist[j].replace('_', ' ')
    else:
        label = keylist[j].replace('_', ' ')+' Z-4B'
    #mplhep.histplot(hist_hltevents_nbjets[j], binwnorm=137000.*xs[j]*hlteffs[j], label=label, ax=ax)
    counts, bins = hist_hltevents_nbjets[j].to_numpy()
    hist_hltevents_nbjets[j] = hist_hltevents_nbjets[j]*137000*xs[j]*hlteffs[j]/np.sum(counts)
    mplhep.histplot(hist_hltevents_nbjets[j], label=label, ax=ax)

plt.gca().set_ylim(bottom=0.1)
ax.set_yscale('log')
ax.set_xlabel('N B-Tagged Jets')
ax.set_ylabel(r'$\mathcal{L}*\sigma*\varepsilon$')
ax.set_title(f'1L HLT, N B-Tagged Jets')
ax.legend()
plt.savefig(f'plots/hist_hltevents_HLT1L_nbjets_comp_yield.pdf')
plt.close(fig)



