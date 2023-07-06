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
from class4VPlots import Plots4V
from classDRPlots import PlotsDR
from classMatchPlots import PlotsMatch
from classTrigPlots import PlotsTrig
from classJetPlots import PlotsJet
import os, psutil
import tracemalloc
import gc
from copy import deepcopy
import matplotlib.ticker as ticker


#tracemalloc.start()
plt.rcParams['text.usetex'] = True

parser = ArgumentParser(prog = 'Friend Plotter',
                        description = 'Preliminary plots for nanotuples + friends')

parser.add_argument('-n', '--nfiles', default=1000, help='Number of file pairs to read', type=int)
parser.add_argument('-p', '--plots', default='all', help='Plots to run: "4V", "DR", "Jet", "Match", "all"', type=str)
parser.add_argument('-d', '--debug', action='store_true', default=False, help='Run the plots at the end or not')
#parser.add_argument('-k', '--key', default='ZZTo4B01j_mc2017UL', help='Dataset key to run: "ZZTo4B01j_mc2017UL", "ZZTo2Q2L_mc2017UL"', type=str)

args = parser.parse_args()
whichplots = 'Match'#args.plots

tmr = timer()
friendtag = 'ZBQ-EPid'
limit_nfiles = args.nfiles

keys = ['ZZTo4B01j_mc2017UL', 'ZZTo2Q2L_mc2017UL', 'WZ_mc2017UL', 'ZGammaToJJ_mc2017UL', 'ZJets_HT_800toInf_mc2017UL']
hist_4B_matchedset = []
for key in keys:    
    
    localfiles = glob.glob('nanos/*'+key+'*')
    thesefiles = xglob_onedir('/store/user/acrobert/Znano/Jan2023/'+t3dirs[key]+'/')
    taggednano = [f for f in thesefiles if key in f and 'nano_ParT' in f]
    friendEPid = [f for f in thesefiles if key in f and 'EPid' in f]
    print(key, 'Tagnano:', len(taggednano), 'Friend:', len(friendEPid))
    
    totZ4b = 0
    
    #initialize hists
    plotter4V = Plots4V(key=key)
    plotterDR = PlotsDR(key=key)
    plotterMatch = PlotsMatch(key=key)
    plotterJet = PlotsJet(key=key)
    plotterTrig = PlotsTrig(key=key)

    if whichplots == 'all':
        plotters = [plotter4V, plotterDR, plotterMatch, plotterJet, plotterTrig]
    if whichplots == '4V':
        plotters = [plotter4V]
    if whichplots == 'DR':
        plotters = [plotterDR]
    if whichplots == 'Jet':
        plotters = [plotterJet]
    if whichplots == 'Match':
        plotters = [plotterMatch]
    if whichplots == 'Trig':
        plotters = [plotterTrig]

    
    has_snap = False
    nfilled = 0
    for i in range(len(friendEPid)):
        friendTname = f'friend_{key}_{friendtag}_file{i}.root'
        tagnanoname = f'nano_ParT_{key}_file{i}.root'
        friendTfile = f'/store/user/acrobert/Znano/Jan2023/{t3dirs[key]}/{friendTname}'
        tagnanofile = f'/store/user/acrobert/Znano/Jan2023/{t3dirs[key]}/{tagnanoname}'
        friendTpath = f'root://cmsdata.phys.cmu.edu/{friendTfile}'
        tagnanopath = f'root://cmsdata.phys.cmu.edu/{tagnanofile}'
    
        if f'nanos/{friendTname}' in localfiles and f'nanos/{tagnanoname}' in localfiles:
    
            friendfile = f'nanos/{friendTname}'
            nanofile = f'nanos/{tagnanoname}'
    
        elif tagnanofile in thesefiles and friendTfile in thesefiles:
    
            friendfile = f'root://cmsdata.phys.cmu.edu/{friendTfile}'
            nanofile = f'root://cmsdata.phys.cmu.edu/{tagnanofile}'
    
        else:
            continue
    
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Zfriend = NanoEventsFactory.from_root(
                    friendfile,
                    treepath='fevt/EvtTree',
                    schemaclass=NanoAODSchema.v6,
                    metadata={"dataset": key},
                ).events()
        except uproot.exceptions.KeyInFileError:
            continue
    
        tagnano = NanoEventsFactory.from_root(
            nanofile,
            schemaclass=NanoAODSchema.v6,
            metadata={"dataset": key},
        ).events()
    
        if len(tagnano) != len(Zfriend):
            continue
        assert len(tagnano) == len(Zfriend)
        print(f'Running file pair {i}; elapsed time {round(tmr.time()/60., 3)}')
        nfilled += 1
    
        tagnano.Zs = Zfriend.Zs
    
        hasZ4B = ak.any(Zfriend.Zs.nB == 4, axis = -1)
        totZ4b += len(Zfriend.Zs[hasZ4B])
        print('Z-4B Fraction:', round(len(Zfriend.Zs[hasZ4B])/len(tagnano), 6), 'N Z-4B:', totZ4b)
    
        tagnano['Zs'] = Zfriend.Zs
        
        hasZ4B = (tagnano.Zs.nB[:,0] == 4)
        hasZ2B = (tagnano.Zs.nB[:,0] == 2)
        nZ4B = ak.sum(hasZ4B)
        nZ2B = ak.sum(hasZ2B)
        print('Current memory:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    
        if nZ4B > 0: 
            for plotter in plotters:
                plotter.process4B(tagnano[hasZ4B])
        if nZ2B > 0:
            for plotter in plotters:
                plotter.process2B(tagnano[hasZ2B])
                
        del tagnano, Zfriend, hasZ4B, hasZ2B
        if nfilled == limit_nfiles:
            break
        gc.collect()

    hist_4B_matchedset.append(deepcopy(plotterMatch.hist_matchedsetSV[0]))

if args.debug:
    exit()
print(f'Plotting; elapsed time {round(tmr.time()/60., 3)}')

jetlabels = ['AK4 Jets', 'AK8 Jets', 'AK15 Jets']
dsetlabels = [key.split('_')[0] for key in keys]
xlabels = ['[]', '[1]', '[2]', '[1,1]', '[3]', '[2,1]', '[1,1,1]', '[4]', '[3,1]', '[2,2]', '[2,1,1]', '[1,1,1,1]']

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(5):
    counts, bins = hist_4B_matchedset[i].to_numpy()
    plt.hist(bins[:-1], bins, weights=counts, density=True, label=dsetlabels[i], histtype='step')
ax.set_ylabel('Normalized Events')
ax.set_title(f'Types of Matched Z-4B Events: {jetlabels[0]}')

ax.xaxis.set_major_locator(ticker.NullLocator())
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for count, x, label in zip(counts, bin_centers, xlabels):
    ax.annotate(label, xy=(x, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -4), textcoords='offset points', va='top', ha='center')
ax.legend()
plt.savefig(f'plots/hist_matchedsetSV_{jetlabels[0].split(" ")[0]}_comp.pdf')
plt.close(fig)
