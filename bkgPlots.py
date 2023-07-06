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
parser.add_argument('-p', '--plots', default='all', help='Plots to run: "4V", "DR", "Jet", "Match", "all"', type=str)
parser.add_argument('-d', '--debug', action='store_true', default=False, help='Run the plots at the end or not')
parser.add_argument('-k', '--key', default='ZZTo4B01j_mc2017UL', help='Dataset key to run: "ZZTo4B01j_mc2017UL", "ZZTo2Q2L_mc2017UL"', type=str)

tmr = timer()
args = parser.parse_args()
plotters = RunCoffea.initplotters(args.key, args.plots, bkg=True)
filepairs = RunCoffea.initnanos(args.key)

friendtag = 'ZBQ-EPid'
totZ4b = 0

limit_nfiles = args.nfiles
nfilled = 0

totwgt = {'all':0.}
for i in range(len(filepairs)):

    nanofile, _ = filepairs[i]
    tagnano = RunCoffea.loadfile(nanofile, metadata={"dataset file": nanofile})
    if tagnano is None:
        continue

    print(f'Running file {i}; elapsed time {round(tmr.time()/60., 3)}')
    nfilled += 1

    #for plotter in plotters:
    #    plotter.processAll(tagnano)

    RunCoffea.runAll(tagnano, plotters, totwgt=totwgt)

    del tagnano
    print('Current memory:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    if nfilled == limit_nfiles:
        break
    gc.collect()

print(totwgt)
if args.debug:
    exit()
print(f'Plotting; elapsed time {round(tmr.time()/60., 3)}')
for plotter in plotters:
    plotter.plot()

