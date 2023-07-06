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
plotters = RunCoffea.initplotters(args.key, args.plots)
filepairs = RunCoffea.initnanos(args.key)

friendtag = 'ZBQ-EPid'
totZ4b = 0

limit_nfiles = args.nfiles

#has_snap = False
nfilled = 0
totwgt={'all':0., 'Z4B':0., 'Z2Q2B':0., 'Z4Q':0., 'Z2B':0., 'Z2Q':0., }

for i in range(len(filepairs)):
 
    nanofile, friendfile = filepairs[i]
    tagnano = RunCoffea.loadpair(nanofile, friendfile)

    if tagnano is None:
        continue
    if 'Zs' not in tagnano.fields:
        continue

    print(f'Running file pair {i}; elapsed time {round(tmr.time()/60., 3)}')
    nfilled += 1

    RunCoffea.runZs(tagnano, plotters, args.key, totwgt=totwgt)

    del tagnano
    print('Current memory:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    if nfilled == limit_nfiles:
        break
    gc.collect()
#    if not has_snap:
#        snapshot1 = tracemalloc.take_snapshot()
#        stats = snapshot1.statistics('traceback')
#        for i in range(5):
#            print(f"{stats[i].count} memory blocks: {stats[i].size/1024} KiB")
#            for line in stats[i].traceback.format():
#                print(line)
#        has_snap = True
#    else:
#        snapshot2 = tracemalloc.take_snapshot()
#        stats = snapshot2.statistics('traceback')
#        for i in range(5):
#            print(f"{stats[i].count} memory blocks: {stats[i].size/1024} KiB")
#            for line in stats[i].traceback.format():
#                print(line)
#
#        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
#        print("[ Top 20 differences ]")
#        for stat in top_stats[:20]:
#            print(stat)

if args.debug:
    exit()
print(f'Plotting; elapsed time {round(tmr.time()/60., 3)}')
print(f'Total wgts {totwgt}')

RunCoffea.plot(plotters)

