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
from class4VPlots import Plots4V
from classDRPlots import PlotsDR
from classMatchPlots import PlotsMatch
from classJetPlots import PlotsJet

plt.rcParams['text.usetex'] = True

#key = 'ZZTo2Q2L_mc2017UL'
key = 'ZZTo4B01j_mc2017UL'
thesefiles = xglob_onedir('/store/user/acrobert/Znano/Jan2023/'+t3dirs[key]+'/')
taggednano = [f for f in thesefiles if key in f and 'nano_ParT' in f]
friendEPid = [f for f in thesefiles if key in f and 'EPid' in f]
print(len(taggednano), len(friendEPid))

tmr = timer()

friendtag = 'ZBQ-EPid'
totZ4b = 0

limit_nfiles = 50

for i in range(min(limit_nfiles, len(friendEPid))):
    friendTfile = f'/store/user/acrobert/Znano/Jan2023/ZZ/friend_{key}_{friendtag}_file{i}.root'
    tagnanofile = f'/store/user/acrobert/Znano/Jan2023/ZZ/nano_ParT_{key}_file{i}.root'

    if not tagnanofile in thesefiles and friendTfile in thesefiles:
        continue

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            friendfile = 'root://cmsdata.phys.cmu.edu/'+friendTfile
            Zfriend = NanoEventsFactory.from_root(
                friendfile,
                treepath='fevt/EvtTree',
                schemaclass=NanoAODSchema.v6,
                metadata={"dataset": key},
            ).events()
    except uproot.exceptions.KeyInFileError:
        continue

    nanofile = 'root://cmsdata.phys.cmu.edu/'+tagnanofile
    tagnano = NanoEventsFactory.from_root(
        nanofile,
        schemaclass=NanoAODSchema.v6,
        metadata={"dataset": key},
    ).events()

    if len(tagnano) != len(Zfriend):
        continue
    assert len(tagnano) == len(Zfriend)
    print(f'Running file pair {i}; elapsed time {round(tmr.time()/60., 3)}')
    
    os.system(f'xrdcp root://cmsdata.phys.cmu.edu//{friendTfile} nanos/.')
    os.system(f'xrdcp root://cmsdata.phys.cmu.edu//{tagnanofile} nanos/.')
