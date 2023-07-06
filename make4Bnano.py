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
from hist import Hist
import hist
import matplotlib.pyplot as plt
import mplhep
import matplotlib.ticker as ticker
from classRunCoffea import RunCoffea
import pandas
import sys

out = pandas.DataFrame()
pairs = RunCoffea.initnanos(sys.argv[1])

events_Z4B = ak.Array([])
nfilled = 0

for (nano, friend) in pairs:

    tagnano = RunCoffea.loadpair(nano, friend)
    tagnano = tagnano[tagnano.Zs.nB[:,0] == 4]

    print(len(tagnano), len(events_Z4B))

    if len(events_Z4B) == 0:
        events_Z4B = tagnano
    else:
        assert events_Z4B.fields == tagnano.fields
        events_Z4B = ak.concatenate([events_Z4B, tagnano], axis=0)
    
    print(len(tagnano), len(events_Z4B))
    del tagnano
    nfilled += 1
    if nfilled == int(sys.argv[2]):
        break

print(len(events_Z4B))
