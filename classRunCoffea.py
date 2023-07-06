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

class RunCoffea:

    def initplotters(key, plots, bkg = False):

        if plots == 'all':
            if not bkg:
                from class4VPlots import Plots4V
                from classDRPlots import PlotsDR
                from classMatchPlots import PlotsMatch
                from classMapPlots import PlotsMap
                from classWindowPlots import PlotsWindow
                plotter4V = Plots4V(key=key)
                plotterDR = PlotsDR(key=key)
                plotterMatch = PlotsMatch(key=key)
                plotterMap = PlotsMap(key=key)
                plotterWindow = PlotsWindow(key=key)

            from classJetPlots import PlotsJet
            from classTrigPlots import PlotsTrig
            plotterJet = PlotsJet(key=key, bkg=bkg)
            plotterTrig = PlotsTrig(key=key, bkg=bkg)

            if not bkg:
                plotters = [plotter4V, plotterDR, plotterMatch, plotterJet, plotterTrig, plotterMap, plotterWindow]
            else:
                plotters = [plotterJet, plotterTrig]

        if plots == '4V':
            from class4VPlots import Plots4V
            plotter4V = Plots4V(key=key)
            plotters = [plotter4V]

        if plots == 'DR':
            from classDRPlots import PlotsDR
            plotterDR = PlotsDR(key=key)
            plotters = [plotterDR]

        if plots == 'Jet':
            from classJetPlots import PlotsJet
            plotterJet = PlotsJet(key=key, bkg=bkg)
            plotters = [plotterJet]

        if plots == 'Match':
            from classMatchPlots import PlotsMatch
            plotterMatch = PlotsMatch(key=key)
            plotters = [plotterMatch]

        if plots == 'Trig':
            from classTrigPlots import PlotsTrig
            plotterTrig = PlotsTrig(key=key, bkg=bkg)
            plotters = [plotterTrig]

        if plots == 'Sel':
            from classSelPlots import PlotsSel
            plotterSel = PlotsSel(key=key, bkg=bkg)
            plotters = [plotterSel]

        if plots == 'Map':
            from classMapPlots import PlotsMap
            plotterMap = PlotsMap(key=key, bkg=bkg)
            plotters = [plotterMap]

        if plots == 'Window':
            from classWindowPlots import PlotsWindow
            plotterWindow = PlotsWindow(key=key, bkg=bkg)
            plotters = [plotterWindow]

        return plotters

    def initnanos(key):

        friendtag = ['ZBQ-EPid', 'MapEcalPFCs']
        localfiles = glob.glob('nanos/*'+key+'*')
        thesefiles = xglob_onedir('/store/user/acrobert/Znano/Jan2023/'+t3dirs[key]+'/')
        taggednano = [f for f in thesefiles if key in f and 'nano_ParT' in f]
        friendtrees = [[f for f in thesefiles if key in f and tag in f] for tag in friendtag]

        matchedpairs = []

        for i in range(len(friendtrees[0])):
            friendTname = [f'friend_{key}_{tag}_file{i}.root' for tag in friendtag]
            tagnanoname = f'nano_ParT_{key}_file{i}.root'
            friendTfile = [f'/store/user/acrobert/Znano/Jan2023/{t3dirs[key]}/{name}' for name in friendTname]
            tagnanofile = f'/store/user/acrobert/Znano/Jan2023/{t3dirs[key]}/{tagnanoname}'
            friendTpath = [f'root://cmsdata.phys.cmu.edu/{file}' for file in friendTfile]
            tagnanopath = f'root://cmsdata.phys.cmu.edu/{tagnanofile}'

            allfriends = all([file in thesefiles for file in friendTfile])
            if tagnanofile in thesefiles and allfriends:

                friendfile = [f'root://cmsdata.phys.cmu.edu/{file}' for file in friendTfile]
                nanofile = f'root://cmsdata.phys.cmu.edu/{tagnanofile}'
                matchedpairs.append((nanofile, friendfile))

        print(f'Key: {key} found {len(matchedpairs)} pairs; missing {(len(friendtrees[0]) - len(matchedpairs))}')
        return matchedpairs

    def loadfile(loc, path='', metadata={}):
        
        if path=='':
            nano = NanoEventsFactory.from_root(
                loc,
                schemaclass=NanoAODSchema.v6,
                metadata=metadata,
            ).events()
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nano = NanoEventsFactory.from_root(
                        loc,
                        treepath=path,
                        schemaclass=NanoAODSchema.v6,
                        metadata=metadata,
                    ).events()
            except uproot.exceptions.KeyInFileError:
                return None

        return nano

    def loadpair(nanofile, friendfile):

        tagnano = RunCoffea.loadfile(nanofile, metadata={"dataset file": nanofile})
        friends = [RunCoffea.loadfile(file, path='fevt/EvtTree', metadata={"dataset file": file}) for file in friendfile]
        
        if friends[0] is None:
            return tagnano

        if friends[1] is None:
            print(f'Skipping null file: {friendfile[1].split("/")[-1]}')
            return None
        
        if len(tagnano) != len(friends[0]) or len(tagnano) != len(friends[1]):
            print(f'Skipping unmatched files: {len(friends[0])} {len(friends[1])} {len(tagnano)}')
            return None

        assert len(tagnano) == len(friends[0])
        assert len(tagnano) == len(friends[1])

        keys = ['Zs', 'map']
        for i in range(len(friends)):
            #print(i, friends[i].fields)
            tagnano[keys[i]] = friends[i][keys[i]]
        #print(f'Running file pair {nanofile}; elapsed time {round(tmr.time()/60., 3)}')
        
        #print(tagnano.SV.fields)
        return tagnano

    def runZs(tagnano, plotters, key, totwgt={'all':0., 'Z4B':0., 'Z2Q2B':0., 'Z4Q':0., 'Z2B':0., 'Z2Q':0., }):

        if key == 'ZZTo2Q2L_mc2017UL':
            tagnano = tagnano[ak.num(tagnano.Zs) == 2]
                
        tagnano['Zs'] = tagnano.Zs[ak.argsort(ak.num(tagnano.Zs.final_Qs_pdgId, axis=-1), ascending=False)]
            
        hasZ4B = tagnano.Zs.nB[:,0] == 4 #ak.any(tagnano.Zs.nB == 4, axis = -1)
        nZ4B = ak.sum(hasZ4B)

        abspdgs = abs(tagnano[hasZ4B].Zs[:,0].final_Qs_pdgId)
        #print(tagnano[hasZ4B].Zs[:,0].final_Qs_pdgId[abspdgs == 5][0:5])
        bspdg = tagnano[hasZ4B].Zs[:,0].final_Qs_pdgId[abspdgs == 5]
        #print(ak.num(bspdg, axis=-1)[0:5])
        #print(bspdg[ak.num(bspdg, axis=-1) != 4])

        hasZ2Q2B = (tagnano.Zs.nB[:,0] == 2) & (ak.num(tagnano.Zs.final_Qs_pdgId, axis=-1)[:,0] == 4)
        nZ2Q2B = ak.sum(hasZ2Q2B)
         
        hasZ4Q = (tagnano.Zs.nB[:,0] == 0) & (ak.num(tagnano.Zs.final_Qs_pdgId, axis=-1)[:,0] == 4)
        nZ4Q = ak.sum(hasZ4Q)
            
        hasZ2B = (tagnano.Zs.nB[:,0] == 2) & (ak.num(tagnano.Zs.final_Qs_pdgId, axis=-1)[:,0] == 2)
        nZ2B = ak.sum(hasZ2B)
        
        hasZ2Q = (tagnano.Zs.nB[:,0] == 0) & (ak.num(tagnano.Zs.final_Qs_pdgId, axis=-1)[:,0] == 2)
        nZ2Q = ak.sum(hasZ2Q)
            
        totwgt['all'] += ak.sum(tagnano.genWeight)
        totwgt['Z4B'] += ak.sum(tagnano[hasZ4B].genWeight)
        totwgt['Z2Q2B'] += ak.sum(tagnano[hasZ2Q2B].genWeight)
        totwgt['Z4Q'] += ak.sum(tagnano[hasZ4Q].genWeight)
        totwgt['Z2B'] += ak.sum(tagnano[hasZ2B].genWeight)
        totwgt['Z2Q'] += ak.sum(tagnano[hasZ2Q].genWeight)

        for plotter in plotters:
            if nZ4B > 0:
                plotter.process4B(tagnano[hasZ4B])
            if nZ2B > 0:
                plotter.process2B(tagnano[hasZ2B])
            try:
                if nZ2Q2B > 0:
                    plotter.process2Q2B(tagnano[hasZ2Q2B])
                if nZ4Q > 0:
                    plotter.process4Q(tagnano[hasZ4Q])
            except:
                pass

    def runAll(tagnano, plotters, totwgt={'all':0.}):

        totwgt['all'] = ak.sum(tagnano.genWeight)

        for plotter in plotters:
            plotter.processAll(tagnano)

    def plot(plotters):
        
        for plotter in plotters:
            plotter.plot()
            
