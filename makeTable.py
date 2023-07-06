import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, TreeMakerSchema, BaseSchema
from coffea.analysis_tools import PackedSelection
from coffea import processor
from coffea.processor import IterativeExecutor
import numpy as np
import glob
from utils import *
import uproot
import warnings
import fnmatch

key = 'ZZTo2Q2L_mc2017UL'
taggednano = xglob_onedir('/store/user/acrobert/Znano/Jan2023/'+t3dirs[key]+'/')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    friendfile = 'CMSSW_10_6_28/src/friend_ZZTo2Q2L_file1.root'
    friend = NanoEventsFactory.from_root(
        friendfile,
        treepath='fevt/EvtTree',
        schemaclass=NanoAODSchema.v6,
        metadata={"dataset": "ZZTo2Q2L"},
    ).events()

nanofile = 'root://cmsdata.phys.cmu.edu//store/user/acrobert/Znano/Jan2023/ZZ/nano_ParT_ZZTo2Q2L_mc2017UL_file1.root'
tagnano = NanoEventsFactory.from_root(
    nanofile,
    schemaclass=NanoAODSchema.v6,
    metadata={"dataset": "ZJetsToQQ"},
).events()


class HLTProcessor(processor.ProcessorABC):
    def __init__(self, flag=False):
        self._flag = flag

    def process(self, events):
        out = {'nevts': len(events)
               #, 'nZto4B': len(events[ak.any(events.Zs.nB == 4, axis=-1)])
               , 'nPassHLT_Mu50': len(events[events.HLT.Mu50])
               , 'nPassHLT_IsoMu24': len(events[events.HLT.IsoMu24])
               , 'nPassHLT_ELe27_Tight': len(events[events.HLT.Ele27_WPTight_Gsf])
               , 'nPassHLT_1L': len(events[np.logical_or(np.logical_or(events.HLT.Mu50, events.HLT.IsoMu24), events.HLT.Ele27_WPTight_Gsf)])
               , 'nPassHLT_Pho200': len(events[events.HLT.Photon200])
               #, 'nPassHLT_Pho110EB': len(events[events.HLT.Photon110EB_TightID_TightIso])
               #, 'nPassHLT_Pho': len(events[np.logical_or(events.HLT.Photon200, events.HLT.Photon110EB_TightID_TightIso)])
               , 'nPassHLT_AK8Jet400': len(events[events.HLT.AK8PFJet400_TrimMass30])
               , 'nPassHLT_Jet500': len(events[events.HLT.PFJet500])
               , 'nPassHLT_HT1050': len(events[events.HLT.PFHT1050])
               , 'nPassHLT_Jet': len(events[np.any(np.array([events.HLT.PFJet500, events.HLT.PFHT1050, events.HLT.AK8PFJet400_TrimMass30]), axis=0)])
               , 'nPassHLT_BTags': len(events[np.any(np.array([events.HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2, events.HLT.PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0, events.HLT.QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1, events.HLT.DoublePFJets116MaxDeta1p6_DoubleCaloBTagCSV_p33]), axis=0)])
               , 'nPassHLT_DoubleEle25_CaloIdL_MW': len(events[events.HLT.DoubleEle25_CaloIdL_MW])
               , 'nPassHLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350': len(events[events.HLT.DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350])
               , 'nPassHLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL': len(events[events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL])
               , 'nPassHLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8': len(events[events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8])
               , 'nPassHLT_DoubleMu43NoFiltersNoVtx': len(events[events.HLT.DoubleMu43NoFiltersNoVtx])
               , 'nPassHLT_DoubleMu8_Mass8_PFHT350': len(events[events.HLT.DoubleMu8_Mass8_PFHT350])
               , 'nPassHLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg': len(events[events.HLT.DoubleMediumChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg])
               , 'nPassHLT_2L': len(events[np.any(np.array([events.HLT.DoubleEle25_CaloIdL_MW, events.HLT.DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350, events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL, events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, events.HLT.DoubleMu43NoFiltersNoVtx, events.HLT.DoubleMu8_Mass8_PFHT350, events.HLT.DoubleMediumChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg]), axis=0)])

        }

        # ...

        return out

    def postprocess(self, accumulator):
        pass

proc = HLTProcessor()

taggednano = [f for f in taggednano if key in f]

nfiles = 5
filelist = {
    'tagnano': ['root://cmsdata.phys.cmu.edu/'+f for f in taggednano[0:nfiles]]
}

executor = IterativeExecutor()
run = processor.Runner(
    executor=executor,
    schema=NanoAODSchema,
    format='root',
)

xsarr = []
errarr = []
os.system('cd CMSSW_10_6_30/src && export SCRAM_ARCH=slc7_amd64_gcc700 && cmsenv')
os.system('pwd')
os.system('curl https://raw.githubusercontent.com/cms-sw/genproductions/master/Utilities/calculateXSectionAndFilterEfficiency/genXsec_cfg.py -o ana.py > /dev/null 2>&1')
for file in taggednano[0:nfiles]:
    os.system('cmsRun ana.py inputFiles="file:root://cms-xrd-global.cern.ch/{}" maxEvents=-1 > xstmp.txt 2>&1'.format(file))

    xs = -1.
    
    with open('xstmp.txt','r') as xsout:
        xslines = xsout.readlines()

        for j in range(len(xslines)):
            if 'After filter: final cross section' in xslines[j]:
                eles = xslines[j].strip('\n').split(' ')
                xsarr.append(float(eles[6]))
                errarr.append(float(eles[8]))
                print('xs',i,float(eles[6]))
                if len(xsarr) >= 10:
                    checkxs = False
    #os.system('rm xstmp.txt')
xsarr = np.array(xsarr)
print(key, 'XS:', xsarr.mean(), xsarr.std())
os.system('cd ../..')

hists = run(filelist, "Events", processor_instance=proc)
print('HLT:', '2L:', hists['nPassHLT_2L']/hists['nevts'], '1L:', hists['nPassHLT_1L']/hists['nevts'], 'Pho:', hists['nPassHLT_Pho200']/hists['nevts'], 'BTags:', hists['nPassHLT_BTags']/hists['nevts'], 'Jet:', hists['nPassHLT_Jet']/hists['nevts'])

exit()

totZ = 0
totEvt = 0
for nano in taggednano:
    #break

    fname = 'root://cmsdata.phys.cmu.edu/'+nano
    print('Loading file: {}'.format(fname))
    events = NanoEventsFactory.from_root(
        fname,
        schemaclass=NanoAODSchema.v6,
        metadata={"dataset": "ZJetsToQQ"},
    ).events()

    print(len(events), len(events.GenPart))

    continue

    print('len', len(friend), len(events))
    print('len2', len(friend[0].GenPart.pdgId), len(events[0].GenPart.pdgId))

    selection = PackedSelection()
   
    selection.add('zto4b', find_z_to_4b(events.GenPart))
    nZ4B = selection.all('zto4b').sum()
    totZ += nZ4B
    totEvt += len(events)
    #print('Number Z->4B events: {}/{}'.format(nZ4B, len(events)))
    #print('Running totals: {}/{}'.format(totZ, totEvt))

    break

#print(f'Z->4B: {totZ}/{totEvt}')

nano = taggednano[4]
fname = 'root://cmsdata.phys.cmu.edu/'+nano
print('Loading file: {}'.format(fname))
events = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v6,
    metadata={"dataset": "ZJetsToQQ"},
).events()

selection = PackedSelection()

selection.add('zto4b', (count_z_to_bs(events.GenPart) == 3))
theseEvts = events[count_z_to_bs(events.GenPart) == 3]
print(f'Found {len(theseEvts)} Z->3b') 

for i in range(len(theseEvts)):

    parts = theseEvts[i].GenPart
    myZs = parts[np.logical_and(parts.pdgId == 23, parts.status == 62)]
    print_gentree(myZs)
    
