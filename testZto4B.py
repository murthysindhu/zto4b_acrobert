import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, TreeMakerSchema, BaseSchema
from coffea.analysis_tools import PackedSelection
import numpy as np
import glob
from utils import *
import uproot
import warnings

def find_z_to_4b(evt_parts):

    Zs = evt_parts[np.logical_and(evt_parts.pdgId == 23, evt_parts.status == 62)]
    ret = countBs(Zs)
    return ak.ravel(ret == 4)

def count_z_to_bs(evt_parts):

    Zs = evt_parts[np.logical_and(evt_parts.pdgId == 23, evt_parts.status == 62)]
    ret = countBs(Zs)
    return ak.ravel(ret)

def countBs(part, foundBs = ak.Array([])):
    
    #print(part.pdgId)
    #print(part.children.pdgId)
    bpdg = 5
    hasbchild = ak.any(abs(part.children.pdgId) == bpdg, axis = -1)
    #print(abs(part.pdgId) == bpdg, part.status == 71, hasbchild)
    #nb = ak.values_astype(np.logical_and(np.logical_and(abs(part.pdgId) == bpdg, part.status == 71), np.logical_not(hasbchild)), "int64")#, axis = 1)
    nb = ak.values_astype(np.logical_and(np.logical_and(abs(part.pdgId) == bpdg, True), np.logical_not(hasbchild)), "int64")#, axis = 1)
    #print(nb)
    print('this',part.pdgId)
    #print('where', np.where(abs(part.pdgId) == bpdg, part, part).pdgId)
    print('bool', abs(part.pdgId) == bpdg)
    #print('allbs', part.pdgId[abs(part.pdgId) == bpdg])
    print('next',part.children.pdgId)
    print('nb',nb)
    print('bool', np.logical_and(abs(part.pdgId) == bpdg, np.logical_not(hasbchild)))
    f1 = ak.ravel(np.logical_and(abs(part.pdgId) == bpdg, np.logical_not(hasbchild)))
    f2 = ak.ravel(part)
    f3 = f2[f1]

    print('f1', f1)
    print('f2', f2)
    print('f3', f2[f1])
    #print('bool',np.logical_and(abs(part.pdgId) == bpdg, np.logical_not(hasbchild)))
    print('B',ak.flatten(part[np.logical_and(abs(part.pdgId) == bpdg, np.logical_not(hasbchild))], axis=0).pdgId)
    theseBs = part[np.logical_and(abs(part.pdgId) == bpdg, np.logical_not(hasbchild))]
    print('Bs',theseBs.pdgId)
    if ak.sum(nb) > 0 and len(ak.flatten(theseBs)) > 0:
        foundBs = ak.concatenate((foundBs, theseBs))

    if ak.sum(part.pt) == 0:
        return nb
    else:
        nextlev = ak.sum(countBs(part.children, foundBs=foundBs), axis=-1)
        return nb + nextlev

#taggednano = glob.glob("taggednano/nano_ParT_ZJets_HT_800toInf_file*.root")

def print_gentree(evt):
        
    if ak.sum(evt.pt) == 0:
        return
    else:
        print(evt.pdgId)
        print_gentree(evt.children)

taggednano = xglob_onedir('/store/user/acrobert/Znano/Jan2023/ZJets/')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #friendfile = 'root://cmsdata.phys.cmu.edu/'+taggednano[0]
    #friendfile = 'root://cmsdata.phys.cmu.edu//store/user/acrobert/egiddata/Nov2022/DoubleEG/pfml_Nov2022_r2ntup_dt1/221111_225858/0000/output_1.root'
    friendfile = 'CMSSW_10_6_28/src/friend_ZJets_HT_800toInf_file3.root'
    friend = NanoEventsFactory.from_root(
        friendfile,
        treepath='fevt/EvtTree',
        schemaclass=NanoAODSchema.v6,
        metadata={"dataset": "ZJetsToQQ"},
    ).events()

nanofile = 'root://cmsdata.phys.cmu.edu//store/user/acrobert/Znano/Jan2023/ZJets/nano_ParT_ZJets_HT_800toInf_file3.root'
events = NanoEventsFactory.from_root(
    nanofile,
    schemaclass=NanoAODSchema.v6,
    metadata={"dataset": "ZJetsToQQ"},
).events()

count_z_to_bs(friend[7].GenPart)
exit()

#print(count_z_to_bs(friend.GenPart)[0])
disagreements = friend[count_z_to_bs(friend.GenPart) != ak.flatten(friend.GenTree.nFinalBs)]
#print(np.where(count_z_to_bs(friend.GenPart) != ak.flatten(friend.GenTree.nFinalBs)))
#print(len(friend), len(disagreements))
#disZs = disagreements.GenPart[np.logical_and(disagreements.GenPart.pdgId == 23, disagreements.GenPart.status == 62)]
#print("nb", count_z_to_bs(disagreements.GenPart)[0], ak.flatten(disagreements.GenTree.nFinalBs)[0])
#print_gentree(disZs[0])
#print("nb", count_z_to_bs(disagreements.GenPart)[1], ak.flatten(disagreements.GenTree.nFinalBs)[1])
#print_gentree(disZs[1])

print('look now')
count_z_to_bs(disagreements[0].GenPart)
count_z_to_bs(disagreements[1].GenPart)

#print(len(events), len(friend))
#nbs = 1
#Zevents = events[count_z_to_bs(events.GenPart) == nbs]
#Zfriend = friend[count_z_to_bs(friend.GenPart) == nbs]
#Zfriend2 = friend[ak.flatten(friend.GenTree.nFinalBs) == nbs]
#print(np.where(count_z_to_bs(events.GenPart) == nbs))
#print(np.where(ak.flatten(friend.GenTree.nFinalBs) == nbs))
#print(len(Zevents), len(Zfriend), len(Zfriend2))
#print(Zevents.GenPart[np.logical_and(Zevents.GenPart.pdgId == 23, Zevents.GenPart.status == 62)][0:20].pt)
#print(Zfriend.GenPart[np.logical_and(Zfriend.GenPart.pdgId == 23, Zfriend.GenPart.status == 62)][0:20].pt)
##print(events.GenPart[np.logical_and(events.GenPart.pdgId == 23, events.GenPart.status == 62)][0:10].pt)
##print(friend.GenPart[np.logical_and(friend.GenPart.pdgId == 23, friend.GenPart.status == 62)][0:10].pt)
#
##print(xglob)
#
#print('checkfields')
#
#ZeventsZ = Zevents.GenPart[np.logical_and(Zevents.GenPart.pdgId == 23, Zevents.GenPart.status == 62)]
#ZfriendZ = Zfriend.GenPart[np.logical_and(Zfriend.GenPart.pdgId == 23, Zfriend.GenPart.status == 62)]
#
#print(ZeventsZ[0].eta,              ZfriendZ[0].eta)
#print(ZeventsZ[0].phi,              ZfriendZ[0].phi)
#print(ZeventsZ[0].pt,               ZfriendZ[0].pt)
#print(ZeventsZ[0].mass,             ZfriendZ[0].mass)
#print(ZeventsZ[0].pdgId,            ZfriendZ[0].pdgId)
#print(ZeventsZ[0].status,           ZfriendZ[0].status)
#print(ZeventsZ[0].statusFlags,      ZfriendZ[0].statusFlags)
#print(ZeventsZ[0].genPartIdxMother, ZfriendZ[0].genPartIdxMother)
#
#events.GenPart = friend.GenPart
#print('new Zs', len(events[count_z_to_bs(events.GenPart) == nbs]))

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
    
