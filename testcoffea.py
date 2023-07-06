import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

fname = "taggednano/nano_ParT_ZJets_HT_800toInf_file72.root" #"https://raw.githubusercontent.com/CoffeaTeam/coffea/master/tests/samples/nano_dy.root"
events = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v6,
    metadata={"dataset": "ZJetsToQQ"},
    treepath='fevt/EvtTree',
).events()

print(len(events.FatJet))
nevts = len(events)
nfj = []
for i in range(nevts):
    if len(events.FatJet.eta[i]) not in nfj:
        nfj.append(len(events.FatJet.eta[i]))

print(nfj)

#print(events.FatJet.eta[2])
#print(events.FatJet.eta[3])
