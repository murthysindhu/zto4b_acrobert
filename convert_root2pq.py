import pyarrow.parquet as pq
import pyarrow as pa # pip install pyarrow==0.7.1
import ROOT
import numpy as np
import glob, os
import sys
import time
#np.set_printoptions(threshold=sys.maxsize)

import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('-i', '--infile', default='output.root', nargs='+', type=str, help='Input root file.')
parser.add_argument('-o', '--outdir', default='parquet', type=str, help='Output pq file dir.')
args = parser.parse_args()

start = time.time()
evtTreeStr = str(args.infile[0])

print(" >> Input file:",evtTreeStr)
evtTree = ROOT.TChain("fevt/EvtTree")
evtTree.AddFile(evtTreeStr)

nEvts = evtTree.GetEntries()
assert nEvts > 0
print(" >> nEvts:",nEvts)

outStr = ['parquet/output.duo.parquet.0', 'parquet/output.trio.parquet.0', 'parquet/output.quad.parquet.0']
print(" >> Output files: {}".format(outStr))

##### EVENT SELECTION START #####

data = {}
duodata = {}
triodata = {}
quaddata = {}
sw = ROOT.TStopwatch()
sw.Start()

keys = ['Z1_nB', 'Z1_E', 'Z1_pT', 'Z1_eta', 'Z1_phi', 'Z2_nB', 'Z2_E', 'Z2_pT', 'Z2_eta', 'Z2_phi',
        'Z1_orig_Bs_E', 'Z1_orig_Bs_pT', 'Z1_orig_Bs_eta', 'Z1_orig_Bs_phi',
        'Z1_final_Bs_E', 'Z1_final_Bs_pT', 'Z1_final_Bs_eta', 'Z1_final_Bs_phi', 
        'Z2_orig_Bs_E', 'Z2_orig_Bs_pT', 'Z2_orig_Bs_eta', 'Z2_orig_Bs_phi', 
        'Z2_final_Bs_E', 'Z2_final_Bs_pT', 'Z2_final_Bs_eta', 'Z2_final_Bs_phi',
        'jet_E', 'jet_pT', 'jet_eta', 'jet_phi', 'jet_btags',
        'fatJet_E', 'fatJet_pT', 'fatJet_eta', 'fatJet_phi']


nduos = 0
ntrios = 0
nquads = 0 
total_ngroups = np.array([0, 0, 0])
data = [{}, {}, {}]
writers = [None, None, None]
ecalkeys = ['duo_ecal_img', 'trio_ecal_img', 'quad_ecal_img']
hcalkeys = ['duo_hcal_img', 'trio_hcal_img', 'quad_hcal_img']
pftkeys = ['duo_pft_img', 'trio_pft_img', 'quad_pft_img']
etakeys = ['duo_eta', 'trio_eta', 'quad_eta']
phikeys = ['duo_phi', 'trio_phi', 'quad_phi']
radkeys = ['duo_rad', 'trio_rad', 'quad_rad']

for iEvt in range(evtTree.GetEntries()):
    
    # Initialize event
    evtTree.GetEntry(iEvt)
    
    if iEvt % 100 == 0:
        print(" .. Processing entry",iEvt,'time:',time.time() - start)
    
    ngroups = np.array([ np.array(evtTree.duo_rad).size, np.array(evtTree.trio_rad).size, np.array(evtTree.quad_rad).size ])
    hasgroup = ngroups > 0

    #if not hasgroup.any():
    #    continue
    
    data = [{}, {}, {}]
    thisdata = {}
    idx = [evtTree.runId, evtTree.lumiId, evtTree.eventId]
    thisdata['idx'] = np.array(idx)
    
    for key in keys:
        
        exec('tempdata = np.array(evtTree.{})'.format(key))
        if tempdata.size == 0:
            tempdata = np.array([[-1.]])
        thisdata[key] = np.array(tempdata)


        
    #create table for this event
    pqdata = [pa.array([d]) if np.isscalar(d) or type(d) == list else pa.array([d.tolist()]) for d in list(thisdata.values())]
    table = pa.Table.from_arrays(pqdata, list(thisdata.keys()))
    
    if iEvt == 0:
        writer = pq.ParquetWriter('parquet/output8f.all.parquet.0', table.schema, compression='snappy')
        
    #write table
    writer.write_table(table)

        
    #if hasgroup[0]:
    #for iG in range(3):
    #
    #    if not hasgroup[iG]:
    #        continue
    #    
    #    exec('ecal_image = np.array(evtTree.{})'.format(ecalkeys[iG]))
    #    assert ecal_image.size > 0
    #    exec('pft_image = np.array(evtTree.{})'.format(pftkeys[iG]))
    #    assert pft_image.size > 0
    #    assert len(ecal_image) == len(pft_image)
    #    exec('hcal_image = np.array(evtTree.{})'.format(hcalkeys[iG]))
    #
    #    for iI in range(len(ecal_image)):
    #
    #        # fill dict for this event
    #        data[iG] = thisdata.copy()
    #        data[iG][ecalkeys[iG]] = ecal_image[iI].reshape(1,160,160) 
    #        data[iG][hcalkeys[iG]] = hcal_image[iI].reshape(1,32,32) 
    #        data[iG][pftkeys[iG]] = pft_image[iI].reshape(1,160,160) 
    #        exec('data[iG][etakeys[iG]] = np.array([evtTree.{}[0][iI]])'.format(etakeys[iG]))
    #        exec('data[iG][phikeys[iG]] = np.array([evtTree.{}[0][iI]])'.format(phikeys[iG]))
    #        exec('data[iG][radkeys[iG]] = np.array([evtTree.{}[0][iI]])'.format(radkeys[iG]))
    #
    #        #create table for this event                                                                                                          
    #        pqdata = [pa.array([d]) if np.isscalar(d) or type(d) == list else pa.array([d.tolist()]) for d in list(data[iG].values())]
    #        table = pa.Table.from_arrays(pqdata, list(data[iG].keys()))
    #
    #        if total_ngroups[iG] == 0:
    #            writers[iG] = pq.ParquetWriter(outStr[iG], table.schema, compression='snappy')
    #
    #        #write table                                                                                                                                        
    #        writers[iG].write_table(table)
    #        
    #        total_ngroups[iG] += 1
    #
    #        #print(" .. len", ' '.join([ str(data[iG][key].shape) for key in data[iG].keys() ]))

     

writer.close()
       
#for iG in range(3):
#    writers[iG].close()

#print(" >> Groups - nduos: {} ntrios: {} nquads: {}".format(total_ngroups[0], total_ngroups[1], total_ngroups[2]))
print(" >> Real time:",sw.RealTime()/60.,"minutes")
print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
print(" >> ======================================")

sw.Stop()
