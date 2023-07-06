import os, sys

nevts = 10
name = 'ZZTo2L4B-0j-t6gp'
#name = 'ZZTo2L2B-0j-t1gp'
key = 'r0'

condition = 'RunIISummer20UL17'
steps = ['wmLHEGEN', 'SIM', 'DIGIPremix', 'HLT', 'RECO', 'MiniAODv2']
CMSSW = ['CMSSW_10_6_28_patch1', 'CMSSW_10_6_17_patch1', 'CMSSW_10_6_17_patch1', 'CMSSW_9_4_14_UL_patch1', 'CMSSW_10_6_17_patch', 'CMSSW_10_6_20']
envs = [f'cd {CMSSW[i]}/src && cmsenv && eval `scram runtime -sh` && scram b -j8 && cd ../..' for i in range(len(CMSSW))]

cmd0 = f'cmsDriver.py Configuration/GenProduction/python/{name}_13TeV_pythia8_fragment.py --python_filename {name}-{key}-{condition}{steps[0]}_step0_cfg.py --eventcontent RAWSIM,LHE --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN,LHE --fileout file:{name}-{key}-{condition}{steps[0]}_step0.root --conditions 106X_mc2017_realistic_v6 --beamspot Realistic25ns13TeVEarly2017Collision --step LHE,GEN --geometry DB:Extended --era Run2_2017 --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[0]}_step0_cfg.py'

#cmd0 = f'cmsDriver.py Configuration/GenProduction/python/SMP-RunIISummer20UL17wmLHEGEN-00316-fragment.py --python_filename {name}-{key}-{condition}{steps[0]}_step0_cfg.py --eventcontent RAWSIM,LHE --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN,LHE --fileout file:{name}-{key}-{condition}{steps[0]}_step0.root --conditions 106X_mc2017_realistic_v6 --beamspot Realistic25ns13TeVEarly2017Collision --step LHE,GEN --geometry DB:Extended --era Run2_2017 --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[0]}_step0_cfg.py'

#cmd0 = f'cmsDriver.py Configuration/GenProduction/python/{name}_13TeV_pythia6_fragment.py --python_filename {name}-{key}-{condition}{steps[0]}_step0_cfg.py --eventcontent RAWSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM --fileout file:{name}-{key}-{condition}{steps[0]}_step0.root --conditions 106X_mc2017_realistic_v6 --beamspot Realistic25ns13TeVEarly2017Collision --step GEN,SIM --geometry DB:Extended --era Run2_2017 --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[0]}_step0_cfg.py'

cmd1 = f'cmsDriver.py  --python_filename {name}-{key}-{condition}{steps[1]}_step1_cfg.py --eventcontent RAWSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM --fileout file:{name}-{key}-{condition}{steps[1]}_step1.root --conditions 106X_mc2017_realistic_v6 --beamspot Realistic25ns13TeVEarly2017Collision --step SIM --geometry DB:Extended --filein file:{name}-{key}-{condition}{steps[0]}_step0.root --era Run2_2017 --runUnscheduled --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[1]}_step1_cfg.py'

cmd2 = f'cmsDriver.py  --python_filename {name}-{key}-{condition}{steps[2]}_step2_cfg.py --eventcontent PREMIXRAW --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM-DIGI --fileout file:{name}-{key}-{condition}{steps[2]}_step2.root --pileup_input "dbs:/Neutrino_E-10_gun/RunIISummer20ULPrePremix-UL17_106X_mc2017_realistic_v6-v3/PREMIX" --conditions 106X_mc2017_realistic_v6 --step DIGI,DATAMIX,L1,DIGI2RAW --procModifiers premix_stage2 --geometry DB:Extended --filein file:{name}-{key}-{condition}{steps[1]}_step1.root --datamix PreMix --era Run2_2017 --runUnscheduled --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[2]}_step2_cfg.py'

cmd3 = f'cmsDriver.py  --python_filename {name}-{key}-{condition}{steps[3]}_step3_cfg.py --eventcontent RAWSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM-RAW --fileout file:{name}-{key}-{condition}{steps[3]}_step3.root --conditions 94X_mc2017_realistic_v15 --customise_commands "process.source.bypassVersionCheck = cms.untracked.bool(True)" --step HLT:2e34v40 --geometry DB:Extended --filein file:{name}-{key}-{condition}{steps[2]}_step2.root --era Run2_2017 --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[3]}_step3_cfg.py'

cmd4 = f'cmsDriver.py  --python_filename {name}-{key}-{condition}{steps[4]}_step4_cfg.py --eventcontent AODSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier AODSIM --fileout file:{name}-{key}-{condition}{steps[4]}_step4.root --conditions 106X_mc2017_realistic_v6 --step RAW2DIGI,L1Reco,RECO,RECOSIM --geometry DB:Extended --filein file:{name}-{key}-{condition}{steps[3]}_step3.root --era Run2_2017 --runUnscheduled --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[4]}_step4_cfg.py'

cmd5 = f'cmsDriver.py  --python_filename {name}-{key}-{condition}{steps[5]}_step5_cfg.py --eventcontent MINIAODSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier MINIAODSIM --fileout file:{name}-{key}-{condition}{steps[5]}_step5.root --conditions 106X_mc2017_realistic_v9 --step PAT --procModifiers run2_miniAOD_UL --geometry DB:Extended --filein file:{name}-{key}-{condition}{steps[4]}_step4.root --era Run2_2017 --runUnscheduled --no_exec --mc -n {nevts} && cmsRun {name}-{key}-{condition}{steps[5]}_step5_cfg.py'

cmds = [cmd0, cmd1, cmd2, cmd3, cmd4, cmd5]

if sys.argv[1] == 'all':
    for i in range(len(cmds)):
        os.system(envs[i])
        os.system(cmds[i])
elif int(sys.argv[1]) in range(len(cmds)):
    idx = int(sys.argv[1])
    print(envs[idx])
    print(cmds[idx])
    os.system(envs[idx])
    os.system(cmds[idx])

