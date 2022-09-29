import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as scspec
import datetime
import scipy.integrate
import scipy.interpolate
import HGEFuncs
import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(description="Parameters for HGExpandChemicalPotential")
parser.add_argument("-f","--forcerecalc", type=int,default=0,help="If 1 then potential HGE coeffs are recalculated even if their table already exists")
parser.add_argument("-i","--initial", type=int, default=0,help="Initial structure to start calculating for multiprocessing")
parser.add_argument("-s","--step", type=int, default=1,help="Stride for slicing for multiprocessing")
args = parser.parse_args()

#    freeEnergySet[:,0] = freeEnergySet[:,0] + np.random.normal( 0, 0.01)
#    freeEnergySet[:,1] = freeEnergySet[:,1] * np.random.normal( 1, 0.1) + np.random.normal( 0, 0.2, len(freeEnergySet))



materialSet = np.genfromtxt("Structures/ChemicalDefinitions.csv",dtype=str,delimiter=",")
if materialSet.ndim == 1:
    materialSet = np.array([materialSet])
    
    
noiseReplicas = 1
nMaxValAll = 20
fitEnergyStart = 25
r0ValAll = 0.25



potentialFolder = "ChemicalPotentials/"

os.makedirs(potentialFolder,exist_ok=True)
os.makedirs("Datasets/ChemicalHGE",exist_ok=True)

outfile=open("Datasets/ChemicalPotentialCoefficients-sep15.csv","w")
noiseoutfile=open("Datasets/ChemicalPotentialCoefficientsNoise"+str(noiseReplicas)+"-sep15.csv","w")

#energyTargetSet = [ 10,15,20,25, 30, 35,40]
energyTargetSet = [25]
energyTargetBase = 25

maxR0 = 1.0
minR0 = 0.05
r0ValRange =  np.arange( minR0, maxR0, 0.01)


#["methane", methaneProbe],
#["cline", clineProbe]
pointProbes=  [ ["C",""],    ["K",""], ["Cl",""] ,["Slab",""], ["C2A",""] ,["C4A",""]  , ["CPlus",""] , ["CMinus",""], ["CMoreLJ",""],["CLessLJ",""]  ,["CEps20",""] ,["CMin",""]    ]
moleculeProbes = [ ["Water","waterfe"]  ,["WaterUCD","waterUCDfe"] , ["Methane","methanefe"] ,["CLine","clinefe"] ,["CarbRing","carbringfe"] ]
allProbes = pointProbes + moleculeProbes
allLabels = []

'''
for probeLabel in allProbes:
    allLabels.append("Chem"+probeLabel+"ProbeR0")
    for i in range(0,nMaxValAll+1):
        allLabels.append("Chem"+probeLabel+"ProbeC"+str(i))
    allLabels.append("Chem"+probeLabel+"ProbeEMin")
    allLabels.append("Chem"+probeLabel+"ProbeRightEMin")    
'''


for probeDef in allProbes:
    probeLabel = probeDef[0]
    probeFile = probeDef[1]
    allLabels.append("Chem"+probeLabel+"ProbeR0")
    for i in range(0,nMaxValAll+1):
        allLabels.append("Chem"+probeLabel+"ProbeC"+str(i))
    allLabels.append("Chem"+probeLabel+"ProbeEMin")
    allLabels.append("Chem"+probeLabel+"ProbeRightEMin")
    allLabels.append("Chem"+probeLabel+"EAtR0")


#headerSet =  [ "SurfID", "shape", "numericShape", "source",  "SurfCProbeR0" ] + CHGELabels + ["SurfKProbeR0"] + KHGELabels + ["SurfClProbeR0"] + ClHGELabels  + ["SurfWaterR0"]+ waterHGELabels
headerSet = ["ChemID", "SMILES" ] + allLabels
outfile.write( ",".join([str(a) for a in headerSet]) +"\n")
noiseoutfile.write( ",".join([str(a) for a in headerSet]) +"\n")


materialNotFoundList = []
for material in materialSet[args.initial::args.step]:
    materialID = material[0]
    chemSMILES = material[1]
    moleculePotentials = {}
    print("Starting chemical "+ materialID,flush=True)
    #load chem-probe potential and HGE
    try:
        freeEnergies0 = np.genfromtxt( potentialFolder+materialID+"_fev6.dat",delimiter=",")
        freeEnergies = freeEnergies0.copy()
        freeEnergiesNames = np.genfromtxt( potentialFolder+materialID+"_fev6.dat",delimiter=",",names=True)
        freeEnergyHeader = list(freeEnergiesNames.dtype.names)
        #print(freeEnergyHeader)
    except:
        print("Free energy file not found for ", materialID)
        materialNotFoundList.append( material)
        continue
    for molProbe in moleculeProbes:
        try:
            moleculeFreeEnergies0 = np.genfromtxt( potentialFolder+materialID+"_"+molProbe[1]+"v6.dat",delimiter=",")
            moleculePotentials[ molProbe[0] ] = moleculeFreeEnergies0
        except:
            print("Could not find ", molProbe[0], "for", materialID)
        #waterFreeEnergies0 = np.genfromtxt( potentialFolder+materialID+"_waterfe.dat",delimiter=",")
        #waterFreeEnergies = waterFreeEnergies0.copy()
        #waterFreeEnergiesUCD0 = np.genfromtxt( potentialFolder+materialID+"_waterUCDfe.dat",delimiter=",")
        #waterFreeEnergiesUCD = waterFreeEnergiesUCD0.copy()


        # except:
        #print("Could not locate water potentials for", materialID)
        #continue      
    skipCalc = 0
    targetOutputPath = "Datasets/ChemicalHGE/"+materialID+"-noise-"+str(noiseReplicas)+".csv"
    if os.path.exists(targetOutputPath) and args.forcerecalc == 0:
        #print("File for ", materialID, "already exists and force recalc = 0, skipping")
        skipCalc =1
        #preloadedData = np.genfromtxt(targetOutputPath,delimiter=",")
        precalcFile = open(targetOutputPath,"r")
        precalcFile.readline()
        for resLine in precalcFile:
            outfile.write(resLine)
        precalcFile.close()
        continue
    chemOutfile=open(targetOutputPath,"w")
    chemOutfile.write( ",".join([str(a) for a in headerSet]) +"\n")
    for r0Val in r0ValRange:   
        for itNum in range(noiseReplicas):
            resSet = [materialID, chemSMILES]   
            #freeEnergies = freeEnergies0.copy()
            #waterFreeEnergies = waterFreeEnergies0.copy()
            for probeDef in allProbes:
                probe = probeDef[0]
                if probeDef[1] != "":
                    probeFreeEnergies = HGEFuncs.getValidRegion( np.copy( moleculePotentials[probeDef[0] ])[:,(2,3)])
                #else:
                #if probe=="Water":
                #    probeFreeEnergies = getValidRegion( waterFreeEnergies[:,(2,3)] )
                #elif probe=="WaterUCD":
                #    probeFreeEnergies = getValidRegion (waterFreeEnergiesUCD[:,(2,3)])
                else:
                    probeHeader = "U"+probe+"dkJmol"
                    probeNumber = freeEnergyHeader.index(probeHeader)
                    #print(r0Val, probeNumber)
                    #print(freeEnergies[:5,(2,probeNumber) ])
                    probeFreeEnergies  = HGEFuncs.getValidRegion( freeEnergies[:,(2,probeNumber)])
                if itNum > 0:
                    probeFreeEnergies = HGEFuncs.applyNoise(probeFreeEnergies,0.01,0.1,0.2)
                probeFinalEnergy = probeFreeEnergies[-1,1]
                probeFreeEnergies[:,1] = probeFreeEnergies[:,1] - probeFinalEnergy
                probeSubsetMask = probeFreeEnergies[:,0] >= r0Val 
                probeFreeEnergiesSubset = probeFreeEnergies[   probeSubsetMask  ]
                probeMinEnergy = np.amin( probeFreeEnergies[:,1])
                probeRightMinEnergy = np.amin( probeFreeEnergiesSubset[:,1])
                #r0Val = 0.2
                #r0Val = max(0.1, estimateValueLocation(probeFreeEnergies,energyTarget)[0])
                probeHGE= HGEFuncs.HGECoeffsInterpolate( probeFreeEnergies, r0Val, nMaxValAll)
                probeHGE.insert(1, probeFinalEnergy)
                probeHGE.append(probeMinEnergy)
                probeHGE.append(probeRightMinEnergy)
                probeHGE.append(probeFreeEnergiesSubset[0,1])
                resSet = resSet + probeHGE
            resLine = ",".join([str(a) for a in resSet])
            noiseoutfile.write(resLine+"\n")
            chemOutfile.write(resLine+"\n")
            #print( resLine )
            if itNum == 0  :
                outfile.write(resLine+"\n")
    chemOutfile.close()
    print("Completed "+materialID,flush=True)
'''



ljHGELabels = []
electroHGELabels = []
waterHGELabels = []
slabHGELabels = []
CHGELabels=[]
KHGELabels=[]
ClHGELabels=[]
for i in range(0,nMaxValAll+1):
    ljHGELabels.append("ChemLJC"+str(i))
    electroHGELabels.append("ChemElC"+str(i))
    waterHGELabels.append("ChemWaterC"+str(i))
    slabHGELabels.append("ChemSlabC"+str(i))
    CHGELabels.append("ChemCProbeC"+str(i))
    KHGELabels.append("ChemKProbeC"+str(i))
    ClHGELabels.append("ChemClProbeC"+str(i))
    
    
headerSet =  [ "ChemID", "SMILES" ,"ChemLJR0" ] + ljHGELabels + ["ChemElR0"] + electroHGELabels  + ["ChemWaterR0"]+ waterHGELabels   + ["ChemSlabR0"]+ slabHGELabels + ["ChemCProbeR0"] + CHGELabels + ["ChemKProbeR0"] + KHGELabels + ["ChemClProbeR0"] + ClHGELabels   
outfile.write( ",".join([str(a) for a in headerSet]) +"\n")
noiseoutfile.write( ",".join([str(a) for a in headerSet]) +"\n")
plotFigs = 1



materialNotFoundList = []
for material in materialSet:
    materialID = material[0]
    chemSMILES = material[1]
    print("Starting chemical ", materialID)
    #load surface-probe potential and HGE
    try:
        freeEnergies = np.genfromtxt( potentialFolder+materialID+"_fev3.dat",delimiter=",")
    except:
        print("Free energy file not found for ", materialID)
        materialNotFoundList.append( material)
        continue
    r0ValAll =  freeEnergies[  np.where(   freeEnergies[:,3] < fitEnergyStart )[0][0],2]
    r0ValSlab = freeEnergies[  np.where(   freeEnergies[:,5] < fitEnergyStart )[0][0],2]
    ljC0 = freeEnergies[-1,3]
    electroC0 = freeEnergies[-1,4]
    slabC0 = freeEnergies[-1,5]

    slabMaskStart =  np.where(   np.isfinite( freeEnergies[:,5] ) )[0][0]
    #print(slabMaskStart)
    slabPotential = freeEnergies[ slabMaskStart:      , (2,5) ]

    #load surface-water potential and HGE
    waterFreeEnergies = np.genfromtxt( potentialFolder+materialID+"_waterfe.dat",delimiter=",")
    waterC0 = waterFreeEnergies[-1,3]
    waterFreeEnergies[:,3] = waterFreeEnergies[:,3] - waterC0
    waterFE = getValidRegion(waterFreeEnergies[:,(2,3)])
    CProbeFE = getValidRegion(freeEnergies[:,(2,6)])
    KProbeFE = getValidRegion(freeEnergies[:,(2,7)])
    ClProbeFE = getValidRegion(freeEnergies[:,(2,8)])
    CProbe0 = CProbeFE[-1,1]
    ClProbe0 = ClProbeFE[-1,1]
    KProbe0 = KProbeFE[-1,1]
    CProbeFE[:,1] = CProbeFE[:,1] - CProbeFE[-1,1]
    ClProbeFE[:,1] = ClProbeFE[:,1] - ClProbeFE[-1,1]
    KProbeFE[:,1] = KProbeFE[:,1] - KProbeFE[-1,1]
    #print(CProbeFE)
    
    for energyTarget in energyTargetSet:
       r0ValC = estimateValueLocation(  CProbeFE, energyTarget)[0]
       r0ValK = estimateValueLocation(  KProbeFE, energyTarget)[0]
       r0ValCl = estimateValueLocation(  ClProbeFE, energyTarget)[0]
       r0ValSlab = estimateValueLocation(  slabPotential, energyTarget)[0]
       r0ValWater = estimateValueLocation(waterFE,energyTarget)[0]
       if r0ValK < r0ValC - 0.05:
           r0ValK = r0ValC- 0.05
       if r0ValCl < r0ValC - 0.05:
           r0ValCl = r0ValC - 0.05
       r0ValSlab = min( r0ValSlab, r0ValC-0.05)
       r0ValWater = min(r0ValWater, r0ValC-0.05)
       ljHGE = HGECoeffs( freeEnergies[:,(2,3)] , r0ValC, nMaxValAll)
       electroHGE = HGECoeffs( freeEnergies[:,(2,4)] , r0ValC, nMaxValAll)
       CProbeHGE = HGECoeffs( CProbeFE , r0ValC, nMaxValAll)
       KProbeHGE = HGECoeffs( KProbeFE , r0ValK, nMaxValAll)
       ClProbeHGE = HGECoeffs( ClProbeFE , r0ValCl, nMaxValAll) 
       slabHGE = HGECoeffs( slabPotential, r0ValSlab, nMaxValAll)
       waterHGE = HGECoeffs( waterFE , r0ValWater, nMaxValAll)
       ljHGE.insert(1,ljC0)
       electroHGE.insert(1,electroC0)
       slabHGE.insert(1,slabC0)
       waterHGE.insert(1,waterC0)
       CProbeHGE.insert(1, CProbe0)
       KProbeHGE.insert(1, KProbe0)
       ClProbeHGE.insert(1, ClProbe0)
       #write out coefficients to a file
       resSet =  [ materialID, chemSMILES] + ljHGE +  electroHGE  + waterHGE + slabHGE + CProbeHGE + KProbeHGE + ClProbeHGE
       resLine = ",".join([str(a) for a in resSet])
       #print( resLine )
       noiseoutfile.write(resLine+"\n")
       if energyTarget == energyTargetBase:
           outfile.write(resLine+"\n")
       if plotFigs == 1 and energyTarget == energyTargetBase:
           plt.figure()
           plt.plot( freeEnergies[::5,2], freeEnergies[::5,3] ,"kx")
           plt.plot( freeEnergies[ freeEnergies[:,2] > ljHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], ljHGE) ,"k-")
           plt.plot( freeEnergies[::5,2], freeEnergies[::5,4] ,"yx")
           plt.plot( freeEnergies[ freeEnergies[:,2] > electroHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], electroHGE) ,"y-")
           plt.plot( freeEnergies[::5,2], freeEnergies[::5,5] ,"ko")
           plt.plot( freeEnergies[ freeEnergies[:,2] > slabHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], slabHGE) ,"k:")
           plt.plot( freeEnergies[::5,2], freeEnergies[::5,7] ,"rx")
           plt.plot( freeEnergies[ freeEnergies[:,2] > KProbeHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], KProbeHGE) ,"r-")
           plt.plot( freeEnergies[::5,2], freeEnergies[::5,8] ,"gx")
           plt.plot( freeEnergies[ freeEnergies[:,2] > ClProbeHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], ClProbeHGE) ,"g-")
           plt.plot( waterFreeEnergies[::,2], waterFreeEnergies[::,3] ,"bx")
           plt.plot( waterFreeEnergies[ waterFreeEnergies[:,2] > waterHGE[0]  ,2], BuildHGEFromCoeffs( waterFreeEnergies[:,2], waterHGE) ,"b-")
           minPlotEnergy = min ( np.amin(waterFreeEnergies[:,3]), np.amin( freeEnergies[:,4]), np.amin(freeEnergies[:,3]))
           plt.xlim(0,2.0)
           plt.ylim(-25,50)
           plt.savefig( potentialFolder+"/"+materialID+"-fitted.png")
'''
outfile.close()

if len(materialNotFoundList) > 0:
    print("Some chemicals were not found ")
    print(materialNotFoundList)
    #for notFoundChem in materialNotFoundList:
    #print("Afterwards re-run chemical potential scripts")
#plt.show()
