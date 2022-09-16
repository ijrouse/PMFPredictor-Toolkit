import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as scspec
import datetime
import scipy.integrate
import scipy.interpolate
import warnings
import argparse
import HGEFuncs

parser = argparse.ArgumentParser(description="Parameters for HGExpandSurfacePotential")
parser.add_argument("-f","--forcerecalc", type=int,default=0,help="If 1 then potential HGE coeffs are recalculated even if their table already exists")
args = parser.parse_args()
warnings.filterwarnings('ignore')

         

 #material ID, shape, source
materialSet = np.genfromtxt("Structures/SurfaceDefinitions.csv",dtype=str,delimiter=",")
if materialSet.ndim == 1:
    materialSet = np.array([materialSet])
    
    
plotFigs = 1

nMaxValAll = 20
r0ValC = 0.2


maxR0 = 1.0
minR0 = 0.05
r0ValRange =  np.arange( minR0, maxR0, 0.01)

noiseReplicas = 1
potentialFolder = "SurfacePotentials/"

outfile=open("Datasets/SurfacePotentialCoefficients-sep15.csv","w")
noiseoutfile=open("Datasets/SurfacePotentialCoefficientsNoise-"+str(noiseReplicas)+"-sep15.csv","w")
ljHGELabels = []
electroHGELabels = []
waterHGELabels = []

CHGELabels=[]
KHGELabels=[]
ClHGELabels=[]
energyTargetBase = 25


'''
["C",0.339,0.3598,0,0],
["K",0.314264522824 ,  0.36401,1,0],
["Cl",0.404468018036 , 0.62760,-1,0],
["C2A",0.2,0.3598,0,0],
["C4A",0.4,0.3598,0,0],
["CPlus",0.339,0.3598,0.5,0],
["CMinus",0.339,0.3598,-0.5,0],
["CMoreLJ",0.339,0.5,0,0],
["CLessLJ",0.339,0.2,0,0]

'''

#energyTargetSet = [5, 10,15,20,25, 30, 35,40]
energyTargetSet = [25]
pointProbes = [ ["C",""] ,  ["K",""], ["Cl",""] ,["C2A",""],["C4A",""] ,["CPlus",""],["CMinus",""],["CMoreLJ",""],["CLessLJ",""] ,["CMin",""], ["CPlusMin",""],["CMinusMin",""]]
moleculeProbes = [ ["Methane","methanefe"]   ,["WaterFull","waterUCDfe"] ,["CarbonRing","carbringfe"], ["CLine3","cline3fe"]]

allProbes = moleculeProbes + pointProbes
allLabels = []

'''
for i in range(0,nMaxValAll+1):
    CHGELabels.append("SurfCProbeC"+str(i))
    KHGELabels.append("SurfKProbeC"+str(i))
    ClHGELabels.append("SurfClProbeC"+str(i))
    waterHGELabels.append("SurfWaterC"+str(i))
'''

offsetDict = {}
offsetDictFile = open("Datasets/SurfaceOffsetData.csv","r")
firstline =0
offsetDictFileLines=offsetDictFile.readlines()
for line in offsetDictFileLines:
    if firstline == 0:
        firstline = 1
        continue
    lineParts =  line.strip().split(",")
    offsetDict[lineParts[0]] = [float(lineParts[2]),float(lineParts[3])] 
offsetDictFile.close()


for probeDef in allProbes:
    probeLabel = probeDef[0]
    probeFile = probeDef[1]
    allLabels.append("Surf"+probeLabel+"ProbeR0")
    for i in range(0,nMaxValAll+1):
        allLabels.append("Surf"+probeLabel+"ProbeC"+str(i))
    allLabels.append("Surf"+probeLabel+"ProbeEMin")
    allLabels.append("Surf"+probeLabel+"ProbeRightEMin")    

#headerSet =  [ "SurfID", "shape", "numericShape", "source",  "SurfCProbeR0" ] + CHGELabels + ["SurfKProbeR0"] + KHGELabels + ["SurfClProbeR0"] + ClHGELabels  + ["SurfWaterR0"]+ waterHGELabels
headerSet = [ "SurfID", "shape", "numericShape", "source","ssdType" ,"SurfAlignDist","SSDRefDist"] + allLabels
outfile.write( ",".join([str(a) for a in headerSet]) +"\n")
noiseoutfile.write( ",".join([str(a) for a in headerSet]) +"\n")

for material in materialSet:
    materialID = material[0]
    print("Starting material ", materialID)
    print("Surface alignment offset", offsetDict[materialID])
    alignOffsets = offsetDict[materialID]
    alignOffset =alignOffsets[0] 
    ssdRefDist =  alignOffsets[1]
    materialShape = material[1]
    materialPMFSource = material[2]
    materialSSDType = material[3]
    numericShape = 0
    moleculePotentials = {}
    if materialShape=="cylinder":
        numericShape = 1
    #load surface-probe potential and HGE
    try:
        freeEnergies0 = np.genfromtxt( potentialFolder+materialID+"_fev5.dat",delimiter=",")
        freeEnergies = freeEnergies0.copy()
        freeEnergiesNames = np.genfromtxt( potentialFolder+materialID+"_fev5.dat",delimiter=",",names=True)
        freeEnergyHeader = list(freeEnergiesNames.dtype.names)
    except:
        print("Could not locate single-bead potentials for", materialID)
        continue
    for molProbe in moleculeProbes:
        try:
            moleculeFreeEnergies0 = np.genfromtxt(potentialFolder+materialID+"_"+molProbe[1]+".dat",delimiter=",")
            moleculePotentials[ molProbe[0] ] = moleculeFreeEnergies0
        except:
            print("Could not find ", molProbe[0], "for", materialID)
    ''''
    try:
        waterFreeEnergies0 = np.genfromtxt( potentialFolder+materialID+"_waterfe.dat",delimiter=",")
        waterFreeEnergies = waterFreeEnergies0.copy()
    except:
        print("Could not locate water potentials for", materialID)
        continue      
    try:
        #print("Methane path: ", potentialFolder+materialID+"_methanefe_eps1.dat" )
        methaneEnergies0 = np.genfromtxt( potentialFolder+materialID+"_methanefe.dat",delimiter=",")
        methaneFreeEnergies = methaneEnergies0.copy()
    except:
        print("Could not locate methane potentials for", materialID)
        continue     
    '''
    if os.path.exists("Datasets/SurfaceHGE/"+materialID+"-noise-"+str(noiseReplicas)+ ".csv") and args.forcerecalc == 0:
        print("File for ", materialID, "already exists and force recalce = 0, skipping")
        continue
    surfaceOutfile=open("Datasets/SurfaceHGE/"+materialID+"-noise-"+str(noiseReplicas)+ ".csv","w")
    surfaceOutfile.write( ",".join([str(a) for a in headerSet]) +"\n")
    for energyTarget in energyTargetSet:   
        for r0Val in r0ValRange:
            for itNum in range(noiseReplicas):
                resSet = [ materialID, materialShape, numericShape ,materialPMFSource,materialSSDType,alignOffset,ssdRefDist]   
                for probeDef in allProbes:
                    probe = probeDef[0]
                    if probeDef[1] != "":
                        probeFreeEnergies = HGEFuncs.getValidRegion(   np.copy( moleculePotentials[probeDef[0]] )[:,(2,3)] )
                        #probeFreeEnergies = getValidRegion( waterFreeEnergies[:,(2,3)] )
                    #elif probe=="Methane":
                    #    probeFreeEnergies = getValidRegion( methaneFreeEnergies[:,(2,3)])
                    else:
                        probeHeader = "U"+probe+"dkJmol"
                        probeNumber = freeEnergyHeader.index(probeHeader)
                        #print(r0Val,probe,probeNumber,freeEnergies[:5,(2,probeNumber)])
                        freeEnergies[:, probeNumber] = freeEnergies[:,probeNumber] - freeEnergies[-1,probeNumber]
                        probeFreeEnergies  = HGEFuncs.getValidRegion( freeEnergies[:,(2,probeNumber)])
                        #print(probeFreeEnergies)
                    if itNum > 0:
                        probeFreeEnergies = HGEFuncs.applyNoise(probeFreeEnergies)
                    probeFinalEnergy = probeFreeEnergies[-1,1]
                    probeFreeEnergies[:,1] = probeFreeEnergies[:,1] - probeFinalEnergy
                    probeSubsetMask = probeFreeEnergies[:,0] >= r0Val 
                    probeFreeEnergiesSubset = probeFreeEnergies[   probeSubsetMask  ]
                    probeMinEnergy = np.amin( probeFreeEnergies[:,1])
                    probeRightMinEnergy = np.amin( probeFreeEnergiesSubset[:,1])
                    #r0Val =0.2 # max(0.1, estimateValueLocation(probeFreeEnergies,energyTarget)[0])
                    probeHGE= HGEFuncs.HGECoeffs( probeFreeEnergies, r0Val, nMaxValAll)
                    probeHGE.insert(1, probeFinalEnergy)
                    probeHGE.append(probeMinEnergy)
                    probeHGE.append(probeRightMinEnergy)
                    resSet = resSet + probeHGE
                resLine = ",".join([str(a) for a in resSet])
                noiseoutfile.write(resLine+"\n")
                surfaceOutfile.write(resLine+"\n")
                #print(energyTarget, itNum, resLine )
                if energyTarget == energyTargetBase and itNum == 0:
                    outfile.write(resLine+"\n")
    surfaceOutfile.close()            
                
                
outfile.close()
