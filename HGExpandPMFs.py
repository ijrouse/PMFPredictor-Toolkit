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
import HGEFuncs



warnings.filterwarnings('ignore')

numReplicas = 5
overrideOffset = 1
randomDownsample = 1

noiseStr = ""
if overrideOffset == 1:
    noiseStr=noiseStr+"nooffset"
if numReplicas > 1:
    noiseStr = noiseStr+"_noise"

if randomDownsample == 1:
    noiseStr = noiseStr+"_downsample"


pmfFolder = "AllPMFs"
pmfsFound = os.listdir(pmfFolder)
targetSet = []
for pmf in pmfsFound:
    targetSet.append( pmfFolder+"/"+pmf )
print(targetSet)
#targetSet = [ "AllPMFsRegularised/Ag100_AFUC.dat","AllPMFsRegularised/Ag100_ALASCA.dat","AllPMFsRegularised/AuFCC100_ALASCA.dat"]
seenMaterials = []

#define parameters for the fitting
nMaxValAll = 20
r0ValAll = 0.25
maximumEnergy = 400
maxR0 = 1.0
minR0 = 0.05

pmfOutputFile = open("Datasets/PMFCoefficientsDiffs-ManualN"+str(numReplicas)+noiseStr+"-aug15.csv","w")

hgeLabels = ["Material","Chemical","TargetE0", "fittedE0","offset","resolution", "r0"] + [ "A"+str(i) for i in range(1,nMaxValAll+1)] + ["NMaxBest", "BestError" ] + ["rEMin", "EMin"] +  [ "D"+str(i) for i in range(1,nMaxValAll+1)]
pmfOutputFile.write( ",".join(hgeLabels) + "\n")

offsetDict = {}
offsetDictFile = open("Datasets/SurfaceOffsetDataFEDists.csv","r")
methaneFEDict = {}
methaneFEDict["default"] =  np.genfromtxt("SurfacePotentials/AuFCC100_methanefe.dat",delimiter=",")

firstline =0
offsetDictFileLines=offsetDictFile.readlines()
for line in offsetDictFileLines:
    if firstline == 0:
        firstline = 1
        continue
    lineParts =  line.strip().split(",")
    offsetDict[lineParts[0]] = float(lineParts[3]) 
offsetDictFile.close()


for target in targetSet:
    #plt.figure()
    material,chemical = target.split("/")[-1].split(".")[0].split("_")  
    if material in methaneFEDict:
        methaneData = methaneFEDict[material]
    else:
        try:
            methaneData = np.genfromtxt("SurfacePotentials/" + material + "_methanefe.dat",delimiter=",")
            methaneFEDict[material] = methaneData
            print("Loaded methane data for", material)
        except:
            print("Methane data not found for", material, "please generate these. Defaulting to AuFCC100")
            methaneData = methaneFEDict["default"]
    if material in offsetDict:
        materialOffsetData = offsetDict[material]
        materialOffsetVal = materialOffsetData
        if material not in seenMaterials:
            print("Offset data found for", material, "using offset", materialOffsetVal)
    else:
        if material not in seenMaterials:
            print("No offset data found for", material, "assuming 0. Please re-run GenerateSurfacePotentials.py")
        materialOffsetVal = 0     
    if material not in seenMaterials:
        seenMaterials.append(material) 
        print("Starting", material)
    PMFData = []
    try:
        pmfText = open(target , "r")
        for line in pmfText:
            if line[0] == "#":
                continue
            if "," in line:
                lineTerms = line.strip().split(",")
            else:
                lineTerms = line.split()
            PMFData.append([float(lineTerms[0]),float(lineTerms[1])])
        pmfText.close()
        foundPMF = 1
        print("Loaded ", material, chemical)
        PMFData = np.array(PMFData)
        PMFData[:,1] = PMFData[:,1] - PMFData[-1,1] #set to 0 at end of PMF
    except: 
        print("Failed to read PMF")
        continue
    #PMFData[:,1] = PMFData[:,1] - PMFData[-1,1]
    if overrideOffset == 0:
        PMFData[:,0] = PMFData[:,0] + materialOffsetVal #offset such that ideally the interesting parts of the PMF are in the 0.2 - 1.0 zone
        offsetApplied = materialOffsetVal
    else:
        offsetApplied = 0
    #plt.plot(PMFData[:,0],PMFData[:,1], "bx")
    #discard PMF with extremely high energies
    #print(PMFData[0])
    #PMFDataOriginal = backfillPMF( PMFData.copy(), 0.18 )
    #backfillPMF( inputPotential, r0Val)
    PMFDataOriginal = PMFData.copy()
    for i in range(numReplicas):
        #PMFData = PMFDataOriginal.copy()
        #PMFData[:,0] = PMFData[:,0] + np.random.uniform( -0.05,0.05)
        #PMFData[:,1] = PMFData[:,1] * np.random.uniform( 0.95,1.05) 
        #PMFData[:,1] = PMFData[:,1] - PMFData[-1,1]
        #closeRangeCutoff = (PMFData[PMFData[:,1] < maximumEnergy ,0])[0]
        #PMFData = PMFData[ PMFData[:,0] >= closeRangeCutoff ]
        #print("Truncating PMF before ", closeRangeCutoff, "  energy here is: ", PMFData[0] )
        r0ValRange =  np.arange( minR0, maxR0, 0.01)
        foundMinima = 0
        for r0Val in r0ValRange:
            if numReplicas > 1:
                PMFData =    HGEFuncs.applyNoise(PMFDataOriginal.copy(), 0.01, 0.01, 0.1)
                if randomDownsample == 1:
                    initialResolution = np.mean(PMFData[2:,0] - PMFData[:-2,0])
                    downsampleWidth = np.random.randint(2,11)
                    downsampleWidth = min(downsampleWidth, int(len(PMFData[:,0])/2) )
                    convWindow = np.ones(downsampleWidth)
                    #rDS = np.convolve( PMFData[:,0], convWindow,'valid')/downsampleWidth
                    #eDS = np.convolve( PMFData[:,1], convWindow,'valid')/downsampleWidth
                    maxBlocks = int( np.floor( len(PMFData)/downsampleWidth  ) )*downsampleWidth
                    rDS = np.mean( np.reshape( PMFData[0:maxBlocks,0], (-1,downsampleWidth)),axis=1)
                    eDS = np.mean( np.reshape(PMFData[0:maxBlocks,1] ,(-1,downsampleWidth)), axis=1)
                    
                    PMFData = np.stack( (rDS,eDS), axis=-1 )
                    #print( "Initial res", initialResolution , "Downsample width: ", downsampleWidth, " new resolution: ", np.mean(PMFData[2:,0] - PMFData[:-2,0]) )
            else:
                PMFData = PMFDataOriginal.copy()
            if r0Val < PMFData[0,0]:
                continue
            closeRangeCutoff = (PMFData[PMFData[:,1] < maximumEnergy ,0])[0]
            PMFData = PMFData[ PMFData[:,0] >= closeRangeCutoff ]
            resolution = np.mean(PMFData[2:,0] - PMFData[:-2,0])
            #PMFData = applyNoise(PMFData)
            #PMFData[:,0] = PMFData[:,0] + np.random.uniform( -0.02,0.02) 
            #PMFData[:,1] = PMFData[:,1]  * np.random.uniform( 1-0.1,1+0.1) + np.random.uniform( -0.1,0.1, len(PMFData))
            PMFData[:,1] = PMFData[:,1] - PMFData[-1,1]
            pmfSubsetMask = PMFData[:,0] >= r0Val 
            pmfSubset = PMFData[   pmfSubsetMask  ]
            #print(pmfSubset)
            #if r0Val < closeRangeCutoff:
            #    print( "r0 < closeRangeCutoff", r0Val, closeRangeCutoff, material, chemical, pmfSubset[0:5])
            #    continue
            if len(pmfSubset) < 2:
                continue
            initialEnergy = pmfSubset[0,1]
            minimaIndex = np.argmin(pmfSubset[:,1])
            rEMinVal = pmfSubset[minimaIndex,0]
            EMinVal = pmfSubset[minimaIndex,1]
            #print(material, chemical, pmfSubset[0:5])
            #if np.any( pmfSubset[2:,1] > initialEnergy):
            #    foundMinima = 1
            #    print("attempting to fit inside a minima, stopping")
            #    break
            hgeSet = HGEFuncs.HGECoeffs(  PMFData, r0Val, nMaxValAll)
            #methaneHGE = HGEFuncs.HGECoeffs( methaneData[:,(2,3)] , r0Val, nMaxValAll)
            #differentialCoeffs = []
            #for i in range(1,nMaxValAll+1):
            #    differentialCoeffs.append( hgeSet[i] - methaneHGE[i])
            #    #print(differentialCoeffs[i-1])
            currentBestVar = np.sum( pmfSubset[:,1]**2)
            currentBestMaxIndex = 0
            #if material=="AuFCC100UCD":
            #    print( BuildHGEFromCoeffs( pmfSubset[:,0], hgeSet[:i+1], 1) )
            #    print( pmfSubset[:,1] )
            for i in range(1,1+nMaxValAll):
                #get PMF mean-square deviation
                pmfVar =  np.trapz( (HGEFuncs.BuildHGEFromCoeffs( pmfSubset[:,0], hgeSet[:i+1], 1) - pmfSubset[:,1])**2,   pmfSubset[:,0])
                if pmfVar < currentBestVar:
                    #print( pmfVar , "beats", currentBestVar, "at index", i)
                    currentBestMaxIndex = i
                    currentBestVar = pmfVar
            resLine = [material,chemical, pmfSubset[0,1] , HGEFuncs.BuildHGEFromCoeffs(pmfSubset[0,0]  , hgeSet,1)[0], offsetApplied,resolution ] + hgeSet  + [currentBestMaxIndex, currentBestVar] + [rEMinVal, EMinVal] 
            #fitData= BuildHGEFromCoeffs(pmfSubset[:,0]  , hgeSet,1)
            if pmfSubset[0,1] < maximumEnergy:
                #plt.plot(pmfSubset[:,0],fitData)
                #print(resLine)
                
                pmfOutputFile.write( ",".join([str(a) for a in resLine])+"\n")
        
pmfOutputFile.close()


