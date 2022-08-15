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


def HGEFunc(r, r0, n):
    return (-1)**(1+n) * np.sqrt( 2*n - 1) * np.sqrt(r0)/r * scspec.hyp2f1(1-n,n,1,r0/r)

def backfillPMF( inputPotential, r0Val):
    if  inputPotential[0,0] > r0Val:
        #Fill-in missing values to ensure the integral extends to at least r0 to get a physically meaningful expansion. we assume this region is purely repulsive and of the form a + b/r^12
        missingSection = np.arange( r0Val - 0.01, inputPotential[0,0], 0.01)
        gradAtStart = (inputPotential[2,1] - inputPotential[0,1])/(inputPotential[2,0] - inputPotential[0,0])
        missingSectionPotential = inputPotential[0,1] + 1.0/12.0 * gradAtStart * inputPotential[0,0]*(1 - ( inputPotential[0,0]/missingSection  )**12 )
        finalInputPotential = np.concatenate( (  np.transpose(np.array([missingSection,missingSectionPotential])   ) , inputPotential  ), axis=0 )
        return finalInputPotential
    else:
        return inputPotential
    
    
def HGECoeffs( inputPotential, r0Val, nmax, backfill = False):
    #r0Actual = max(np.amin(inputPotential[:,0]), r0Val)
    r0Actual = r0Val
    if backfill == True and inputPotential[0,0] > r0Val:
        #Fill-in missing values to ensure the integral extends to at least r0 to get a physically meaningful expansion. we assume this region is purely repulsive and of the form a + b/r^12
        missingSection = np.arange( r0Val - 0.01, inputPotential[0,0], 0.01)
        gradAtStart = (inputPotential[2,1] - inputPotential[0,1])/(inputPotential[2,0] - inputPotential[0,0])
        missingSectionPotential = inputPotential[0,1] + 1.0/12.0 * gradAtStart * inputPotential[0,0]*(1 - ( inputPotential[0,0]/missingSection  )**12 )
        finalInputPotential = np.concatenate( (  np.transpose(np.array([missingSection,missingSectionPotential])   ) , inputPotential  ), axis=0 )
        print("Backfilling: ", missingSection, missingSectionPotential)
        print("To join: ", inputPotential[:10,0], inputPotential[:10,1])
    else:
        finalInputPotential = inputPotential
    potentialInterpolated = scipy.interpolate.interp1d(finalInputPotential[:,0],finalInputPotential[:,1], fill_value = (2*finalInputPotential[0,1],  0))
    #start from  just before r0 or the second recorded point, whichever is higher and to ensure the gradients are still somewhat sensible
    rminVal =  max( r0Actual, inputPotential[1,0])
    rmaxVal = inputPotential[-1,0]
    #rRange = np.arange( max( r0Actual, inputPotential[0,0]), inputPotential[-1,0], 0.000001)
    #potentialUpscaled = potentialInterpolated(rRange)
    
    #inputRange = inputPotential [ np.logical_and( inputPotential[:,0] > r0Actual ,inputPotential[:,0] <= 1.5 ) ]
    #print("Integrating over ", rRange[0] , " to ", rRange[-1])
    hgeCoeffRes = [r0Actual]
    for n in range(1,nmax+1):
        integrand = lambda x: potentialInterpolated(x) * HGEFunc(x, r0Actual, n)
        #hgeCoeff =  scipy.integrate.trapz( potentialUpscaled*HGEFunc( rRange,r0Actual, n),  rRange )
        hgeCoeffGaussian = scipy.integrate.quadrature( integrand, rminVal, rmaxVal, maxiter=200)
        hgeCoeffRes.append(hgeCoeffGaussian[0])
    #print(hgeCoeffRes)
    return hgeCoeffRes

def BuildHGEFromCoeffs(r , coeffSet, forceValid = 0):
    r0Val = coeffSet[0]
    if forceValid == 0:
        validRegion = r > r0Val
    else:
        validRegion = r > 0
    funcVal = np.zeros_like(r[validRegion])
    for i in range(1,len(coeffSet)):
        funcVal += HGEFunc(r[validRegion], r0Val, i) * coeffSet[i]
    return funcVal

def applyNoise(freeEnergySet):
    #translate with probability 0.5
    freeEnergySet[:,0] = freeEnergySet[:,0] + np.random.normal( 0, 0.01)
    freeEnergySet[:,1] = freeEnergySet[:,1] * np.random.normal( 1, 0.01) + np.random.normal( 0, 0.1, len(freeEnergySet))
    return freeEnergySet


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
                PMFData =  applyNoise( PMFDataOriginal.copy() )
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
            hgeSet = HGECoeffs(  PMFData, r0Val, nMaxValAll)
            methaneHGE = HGECoeffs( methaneData[:,(2,3)] , r0Val, nMaxValAll)
            differentialCoeffs = []
            for i in range(1,nMaxValAll+1):
                differentialCoeffs.append( hgeSet[i] - methaneHGE[i])
                #print(differentialCoeffs[i-1])
            currentBestVar = np.sum( pmfSubset[:,1]**2)
            currentBestMaxIndex = 0
            #if material=="AuFCC100UCD":
            #    print( BuildHGEFromCoeffs( pmfSubset[:,0], hgeSet[:i+1], 1) )
            #    print( pmfSubset[:,1] )
            for i in range(1,1+nMaxValAll):
                #get PMF mean-square deviation
                pmfVar =  np.trapz( (BuildHGEFromCoeffs( pmfSubset[:,0], hgeSet[:i+1], 1) - pmfSubset[:,1])**2,   pmfSubset[:,0])
                if pmfVar < currentBestVar:
                    #print( pmfVar , "beats", currentBestVar, "at index", i)
                    currentBestMaxIndex = i
                    currentBestVar = pmfVar
            resLine = [material,chemical, pmfSubset[0,1] , BuildHGEFromCoeffs(pmfSubset[0,0]  , hgeSet,1)[0], offsetApplied,resolution ] + hgeSet  + [currentBestMaxIndex, currentBestVar] + [rEMinVal, EMinVal] + differentialCoeffs
            #fitData= BuildHGEFromCoeffs(pmfSubset[:,0]  , hgeSet,1)
            if pmfSubset[0,1] < maximumEnergy:
                #plt.plot(pmfSubset[:,0],fitData)
                #print(resLine)
                
                pmfOutputFile.write( ",".join([str(a) for a in resLine])+"\n")
        
pmfOutputFile.close()
#plt.show()
'''
materialSet = np.genfromtxt("Structures/ChemicalDefinitions.csv",dtype=str,delimiter=",")
if materialSet.ndim == 1:
    materialSet = np.array([materialSet])
'''
    
    

'''
potentialFolder = "ChemicalPotentials/"

outfile=open("Datasets/ChemicalPotentialCoefficients.csv","w")

ljHGELabels = []
electroHGELabels = []
waterHGELabels = []
for i in range(1,nMaxValAll+1):
    ljHGELabels.append("ChemLJC"+str(i))
    electroHGELabels.append("ChemElC"+str(i))
    waterHGELabels.append("ChemWaterC"+str(i))

headerSet =  [ "ChemID", "SMILES" ,"ChemLJR0" ] + ljHGELabels + ["ChemElR0"] + electroHGELabels  + ["ChemWaterR0"]+ waterHGELabels
outfile.write( ",".join([str(a) for a in headerSet]) +"\n")

plotFigs = 1

for material in materialSet:
    materialID = material[0]
    chemSMILES = material[1]
    print("Starting material ", materialID)
    #load surface-probe potential and HGE
    freeEnergies = np.genfromtxt( potentialFolder+materialID+"_fev2.dat",delimiter=",")
    ljHGE = HGECoeffs( freeEnergies[:,(2,3)] , r0ValAll, nMaxValAll)
    electroHGE = HGECoeffs( freeEnergies[:,(2,4)] , r0ValAll, nMaxValAll)
    #load surface-water potential and HGE
    waterFreeEnergies = np.genfromtxt( potentialFolder+materialID+"_waterfe.dat",delimiter=",")
    waterHGE = HGECoeffs( waterFreeEnergies[:,(2,3)] , r0ValAll, nMaxValAll)
    #write out coefficients to a file
    resSet =  [ materialID, chemSMILES] + ljHGE +  electroHGE  + waterHGE
    resLine = ",".join([str(a) for a in resSet])
    #print( resLine )
    outfile.write(resLine+"\n")
    if plotFigs == 1:
        plt.figure()
        plt.plot( freeEnergies[::5,2], freeEnergies[::5,3] ,"kx")
        plt.plot( freeEnergies[ freeEnergies[:,2] > ljHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], ljHGE) ,"k-")
        plt.plot( freeEnergies[::5,2], freeEnergies[::5,4] ,"rx")
        plt.plot( freeEnergies[ freeEnergies[:,2] > electroHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], electroHGE) ,"r-")
        plt.plot( waterFreeEnergies[::,2], waterFreeEnergies[::,3] ,"bx")
        plt.plot( waterFreeEnergies[ waterFreeEnergies[:,2] > waterHGE[0]  ,2], BuildHGEFromCoeffs( waterFreeEnergies[:,2], waterHGE) ,"b-")
        minPlotEnergy = min ( np.amin(waterFreeEnergies[:,3]), np.amin( freeEnergies[:,4]), np.amin(freeEnergies[:,3]))
        plt.ylim(minPlotEnergy-5,50)
        plt.savefig( potentialFolder+"/"+materialID+"-fitted.png")
outfile.close()

#plt.show()
'''
