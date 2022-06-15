import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as scspec
import datetime
import scipy.integrate
import scipy.interpolate

def HGEFunc(r, r0, n):
    return (-1)**(1+n) * np.sqrt( 2*n - 1) * np.sqrt(r0)/r * scspec.hyp2f1(1-n,n,1,r0/r)
    
    
def HGECoeffs( inputPotential, r0Val, nmax):
    #r0Actual = max(np.amin(inputPotential[:,0]), r0Val)
    r0Actual = r0Val
    #print(inputPotential)
    potentialInterpolated = scipy.interpolate.interp1d(inputPotential[:,0],inputPotential[:,1], fill_value = (10*inputPotential[0,1],  0))
    #start from either r0+0.001 or the first recorded point, whichever is higher 
    rRange = np.arange( max( r0Actual+0.00001, inputPotential[0,0]), min(1.5,inputPotential[-1,0]), 0.0001)
    potentialUpscaled = potentialInterpolated(rRange)
    
    #inputRange = inputPotential [ np.logical_and( inputPotential[:,0] > r0Actual ,inputPotential[:,0] <= 1.5 ) ]
    #print("Integrating over ", rRange[0] , " to ", rRange[-1])
    hgeCoeffRes = [r0Actual]
    for n in range(1,nmax+1):
        hgeCoeff =  scipy.integrate.trapz( potentialUpscaled*HGEFunc( rRange,r0Actual, n),  rRange )
        hgeCoeffRes.append(hgeCoeff)
    return hgeCoeffRes

def BuildHGEFromCoeffs(r , coeffSet):
    r0Val = coeffSet[0]
    validRegion = r > r0Val
    funcVal = np.zeros_like(r[validRegion])
    for i in range(1,len(coeffSet)):
        funcVal += HGEFunc(r[validRegion], r0Val, i) * coeffSet[i]
    return funcVal

    
 #material ID, shape, source
materialSet = [

    ["AuFCC100",   "plane",  0] ,
["GoldBrush","plane",0]
     
]  

materialSet = np.genfromtxt("Structures/ChemicalDefinitions.csv",dtype=str,delimiter=",")
if materialSet.ndim == 1:
    materialSet = np.array([materialSet])
    
    
    
nMaxValAll = 18
fitEnergyStart = 25
r0ValAll = 0.25

potentialFolder = "ChemicalPotentials/"

outfile=open("Datasets/ChemicalPotentialCoefficients.csv","w")

ljHGELabels = []
electroHGELabels = []
waterHGELabels = []
slabHGELabels = []
for i in range(1,nMaxValAll+1):
    ljHGELabels.append("ChemLJC"+str(i))
    electroHGELabels.append("ChemElC"+str(i))
    waterHGELabels.append("ChemWaterC"+str(i))
    slabHGELabels.append("ChemSlabC"+str(i))

headerSet =  [ "ChemID", "SMILES" ,"ChemLJR0" ] + ljHGELabels + ["ChemElR0"] + electroHGELabels  + ["ChemWaterR0"]+ waterHGELabels   + ["ChemSlabR0"]+ slabHGELabels
outfile.write( ",".join([str(a) for a in headerSet]) +"\n")

plotFigs = 1
materialNotFoundList = []
for material in materialSet:
    materialID = material[0]
    chemSMILES = material[1]
    print("Starting material ", materialID)
    #load surface-probe potential and HGE
    try:
        freeEnergies = np.genfromtxt( potentialFolder+materialID+"_fev2.dat",delimiter=",")
    except:
        print("Free energy file not found for ", materialID)
        materialNotFoundList.append( material)
        continue
    r0ValAll =  freeEnergies[  np.where(   freeEnergies[:,3] < fitEnergyStart )[0][0],2]
    r0ValSlab = freeEnergies[  np.where(   freeEnergies[:,5] < fitEnergyStart )[0][0],2]
    ljHGE = HGECoeffs( freeEnergies[:,(2,3)] , r0ValAll, nMaxValAll)
    electroHGE = HGECoeffs( freeEnergies[:,(2,4)] , r0ValAll, nMaxValAll)
    slabMaskStart =  np.where(   np.isfinite( freeEnergies[:,5] ) )[0][0]
    #print(slabMaskStart)
    slabPotential = freeEnergies[ slabMaskStart:      , (2,5) ]
    slabHGE = HGECoeffs( slabPotential, r0ValSlab, nMaxValAll)
    #load surface-water potential and HGE
    waterFreeEnergies = np.genfromtxt( potentialFolder+materialID+"_waterfe.dat",delimiter=",")
    waterHGE = HGECoeffs( waterFreeEnergies[:,(2,3)] , r0ValAll, nMaxValAll)
    #write out coefficients to a file
    resSet =  [ materialID, chemSMILES] + ljHGE +  electroHGE  + waterHGE + slabHGE
    resLine = ",".join([str(a) for a in resSet])
    #print( resLine )
    outfile.write(resLine+"\n")
    if plotFigs == 1:
        plt.figure()
        plt.plot( freeEnergies[::5,2], freeEnergies[::5,3] ,"kx")
        plt.plot( freeEnergies[ freeEnergies[:,2] > ljHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], ljHGE) ,"k-")
        plt.plot( freeEnergies[::5,2], freeEnergies[::5,4] ,"rx")
        plt.plot( freeEnergies[ freeEnergies[:,2] > electroHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], electroHGE) ,"r-")
        plt.plot( freeEnergies[::5,2], freeEnergies[::5,5] ,"gx")
        plt.plot( freeEnergies[ freeEnergies[:,2] > slabHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], slabHGE) ,"g-")


        plt.plot( waterFreeEnergies[::,2], waterFreeEnergies[::,3] ,"bx")
        plt.plot( waterFreeEnergies[ waterFreeEnergies[:,2] > waterHGE[0]  ,2], BuildHGEFromCoeffs( waterFreeEnergies[:,2], waterHGE) ,"b-")
        minPlotEnergy = min ( np.amin(waterFreeEnergies[:,3]), np.amin( freeEnergies[:,4]), np.amin(freeEnergies[:,3]))
        plt.ylim(minPlotEnergy-5,50)
        plt.savefig( potentialFolder+"/"+materialID+"-fitted.png")
outfile.close()

if len(materialNotFoundList) > 0:
    print("Some chemicals were not found ")
    print(materialNotFoundList)
    #for notFoundChem in materialNotFoundList:
    #print("Afterwards re-run chemical potential scripts")
#plt.show()
