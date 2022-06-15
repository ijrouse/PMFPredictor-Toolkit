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
    potentialInterpolated = scipy.interpolate.interp1d(inputPotential[:,0],inputPotential[:,1], fill_value = (10*inputPotential[0,1],  0))
    #start from either r0+0.001 or the first recorded point, whichever is higher 
    rRange = np.arange( max( r0Actual+0.00001, inputPotential[0,0]), inputPotential[-1,0], 0.0005)
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


materialSet = np.genfromtxt("Structures/SurfaceDefinitions.csv",dtype=str,delimiter=",")
if materialSet.ndim == 1:
    materialSet = np.array([materialSet])
    
    
plotFigs = 1

nMaxValAll = 18
r0ValAll = 0.2

potentialFolder = "SurfacePotentials/"

outfile=open("Datasets/SurfacePotentialCoefficients.csv","w")
ljHGELabels = []
electroHGELabels = []
waterHGELabels = []
for i in range(1,nMaxValAll+1):
    ljHGELabels.append("SurfLJC"+str(i))
    electroHGELabels.append("SurfElC"+str(i))
    waterHGELabels.append("SurfWaterC"+str(i))

headerSet =  [ "SurfID", "shape", "numericShape", "source",  "SurfLJR0" ] + ljHGELabels + ["SurfElR0"] + electroHGELabels  + ["SurfWaterR0"]+ waterHGELabels
outfile.write( ",".join([str(a) for a in headerSet]) +"\n")
for material in materialSet:
    materialID = material[0]
    print("Starting material ", materialID)
    materialShape = material[1]
    materialPMFSource = material[6]
    #load surface-probe potential and HGE
    try:
        freeEnergies = np.genfromtxt( potentialFolder+materialID+"_fev2.dat",delimiter=",")
    except:
        print("Could not locate potentials for", materialID)
        continue
    try:
        waterFreeEnergies = np.genfromtxt( potentialFolder+materialID+"_waterfe.dat",delimiter=",")
    except:
        print("Could not locate water potentials for", materialID)
        continue        
    ljHGE = HGECoeffs( freeEnergies[:,(2,3)] , r0ValAll, nMaxValAll)
    electroHGE = HGECoeffs( freeEnergies[:,(2,4)] , r0ValAll, nMaxValAll)
    #load surface-water potential and HGE

    waterHGE = HGECoeffs( waterFreeEnergies[:,(2,3)] , r0ValAll, nMaxValAll)
    #write out coefficients to a file
    numericShape = 0
    if materialShape=="cylinder":
        numericShape = 1
    resSet =  [ materialID, materialShape, numericShape ,materialPMFSource ] + ljHGE + electroHGE  +  waterHGE
    resLine = ",".join([str(a) for a in resSet])
    print( resLine )
    outfile.write(resLine+"\n")
    if plotFigs == 1:
        plt.figure()
        plt.plot( freeEnergies[::5,2], freeEnergies[::5,3] ,"kx")
        plt.plot( freeEnergies[ freeEnergies[:,2] > ljHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], ljHGE) ,"k-")
        plt.plot( freeEnergies[::5,2], freeEnergies[::5,4] ,"rx")
        plt.plot( freeEnergies[ freeEnergies[:,2] > electroHGE[0]  ,2], BuildHGEFromCoeffs( freeEnergies[:,2], electroHGE) ,"r-")
        #print(waterFreeEnergies[::,2], waterFreeEnergies[::,3])
        plt.plot( waterFreeEnergies[::,2], waterFreeEnergies[::,3] ,"bx")
        plt.plot( waterFreeEnergies[ waterFreeEnergies[:,2] > waterHGE[0]  ,2], BuildHGEFromCoeffs( waterFreeEnergies[:,2], waterHGE) ,"b-")
        minPlotEnergy = min ( np.amin(waterFreeEnergies[:,3]), np.amin( freeEnergies[:,4]), np.amin(freeEnergies[:,3]))
        plt.ylim(minPlotEnergy-5,50)
        plt.savefig( potentialFolder+"/"+materialID+"-fitted.png")
outfile.close()

#plt.show()
