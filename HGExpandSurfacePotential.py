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
    


def HGECoeffsInterpolate( inputPotential, r0Val, nmax):
    r0Actual = r0Val
    potentialInterpolated = scipy.interpolate.interp1d(inputPotential[:,0],inputPotential[:,1],  bounds_error=False,fill_value="extrapolate")
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
        hgeCoeffGaussian = scipy.integrate.quadrature( integrand, rminVal, rmaxVal, maxiter=100)
        hgeCoeffRes.append(hgeCoeffGaussian[0])
    #print(hgeCoeffRes)
    return hgeCoeffRes

def HGECoeffs( inputPotential, r0Val, nmax):
    #r0Actual = max(np.amin(inputPotential[:,0]), r0Val)
    r0Actual = r0Val
    potentialInterpolated = scipy.interpolate.interp1d(inputPotential[:,0],inputPotential[:,1], bounds_error=False, fill_value = (10*inputPotential[0,1],  0),kind='slinear')
    #start from either r0+0.001 or the first recorded point, whichever is higher 
    #print(inputPotential[-1,0])
    #print(max( r0Actual+0.00001, inputPotential[0,0]))
    rRange = np.arange( max( r0Actual, inputPotential[0,0]), inputPotential[-1,0], 0.00005)
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
    for i in range(2,len(coeffSet)):
        funcVal += HGEFunc(r[validRegion], r0Val, i-1) * coeffSet[i]
    return funcVal

def estimateValueLocation( potential, target):
    if np.all( potential[:,1] > target):
        return (0.2,500) #no matches found - return a default value
    firstIndex =   np.nonzero( potential[:,1] < target)[0][0] 
    if firstIndex < 1:
        return (potential[firstIndex,0],potential[firstIndex,0])
    pointa = potential[firstIndex - 1]
    pointb = potential[firstIndex]
    mEst = (pointb[1] - pointa[1])/(pointb[0] - pointa[0])
    cEst = -( ( pointb[0]*pointa[1] - pointa[0]*pointb[1]  )/(  pointa[0] - pointb[0] )  )
    crossingEst = (target - cEst) / mEst
    return (crossingEst,target)
    
def getValidRegion(potential,rmin=0.05):
    MaskStart =  np.where(  np.logical_and(  potential[:,0] >= rmin  ,np.logical_and(np.logical_and(    np.isfinite( potential[:,1] )     , potential[:,1] > -1000)  , potential[:,1] < 1000     ) ))[0][0]
    MaskEnd = np.where(  potential[:,0] > 1.5)[0][0]
    return potential[ MaskStart:MaskEnd ]

     
 #material ID, shape, source

warnings.filterwarnings('ignore')
materialSet = np.genfromtxt("Structures/SurfaceDefinitions.csv",dtype=str,delimiter=",")
if materialSet.ndim == 1:
    materialSet = np.array([materialSet])
    
    
plotFigs = 1

nMaxValAll = 18
r0ValC = 0.2

potentialFolder = "SurfacePotentials/"

outfile=open("Datasets/SurfacePotentialCoefficients.csv","w")
ljHGELabels = []
electroHGELabels = []
waterHGELabels = []

CHGELabels=[]
KHGELabels=[]
ClHGELabels=[]

for i in range(0,nMaxValAll+1):
    CHGELabels.append("SurfCProbeC"+str(i))
    KHGELabels.append("SurfKProbeC"+str(i))
    ClHGELabels.append("SurfClProbeC"+str(i))
    waterHGELabels.append("SurfWaterC"+str(i))

headerSet =  [ "SurfID", "shape", "numericShape", "source",  "SurfCProbeR0" ] + CHGELabels + ["SurfKProbeR0"] + KHGELabels + ["SurfClProbeR0"] + ClHGELabels  + ["SurfWaterR0"]+ waterHGELabels
outfile.write( ",".join([str(a) for a in headerSet]) +"\n")
for material in materialSet:
    materialID = material[0]
    print("Starting material ", materialID)
    materialShape = material[1]
    materialPMFSource = material[6]
    #load surface-probe potential and HGE
    try:
        freeEnergies = np.genfromtxt( potentialFolder+materialID+"_fev3.dat",delimiter=",")
    except:
        print("Could not locate potentials for", materialID)
        continue
    try:
        waterFreeEnergies = np.genfromtxt( potentialFolder+materialID+"_waterfe.dat",delimiter=",")
    except:
        print("Could not locate water potentials for", materialID)
        continue        
    energyTarget = 35
    CProbeFE = getValidRegion(freeEnergies[:,(2,3)])
    KProbeFE = getValidRegion(freeEnergies[:,(2,4)])
    ClProbeFE = getValidRegion(freeEnergies[:,(2,5)])
    CProbe0 = CProbeFE[-1,1]
    ClProbe0 = ClProbeFE[-1,1]
    KProbe0 = KProbeFE[-1,1]
    CProbeFE[:,1] = CProbeFE[:,1] - CProbe0
    ClProbeFE[:,1] = ClProbeFE[:,1] - ClProbe0
    KProbeFE[:,1] = KProbeFE[:,1] - KProbe0
    #print(CProbeFE)
    r0ValC = estimateValueLocation(  CProbeFE, energyTarget)[0]
    r0ValK = estimateValueLocation(  KProbeFE, energyTarget)[0]
    r0ValCl = estimateValueLocation(  ClProbeFE, energyTarget)[0]
    if r0ValK < r0ValC - 0.15:
        r0ValK = r0ValC- 0.15
    if r0ValCl < r0ValC - 0.15:
        r0ValCl = r0ValC - 0.15
    
    CProbeHGE = HGECoeffs( CProbeFE , r0ValC, nMaxValAll)
    KProbeHGE = HGECoeffs( KProbeFE , r0ValK, nMaxValAll)
    ClProbeHGE = HGECoeffs( ClProbeFE , r0ValCl, nMaxValAll)
    CProbeHGE.insert(1, CProbe0)
    KProbeHGE.insert(1, KProbe0)
    ClProbeHGE.insert(1, ClProbe0)
    #electroHGE = HGECoeffs( freeEnergies[:,(2,4)] , r0ValAll, nMaxValAll)
    #load surface-water potential and HGE
    waterFreeEnergies = getValidRegion( waterFreeEnergies[:,(2,3)] )
    r0ValWater = estimateValueLocation( waterFreeEnergies, energyTarget)[0]
    if r0ValWater < r0ValC - 0.05:
        r0ValWater = r0ValC - 0.05
    waterProbe0 = waterFreeEnergies[-1,1]
    waterFreeEnergies[:,1] = waterFreeEnergies[:,1] - waterProbe0
    waterHGE = HGECoeffs( waterFreeEnergies , r0ValWater, nMaxValAll)
    waterHGE.insert(1, waterProbe0)
    #write out coefficients to a file
    numericShape = 0
    if materialShape=="cylinder":
        numericShape = 1
    resSet =  [ materialID, materialShape, numericShape ,materialPMFSource ] + CProbeHGE + KProbeHGE + ClProbeHGE  +  waterHGE
    resLine = ",".join([str(a) for a in resSet])
    #print( resLine )
    outfile.write(resLine+"\n")
    if plotFigs == 1:
        plt.figure()
        plt.plot( CProbeFE[::5,0], CProbeFE[::5,1] ,"kx")
        #print(CProbeFE[ CProbeFE[:,0] > CProbeHGE[0]  ,0])
        plt.plot( CProbeFE[ CProbeFE[:,0] > CProbeHGE[0]  ,0], BuildHGEFromCoeffs( CProbeFE[:,0], CProbeHGE) ,"k-")
        plt.plot( KProbeFE[::5,0], KProbeFE[::5,1] ,"rx")
        plt.plot( KProbeFE[ KProbeFE[:,0] > KProbeHGE[0]  ,0], BuildHGEFromCoeffs(KProbeFE[:,0], KProbeHGE) ,"r-")
        plt.plot( ClProbeFE[::5,0], ClProbeFE[::5,1] ,"gx")
        plt.plot( ClProbeFE[ ClProbeFE[:,0] > ClProbeHGE[0]  ,0], BuildHGEFromCoeffs( ClProbeFE[:,0], ClProbeHGE) ,"g-")
        #print(waterFreeEnergies[::,2], waterFreeEnergies[::,3])
        plt.plot( waterFreeEnergies[::,0], waterFreeEnergies[::,1] ,"bx")
        waterRange = np.linspace( waterHGE[0]+0.01, 1.5, 100)
        plt.plot(waterRange, BuildHGEFromCoeffs( waterRange, waterHGE) ,"b-")
        minPlotEnergy = min ( np.amin(waterFreeEnergies[:,1]), np.amin( freeEnergies[:,4]), np.amin(freeEnergies[:,3]))
        #plt.ylim(minPlotEnergy-5,50)
        plt.xlim(0,1.5)
        plt.ylim(top=50)
        plt.savefig( potentialFolder+"/"+materialID+"-fitted.png")
outfile.close()

#plt.show()
