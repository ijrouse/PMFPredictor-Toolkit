#Generates the free energy map of a probe atom around a chemical
#conventions: energies are in units kJ/mol, distances are in nm.
#Ian Rouse, ian.rouse@ucd.ie , 13/06/2022


import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special as scspec
import scipy.integrate
import argparse


parser = argparse.ArgumentParser(description="Parameters for GenerateChemicalPotentials")
parser.add_argument("-f","--forcerecalc", type=int,default=0,help="If 1 then potentials are recalculated even if their table already exists")
args = parser.parse_args()

if args.forcerecalc == 1:
    print("Recalculating all potentials")

def estimateValueLocation( potential, target):
    firstIndex =   np.nonzero( potential[:,1] < target)[0][0] 
    if firstIndex < 1:
        return (potential[firstIndex,0],potential[firstIndex,0])
    pointa = potential[firstIndex - 1]
    pointb = potential[firstIndex]
    mEst = (pointb[1] - pointa[1])/(pointb[0] - pointa[0])
    cEst = -( ( pointb[0]*pointa[1] - pointa[0]*pointb[1]  )/(  pointa[0] - pointb[0] )  )
    crossingEst = (target - cEst) / mEst
    return (crossingEst,target)



def rotateMatrix(theta,phi):
    return np.array([ [ np.cos(theta) * np.cos(phi) , -1 * np.cos(theta) * np.sin(phi) , np.sin(theta) ],
      [  np.sin(phi)                 ,   np.cos(phi)                    , 0 ],
      [ -1 * np.sin(theta) * np.cos(phi) ,   np.sin(theta) * np.sin(phi) , np.cos(theta) ]
    ])

def HGEFunc(r, r0, n):
    return (-1)**(1+n) * np.sqrt( 2*n - 1) * np.sqrt(r0)/r * scspec.hyp2f1(1-n,n,1,r0/r)
    
    
def HGECoeffs( inputPotential, r0Val, nmax):
    r0Actual = max(np.amin(inputPotential[:,0]), r0Val)
    hgeCoeffRes = [r0Actual]
    for n in range(1,nmax+1):
        hgeCoeff =  scipy.integrate.simpson( inputPotential[:,1]*HGEFunc( inputPotential[:,0] ,r0Actual, n),  inputPotential[:,0] )
        hgeCoeffRes.append(hgeCoeff)
    return hgeCoeffRes
    
def centerProbe( probeDef):
    m=0
    cx =0
    cy =0
    cz = 0
    for i in range(len(probeDef)):
        m+= probeDef[i][4]
        cx += probeDef[i][0] * probeDef[i][4]
        cy += probeDef[i][1] * probeDef[i][4]
        cz += probeDef[i][2] * probeDef[i][4]
    for i in range(len(probeDef)):
        probeDef[i][0] =probeDef[i][0]  - cx/m
        probeDef[i][1] = probeDef[i][1]  - cy/m
        probeDef[i][2] = probeDef[i][2]  - cz/m
    return probeDef
    
    
#define parameters used for the free energy calculation. probeEpsilon, probeSigma
temperature = 300.0
probeEpsilon = 0.3598
probeSigma = 0.339

slabBeadEpsilon = 0.3598
slabBeadSigma = 0.339

dielectricConst = 1

pointProbes =[
["C",0.339,0.3598,0],
["K",0.314264522824 ,  0.36401,1],
["Cl",0.404468018036 , 0.62760,-1],
["C2A",0.2,0.3598,0],
["C4A",0.4,0.3598,0],
["C6A",0.6,0.3598,0],
["C8A",0.8,0.3598,0],
["C10A",1.0,0.3598,0]
]



waterProbe =  [
[0.00305555555556,-0.00371666666667,0.00438888888889,  -0.834, 16, 0.315 , 0.636],
[0.01195555555556,0.09068333333333,-0.00881111111111,    0.417, 1, 0.0,0.0],
[-0.06084444444444,-0.03121666666667,-0.06141111111111,   0.417, 1, 0.0, 0.0]
]

waterUCDProbe =  [
[0.2531 ,  0.0596,  -0.2477,  -0.834, 16, 0.315057422683 ,0.63639],
[0.2620 ,  0.1540,  -0.2609,    0.417, 1, 0.040001352445, 0.19246],
[0.1892 ,   0.0321,  -0.3135,   0.417, 1, 0.040001352445,  0.19246]
]

methaneProbe=[
[0.108,0.006,0.001,-0.106800,12.01000,3.39771e-01,4.51035e-01],
[0.072,0.109,0.000,0.026700,1.00800,2.60018e-01,8.70272e-02],
[0.072,-0.047,-0.087,0.026700,1.00800,2.60018e-01,8.70272e-02],
[0.072,-0.045,0.091,0.026700,1.00800,2.60018e-01,8.70272e-02],
[0.217,0.006,0.001,0.026700,1.00800,2.60018e-01,8.70272e-02]
]

moleculeProbes = [
["water", waterProbe], 
["waterUCD", waterUCDProbe]
]

feFileHeader = "r[nm],d[nm],daligned[nm],U(d)[kj/mol],V(d)[e/nm],USlab(d)[kJ/mol]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe[0]+"(d)[kJ/mol]"



conversionFactor = ( 8.314/1000.0) * temperature

inputFolder = "Structures/Chemicals"
outputFolder = "ChemicalPotentials"
os.makedirs(outputFolder, exist_ok=True)
debyeLength = 0.7
avogadroNum = 6.022e23
electroToKJMol = 1.0/(dielectricConst) *   1.4399 * 1.6e-19 *avogadroNum/1000.0
#1.4399 arises from (1/4 pi eps0) * 1 elementary charge/1nm to give scaling factor to V , 1.6e-19 is the second elementary charge to give an energy in J, multiply by atoms/mol , divide by 1000 to get kJ/mol
#this is quite a large number! 
slabPackingEfficiency = 1.0
slabDensity = 6 * slabPackingEfficiency/(np.pi * slabBeadSigma**3)

#label, structure file
targetSet = np.genfromtxt("Structures/ChemicalDefinitions.csv",dtype=str,delimiter=",")
if targetSet.ndim == 1:
    targetSet = np.array([targetSet])

surfaceType = "sphere"
r0Start = 0
for target in targetSet:
    foundPMF = 0
    targetName = target[0]
    '''
    canSkip = 0
    if args.forcerecalc == 0:
        feOutput = outputFolder+"/" +targetName+"_fev3.dat"
        waterOutput = outputFolder+"/" + targetName+"_waterfe.dat"
        if os.path.exists(feOutput) and os.path.exists(waterOutput):
            print("Both tables exist for", targetName, ", skipping")
            canSkip = 1
    if canSkip == 1:
        continue
    '''


    targetInputFile =  target[0]+"_combined.csv"
    targetFormalCharge = target[1]
    # indexes: 0 = numeric ID, 1 = atomID , 2= atom type, 3= x, 4 = y, 5 = z, 6 = charge, 7= mass, 8 = sigma, 9 = epsilon
    nmData = np.genfromtxt(inputFolder+"/"+targetInputFile, skip_header=1,delimiter=",",dtype=str) 
    atomNumericData = nmData[:, 3:].astype(float)
    print("Loaded ", len(atomNumericData), "atoms from file")


    #indexes: 0,1,2 = x,y,z, 3=charge, 4=mass, 5=sigma, 6=epsilon
    #set the centre of mass to 0,0,0
    atomNumericData[:,0] = atomNumericData[:,0] - np.sum(atomNumericData[:,0] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
    atomNumericData[:,1] = atomNumericData[:,1] - np.sum(atomNumericData[:,1] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
    atomNumericData[:,2] = atomNumericData[:,2] - np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
    rRange =  np.arange(0.0, 1.6, 0.0025)
    #sigmaBroadcast, epsBroadcast, chargeBroadcast are all unmixed values for each atom in the molecule. sigmaBroadcastSlab, epsBroadcastSlab are calculated using mixing rules
    if surfaceType == "sphere":
        numc1=16
        numc2=9
        c1Range = np.linspace(0,2*np.pi, num = numc1, endpoint=False)
        c2Range = np.linspace(0, np.pi, num = numc2, endpoint=True)
        c1grid,c2grid = np.meshgrid(   c1Range, c2Range) 
        sigmaBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,5] , (numc1,numc2,len(atomNumericData)) ))
        epsBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,6] , (numc1,numc2,len(atomNumericData)) ))
        chargeBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,3] , (numc1,numc2,len(atomNumericData)) ))
        sigmaBroadcastSlab = 0.5*(sigmaBroadcast + slabBeadSigma)
        epsBroadcastSlab = np.sqrt( epsBroadcast*slabBeadEpsilon)
    else:
        numc1=23
        numc2=23
        c1Range = np.linspace(-1.0, 1.0, num = numc1, endpoint=True)
        c2Range = np.linspace(-1.0, 1.0, num = numc2, endpoint=True)
        areaTerm = (c2Range[-1] - c2Range[0]) * (c1Range[-1] - c1Range[0])
        c1grid,c2grid = np.meshgrid(c1Range,c2Range)
        sigmaBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,5] , (numc1,numc2,len(atomNumericData)) ))
        epsBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,6] , (numc1,numc2,len(atomNumericData)) ))
        chargeBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,3] , (numc1,numc2,len(atomNumericData)) ))
        sigmaBroadcastSlab = 0.5*(sigmaBroadcast + slabBeadSigma)
        epsBroadcastSlab = np.sqrt( epsBroadcast*slabBeadEpsilon)
    resList = []
    waterResList = []
    lastInfPoint = rRange[0] - 1
    lastWaterInfPoint = lastInfPoint
    
    
    feOutput = outputFolder+"/" +targetName+"_fev3.dat"
    skipProbes = 0
    if os.path.exists(feOutput) and args.forcerecalc == 0:
        skipProbes = 1
        print("Found probes for", target[0])
    else:
        print("Starting point probes for ", target[0])    
        for r in rRange:
            #vdw and electrostatic probes
            distList = []
            slabDistList = []
            for i in range(len(atomNumericData)): #There is probably a more efficient way of doing this but in the interest of clarity this is easier to debug.
                if surfaceType == "sphere":
                    atomDist = np.sqrt( ( r* np.cos(c1grid) *np.sin(c2grid)- atomNumericData[i,0] )**2 + ( r* np.sin(c1grid)*np.sin(c2grid) - atomNumericData[i,1] )**2  + ( r*np.cos(c2grid) - atomNumericData[i,2] )**2 )
                    distList.append(atomDist)
                    #dist from the atom to an infinite half-slab with "surface centre" at {r cos[phi] sin[theta], r sin[phi] sin[theta] , r cos[theta]
                    slabDist = np.sqrt(  (  atomNumericData[i,0] * np.cos(c1grid) * np.sin(c2grid)    + atomNumericData[i,1] * np.sin(c1grid)*np.sin(c2grid) + atomNumericData[i,2] * np.cos(c2grid) - r   )**2)
                    slabDistList.append(slabDist)
                else:
                    atomDist = np.sqrt(   ( c1grid - atomNumericData[i,0])**2 + (c2grid - atomNumericData[i,1])**2 + (r - atomNumericData[i,2])**2  )
                    distList.append(atomDist)
            distArray = np.array(distList)
            slabDistArray = np.array(slabDistList)
            electricContributions = chargeBroadcast  / distArray
            scaledDists = sigmaBroadcastSlab/distArray
            scaledDistsSlab = sigmaBroadcastSlab/slabDistArray 
            allContributions = np.sum( 4*epsBroadcastSlab*( scaledDists**12 - scaledDists**6) , axis=0)
            #slabPackingEfficiency
            slabPotentialTerms = np.sum(  2*epsBroadcastSlab * (sigmaBroadcastSlab**6)* np.pi * slabDensity *(2 * sigmaBroadcastSlab**6 - 15*slabDistArray**6)/(45 * slabDistArray**9)            ,axis=0)
            ones = 1 + np.zeros_like(slabPotentialTerms)
        
        
            freeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -allContributions / conversionFactor) )  / np.sum(ones) )
            if np.any( slabPotentialTerms < -300) or np.all(slabPotentialTerms > 200):
               slabFreeEnergy = np.amin(slabPotentialTerms)
            else:
               slabFreeEnergy =-conversionFactor * np.log( np.sum(   np.exp( -slabPotentialTerms / conversionFactor) )  / np.sum(ones) )
            electrostatic = np.sum( electricContributions * np.exp( -allContributions / conversionFactor) ) / np.sum(np.exp( -allContributions / conversionFactor))
            
            probeFESet = []
            foundInf = 0
            allInf = 1
            cInf = 0
            for probe in pointProbes:  ##name, sigma,epsilon,   charge
                epsCombined = np.sqrt(epsBroadcast * probe[2])
                sigmaCombined = 0.5*(sigmaBroadcast + probe[1])
                electrostaticPotential = np.sum( electroToKJMol*probe[3]*electricContributions   , axis=0)
                scaledDists = sigmaCombined/distArray 
                ljPotential = np.sum( 4 * epsCombined * (scaledDists**12 - scaledDists**6 ), axis=0) #sum over atoms
                totalPotential = electrostaticPotential + ljPotential
            
            
                if np.any( totalPotential < -300     ):
                    probeFreeEnergy = np.amin(totalPotential)
                elif np.all(totalPotential > 200):
                    probeFreeEnergy = np.amin(totalPotential)
                else:
                    probeFreeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones) )            
                
                #probeFreeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones) )
                #print(r, probe, probeFreeEnergy)
                probeFESet.append(probeFreeEnergy)
                #if np.isfinite(freeEnergy):
                #print(r-r0Start,freeEnergy)
            resList.append( [r,r-r0Start, r-r0Start, freeEnergy,electrostatic,slabFreeEnergy]+ probeFESet)
            if not np.isfinite( probeFESet[0]):
                lastInfPoint = r
        resArray = np.array(resList)
        resArray = resArray[    resArray[:,0] > lastInfPoint ]
        np.savetxt( outputFolder+"/" +targetName+"_fev3.dat", resArray, fmt='%2.7f',delimiter=",", header=feFileHeader)
    

    for moleculeProbeDef in moleculeProbes:
        moleculeTag = moleculeProbeDef[0]
        moleculeStructure = centerProbe(moleculeProbeDef[1])
        outputLoc = outputFolder+"/" +     targetName+"_"+moleculeTag+"fe.dat"
        if args.forcerecalc == 0 and os.path.exists(outputLoc):
            print("Found", moleculeTag, " for " , targetName)
            continue
        else:
            print("Starting", moleculeTag)
        rRangeWater =  np.arange(0, 1.6,  0.005)  
        waterResList = []         
        for r in rRangeWater:            
            runningNumerator = 0
            runningDenominator = 0
            waterMin = 1e12
            for theta in  np.linspace(0,np.pi, num = 5, endpoint=True) :
                for phi in np.linspace(0,2*np.pi, num = 8, endpoint=False):
                    rotateMatrixInternal = rotateMatrix(np.pi - theta,-phi)
                    allContributions = 0
                    for atom in moleculeStructure:
                        #print(theta,phi,atom)
                        epsCombined = np.sqrt(epsBroadcast * atom[6])
                        sigmaCombined = 0.5*(sigmaBroadcast + atom[5])
                        ax = atom[0] * rotateMatrixInternal[0,0] + atom[1]*rotateMatrixInternal[0,1] + atom[2]*rotateMatrixInternal[0,2]
                        ay = atom[0] * rotateMatrixInternal[1,0] + atom[1]*rotateMatrixInternal[1,1] + atom[2]*rotateMatrixInternal[1,2]
                        az = atom[0] * rotateMatrixInternal[2,0] + atom[1]*rotateMatrixInternal[2,1] + atom[2]*rotateMatrixInternal[2,2]
                        distList = []
                        for i in range(len(atomNumericData)): #There is probably a more efficient way of doing this but in the interest of clarity this is easier to debug.
                            if surfaceType == "sphere":
                                atomDist = np.sqrt( ( r* np.cos(c1grid) *np.sin(c2grid)- atomNumericData[i,0] )**2 + ( r* np.sin(c1grid)*np.sin(c2grid) - atomNumericData[i,1] )**2  + ( r*np.cos(c2grid) - atomNumericData[i,2] )**2 )
                                distList.append(atomDist)
                            else:
                                atomDist = np.sqrt(   ( c1grid + ax - atomNumericData[i,0])**2 + (c2grid + ay - atomNumericData[i,1])**2 + (r + az - atomNumericData[i,2])**2  )
                                distList.append(atomDist)
                        distArray = np.array(distList)
                        electricContributions = np.sum( chargeBroadcast   / distArray , axis=0 )
                        scaledDists = sigmaCombined/distArray 
                        ljContributions =  np.sum( 4*epsCombined*( scaledDists**12 - scaledDists**6) , axis=0)
                        allContributions += (electroToKJMol*atom[3]*electricContributions + ljContributions)
                    waterMin = min( waterMin, np.amin(allContributions)) #this is used only if there's no better value under the assumption its either the least bad or most favourable
                    ones = (  1 + np.zeros_like(allContributions) )
                    runningNumerator += np.sum(  np.exp(-allContributions/conversionFactor)     )* np.sin(theta)
                    runningDenominator += np.sum(ones * np.sin(theta))
        
            waterFreeEnergy=-conversionFactor * np.log( runningNumerator/runningDenominator)
            if not np.isfinite(waterFreeEnergy):
                waterFreeEnergy = waterMin
            if np.isfinite(waterFreeEnergy):
                #print(r-r0Start,waterFreeEnergy)
                waterResList.append( [r,r-r0Start, r-r0Start, waterFreeEnergy])            
            else:
                lastWaterInfPoint = r    
        waterResArray = np.array(waterResList)
        waterResArray = waterResArray[ waterResArray[:,0] > lastWaterInfPoint ]
        np.savetxt( outputLoc, waterResArray, fmt='%2.7f',delimiter=",", header="r[nm],d[nm],daligned[nm],U"+moleculeTag+"(d)[kj/mol]")

