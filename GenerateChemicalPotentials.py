#Generates the free energy map of a probe atom around a chemical
#conventions: energies are in units kJ/mol, distances are in nm.
#Ian Rouse, ian.rouse@ucd.ie , 13/06/2022


import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special as scspec
import scipy.integrate
import argparse
import HGEFuncs
import GeometricFuncs
import PotentialProbes

parser = argparse.ArgumentParser(description="Parameters for GenerateChemicalPotentials")
parser.add_argument("-f","--forcerecalc", type=int,default=0,help="If 1 then potentials are recalculated even if their table already exists")
parser.add_argument("-i","--initial", type=int, default=0,help="Initial structure to start calculating for multiprocessing")
parser.add_argument("-s","--step", type=int, default=1,help="Stride for slicing for multiprocessing")

args = parser.parse_args()

if args.forcerecalc == 1:
    print("Recalculating all potentials")

    
#define parameters used for the free energy calculation. probeEpsilon, probeSigma
temperature = 300.0
probeEpsilon = 0.3598
probeSigma = 0.339

slabBeadEpsilon = 0.3598
slabBeadSigma = 0.339

dielectricConst = 1



pointProbes = PotentialProbes.getPointProbeSet([  "C","K","Cl","C2A","C4A","CPlus","CMinus","CMoreLJ","CLessLJ","CEps20"])

moleculeProbes = [
["water", PotentialProbes.waterProbe], 
["waterUCD", PotentialProbes.waterUCDProbe],
["methane", PotentialProbes.methaneProbe],
["cline", PotentialProbes.clineProbe],
["carbring", PotentialProbes.sixcarbProbe]

]

feFileHeader = "r[nm],d[nm],daligned[nm],U(d)[kj/mol],V(d)[e/nm],USlab(d)[kJ/mol]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe.name+"(d)[kJ/mol]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe.name+"Min(d)[kJ/mol]"



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
for target in targetSet[args.initial::args.step]:
    foundPMF = 0
    targetName = target[0]

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
    #if surfaceType == "sphere":
    #    numc1=16
    #    thetaDelta = 15.0
    #    c1Range = np.linspace(0,2*np.pi, num = numc1, endpoint=False)
    #    c2Range = np.arange( thetaDelta, 180.0 - thetaDelta, thetaDelta)*np.pi / 180.0
    #    numc2 = len(c2Range)
    c1Range,c2Range = PotentialProbes.getGridRanges("sphere")
    atomNumRange = np.arange(len(atomNumericData))
    c1grid,c2grid,atomIndexGrid = np.meshgrid(   c1Range, c2Range,atomNumRange) 
    pointWeights = np.sin(c2grid[:,:,0])

    sigmaBroadcast = atomNumericData[atomIndexGrid,5]
    epsBroadcast = atomNumericData[atomIndexGrid,6]
    chargeBroadcast = atomNumericData[atomIndexGrid,3]

    sigmaBroadcastSlab = 0.5*(sigmaBroadcast + slabBeadSigma)
    epsBroadcastSlab = np.sqrt( epsBroadcast*slabBeadEpsilon)

    resList = []
    waterResList = []
    lastInfPoint = rRange[0] - 1
    lastWaterInfPoint = lastInfPoint
    
    
    feOutput = outputFolder+"/" +targetName+"_fev6.dat"
    skipProbes = 0
    if os.path.exists(feOutput) and args.forcerecalc == 0:
        skipProbes = 1
        print("Found probes for", target[0])
    else:
        print("Starting point probes for ", target[0])    
        for r in rRange:
            #vdw and electrostatic probes
            #distList = []
            #slabDistList = []
            #for i in range(len(atomNumericData)): #There is probably a more efficient way of doing this but in the interest of clarity this is easier to debug.
            #    if surfaceType == "sphere":
            #        atomDist = np.sqrt( ( r* np.cos(c1grid) *np.sin(c2grid)- atomNumericData[atomIndexGrid,0] )**2 + ( r* np.sin(c1grid)*np.sin(c2grid) - atomNumericData[atomIndexGrid,1] )**2  + ( r*np.cos(c2grid) - atomNumericData[atomIndexGrid,2] )**2 )
            #        distList.append(atomDist)
            #        #dist from the atom to an infinite half-slab with "surface centre" at {r cos[phi] sin[theta], r sin[phi] sin[theta] , r cos[theta]
            #        slabDist = np.sqrt(  (  atomNumericData[i,0] * np.cos(c1grid) * np.sin(c2grid)    + atomNumericData[i,1] * np.sin(c1grid)*np.sin(c2grid) + atomNumericData[i,2] * np.cos(c2grid) - r   )**2)
            #        slabDistList.append(slabDist)
            #    else:
            #        atomDist = np.sqrt(   ( c1grid - atomNumericData[i,0])**2 + (c2grid - atomNumericData[i,1])**2 + (r - atomNumericData[i,2])**2  )
            #        distList.append(atomDist)

            distArray = np.sqrt( ( r* np.cos(c1grid) *np.sin(c2grid)- atomNumericData[atomIndexGrid,0] )**2 + ( r* np.sin(c1grid)*np.sin(c2grid) - atomNumericData[atomIndexGrid,1] )**2  + ( r*np.cos(c2grid) - atomNumericData[atomIndexGrid,2] )**2 )

            slabDistArray = np.sqrt(  (  atomNumericData[atomIndexGrid,0]  * np.cos(c1grid) * np.sin(c2grid)    + atomNumericData[atomIndexGrid,1]  * np.sin(c1grid)*np.sin(c2grid) + atomNumericData[atomIndexGrid,2] * np.cos(c2grid) - r   )**2)
            electricContributions = np.sum(chargeBroadcast  / distArray, axis=-1)
            scaledDists = sigmaBroadcastSlab/distArray
            scaledDistsSlab = sigmaBroadcastSlab/slabDistArray 
            allContributions = np.sum( 4*epsBroadcastSlab*( scaledDists**12 - scaledDists**6) , axis=-1)
            #slabPackingEfficiency
            slabPotentialTerms = np.sum(  2*epsBroadcastSlab * (sigmaBroadcastSlab**6)* np.pi * slabDensity *(2 * sigmaBroadcastSlab**6 - 15*slabDistArray**6)/(45 * slabDistArray**9)            ,axis=-1)
            ones =  np.ones_like(slabPotentialTerms)
        
        
            freeEnergy=-conversionFactor * np.log( np.sum( pointWeights *  np.exp( -allContributions / conversionFactor) )  / np.sum(pointWeights ) )
            slabFreeEnergy =-conversionFactor * np.log( np.sum(   np.exp( -slabPotentialTerms / conversionFactor) )  / np.sum(ones) )
            if not np.isfinite(slabFreeEnergy):
                slabFreeEnergy = np.amin(slabPotentialTerms)

            
            electrostatic = np.sum( electricContributions * np.exp( -allContributions / conversionFactor) ) / np.sum(np.exp( -allContributions / conversionFactor))
            
            probeFESet = []
            probeSimpleSet = []
            foundInf = 0
            allInf = 1
            cInf = 0

            for probe in pointProbes:  ##name, sigma,epsilon,   charge
                probeTotalPotential,probeFreeEnergy,probeSimpleEnergy = PotentialProbes.getProbeEnergy(chargeBroadcast, epsBroadcast,sigmaBroadcast, distArray, probe,conversionFactor,pointWeights)
                probeFESet.append(probeFreeEnergy)
                probeSimpleSet.append(probeSimpleEnergy)      
            resList.append( [r,r-r0Start, r-r0Start, freeEnergy,electrostatic,slabFreeEnergy]+ probeFESet + probeSimpleSet)
            if not np.isfinite( probeFESet[0]):
                lastInfPoint = r
        resArray = np.array(resList)
        resArray = resArray[    resArray[:,0] > lastInfPoint ]
        np.savetxt( feOutput, resArray, fmt='%2.7f',delimiter=",", header=feFileHeader)
        print("Completed point probe potentials",flush=True)

    for moleculeProbeDef in moleculeProbes:
        moleculeTag = moleculeProbeDef[0]
        moleculeStructure = moleculeProbeDef[1].getAtomSet()
        outputLoc = outputFolder+"/" +     targetName+"_"+moleculeTag+"fev6.dat"
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
            waterMin = 1e50
            thetaDelta = 15
            thetaRange = np.arange( thetaDelta, 180 , thetaDelta)*np.pi/180.0
            for theta in  thetaRange:
                for phi in np.linspace(0,2*np.pi, num = 8, endpoint=False):
                    rotateMatrixInternal = GeometricFuncs.UARotateMatrix(np.pi - theta,-phi)
                    allContributions = 0
                    for atom in moleculeStructure:
                        ax = atom.x * rotateMatrixInternal[0,0] + atom.y*rotateMatrixInternal[0,1] + atom.z*rotateMatrixInternal[0,2]
                        ay = atom.x * rotateMatrixInternal[1,0] + atom.y*rotateMatrixInternal[1,1] + atom.z*rotateMatrixInternal[1,2]
                        az = atom.x * rotateMatrixInternal[2,0] + atom.y*rotateMatrixInternal[2,1] + atom.z*rotateMatrixInternal[2,2]
                        distArray = np.sqrt( ( r* np.cos(c1grid) *np.sin(c2grid)  + ax  - atomNumericData[atomIndexGrid,0] )**2 + ( r* np.sin(c1grid)*np.sin(c2grid) + ay - atomNumericData[atomIndexGrid,1] )**2  + ( r*np.cos(c2grid) + az- atomNumericData[atomIndexGrid,2] )**2 )
                        #distArray = np.array(distList)
                        totalPotential,probeFreeEnergy,probeSimpleEnergy = PotentialProbes.getProbeEnergy( chargeBroadcast, epsBroadcast,sigmaBroadcast, distArray, atom,conversionFactor,pointWeights)
                        allContributions += totalPotential
                    waterMin = min( waterMin, np.amin(allContributions)) #this is used only if there's no better value under the assumption its either the least bad or most favourable
                    runningNumerator += np.sum( pointWeights *  np.exp(-allContributions/conversionFactor)     )* np.sin(theta)
                    runningDenominator += np.sum(pointWeights  * np.sin(theta))
        
            waterFreeEnergy=-conversionFactor * np.log( runningNumerator/runningDenominator)
            if not np.isfinite(waterFreeEnergy):
                waterFreeEnergy = waterMin
            if np.isfinite(waterFreeEnergy):
                #print(r-r0Start,waterFreeEnergy)
                waterResList.append( [r,r-r0Start, r-r0Start, waterFreeEnergy,waterMin])            
            else:
                lastWaterInfPoint = r    
        waterResArray = np.array(waterResList)
        waterResArray = waterResArray[ waterResArray[:,0] > lastWaterInfPoint ]
        np.savetxt( outputLoc, waterResArray, fmt='%2.7f',delimiter=",", header="r[nm],d[nm],daligned[nm],U"+moleculeTag+"(d)[kj/mol],UMin"+moleculeTag+"(d)[kj/mol]")
        print("Completed "+moleculeTag,flush=True)
    print("Completed "+targetName,flush=True)
