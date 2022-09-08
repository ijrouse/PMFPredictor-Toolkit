#Generates the free energy map of a probe atom above a nanomaterial surface
#conventions: energies are in units kJ/mol, distances are in nm.
#Ian Rouse, ian.rouse@ucd.ie , 24/05/2022


import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special as scspec
import scipy.integrate
import argparse
import numpy.linalg as npla
import HGEFuncs
import GeometricFuncs
import PotentialProbes

parser = argparse.ArgumentParser(description="Parameters for GenerateChemicalPotentials")
parser.add_argument("-f","--forcerecalc", type=int,default=0,help="If 1 then potentials are recalculated even if their table already exists")
parser.add_argument("-i","--initial", type=int, default=0,help="Initial structure to start calculating for multiprocessing")
parser.add_argument("-s","--step", type=int, default=1,help="Stride for slicing for multiprocessing")
args = parser.parse_args()




#define parameters used for the free energy calculation
temperature = 300.0
probeEpsilon = 0.3598
probeSigma = 0.339
dielectricConst = 1 


#point probes. the first must be left unchanged because this is used to determine the surface level.
#name, sigma,epsilon,   charge, LJ model (=0 for point, =1 for flat disk of radius 0.5nm)


 

pointProbes = PotentialProbes.getPointProbeSet( ["C","K","Cl","C2A","C4A","CPlus","CMinus","CMoreLJ","CLessLJ"]  )



moleculeProbes = [
["water", PotentialProbes.waterProbe], 
["methane",PotentialProbes.methaneProbe], 
["waterUCD", PotentialProbes.waterUCDProbe],
["carbring", PotentialProbes.sixcarbProbe],
["cline", PotentialProbes.clineProbe],
["cline3",PotentialProbes.cline3Probe]
]

feFileHeader = "r[nm],d[nm],daligned[nm]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe[0]+"(d)[kJ/mol]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe[0]+"Min(d)[kJ/mol]"
conversionFactor = ( 8.314/1000.0) * temperature

planeScanHalfLength = 0.5

inputFolder = "Structures/Surfaces"
outputFolder = "SurfacePotentials"
os.makedirs(outputFolder, exist_ok=True)
debyeLength = 0.7
avogadroNum = 6.022e23
electroToKJMol =  (1.4399/dielectricConst) * 1.6e-19 *avogadroNum/1000.0
print("Electric to kJ/mol conversion: ",electroToKJMol)
#1.4399 arises from (1/4 pi eps0) * 1 elementary charge/1nm to give scaling factor to V , 1.6e-19 is the second elementary charge to give an energy in J, multiply by atoms/mol , divide by 1000 to get kJ/mol
#this is quite a large number! 

#define the alignment point
#the free energy is calculated as a function of distance relative to the unshifted co-ordinates provided and the point at which U(r) = alignPointEnergy found
#this is taken to be the r = alignPointDist point and the potentials shifted accordingly.
alignPointEnergyDefault = 35.0
alignPointDist = 0.2
pmfAlignEnergyDefault = 35.0


#offset file: record the material, distance from the initial co-ordinates to level the uppermost atom to z=0, offset required to level to the UC(r=0.2) = 35 kJ/mol reference, position of the reference plane relative to z = 0
offsetResultFile = open("Datasets/SurfaceOffsetData.csv","w")
offsetResultFile.write("Material,ZeroLevelOffset[nm],FEOffset[nm],SSDRefDist[nm]\n")



#input data has the form
##ID,shape,source,ssdType
surfaceTargetSet = np.genfromtxt("Structures/SurfaceDefinitions.csv",dtype=str,delimiter=",")
if surfaceTargetSet.ndim == 1:
    surfaceTargetSet = np.array([surfaceTargetSet])

for surfaceTarget in surfaceTargetSet[args.initial::args.step] :
    surfaceName = surfaceTarget[0]

    ssdRefDist = 0
    surfaceType = surfaceTarget[1]
    sourceType = int(surfaceTarget[2])
    ssdDefType = int(surfaceTarget[3]) #0 = PMFs are defined relative to a flat surface, 1 = PMFs are defined using the minimum-z-distance convention, 2 = PMFs are defined relative to the COM-COM(Plane) distance - half a slab width, 3 = PMFs are defined relative to the surface atom COM

    surfaceInputFile = surfaceName+"_combined.csv"
    if surfaceType=="cylinder":
        r0Start = 0.75
    else:
        r0Start = 0.00
    # indexes: 0 = numeric ID, 1 = atomID , 2= atom type, 3= x, 4 = y, 5 = z, 6 = charge, 7= mass, 8 = sigma, 9 = epsilon
    nmData = np.genfromtxt(inputFolder+"/"+surfaceInputFile, skip_header=1,delimiter=",",dtype=str) 
    atomNumericData = nmData[:, 3:].astype(float)
    print("Loaded ", len(atomNumericData), "atoms from file")

    alignSurface = 1
    zUpperSurface  = 0
    if alignSurface == 1:
        #indexes: 0,1,2 = x,y,z, 3=charge, 4=mass, 5=sigma, 6=epsilon
        #print(atomNumericData)
        #planes: zero-center the COM, then level the uppermost atom  to the z=0 plane, excluding small atoms and water . This offset gets updated later.
        #cylinders: these are already aligned around z=0, so just set the centre of mass to the z=0 plane
        atomNumericData[:,2] = atomNumericData[:,2] - np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
        #zeroLevelOffset = np.amax(atomNumericData[   atomNumericData[:,6] > 0.1 ][:,2]  )   
        zeroLevelAtoms = np.logical_and(   np.logical_and( atomNumericData[:,6] > 0.05 , nmData[:,2] != "OHW"   ) , nmData[:,2] != "OFW")
        zeroLevelOriginalOffset = np.amax(atomNumericData[  atomNumericData[:,6] > 0.05  ][:,2]  )
        zeroLevelOffset = np.amax(atomNumericData[  zeroLevelAtoms ][:,2]  ) 

        if surfaceType == "plane":
            atomNumericData[:,0] = atomNumericData[:,0] - np.sum(atomNumericData[:,0] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
            atomNumericData[:,1] = atomNumericData[:,1] - np.sum(atomNumericData[:,1] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
            atomNumericData[:,2] = atomNumericData[:,2] - zeroLevelOffset  #zero level for the purpose of generating the potential to make sure we start "inside" the NP, this is recorded for archival purposes. At this point, z=0 is the upper layer of heavy atoms
            slabWidth = np.amax(  atomNumericData[:,2] ) - np.amin( atomNumericData[:,2])
            zUpperSurface = np.amax( atomNumericData[:,2]) 

            
        newZCOM = np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
        if ssdDefType == 0: #defined relative to upper surface
            ssdRefDist = 0
        if ssdDefType == 1: #defined relative to upper surface 
            ssdRefDist = 0        
        if ssdDefType == 2:  #SSD is given by the distance from a point to the COM of the slab , then subtrac slabWidth/2. 
            print( surfaceName, zUpperSurface, newZCOM, slabWidth/2.0)
            ssdRefDist = zUpperSurface  - newZCOM - slabWidth/2.0  
        if ssdDefType == 3:
            #this SSD type so far is applied only to graphene-like sheets and is the "centre of mass of surface atoms", which presumably includes both carbon and oxygen for go/rgo and explicitly includes only the upper layer. to automatically process this, we find the set of uppermost carbon atoms, use this to find a cutoff for the "upper layer" and compute the offset between the upper surface and the com-surface.
            zCarbonAtoms = atomNumericData[ np.logical_and(  atomNumericData[:,4] > 11 , atomNumericData[:,4] < 13 ) ] 
            zCarbonOffset = np.amax( zCarbonAtoms[:,2])
            zSurfaceAtoms = atomNumericData[ atomNumericData[:,2] > zCarbonOffset - 0.1]
            zSurfaceCOM = np.sum(zSurfaceAtoms[:,2] * zSurfaceAtoms[:,4])/np.sum(zSurfaceAtoms[:,4])
            ssdRefDist = zUpperSurface - zSurfaceCOM
        #print(newZCOM)
        rRange =r0Start + np.arange(newZCOM, 2.0, 0.005) #actual resolution  0.005
        if surfaceType == "cylinder":
            numc1=16
            numc2=23
            c1Range = np.linspace(0,2*np.pi, num = numc1, endpoint=False)
            c2Range = np.linspace(-1.0, 1.0, num = numc2, endpoint=True)
            c1grid,c2grid = np.meshgrid(   c1Range, c2Range) 
            #sigmaBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,5] , (numc1,numc2,len(atomNumericData)) ))
            #epsBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,6] , (numc1,numc2,len(atomNumericData)) ))
            #chargeBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,3] , (numc1,numc2,len(atomNumericData)) ))
            #sigmaBroadcast = 0.5*(sigmaBroadcast + probeSigma)
            #epsBroadcast = np.sqrt( epsBroadcast*probeEpsilon)
        else:
            numc1=23
            numc2=23
            c1Range = np.linspace(-planeScanHalfLength, planeScanHalfLength, num = numc1, endpoint=True)
            c2Range = np.linspace(-planeScanHalfLength, planeScanHalfLength, num = numc2, endpoint=True)
            areaTerm = (c2Range[-1] - c2Range[0]) * (c1Range[-1] - c1Range[0])
            c1grid,c2grid = np.meshgrid(c1Range,c2Range)
            #sigmaBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,5] , (numc1,numc2,len(atomNumericData)) ))
            #epsBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,6] , (numc1,numc2,len(atomNumericData)) ))
            #chargeBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,3] , (numc1,numc2,len(atomNumericData)) ))
            #sigmaBroadcast = 0.5*(sigmaBroadcast + probeSigma)
            #epsBroadcast = np.sqrt( epsBroadcast*probeEpsilon)
        sigmaBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,5] , (  len(c1Range),len(c2Range),len(atomNumericData)) ))
        epsBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,6] , (len(c1Range),len(c2Range),len(atomNumericData)) ))
        chargeBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,3] , (len(c1Range),len(c2Range),len(atomNumericData)) ))
        lastInfPoint = rRange[0] - 1
        lastAllInfPoint = lastInfPoint
        lastWaterInfPoint = lastInfPoint
        
    skipPointProbe = 0
    outputTarget = outputFolder+"/" + surfaceName+"_fev5.dat"
    if os.path.exists( outputTarget) and args.forcerecalc == 0:
        skipPointProbe = 1
        print(surfaceName+": Skipping point probe")
    else:
        print("Starting point probes for ", surfaceTarget[0])
        
    resList = []
            
    if skipPointProbe == 0:       
        for r in rRange:
            #vdw and electrostatic probes
            distList = []
            for i in range(len(atomNumericData)): #There is probably a more efficient way of doing this but in the interest of clarity this is easier to debug.
                if surfaceType == "cylinder":
                    atomDist = np.sqrt( ( r* np.cos(c1grid) - atomNumericData[i,0] )**2 + ( r* np.sin(c1grid) - atomNumericData[i,1] )**2  + ( c2grid - atomNumericData[i,2] )**2 )
                    distList.append(atomDist)
                else:
                    atomDist = np.sqrt(   ( c1grid - atomNumericData[i,0])**2 + (c2grid - atomNumericData[i,1])**2 + (r - atomNumericData[i,2])**2  )
                    distList.append(atomDist)
            distArray = np.array(distList)
            electricContributions = chargeBroadcast  / distArray
            scaledDists = sigmaBroadcast/distArray 
            allContributions = np.sum( 4*epsBroadcast*( scaledDists**12 - scaledDists**6) , axis=0)
            ones = 1 + np.zeros_like(allContributions)
            #print(ones)
            #freeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -allContributions / conversionFactor) )  / np.sum(ones) )
            #electrostatic = np.sum( electricContributions * np.exp( -allContributions / conversionFactor) ) / np.sum(np.exp( -allContributions / conversionFactor))
            probeFESet = []
            probeSimpleSet = []
            foundInf = 0
            allInf = 1
            cInf = 0
            for probe in pointProbes:  ##name, sigma,epsilon,   charge
                epsCombined = np.sqrt(epsBroadcast * probe[2])
                sigmaCombined = 0.5*(sigmaBroadcast + probe[1])
                #print(electroToKJMol*probe[3]*electricContributions )
                electrostaticPotential = np.sum( electroToKJMol*probe[3]*electricContributions   , axis=0)
                scaledDists = sigmaCombined/distArray 
                if probe[4] == 1:
                    beadRadius = 0.5
                    hamterm1 = (4 * beadRadius**3) /(  ( beadRadius**2 - distArray**2  )**3  )
                    hamterm2 = (sigmaCombined**6/(120.0 * distArray))*(  ( distArray + 9 * beadRadius )/(  (beadRadius+distArray)**9    )    -  (  distArray-9*beadRadius   )/(   ( distArray - beadRadius  )**9    )     )
                    ljPotential = np.sum( 4.0/3.0 * np.pi * epsCombined * sigmaCombined**6 * (hamterm1 + hamterm2)     , axis=0)
                else:
                    ljPotential = np.sum( 4 * epsCombined * (scaledDists**12 - scaledDists**6 ), axis=0) #sum over atoms
                #print( (4 * epsCombined * (scaledDists**12 - scaledDists**6 )).shape, (electroToKJMol*probe[3]*electricContributions).shape)
                #print(probe)
                #print("LJ")
                #print(ljPotential)
                #print("Electro")
                #print(electrostaticPotential)

                totalPotential = electrostaticPotential + ljPotential
                probeFreeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones) )
                if not np.isfinite(probeFreeEnergy):
                    probeFreeEnergy = np.amin(totalPotential)
                probeSimpleEnergy = np.amin(totalPotential)
                #print(np.sum(totalPotential))
                #print(r, np.sum(   np.exp( -totalPotential / conversionFactor) ) , -conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones)))
                #calculate the free energy using the standard expression if it's safe to do so and min/max if not. 
                #if np.any( totalPotential < -300     ):
                #    probeFreeEnergy = np.amin(totalPotential)
                #elif np.all(totalPotential > 200):
                #    probeFreeEnergy = np.amin(totalPotential)
                #else:
                #    probeFreeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones) )
                
                #print(r, probe, probeFreeEnergy)
                probeFESet.append(probeFreeEnergy)
                probeSimpleSet.append(probeSimpleEnergy)                
                #print( r, probe[0], probeFreeEnergy)
            if not np.isfinite( probeFESet[0]):
                lastInfPoint = r
            resList.append( [r,r-r0Start, r-r0Start ] + probeFESet + probeSimpleSet)
            resArray = np.array(resList)
            resArray = resArray[    resArray[:,0] > lastInfPoint ]
    else:
        #load in the already calculated probe potentials
        resArray = np.genfromtxt(outputTarget,delimiter=",",skip_header=1)
        lastInfPoint = resArray[0,0] #FIX
        resArray[:,2] = resArray[:,1]

    #Find the point at which the carbon probe has an energy of 35 kJ/mol, which we then define to be the point at which r = 0.2
    (freeEnergyAlignPoint, freeEnergyAlignActual) = HGEFuncs.estimateValueLocation(resArray[:, (2,3)],alignPointEnergyDefault)    
    #alignPointDist is by default 0.2 . if the free energy align point is found at 0.5, we need to move the potential by -0.3 to align them.
    requiredOffsetFE = alignPointDist - freeEnergyAlignPoint    
    resArray[:,2] = resArray[:,2] + requiredOffsetFE

    #While we're processing the structure we also harmonise the PMFs to the definition of the surface given here
    #    ssdDefType #0 = PMFs are defined relative to a flat surface, 1 = PMFs are defined using the minimum-z-distance convention, 2 = PMFs are defined relative to the COM-COM distance - half a slab width.
    #Update: type 2 is essentially type 1 as it's a flat plane, just potentially defined at a different point
    #Convention: before fitting the PMF, we subtract PMFDelta from all r values to shift into the frame of reference relative to the potential-plane.
    

    if surfaceType == "plane":
        ssdDelta = ssdRefDist
    else:
        ssdDelta = 0
    pmfDelta = requiredOffsetFE + ssdDelta
    offsetResultFile.write(surfaceName +","+str(zeroLevelOffset)+","   +str(requiredOffsetFE)+","+str(ssdRefDist) +"\n")    
    np.savetxt(outputTarget, resArray, fmt='%2.7f',delimiter=",", header=feFileHeader)

    #calculate extra molecules and save them out to individual files
    for moleculeProbeDef in moleculeProbes:
        moleculeTag = moleculeProbeDef[0]
        moleculeStructure = PotentialProbes.centerProbe(moleculeProbeDef[1])
        outputLoc = outputFolder+"/" +     surfaceName+"_"+moleculeTag+"fe.dat"
        if args.forcerecalc == 0 and os.path.exists(outputLoc):
            continue
        else:
            print("Starting", moleculeTag)
        rRangeWater =r0Start + np.arange(r0Start - requiredOffsetFE, 1.6,  0.01 ) #usual resolution 0.01   , starting point:  r-r0Start   + requiredOffsetFE = 0 -> r = r0Start - requiredOffsetFE
        #rRangeWater = np.arange(0.1,0.3,0.005) 
        waterResList = []   
        for r in rRangeWater:      
            waterMin = 1e20      
            runningNumerator = 0
            runningDenominator = 0
            thetaDelta = 30
            phiNum = 8
            if len(moleculeStructure) > 14:
                thetaDelta = 15
                phiNum = 16
            thetaRange = np.arange( thetaDelta, 180 , thetaDelta)*np.pi/180.0
            minEnergy = 1e20
            minData = [0,0,0] #phi,theta
            for theta in  thetaRange: #np.linspace(0,np.pi, num = 5, endpoint=True) : #usual 5 values
                for phi in np.linspace(0,2*np.pi, num =phiNum, endpoint=False): #usual 8 values
                    rotateMatrixInternal = GeometricFuncs.UARotateMatrix(np.pi - theta,-phi)
                    allContributions = 0
                    minz = 20
                    for atom in moleculeStructure:
                        epsCombined = np.sqrt(epsBroadcast * atom[6])
                        sigmaCombined = 0.5*(sigmaBroadcast + atom[5])
                        #print(theta,phi,atom)
                        ax = atom[0] * rotateMatrixInternal[0,0] + atom[1]*rotateMatrixInternal[0,1] + atom[2]*rotateMatrixInternal[0,2]
                        ay = atom[0] * rotateMatrixInternal[1,0] + atom[1]*rotateMatrixInternal[1,1] + atom[2]*rotateMatrixInternal[1,2]
                        az = atom[0] * rotateMatrixInternal[2,0] + atom[1]*rotateMatrixInternal[2,1] + atom[2]*rotateMatrixInternal[2,2]
                        distList = []
                        for i in range(len(atomNumericData)): #There is probably a more efficient way of doing this but in the interest of clarity this is easier to debug. ax = probe atom x, atomNumericData[i,0] is structure atom i
                            if surfaceType == "cylinder":
                                atomDist = np.sqrt( ( r* np.cos(c1grid) + ax - atomNumericData[i,0] )**2 + ( r* np.sin(c1grid) + ay - atomNumericData[i,1] )**2  + ( c2grid + az -atomNumericData[i,2] )**2 )
                                minz = min(minz, np.sqrt(np.amin( ( r* np.cos(c1grid) + ax - atomNumericData[i,0] )**2 + ( r* np.sin(c1grid) + ay - atomNumericData[i,1] )**2)))
                                distList.append(atomDist)
                            else:
                                atomDist = np.sqrt(   ( c1grid + ax - atomNumericData[i,0])**2 + (c2grid + ay - atomNumericData[i,1])**2 + (r + az - atomNumericData[i,2])**2  )
                                minz = min(minz, r + az - atomNumericData[i,2]) 
                                distList.append(atomDist)
                        distArray = np.array(distList)
                        #print( np.amin(distArray) )
                        electricContributions = np.sum( chargeBroadcast  / distArray , axis=0 )
                        scaledDists = sigmaCombined/distArray 
                        ljContributions =  np.sum( 4*epsCombined*( scaledDists**12 - scaledDists**6) , axis=0)
                        allContributions += (electroToKJMol*atom[3]*electricContributions + ljContributions)
                    waterMin = min( waterMin, np.amin(allContributions)) #this is used only if there's no better value under the assumption its either the least bad or most favourable
                    if np.amin(allContributions) < minEnergy:
                        minEnergy = np.amin(allContributions)
                        minData[0] = theta
                        minData[1] = phi
                        minData[2] = minz
                    ones = (  1 + np.zeros_like(allContributions) )
                    runningNumerator += np.sum(  np.exp(-allContributions/conversionFactor)     )* np.sin(theta)
                    runningDenominator += np.sum(ones * np.sin(theta))
                    #thetaNumTerms.append(np.sum(  np.exp(-allContributions/conversionFactor)     ))
                    #thetaDenomTerms.append(  np.sum(ones ))
            #print( r, minData  )
                    
            #runningNumerator = sum (thetaNumTerms * sintheta)
            #runningDenominator = sum(thetaDenomTerms * sintheta)        
                    
            waterFreeEnergy=-conversionFactor * np.log( runningNumerator/runningDenominator)
            if not np.isfinite(waterFreeEnergy):
                #print("Overriding free energy with minimum")
                waterFreeEnergy = waterMin
            if np.isfinite(waterFreeEnergy):
                #print(r-r0Start,waterFreeEnergy)
                waterResList.append( [r,r-r0Start, r-r0Start, waterFreeEnergy])  
                #print( r,waterFreeEnergy)          
            else:
                lastWaterInfPoint = r    

        waterResArray = np.array(waterResList)
        waterResArray = waterResArray[ waterResArray[:,0] > lastWaterInfPoint ]
        waterResArray[:,2] = waterResArray[:,2] + requiredOffsetFE
        np.savetxt(outputLoc, waterResArray, fmt='%2.7f',delimiter=",", header="r[nm],d[nm],daligned[nm],U"+moleculeTag+"(d)[kj/mol]")    

offsetResultFile.close()
