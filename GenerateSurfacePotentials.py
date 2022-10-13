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

parser = argparse.ArgumentParser(description="Parameters for GenerateSurfacePotentials")
parser.add_argument("-f","--forcerecalc", type=int,default=0,help="If 1 then potentials are recalculated even if their table already exists")
parser.add_argument("-i","--initial", type=int, default=0,help="Initial structure to start calculating for multiprocessing")
parser.add_argument("-s","--step", type=int, default=1,help="Stride for slicing for multiprocessing")
args = parser.parse_args()








#define parameters used for the free energy calculation
temperature = 300.0
dielectricConst = 1 


#point probes. the first must be left unchanged because this is used to determine the surface level.
#name, sigma,epsilon,   charge, LJ model (=0 for point, =1 for flat disk of radius 0.5nm)


 

pointProbes = PotentialProbes.getPointProbeSet( ["C","K","Cl","C2A","C4A","CPlus","CMinus","CMoreLJ","CLessLJ"]  )



moleculeProbes = [
["water", PotentialProbes.waterProbe], 
["methane",PotentialProbes.methaneProbe], 
["waterUCD", PotentialProbes.waterUCDProbe],
["carbring", PotentialProbes.sixcarbProbe],
["cline3",PotentialProbes.cline3Probe]
]

feFileHeader = "r[nm],d[nm],daligned[nm]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe.name+"(d)[kJ/mol]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe.name+"Min(d)[kJ/mol]"
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
offsetResultFile.write("Material,ZeroLevelOffset[nm],FEOffset[nm],SSDRefDist[nm],MethaneOffset[nm]\n")




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
        #planes: zero-center the COM, then level the uppermost atom  to the z=0 plane, excluding small atoms  . This offset gets updated later.
        #cylinders: these are already aligned around z=0, so just set the centre of mass to the z=0 plane
        atomNumericData[:,2] = atomNumericData[:,2] - np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
        #zeroLevelOffset = np.amax(atomNumericData[   atomNumericData[:,6] > 0.1 ][:,2]  )   
        #zeroLevelAtoms = np.logical_and(   np.logical_and( atomNumericData[:,6] > 0.05 , nmData[:,2] != "OHW"   ) , nmData[:,2] != "OFW")
        zeroLevelAtoms = atomNumericData[:,6] > 0.01
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
        if ssdDefType == 1: #reference distance is upper surface to the first layer of heavy atoms(?)
            heavyAtomMass = np.amax( atomNumericData[:,4])
            heavyAtomZ = np.amax(atomNumericData[ (atomNumericData[:,4] > heavyAtomMass - 1.0), 2 ])
            ssdRefDist = zUpperSurface  - heavyAtomZ
        if ssdDefType == 2:  #SSD is given by the distance from a point to the COM of the slab , then subtract slabWidth/2. 
            print( surfaceName, zUpperSurface, newZCOM, slabWidth/2.0)
            if surfaceType == "plane":
                ssdRefDist = zUpperSurface  - newZCOM - slabWidth/2.0  
            else:
                ssdRefDist = 0
        if ssdDefType == 3:
            #this SSD type so far is applied only to graphene-like sheets and is the "centre of mass of surface atoms", 
            # which seemingly only includes carbon based on the fact small molecules on GO can't penetrate past the COM expected otherwise. and explicitly includes only the upper layer. 
            # #to automatically process this, we find the set of uppermost carbon atoms, use this to find a cutoff for the "upper layer" and compute the offset between the upper surface and the com-surface.
            zCarbonAtoms = atomNumericData[ np.logical_and(  atomNumericData[:,4] > 11 , atomNumericData[:,4] < 13 ) ] 
            zCarbonOffset = np.amax( zCarbonAtoms[:,2])
            zSurfaceAtoms = zCarbonAtoms[zCarbonAtoms[:,2] > zCarbonOffset - 0.1]
            zSurfaceCOM = np.sum(zSurfaceAtoms[:,2] * zSurfaceAtoms[:,4])/np.sum(zSurfaceAtoms[:,4])
            ssdRefDist = zUpperSurface - zSurfaceCOM
        #print(newZCOM)
    rRange =r0Start + np.arange(newZCOM, 2.0, 0.005) #actual resolution  0.005
    c1Range,c2Range = PotentialProbes.getGridRanges(surfaceType,planeScanHalfLength)
    atomNumRange = np.arange(len(atomNumericData))
    c1grid,c2grid,atomIndexGrid = np.meshgrid(   c1Range, c2Range,atomNumRange) 
    pointWeights = np.ones_like(c2grid[:,:,0])
    sigmaBroadcast = atomNumericData[atomIndexGrid,5]
    epsBroadcast = atomNumericData[atomIndexGrid,6]
    chargeBroadcast = atomNumericData[atomIndexGrid,3]
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
            if surfaceType == "cylinder":
                distArray = np.sqrt( ( r* np.cos(c1grid) - atomNumericData[atomIndexGrid,0] )**2 + ( r* np.sin(c1grid) - atomNumericData[atomIndexGrid,1] )**2  + ( c2grid - atomNumericData[atomIndexGrid,2] )**2 )
            else:
                distArray = np.sqrt(   ( c1grid - atomNumericData[atomIndexGrid,0])**2 + (c2grid - atomNumericData[atomIndexGrid,1])**2 + (r - atomNumericData[atomIndexGrid,2])**2  )
            probeFESet = []
            probeSimpleSet = []
            foundInf = 0
            allInf = 1
            cInf = 0
            for probe in pointProbes:  ##name, sigma,epsilon,   charge
                probeTotalPotential,probeFreeEnergy,probeSimpleEnergy = PotentialProbes.getProbeEnergy(chargeBroadcast, epsBroadcast,sigmaBroadcast, distArray, probe,conversionFactor,pointWeights)
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
    #type 1 may not even be used looking at the PMF metadynamics input

    if surfaceType == "plane":
        ssdDelta = ssdRefDist
    else:
        ssdDelta = 0
    pmfDelta = requiredOffsetFE + ssdDelta
    #offsetResultFile.write(surfaceName +","+str(zeroLevelOffset)+","   +str(requiredOffsetFE)+","+str(ssdRefDist) +"\n")    
    np.savetxt(outputTarget, resArray, fmt='%2.7f',delimiter=",", header=feFileHeader)
    print("Completed point potentials",flush=True)
    methaneData = [ -1 ]
    #calculate extra molecules and save them out to individual files
    for moleculeProbeDef in moleculeProbes:
        moleculeTag = moleculeProbeDef[0]
        moleculeStructure = moleculeProbeDef[1].getAtomSet()
        #moleculeStructure = PotentialProbes.centerProbe(moleculeProbeDef[1])
        outputLoc = outputFolder+"/" +     surfaceName+"_"+moleculeTag+"fe.dat"
        skipMolCalc = 0
        if args.forcerecalc == 0 and os.path.exists(outputLoc):
            skipMolCalc = 1
            waterResArray = np.genfromtxt(outputLoc,delimiter=",")
        else:
            print("Starting", moleculeTag)
            #moleculeStartPoint = -requiredOffsetFE 
            rRangeWater =r0Start + np.arange(-requiredOffsetFE , 1.6,  0.01 ) #usual resolution 0.01   , starting point:  r-r0Start   + requiredOffsetFE = 0 -> r = r0Start - requiredOffsetFE
            #rRangeWater = np.arange(0.1,0.3,0.005) 
            waterResList = []   
            thetaDelta = 30
            phiNum = 8
            thetaRange = np.arange( thetaDelta, 180 , thetaDelta)*np.pi/180.0       
            phiRange = np.linspace(0,2*np.pi, num =phiNum, endpoint=False)
    
            
            for r in rRangeWater:      
                waterMin = 1e50      
                runningNumerator = 0
                runningDenominator = 0
                minEnergy = 1e20
                minData = [0,0,0] #phi,theta
                for theta in thetaRange: #thetaRange  [0]: #np.linspace(0,np.pi, num = 5, endpoint=True) : #usual 5 values
                    for phi in phiRange: #phiRange  #usual 8 values
                        rotateMatrixInternal = GeometricFuncs.UARotateMatrix(np.pi - theta,-phi)
                        allContributions = 0
                        minz = 20
                        for atom in moleculeStructure:
                            ax = atom.x * rotateMatrixInternal[0,0] + atom.y*rotateMatrixInternal[0,1] + atom.z*rotateMatrixInternal[0,2]
                            ay = atom.x * rotateMatrixInternal[1,0] + atom.y*rotateMatrixInternal[1,1] + atom.z*rotateMatrixInternal[1,2]
                            az = atom.x * rotateMatrixInternal[2,0] + atom.y*rotateMatrixInternal[2,1] + atom.z*rotateMatrixInternal[2,2]

                            if surfaceType == "cylinder":
                                distArray = np.sqrt( ( r* np.cos(c1grid) + ax - atomNumericData[atomIndexGrid,0] )**2 + ( r* np.sin(c1grid) + ay - atomNumericData[atomIndexGrid,1] )**2  + ( c2grid + az -atomNumericData[atomIndexGrid,2] )**2 )
                            else:
                                distArray = np.sqrt(   ( c1grid + ax - atomNumericData[atomIndexGrid,0])**2 + (c2grid + ay - atomNumericData[atomIndexGrid,1])**2 + (r + az - atomNumericData[atomIndexGrid,2])**2  )

                            #print( np.amin(distArray)   ) 
                            #electricContributions = np.sum( chargeBroadcast  / distArray , axis=2 )
                            #scaledDists = sigmaCombined/distArray 
                            #ljContributions =  np.sum( 4*epsCombined*( scaledDists**12 - scaledDists**6) , axis=2)
                            #probeDef[1] = sigma, probeDef[2] = epsilon, probDef[3] = charge
                            #probeDef = PotentialProbes.AtomProbe("molcomponent",atom[5],atom[6],atom[3])
                            totalPotential,probeFreeEnergy,probeSimpleEnergy = PotentialProbes.getProbeEnergy( chargeBroadcast, epsBroadcast,sigmaBroadcast, distArray, atom,conversionFactor,pointWeights)
                            allContributions += totalPotential
                        waterMin = min( waterMin, np.amin(allContributions)) #this is used only if there's no better value under the assumption its either the least bad or most favourable
                        runningNumerator += np.sum(  np.exp(-allContributions/conversionFactor)     * np.sin(theta) )
                        runningDenominator += np.sum(    np.ones_like(allContributions) *np.sin(theta))

                        
                waterFreeEnergy=-conversionFactor * np.log( runningNumerator/runningDenominator)
                if not np.isfinite(waterFreeEnergy):
                    #print("Overriding free energy with minimum")
                    waterFreeEnergy = waterMin
                if np.isfinite(waterFreeEnergy):
                    #print(r-r0Start,waterFreeEnergy)
                    waterResList.append( [r,r-r0Start, r-r0Start, waterFreeEnergy,waterMin])  
                    #print( r,waterFreeEnergy)          
                else:
                    lastWaterInfPoint = r    

            waterResArray = np.array(waterResList)
            waterResArray = waterResArray[ waterResArray[:,0] > lastWaterInfPoint ]
        waterResArray[:,2] = waterResArray[:,1] + requiredOffsetFE
        if moleculeTag == "methane":
            methaneData = waterResArray[:,(2,3)]
        np.savetxt(outputLoc, waterResArray, fmt='%2.7f',delimiter=",", header="r[nm],d[nm],daligned[nm],U"+moleculeTag+"(d)[kj/mol],UMin"+moleculeTag+"(d)[kj/mol]")    
        print("Completed "+moleculeTag,flush=True)
    plotFig = 0
    foundPMF = 0
    if os.path.exists("AllPMFs/"+surfaceName+"_ALASCA-AC.dat"):
        alaPMFData = HGEFuncs.loadPMF("AllPMFs/"+surfaceName+"_ALASCA-AC.dat")
        foundPMF = 1
    elif os.path.exists("AllPMFs/"+surfaceName+"_ALASCA-JS.dat"):
        alaPMFData = HGEFuncs.loadPMF("AllPMFs/"+surfaceName+"_ALASCA-JS.dat")
        foundPMF = 1
    else:
        foundPMF = 0
    targetEnergyOverride = False
    targetEnergyOverrideVal = 50
    #Manual overrides for generating offsets for the less believable auto-generated ones.
    if surfaceName == "AuFCC110UCD":
        alaPMFData = HGEFuncs.loadPMF("AllPMFs/"+"Ag110"+"_ALASCA-JS.dat")
        foundPMF = 1
    if surfaceName == "CdSeWurtzite2-10":
        foundPMF = 0
    if surfaceName == "TiO2-ana-100":
        targetEnergyOverride = True
        targetEnergyOverrideVal =  17.5
    pmfDelta = 0
    if foundPMF == 1:
        #found an alanine PMF so attempt to calculate the offset used for this particular set of PMFs
        pmf = alaPMFData
        probe = methaneData
        #pmf = np.clip( pmf, -50,50)
        #probe[:,1] = np.clip(probe[:,1],10,50)
        if plotFig == 1:
            plt.figure()
            plt.plot(pmf[:,0],pmf[:,1],"b-")
            plt.plot(probe[:,0],probe[:,1],"r:")
            plt.xlim(0,1.5)
            plt.ylim( min(np.amin(pmf[:,1]), np.amin(probe[:,1])) -1 , np.amax(pmf[:,1]))
        targetEnergy = min(50, np.amax(pmf[:,1]))
        if targetEnergyOverride == True:
            targetEnergy = targetEnergyOverrideVal
        alignPointIndex = (np.where( np.diff(np.sign(pmf[:,1] - targetEnergy)))[0])[0]
        #pmfMaxLoc = pmf[ np.argmax(pmf[:,1]) ,0]
        pmfMaxLoc =  pmf[ alignPointIndex,0]
        #probeMaxLoc = probe[ np.argmin( ( probe[:,1] - np.amax(pmf[:,1]) )**2    ), 0]
        probeAlignIndex = (np.where( np.diff(np.sign(probe[:,1] - targetEnergy)))[0])[0]
        probeMaxLoc = probe[probeAlignIndex,0]
        probeShift = pmfMaxLoc - probeMaxLoc
        if plotFig == 1:
            plt.plot( probe[:,0] + probeShift, probe[:,1], "kx")
        #bestDelta = 0
        #bestKL = 10e20
        #bestPotential = probe
        #for delta in np.arange(-0.1,0.1,0.05):
        #    shiftedProbePotential = np.stack(( pmf[:,0],  np.interp( pmf[:,0]  , probe[:,0] + delta, probe[:,1]  )),axis= -1)
        #    shiftedKL =  PotentialProbes.klPotentialDivergence(pmf, shiftedProbePotential, kbTVal=10)
        #    #shiftedKL = np.sum( (pmf[:,1] - shiftedProbePotential[:,1])**2 )
        #    print(delta,shiftedKL)
        #    if shiftedKL < bestKL:
        #        bestDelta = delta
        #        bestPotential = shiftedProbePotential
        #        bestKL = shiftedKL
        #    #plt.plot(shiftedProbePotential[:,0],shiftedProbePotential[:,1],"k:")
        #plt.plot( bestPotential[:,0],bestPotential[:,1],"rx")
        print(surfaceName, probeShift)
        if plotFig == 1:
            plt.show()
        methaneOffset = probeShift
    else:
        methaneOffset = 0
    offsetResultFile.write(surfaceName +","+str(zeroLevelOffset)+","   +str(requiredOffsetFE)+","+str(ssdRefDist) +","+str(methaneOffset)     +"\n")
    print("Completed "+surfaceName,flush=True)
offsetResultFile.close()
