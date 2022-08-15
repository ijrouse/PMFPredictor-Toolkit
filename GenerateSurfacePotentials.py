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

parser = argparse.ArgumentParser(description="Parameters for GenerateChemicalPotentials")
parser.add_argument("-f","--forcerecalc", type=int,default=0,help="If 1 then potentials are recalculated even if their table already exists")
args = parser.parse_args()


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

def getTransformMatrix(coordArr):
    ixx = np.sum(coordArr[:,1]**2 + coordArr[:,2]**2)
    ixy = np.sum(- coordArr[:,0]*coordArr[:,1])
    ixz = np.sum(-coordArr[:,0]*coordArr[:,2])
    iyy = np.sum(coordArr[:,0]**2 + coordArr[:,2]**2)
    iyx = ixy
    iyz = np.sum(-coordArr[:,1]*coordArr[:,2])
    izz = np.sum(coordArr[:,0]**2 + coordArr[:,1]**2)
    izx = ixz
    izy = iyz
    inertialArray = np.array([ [ixx, ixy,ixz],[iyx,iyy,iyz],[izx,izy,izz] ])
    eigvals,eigvecs = npla.eig(inertialArray)
    #invarr = npla.inv(eigvecs)
    sortIndex = eigvals.argsort()[::-1]
    invarr = npla.inv(eigvecs[:,sortIndex])
    return invarr


def centerProbe( probeDef):
    probeCoordList = []
    for i in range(len(probeDef)):
        probeCoordList.append([ probeDef[i][0] , probeDef[i][1], probeDef[i][2], probeDef[i][4] ])
    probeCoords = np.array(probeCoordList)
    m = np.sum( probeCoords[:,3])
    cx = np.sum( probeCoords[:,3] * probeCoords[:,0]) / m
    cy =np.sum( probeCoords[:,3] * probeCoords[:,1]) / m
    cz =np.sum( probeCoords[:,3] * probeCoords[:,2]) / m
    probeCoords[:,0] = probeCoords[:,0] - cx
    probeCoords[:,1] = probeCoords[:,1] - cy
    probeCoords[:,2] = probeCoords[:,2] - cz
    print("centered")
    for i in range(len(probeDef)):
        probeDef[i][0] = probeCoords[i,0]
        probeDef[i][1] = probeCoords[i,1]
        probeDef[i][2] = probeCoords[i,2] 
        print(probeDef[i])
    return probeDef

#define parameters used for the free energy calculation
temperature = 300.0
probeEpsilon = 0.3598
probeSigma = 0.339
dielectricConst = 1 


#point probes. the first must be left unchanged because this is used to determine the surface level.
#name, sigma,epsilon,   charge, LJ model (=0 for point, =1 for flat disk of radius 0.5nm)
pointProbes =[
["C",0.339,0.3598,0,0 ],
["K",0.314264522824 ,  0.36401,1,0],
["Cl",0.404468018036 , 0.62760,-1,0],
["C2A",0.2,0.3598,0,0],
["C4A", 0.4, 0.3598,0,0],
["C6A", 0.6, 0.3598,0,0],
["C8A", 0.8, 0.3598,0,0],
["C10A",1.0,0.3598,0,0]
]

#Define the molecular probe separately

#x,y,z,   charge, mass, sigma, epsilon



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

sixcarbProbe=[

[0.099,0.000,0.001,0,12.01000,3.31521e-01,4.13379e-01],
[0.028,-0.120,0.001,0,12.01000,3.31521e-01,4.13379e-01],
[-0.112,-0.120,-0.0005,0,12.01000,3.31521e-01,4.13379e-01],
[-0.182,0.000,-0.001,0,12.01000,3.31521e-01,4.13379e-01],
[-0.112,0.121,-0.0005,0,12.01000,3.31521e-01,4.13379e-01],
[0.028,0.121,0.0005,0,12.01000,3.31521e-01,4.13379e-01]

]

moleculeProbes = [
["water", waterProbe], 
["methane",methaneProbe], 
["waterUCD", waterUCDProbe],
["carbring", sixcarbProbe]
]

feFileHeader = "r[nm],d[nm],daligned[nm]"
for probe in pointProbes:
    feFileHeader = feFileHeader+",U"+probe[0]+"(d)[kJ/mol]"

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
#input data has the form
#label, shape,  energy alignment value, skip energy alignment,   manual energy point at which to align the PMF, skip PMF alignment
offsetResultFile = open("Datasets/SurfaceOffsetData.csv","w")
offsetResultFile.write("Material,ZeroLevelOffset[nm],FEOffset[nm],PMFOffset[nm]\n")

surfaceTargetSet = np.genfromtxt("Structures/SurfaceDefinitions.csv",dtype=str,delimiter=",")
if surfaceTargetSet.ndim == 1:
    surfaceTargetSet = np.array([surfaceTargetSet])

for surfaceTarget in surfaceTargetSet[21::5] :
    surfaceName = surfaceTarget[0]


    surfaceType = surfaceTarget[1]
    alignPointEnergy = float(surfaceTarget[2])
    surfaceAlignOverride = int(surfaceTarget[3])
    pmfAlignEnergy = float(surfaceTarget[4])
    pmfAlignOverride = int(surfaceTarget[5])
    ssdDefType = int(surfaceTarget[7]) #0 = PMFs are defined relative to a flat surface, 1 = PMFs are defined using the minimum-z-distance convention, 2 = PMFs are defined relative to the COM-COM distance - half a slab width.
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

    if alignSurface == 1:
        #indexes: 0,1,2 = x,y,z, 3=charge, 4=mass, 5=sigma, 6=epsilon
        #print(atomNumericData)
        #planes: zero-center the COM, then level the z-direction to the z=0 plane. This offset gets updated later.
        #cylinders: these are already aligned around z=0, so just set the centre of mass to the z=0 plane
        atomNumericData[:,2] = atomNumericData[:,2] - np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
        #zeroLevelOffset = np.amax(atomNumericData[   atomNumericData[:,6] > 0.1 ][:,2]  )   
        zeroLevelAtoms = np.logical_and(   np.logical_and( atomNumericData[:,6] > 0.1 , nmData[:,2] != "OHW"   ) , nmData[:,2] != "OFW")
        zeroLevelOriginalOffset = np.amax(atomNumericData[  atomNumericData[:,6] > 0.1  ][:,2]  )
        zeroLevelOffset = np.amax(atomNumericData[  zeroLevelAtoms ][:,2]  ) 

        if surfaceType == "plane":
            atomNumericData[:,0] = atomNumericData[:,0] - np.sum(atomNumericData[:,0] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
            atomNumericData[:,1] = atomNumericData[:,1] - np.sum(atomNumericData[:,1] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
            atomNumericData[:,2] = atomNumericData[:,2] - zeroLevelOffset  #zero level for the purpose of generating the potential to make sure we start "inside" the NP, this is recorded for archival purposes 
        newZCOM = np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
        #print(newZCOM)
        rRange =r0Start + np.arange(newZCOM, 2.0, 0.005) #actual resolution  0.005
        if surfaceType == "cylinder":
            numc1=16
            numc2=23
            c1Range = np.linspace(0,2*np.pi, num = numc1, endpoint=False)
            c2Range = np.linspace(-1.0, 1.0, num = numc2, endpoint=True)
            c1grid,c2grid = np.meshgrid(   c1Range, c2Range) 
            sigmaBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,5] , (numc1,numc2,len(atomNumericData)) ))
            epsBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,6] , (numc1,numc2,len(atomNumericData)) ))
            chargeBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,3] , (numc1,numc2,len(atomNumericData)) ))
            #sigmaBroadcast = 0.5*(sigmaBroadcast + probeSigma)
            #epsBroadcast = np.sqrt( epsBroadcast*probeEpsilon)
        else:
            numc1=23
            numc2=23
            c1Range = np.linspace(-planeScanHalfLength, planeScanHalfLength, num = numc1, endpoint=True)
            c2Range = np.linspace(-planeScanHalfLength, planeScanHalfLength, num = numc2, endpoint=True)
            areaTerm = (c2Range[-1] - c2Range[0]) * (c1Range[-1] - c1Range[0])
            c1grid,c2grid = np.meshgrid(c1Range,c2Range)
            sigmaBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,5] , (numc1,numc2,len(atomNumericData)) ))
            epsBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,6] , (numc1,numc2,len(atomNumericData)) ))
            chargeBroadcast = np.transpose( np.broadcast_to( atomNumericData[:,3] , (numc1,numc2,len(atomNumericData)) ))
            #sigmaBroadcast = 0.5*(sigmaBroadcast + probeSigma)
            #epsBroadcast = np.sqrt( epsBroadcast*probeEpsilon)

        lastInfPoint = rRange[0] - 1
        lastAllInfPoint = lastInfPoint
        lastWaterInfPoint = lastInfPoint
        
    skipPointProbe = 0
    if os.path.exists( outputFolder+"/" +surfaceName+"_fev3.dat") and args.forcerecalc == 0:
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
                #print(np.sum(totalPotential))
                #print(r, np.sum(   np.exp( -totalPotential / conversionFactor) ) , -conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones)))
                #calculate the free energy using the standard expression if it's safe to do so and min/max if not. 
                if np.any( totalPotential < -300     ):
                    probeFreeEnergy = np.amin(totalPotential)
                elif np.all(totalPotential > 200):
                    probeFreeEnergy = np.amin(totalPotential)
                else:
                    probeFreeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones) )
                
                #print(r, probe, probeFreeEnergy)
                probeFESet.append(probeFreeEnergy)
                #print( r, probe[0], probeFreeEnergy)
            if not np.isfinite( probeFESet[0]):
                lastInfPoint = r
            resList.append( [r,r-r0Start, r-r0Start ] + probeFESet)
            resArray = np.array(resList)
            resArray = resArray[    resArray[:,0] > lastInfPoint ]
    else:
        #load in the already calculated probe potentials
        resArray = np.genfromtxt(outputFolder+"/" + surfaceName+"_fev3.dat",delimiter=",",skip_header=1)
        lastInfPoint = resArray[0,0] #FIX
        resArray[:,2] = resArray[:,1]

    #Find the point at which the carbon probe has an energy of 35 kJ/mol, which we then define to be the point at which r = 0.2
    (freeEnergyAlignPoint, freeEnergyAlignActual) = estimateValueLocation(resArray[:, (2,3)],alignPointEnergyDefault)    
    #alignPointDist is by default 0.2 . if the free energy align point is found at 0.5, we need to move the potential by -0.3 to align them.
    requiredOffsetFE = alignPointDist - freeEnergyAlignPoint    
    resArray[:,2] = resArray[:,2] + requiredOffsetFE

    #While we're processing the structure we also harmonise the PMFs to the definition of the surface given here
    #    ssdDefType #0 = PMFs are defined relative to a flat surface, 1 = PMFs are defined using the minimum-z-distance convention, 2 = PMFs are defined relative to the COM-COM distance - half a slab width.
    #Convention: before fitting the PMF, we subtract PMFDelta from all r values to shift into the frame of reference relative to the potential-plane.
    if ssdDefType == 0:
        pmfDelta = requiredOffsetFE
    elif ssdDefType == 1:
        print(surfaceName, " zero level original offset", zeroLevelOriginalOffset, " with exclusions" , zeroLevelOffset)
        #this case is more complicated due to the use of the minimum z distance - there is no well-defined relation between the SSD given in the PMF and the actual z distance. we therefore record both the offset and the zero-distance
        pmfDelta = requiredOffsetFE
    elif ssdDefType == 2:
        slabWidth = np.amax(  atomNumericData[:,2] ) - np.amin( atomNumericData[:,2])
        zUpperSurface = np.amax( atomNumericData[:,2]) 
        pmfDelta = requiredOffsetFE  + zUpperSurface  - newZCOM - slabWidth/2.0
    else:
        print("Unrecognised SSD type, defaulting to type 0")
        pmfDelta = requiredOffsetFE  
    print( surfaceName, zeroLevelOffset, requiredOffsetFE, pmfDelta)
    offsetResultFile.write(surfaceName +","+str(zeroLevelOffset)+","   +str(requiredOffsetFE)+","+str(pmfDelta)+"\n")

        
    np.savetxt( outputFolder+"/" + surfaceName+"_fev3.dat", resArray, fmt='%2.7f',delimiter=",", header=feFileHeader)
    

    #calculate extra molecules and save them out to individual files
    for moleculeProbeDef in moleculeProbes:
        moleculeTag = moleculeProbeDef[0]
        moleculeStructure = centerProbe(moleculeProbeDef[1])
        outputLoc = outputFolder+"/" +     surfaceName+"_"+moleculeTag+"fe.dat"
        if args.forcerecalc == 0 and os.path.exists(outputLoc):
            continue
        else:
            print("Starting", moleculeTag)
        rRangeWater =r0Start + np.arange(lastInfPoint-r0Start, 1.6,  0.01 ) #usual resolution 0.01     
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
                    rotateMatrixInternal = rotateMatrix(np.pi - theta,-phi)
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
