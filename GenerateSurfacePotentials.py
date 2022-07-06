#Generates the free energy map of a probe atom above a nanomaterial surface
#conventions: energies are in units kJ/mol, distances are in nm.
#Ian Rouse, ian.rouse@ucd.ie , 24/05/2022


import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special as scspec
import scipy.integrate
import argparse


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

#define parameters used for the free energy calculation
temperature = 300.0
probeEpsilon = 0.3598
probeSigma = 0.339
dielectricConst = 80 


#point probes. the first must be left unchanged because this is used to determine the surface level.
#name, sigma,epsilon,   charge
pointProbes =[
["C",0.339,0.3598,0],
["K",0.314264522824 ,  0.36401,1],
["Cl",0.404468018036 , 0.62760,-1]
]

#Define the molecular probe separately

#x,y,z,   charge, mass, sigma, epsilon
waterProbe =  [
[0.00305555555556,-0.00371666666667,0.00438888888889,  -0.834, 16, 0.315 , 0.636],
[0.01195555555556,0.09068333333333,-0.00881111111111,    0.417, 1, 0.0,0.0],
[-0.06084444444444,-0.03121666666667,-0.06141111111111,   0.417, 1, 0.0, 0.0]
]




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
offsetResultFile.write("Material,ZeroLevelOffset[nm],FEOffset[nm],AlanineOffset[nm]\n")



surfaceTargetSet = np.genfromtxt("Structures/SurfaceDefinitions.csv",dtype=str,delimiter=",")
if surfaceTargetSet.ndim == 1:
    surfaceTargetSet = np.array([surfaceTargetSet])

for surfaceTarget in surfaceTargetSet:
    surfaceName = surfaceTarget[0]

    canSkip = 0
    if args.forcerecalc == 0:
        feOutput = outputFolder+"/" +surfaceName+"_fev3.dat"
        waterOutput = outputFolder+"/" +     surfaceName+"_waterfe.dat"
        if os.path.exists(feOutput) and os.path.exists(waterOutput):
            print("Both tables exist for", surfaceName, ", skipping (but writing offset data)")
            canSkip = 1
    foundPMF = 0
    print("Starting point probes for ", surfaceTarget[0])

    surfaceType = surfaceTarget[1]
    alignPointEnergy = float(surfaceTarget[2])
    surfaceAlignOverride = int(surfaceTarget[3])
    pmfAlignEnergy = float(surfaceTarget[4])
    pmfAlignOverride = int(surfaceTarget[5])
    surfaceInputFile = surfaceName+"_combined.csv"
    if surfaceType=="cylinder":
        r0Start = 0.75
    else:
        r0Start = 0.00
    # indexes: 0 = numeric ID, 1 = atomID , 2= atom type, 3= x, 4 = y, 5 = z, 6 = charge, 7= mass, 8 = sigma, 9 = epsilon
    nmData = np.genfromtxt(inputFolder+"/"+surfaceInputFile, skip_header=1,delimiter=",",dtype=str) 
    atomNumericData = nmData[:, 3:].astype(float)
    print("Loaded ", len(atomNumericData), "atoms from file")
    alaPMFData = []
    try:
        pmfPath = "AllPMFsRegularised/"+surfaceName+"_ALASCA-JS.dat"
        if not os.path.exists(pmfPath):
            pmfPath = "AllPMFsRegularised/"+surfaceName+"_ALASCA-AC.dat"
        pmfText = open(pmfPath , "r")
        for line in pmfText:
            if line[0] == "#":
                continue
            if "," in line:
                lineTerms = line.strip().split(",")
            else:
                lineTerms = line.split()
            alaPMFData.append([float(lineTerms[0]),float(lineTerms[1])])
        pmfText.close()
        foundPMF = 1
        alaPMFData = np.array(alaPMFData)
        alaPMFData[:,1] = alaPMFData[:,1] - alaPMFData[-1,1]

    except: 
        print("Failed to read PMF, attempted", pmfPath)
        alaAlignPoint = 0

    if canSkip == 0:
        #indexes: 0,1,2 = x,y,z, 3=charge, 4=mass, 5=sigma, 6=epsilon
        #print(atomNumericData)
        #planes: zero-center the COM, then level the z-direction to the z=0 plane. This offset gets updated later.
        #cylinders: these are already aligned around z=0, so just set the centre of mass to the z=0 plane
        atomNumericData[:,2] = atomNumericData[:,2] - np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
        zeroLevelOffset = np.amax(atomNumericData[   atomNumericData[:,6] > 0.1 ][:,2]  )
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
        resList = []
        waterResList = []
        lastInfPoint = rRange[0] - 1
        lastAllInfPoint = lastInfPoint
        lastWaterInfPoint = lastInfPoint
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
                probeFreeEnergy=-conversionFactor * np.log( np.sum(   np.exp( -totalPotential / conversionFactor) )  / np.sum(ones) )
                #print(r, probe, probeFreeEnergy)
                probeFESet.append(probeFreeEnergy)
                #print( r, probe[0], probeFreeEnergy)
            if not np.isfinite( probeFESet[0]):
                lastInfPoint = r
            resList.append( [r,r-r0Start, r-r0Start ] + probeFESet)

##x,y,z,   charge, mass, sigma, epsilon
#waterProbe =  [
#[0.00305555555556,-0.00371666666667,0.00438888888889,  -0.834, 16, 0.315 , 0.636],
#                
        print("Starting water")
        rRangeWater =r0Start + np.arange(lastInfPoint-r0Start, 2.0,  0.01 ) #actual resolution 0.01       
        for r in rRangeWater:            
            runningNumerator = 0
            runningDenominator = 0
            for theta in  np.linspace(0,np.pi, num = 5, endpoint=True) :
                for phi in np.linspace(0,2*np.pi, num = 8, endpoint=False):
                    rotateMatrixInternal = rotateMatrix(np.pi - theta,-phi)
                    allContributions = 0
                    for atom in waterProbe:
                        epsCombined = np.sqrt(epsBroadcast * atom[6])
                        sigmaCombined = 0.5*(sigmaBroadcast + atom[5])
                        #print(theta,phi,atom)
                        ax = atom[0] * rotateMatrixInternal[0,0] + atom[1]*rotateMatrixInternal[0,1] + atom[2]*rotateMatrixInternal[0,2]
                        ay = atom[0] * rotateMatrixInternal[1,0] + atom[1]*rotateMatrixInternal[1,1] + atom[2]*rotateMatrixInternal[1,2]
                        az = atom[0] * rotateMatrixInternal[2,0] + atom[1]*rotateMatrixInternal[2,1] + atom[2]*rotateMatrixInternal[2,2]
                        distList = []
                        for i in range(len(atomNumericData)): #There is probably a more efficient way of doing this but in the interest of clarity this is easier to debug.
                            if surfaceType == "cylinder":
                                atomDist = np.sqrt( ( r* np.cos(c1grid) + ax - atomNumericData[i,0] )**2 + ( r* np.sin(c1grid) + ay - atomNumericData[i,1] )**2  + ( c2grid + az -atomNumericData[i,2] )**2 )
                                distList.append(atomDist)
                            else:
                                atomDist = np.sqrt(   ( c1grid + ax - atomNumericData[i,0])**2 + (c2grid + ay - atomNumericData[i,1])**2 + (r + az - atomNumericData[i,2])**2  )
                                distList.append(atomDist)
                        distArray = np.array(distList)
                        electricContributions = np.sum( chargeBroadcast  / distArray , axis=0 )
                        scaledDists = sigmaCombined/distArray 
                        ljContributions =  np.sum( 4*epsCombined*( scaledDists**12 - scaledDists**6) , axis=0)
                        allContributions += (electroToKJMol*atom[3]*electricContributions + ljContributions)
                    ones = (  1 + np.zeros_like(allContributions) )
                    runningNumerator += np.sum(  np.exp(-allContributions/conversionFactor)     )* np.sin(theta)
                    runningDenominator += np.sum(ones * np.sin(theta))
            
            waterFreeEnergy=-conversionFactor * np.log( runningNumerator/runningDenominator)
            if np.isfinite(waterFreeEnergy):
                #print(r-r0Start,waterFreeEnergy)
                waterResList.append( [r,r-r0Start, r-r0Start, waterFreeEnergy])            
            else:
                lastWaterInfPoint = r    

        resArray = np.array(resList)
        resArray = resArray[    resArray[:,0] > lastInfPoint ]
        waterResArray = np.array(waterResList)
        waterResArray = waterResArray[ waterResArray[:,0] > lastWaterInfPoint ]
    
        minEnergy = min( np.amin(resArray[:,3]),np.amin(waterResArray[:,3]) )
        requiredOffset = 0
        alanineOffset = 0
        if surfaceAlignOverride == 0:
            #step 1: align the free energy to the target point
            (freeEnergyAlignPoint, freeEnergyAlignActual) = estimateValueLocation(resArray[:,2:], alignPointEnergy)
            #alignPointDist is by default 0.2 . if the free energy align point is found at 0.5, we need to move the potential by -0.3 to align them.
            requiredOffset = alignPointDist - freeEnergyAlignPoint
            resArray[:,2] = resArray[:,2] + requiredOffset #(alaAlignPoint  - freeEnergyAlignPoint)
            waterResArray[:,2] = waterResArray[:,2] + requiredOffset
    else:
        #load in the already calculated potentials and recalculate the offsets
        resArray = np.genfromtxt(outputFolder+"/" + surfaceName+"_fev3.dat",delimiter=",",skip_header=1)
        waterResArray = np.genfromtxt( outputFolder+"/" + surfaceName+"_waterfe.dat", delimiter=",",skip_header=1)
        alanineOffset = 0
        
        atomNumericData[:,2] = atomNumericData[:,2] - np.sum(atomNumericData[:,2] * atomNumericData[:,4])/np.sum(atomNumericData[:,4])
        zeroLevelOffset = np.amax(atomNumericData[   atomNumericData[:,6] > 0.1 ][:,2]  )
        minEnergy = min( np.amin(resArray[:,3]),np.amin(waterResArray[:,3]) )
        requiredOffset = (resArray[:,2] - resArray[:,1])[0]
    if pmfAlignOverride == 0:
        #step 2: align alanine to the target point if possible and if not to the matching point in the free energy
        #(alaAlignPoint, alaAlignVal) = estimateValueLocation(alaPMFData, min(alignPointEnergy, alaPMFData[0,1]))
        #print("Alanine target energy found at: ", alaAlignPoint, " actual value: ", alaAlignVal)
        #
        #print("Alanine PMF is "+str(alignPointEnergy)+" at ", alaAlignPoint, "free energy is at align target at ", freeEnergyAlignPoint)#, " offsetting by ", alaAlignPoint - freeEnergyAlignPoint)
        #
        #next we align alanine to the same point, i.e. we ewant U_ALA( freeEnergyAlignPoint) = alignPointEnergy if possible.
        if foundPMF == 1:
            (alaAlignPoint, alaAlignVal) = estimateValueLocation(alaPMFData, pmfAlignEnergy  ) #get the point in alanine matching the target energy
            (freeEnergyAlaAlignPoint, freeEnergyAlaAlignVal) = estimateValueLocation( resArray[:,2:] , pmfAlignEnergy ) #get the point in the generated free energy matching the target energy
            alanineOffset = freeEnergyAlaAlignPoint - alaAlignPoint
            alaPMFData[:,0] = alaPMFData[:,0]   + alanineOffset #NOTE THE SIGN CONVENTION: if alanineOffset is positive then the PMF is shifted rightwards before fitting takes place and the fitted PMFs are therefore offset to the right and must be shifted left for use
            minEnergy = min(   minEnergy, np.amin(alaPMFData))
    offsetResultFile.write(surfaceName +","+str(zeroLevelOffset)+","   +str(requiredOffset)+","+str(alanineOffset)+"\n")

    plt.figure()
    for probeNum in range(len(pointProbes)):
        plt.plot(   resArray[:,2],resArray[:,3+probeNum] ,label=pointProbes[probeNum][0] )
    #plt.plot(resArray[:,1], resArray[:,3],'k:')
    #plt.plot(resArray[:,2], resArray[:,3],label='F.e.')
    #plt.plot(resArray[:,2], electroToKJMol*resArray[:,4] , label='E.s.')
    plt.plot(waterResArray[:,2], waterResArray[:,3] ,label='water')
    plt.xlim( 0, 1.5)
    plt.ylim( minEnergy-1 , 50)
    if foundPMF== 1:
        plt.plot(alaPMFData[:,0],alaPMFData[:,1],label='ALA')
        #plt.hlines(alignPointEnergy,0,1.5 )
    plt.legend()
    feFileHeader = "r[nm],d[nm],daligned[nm]"
    for probe in pointProbes:
        feFileHeader = feFileHeader+",U"+probe[0]+"(d)[kJ/mol]"
    plt.savefig(outputFolder+"/" +surfaceName +  "_potentials.png")
    np.savetxt( outputFolder+"/" + surfaceName+"_fev3.dat", resArray, fmt='%2.7f',delimiter=",", header=feFileHeader)
    np.savetxt( outputFolder+"/" + surfaceName+"_waterfe.dat", waterResArray, fmt='%2.7f',delimiter=",", header="r[nm],d[nm],daligned[nm],UWater(d)[kj/mol]")
    #plt.show()
offsetResultFile.close()
