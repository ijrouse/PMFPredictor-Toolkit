import numpy as np


temperature = 300.0
dielectricConst = 1 
avogadroNum = 6.022e23
electroToKJMol =  (1.4399/dielectricConst) * 1.6e-19 *avogadroNum/1000.0

def getGridRanges(surfaceType,planeScanHalfLength=0.5):
    if surfaceType == "cylinder":
        numc1=16
        numc2=23
        c1Range = np.linspace(0,2*np.pi, num = numc1, endpoint=False)
        c2Range = np.linspace(-1.0, 1.0, num = numc2, endpoint=True)
    elif surfaceType=="sphere":
        numc1=16
        thetaDelta = 15.0
        c1Range = np.linspace(0,2*np.pi, num = numc1, endpoint=False)
        c2Range = np.arange( thetaDelta, 180.0 - thetaDelta, thetaDelta)*np.pi / 180.0
        numc2 = len(c2Range)
    else:
        numc1=23
        numc2=23
        c1Range = np.linspace(-planeScanHalfLength, planeScanHalfLength, num = numc1, endpoint=True)
        c2Range = np.linspace(-planeScanHalfLength, planeScanHalfLength, num = numc2, endpoint=True)
    return c1Range,c2Range


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
    

def centerProbe2( probeDef):
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
    #print("centered")
    for i in range(len(probeDef)):
        probeDef[i][0] = probeCoords[i,0]
        probeDef[i][1] = probeCoords[i,1]
        probeDef[i][2] = probeCoords[i,2] 
        #print(probeDef[i])
    return probeDef


def klPotentialDivergence(potTrue,potPred,kbTVal = 1,rmaxVal=1.5):
    '''Given potential 1 (true potential) and potential 2 (approximate potential) calculate the KL divergence via the probability distribution, taking kT=1 by default'''
    #print(potTrue.shape)
    #print(potPred.shape)
    eadsTrue = - kbTVal * np.log(  np.trapz(np.exp(- potTrue[:,1]/kbTVal) , potTrue[:,0])/np.trapz(np.ones_like(potTrue[:,0]), potTrue[:,0])   )
    eadsPred = - kbTVal * np.log(  np.trapz(np.exp(- potPred[:,1]/kbTVal), potPred[:,0])/np.trapz(np.ones_like(potPred[:,0]), potPred[:,0])   )
    hgePotsTrueShift = eadsTrue - potTrue[:,1]
    hgePotsPredShift = eadsPred - potPred[:,1]
    weightFunc =  np.exp( (hgePotsTrueShift)/kbTVal)/(kbTVal*rmaxVal)
    weightFunc = np.where( np.isfinite(weightFunc), weightFunc,1e-12) 
    klLoss= np.trapz(  weightFunc  * (hgePotsTrueShift  - hgePotsPredShift)    , potTrue[:,0])
    return klLoss



class MoleculeProbe:
    def __init__(self,name,atomSet):
        self.name = name
        cx = 0
        cy = 0
        cz = 0
        mtot = 0
        for atom in atomSet:
            mtot+= atom.mass
            cx += atom.x * atom.mass
            cy += atom.y * atom.mass
            cz += atom.z * atom.mass
        for i in range(len(atomSet)):
            atomSet[i].x = atomSet[i].x - cx/mtot
            atomSet[i].y = atomSet[i].y - cy/mtot
            atomSet[i].z = atomSet[i].z - cz/mtot
        self.atomSet = atomSet
    def getNumAtoms(self):
        return len(self.atomSet)
    def getAtomSet(self):
        return self.atomSet

class AtomProbe:
    def __init__(self,name,sigma,eps,charge,x=0.0,y=0.0,z=0.0,mass=0.0):
        self.name = name
        self.sigma = sigma
        self.epsilon = eps
        self.charge = charge
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass
    def setCoordinates(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

def getProbeEnergy( chargeBroadcast, epsBroadcast,sigmaBroadcast, distArray, probe,conversionFactor,pointWeights):
    '''
    Calculates the energy between a point probe and all atoms in the structure. 
    probeFreeEnergy=-conversionFactor * np.log( np.sum( pointWeights *  np.exp( -totalPotential / conversionFactor) )  / np.sum( pointWeights *  ones) )
    '''

    epsCombined = np.sqrt(epsBroadcast * probe.epsilon)
    sigmaCombined = 0.5*(sigmaBroadcast + probe.sigma)
    electricContributions = chargeBroadcast  / distArray
    electrostaticPotential = np.sum( electroToKJMol*probe.charge*electricContributions   , axis=-1)
    scaledDists = sigmaCombined/distArray 
    ljPotential = np.sum( 4 * epsCombined * (scaledDists**12 - scaledDists**6 ), axis=-1) #sum over atoms
    totalPotential = electrostaticPotential + ljPotential
    probeFreeEnergy=-conversionFactor * np.log( np.sum(   pointWeights * np.exp( -totalPotential / conversionFactor) )   /np.sum(pointWeights)  )
    #probeSimpleEnergy =np.sum( totalPotential*pointWeights)/np.sum(pointWeights)
    if not np.isfinite(probeFreeEnergy):
        probeFreeEnergy = np.amin(totalPotential)
    probeMinEnergy = np.amin(totalPotential)
    return totalPotential,probeFreeEnergy,probeMinEnergy

#deprecated, remove once tests are done
pointProbesList =[
["C",0.339,0.3598,0],
["K",0.314264522824 ,  0.36401,1],
["Cl",0.404468018036 , 0.62760,-1],
["C2A",0.2,0.3598,0],
["C4A",0.4,0.3598,0],
["CPlus",0.339,0.3598,0.5],
["CMinus",0.339,0.3598,-0.5],
["CMoreLJ",0.339,0.5,0],
["CLessLJ",0.339,0.2,0],
["CEps20",0.399,20,0]
]

pointProbes =[
AtomProbe("C",0.339,0.3598,0),
AtomProbe("K",0.314264522824 ,  0.36401,1),
AtomProbe("Cl",0.404468018036 , 0.62760,-1),
AtomProbe("C2A",0.2,0.3598,0),
AtomProbe("C4A",0.4,0.3598,0),
AtomProbe("CPlus",0.339,0.3598,0.5),
AtomProbe("CMinus",0.339,0.3598,-0.5),
AtomProbe("CMoreLJ",0.339,0.5,0),
AtomProbe("CLessLJ",0.339,0.2,0),
AtomProbe("CEps20",0.399,20,0)
]

def getPointProbeSet(targetProbes):
    return [probe for probe in pointProbes if probe.name in targetProbes] 

def getPointProbeSetList(targetProbes):
    return [probe for probe in pointProbesList if probe.name in targetProbes] 


waterProbeList =  [
[0.00305555555556,-0.00371666666667,0.00438888888889,  -0.834, 16, 0.315 , 0.636],
[0.01195555555556,0.09068333333333,-0.00881111111111,    0.417, 1, 0.0,0.0],
[-0.06084444444444,-0.03121666666667,-0.06141111111111,   0.417, 1, 0.0, 0.0]
]


waterProbe = MoleculeProbe("Water",  [
AtomProbe("O",0.315 , 0.636 ,  -0.834,0.00305555555556,-0.00371666666667,0.00438888888889,16 ),
AtomProbe( "HL", 0,0,0.417,0.01195555555556,0.09068333333333,-0.00881111111111,1),
AtomProbe( "HL", 0,0,0.417,-0.06084444444444,-0.03121666666667,-0.06141111111111,1)
])

waterUCDProbeList =  [
[0.2531 ,  0.0596,  -0.2477,  -0.834, 16, 0.315057422683 ,0.63639],
[0.2620 ,  0.1540,  -0.2609,    0.417, 1, 0.040001352445, 0.19246],
[0.1892 ,   0.0321,  -0.3135,   0.417, 1, 0.040001352445,  0.19246]
]

waterUCDProbe = MoleculeProbe("WaterUCD",  [
AtomProbe("O",0.315057422683 ,0.63639 ,  -0.834,0.2531 ,  0.0596,  -0.2477,16 ),
AtomProbe( "HH", 0.040001352445, 0.19246,0.417,0.2620 ,  0.1540,  -0.2609,1),
AtomProbe( "HH", 0.040001352445, 0.19246,0.417,0.1892 ,   0.0321,  -0.3135,1)
])



methaneProbeList=[
[0.108,0.006,0.001,-0.106800,12.01000,3.39771e-01,4.51035e-01],
[0.072,0.109,0.000,0.026700,1.00800,2.60018e-01,8.70272e-02],
[0.072,-0.047,-0.087,0.026700,1.00800,2.60018e-01,8.70272e-02],
[0.072,-0.045,0.091,0.026700,1.00800,2.60018e-01,8.70272e-02],
[0.217,0.006,0.001,0.026700,1.00800,2.60018e-01,8.70272e-02]
]


methaneProbe = MoleculeProbe("Methane",[
AtomProbe("C",3.39771e-01,4.51035e-01  ,-0.106800 , 0.108,0.006,0.001,12.01000),
AtomProbe("H",2.60018e-01,8.70272e-02,0.026700,0.072,0.109,0.000,1.00800),
AtomProbe("H",2.60018e-01,8.70272e-02,0.026700,0.072,-0.047,-0.087,1.00800),
AtomProbe("H",2.60018e-01,8.70272e-02,0.026700,0.072,-0.045,0.091,1.00800),
AtomProbe("H",2.60018e-01,8.70272e-02,0.026700,0.217,0.006,0.001,1.00800)
])


clineProbeList =[
[0.0, 0.0, -0.75, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, -0.5, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, -0.25, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.25, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.5, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.75, 0, 12.0100, 3.39771e-01,4.51035e-01]
]


clineProbe=MoleculeProbe("C7Line",
[
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, -0.75, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, -0.5, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, -0.25, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, 0.0, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, 0.25, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, 0.5, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, 0.75, 12.01 )
]

)

sixcarbProbeList=[
[0.099,0.000,0.001,0,12.01000,3.31521e-01,4.13379e-01],
[0.028,-0.120,0.001,0,12.01000,3.31521e-01,4.13379e-01],
[-0.112,-0.120,-0.0005,0,12.01000,3.31521e-01,4.13379e-01],
[-0.182,0.000,-0.001,0,12.01000,3.31521e-01,4.13379e-01],
[-0.112,0.121,-0.0005,0,12.01000,3.31521e-01,4.13379e-01],
[0.028,0.121,0.0005,0,12.01000,3.31521e-01,4.13379e-01]

]

sixcarbProbe = MoleculeProbe("C6Ring", [
    AtomProbe("C", 3.31521e-01,4.13379e-01, 0, 0.099,0.000,0.001,12.01),
    AtomProbe("C", 3.31521e-01,4.13379e-01, 0, 0.028,-0.120,0.001,12.01),
    AtomProbe("C", 3.31521e-01,4.13379e-01, 0, -0.112,-0.120,-0.0005,12.01),
    AtomProbe("C", 3.31521e-01,4.13379e-01, 0, -0.182,0.000,-0.001,12.01),
    AtomProbe("C", 3.31521e-01,4.13379e-01, 0, -0.112,0.121,-0.0005,12.01),
    AtomProbe("C", 3.31521e-01,4.13379e-01, 0, 0.028,0.121,0.0005,12.01)
    ])


cline3ProbeList =[
[0.0, 0.0, -0.25, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.25, 0, 12.0100, 3.39771e-01,4.51035e-01]
]
cline3Probe=MoleculeProbe("C3Line",
[
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, -0.25, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, 0.0, 12.01 ),
    AtomProbe("C",3.39771e-01,4.51035e-01,0,  0.0, 0.0, 0.25, 12.01 )
]

)
