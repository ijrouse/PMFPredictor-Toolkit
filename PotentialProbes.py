import numpy as np

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

pointProbes =[
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

def getPointProbeSet(targetProbes):
    return [probe for probe in pointProbes if probe[0] in targetProbes] 

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

clineProbe =[
[0.0, 0.0, -0.75, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, -0.5, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, -0.25, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.25, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.5, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.75, 0, 12.0100, 3.39771e-01,4.51035e-01]
]



sixcarbProbe=[
[0.099,0.000,0.001,0,12.01000,3.31521e-01,4.13379e-01],
[0.028,-0.120,0.001,0,12.01000,3.31521e-01,4.13379e-01],
[-0.112,-0.120,-0.0005,0,12.01000,3.31521e-01,4.13379e-01],
[-0.182,0.000,-0.001,0,12.01000,3.31521e-01,4.13379e-01],
[-0.112,0.121,-0.0005,0,12.01000,3.31521e-01,4.13379e-01],
[0.028,0.121,0.0005,0,12.01000,3.31521e-01,4.13379e-01]

]

cline3Probe =[
[0.0, 0.0, -0.25, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0, 0, 12.0100, 3.39771e-01,4.51035e-01],
[0.0, 0.0, 0.25, 0, 12.0100, 3.39771e-01,4.51035e-01]
]
