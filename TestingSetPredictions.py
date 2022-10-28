import numpy as np
import matplotlib.pyplot as plt
import os
import HGEFuncs
import argparse



parser = argparse.ArgumentParser(description="Parameters for BuildPredictedPMFs")
parser.add_argument("-b","--bootstrap", type=int,default=0, help="If zero, use cluster results. Else bootstrapping")

parser.add_argument("-m","--match", type=int,default=0, help="If zero, use default parameters for SSD, source, methane offset. Else use predefined.")
parser.add_argument("-c","--complete",type=int,default=0,help="Include also PMFs with no real equivalent")
args = parser.parse_args()


allTrainingPMFsScan = os.listdir("AllPMFs")
allTestingPMFsScan = os.listdir("TestingPMFs251022")

allTrainingPMFsScan.sort()
allTestingPMFsScan.sort()
kbTVal = 2.48

knownMaterials = []
knownChemicals = []

for trainingPMF in allTrainingPMFsScan:
    trainingPMFTerms= trainingPMF.split(".")[0].split("_")  
    knownMaterials.append(trainingPMFTerms[0])
    knownChemicals.append(trainingPMFTerms[1])
knownMaterials = list(set(knownMaterials))
knownChemicals = list(set(knownChemicals))


useBootstrap = True
useMatching = True

if args.bootstrap == 0:
    useBootstrap = False
if args.match == 0:
    useMatching = False



if useBootstrap == True:
    targetModel = "PMFPredictor-oct13-simplesplit-bootstrapped-ensemble"
    modelsForAverage = [1,2,3,4,5,6,7,8,9,10]
else:
    targetModel = "PMFPredictor-oct13-clustersplit-ensemble"
    modelsForAverage = [1,2,5]

if useMatching == True:
    matchMode = "matched"
else:
    matchMode = "canonical"


isSingleton = False
singleString = ""
if isSingleton == True:
    singleString="_one"
    modelsForAverage = [1]

allPMFs = allTrainingPMFsScan + allTestingPMFsScan
allPMFs.sort()
if args.complete == 1:
    allPMFs = []
    targetDir = "predicted_avg_pmfs/"+targetModel+str(1)+"_"+matchMode
    allFolders = os.listdir(targetDir)
    allMaterials = []
    for folder in allFolders:
        folderTerms = folder.split("_")
        if folderTerms[-1] == "simple":
            allMaterials.append(folder)
    allMaterials = list(set(allMaterials))
    allMaterials.sort()
    for materialName in allMaterials:
        xchemNames = os.listdir(targetDir+"/"+materialName)
        xchemNames.sort()
        for chemName in xchemNames:
            allPMFs = allPMFs + [ targetDir+"/"+materialName+"/"+chemName  ]

#allPMFs.sort()
#allPMFs = list(set(allPMFs))
#allPMFs.sort()

if args.complete == 1:
    outFile = open(targetModel+"_"+matchMode+singleString+"_all_eads.csv","w")
else:
    outFile = open(targetModel+"_"+matchMode+singleString+"_eads.csv","w")


outFile.write("#Material,Chemical,Class, EAdsMD[kJ/mol], EadsAvgPred[kJ/mol], MeanEadsPred[kJ/mol],SDevEadsPred[kJ/mol] \n")
for testPMF in allPMFs:
    PMFTerms= testPMF.split(".")[0].split("_")  
    #print(PMFTerms)
    if len(PMFTerms) == 2:
        materialName = PMFTerms[0]
        chemName = PMFTerms[1]
        targetChemName = chemName
        targetPredPMF = "predicted_pmfs/"+targetModel+str(1)+"_"+matchMode+"/"+materialName+"/"+targetChemName+".dat"
    else:
        materialName = PMFTerms[3].split("/")[-1]
        chemName = PMFTerms[4].split("/")[-1]
        targetPredPMF = testPMF
    targetChemName = chemName

    classLabel = ""
    if materialName in knownMaterials:
        classLabel = classLabel+"TM"
    else:
        classLabel = classLabel+"NM"
    if chemName in knownChemicals:
        classLabel = classLabel+"TC"
    else:
        classLabel = classLabel+"NC"
    if not os.path.exists( targetPredPMF):
        print("Not found: ", targetPredPMF)
        continue
    if classLabel=="TMTC":
        loadedTargetPMF = HGEFuncs.loadPMF("AllPMFs/"+materialName+"_"+chemName+".dat",False)
    else:
        loadedTargetPMF = HGEFuncs.loadPMF("TestingPMFs251022/"+materialName+"_"+chemName+".dat",False)
    
    if len(loadedTargetPMF) > 1:
        rmax = loadedTargetPMF[-1,0] - loadedTargetPMF[0,0]
        deltar = np.mean( loadedTargetPMF[1:,0] - loadedTargetPMF[:-1,0])
        eadsTarget = (- kbTVal * np.log( np.sum(deltar*  np.exp(-loadedTargetPMF[:,1]/kbTVal) )  /rmax ))
    else:
        eadsTarget = "x"
        classLabel = classLabel+"X"
    if isSingleton == False:
        targetPredPMF = "predicted_avg_pmfs/"+targetModel+str(1)+"_"+matchMode+"/"+materialName+"_simple/"+targetChemName+".dat"
    else:
        targetPredPMF = "predicted_pmfs/"+targetModel+str(1)+"_"+matchMode+"/"+materialName+"/"+targetChemName+".dat"
    loadedPredPMF = HGEFuncs.loadPMF(targetPredPMF,False)
    rmax = loadedPredPMF[-1,0] - loadedPredPMF[0,0]
    deltar = np.mean(loadedPredPMF[1:,0] - loadedPredPMF[:-1,0])
    eadsAvgPred = (- kbTVal * np.log( np.sum(deltar*  np.exp(-loadedPredPMF[:,1]/kbTVal)  )  /rmax ))
    absMin = 5
    energySet = []
    for i in modelsForAverage:  #range(1,11):
        targetPredPMF = "predicted_pmfs/"+targetModel+str(i)+"_"+matchMode+"/"+materialName+"/"+targetChemName+".dat"
        loadedPredPMF = HGEFuncs.loadPMF(targetPredPMF,False)
        rmax = loadedPredPMF[-1,0] - loadedPredPMF[0,0]
        deltar = np.mean(loadedPredPMF[1:,0] - loadedPredPMF[:-1,0])
        eadsPred = (- kbTVal * np.log( np.sum(deltar*  np.exp(-loadedPredPMF[:,1]/kbTVal)  )  /rmax ))
        #print( np.amin(loadedPredPMF[:,1]), eadsPred )
        if np.amin(loadedPredPMF[:,1]) < absMin:
            absMin = np.amin(loadedPredPMF[:,1])
        energySet.append(eadsPred)
    if len(energySet) > 1:
        meanEnergy = np.mean(energySet)
        stdEnergy = np.std(energySet)
    else:
        meanEnergy = eadsAvgPred
        stdEnergy = 0
    resLine = [materialName, chemName, classLabel,eadsTarget,eadsAvgPred,meanEnergy, stdEnergy]
    resString = ",".join( [str(a) for a in resLine]    )
    outFile.write(resString+"\n")
    print(materialName, chemName, classLabel,eadsTarget,eadsAvgPred,meanEnergy, stdEnergy,absMin)
outFile.close()
