import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.backend import cast
from tensorflow import strings
from tensorflow.keras.utils import plot_model
import scipy.special as scspec
import datetime

def loadPMF(target):
    PMFData = []
    foundAC = 1
    foundJS = 1
    try:
        pmfText = open(target , "r")
        for line in pmfText:
            if line[0] == "#":
                continue
            if "," in line:
                lineTerms = line.strip().split(",")
            else:
                lineTerms = line.split()
            PMFData.append([float(lineTerms[0]),float(lineTerms[1])])
        pmfText.close()
        foundPMF = 1
        print("Loaded ", target)
        PMFData = np.array(PMFData)
        PMFData[:,1] = PMFData[:,1] - PMFData[-1,1] #set to 0 at end of PMF
    except: 
        print("Failed to read PMF (AC)", target)
        foundAC = 0
    if foundAC == 1:
        return PMFData    
    try:
        JSTarget = target[:-6]+"JS.dat"
        print(JSTarget)
        pmfText = open(JSTarget , "r")
        for line in pmfText:
            if line[0] == "#":
                continue
            if "," in line:
                lineTerms = line.strip().split(",")
            else:
                lineTerms = line.split()
            PMFData.append([float(lineTerms[0]),float(lineTerms[1])])
        pmfText.close()
        foundPMF = 1
        print("Loaded ", JSTarget)
        PMFData = np.array(PMFData)
        PMFData[:,1] = PMFData[:,1] - PMFData[-1,1] #set to 0 at end of PMF
    except:    
        foundJS = 0
    if foundJS == 1:
        return PMFData
    return np.array( [ [0,0] , [1.5,0]])

 
        
      
targetModel ="pmfpredict-july25-order20-cosinefixed-moredata-censoredenergies-scaledmixerr-v2-desatfinal-eadsdesat"
#targetModel = "june14-roughonly-predictr0"
targetModel="pmfpredict-july25-order20-cosinefixed-moredata-censoredenergies-scaledmixerr-v2-desatlogactivation-morenoise-rmsadd0p3"
targetModel="pmfpredict-july27-deconvresid-1-unscaledmse-noe0-extracorrection-wider-prescale-e0est-extranoise-noscaffolds"




targetModel="pmfpredict-aug02-gatealpha0p1-nresid8-mixing5"

targetModel = "pmfpredict-aug29-mixtrainval"
def scaledMSE(scaleVal):
    def loss(y_true,y_pred):
        return tf.keras.losses.mean_squared_error( y_true/scaleVal, y_pred/scaleVal) 
    return loss
    
#datasetAll= pd.read_csv("Datasets/TrainingData.csv")

os.makedirs("predicted_pmfs/"+targetModel,exist_ok=True)

UnitedAtomNames = {
"ALASCA-AC":"ALA",
"ARGSCA-AC":"ARG",
"LYSSCA-AC":"LYS",
"HIDSCA-AC":"HID",
"HIESCA-AC":"HIE",
"ASPSCA-AC":"ASP",
"GLUSCA-AC":"GLU",
"SERSCA-AC":"SER",
"THRSCA-AC":"THR",
"ASNSCA-AC":"ASN",
"GLNSCA-AC":"GLN",
"CYSSCA-AC":"CYS",
 "GLY-AC":"GLY",
 "PRO-AC":"PRO",
 "VALSCA-AC":"VAL",
 "ILESCA-AC":"ILE",
 "LEUSCA-AC":"LEU",
 "METSCA-AC":"MET",
 "PHESCA-AC":"PHE",
 "TRYSCA-AC":"TYR",
 "TRPSCA-AC":"TRP",
 "HIPSCA-AC":"HIP",
 "ETA-AC":"ETA",
 "PHO-AC":"PHO",
 "CHL-AC":"CHL",
 "DGL-AC":"DGL",
 "EST-AC":"EST"
}



#Write the training set predictions out

varsetFile = open("models/"+targetModel+"/varset.txt","r")
aaVarSet = varsetFile.readline().strip().split(",")
varsetFile.close()
loadedModel = tf.keras.models.load_model("models/"+targetModel+"/checkpoints/checkpoint-train"    , custom_objects={ 'loss': scaledMSE(1) ,'aaVarSet':aaVarSet },compile=False)


outputVarset = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16", "A17","A18","A19","A20" ,"e0predict","Emin"]
'''

predictionSetAll =  ( np.array( loadedModel.predict([    datasetAll[aaVarSet] ]))[:,:,0] ).T
print(predictionSetAll[1])
outputVarset = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict"]
for i in range(len(outputVarset)):
    datasetAll[ outputVarset[i]+"_regpredict" ] =predictionSetAll[:,i].flatten()

datasetAll.to_csv("predicted_pmfs/"+targetModel+"/checkpointpredictallCoeffs.csv")

'''


'''
targetE0=10
datasetSingle = pd.read_csv("predicted_pmfs/"+targetModel+"/uniquepredictionsCoeffs.csv")
datasetSingle["fittedE0"]=targetE0
predictionSetSingle =  ( np.array( loadedModel.predict([    datasetSingle[aaVarSet] ]))[:,:,0] ).T
for i in range(len(outputVarset)):
    datasetSingle[ outputVarset[i]+"_regpredict" ] =predictionSetSingle[:,i].flatten()

datasetSingle.to_csv("predicted_pmfs/"+targetModel+"/checkpointpredict_targetinput_"+str(targetE0)+".csv")
'''







#Generate the actual tabulated PMFs
def HGEFunc(r, r0, n):
    return (-1)**(1+n) * np.sqrt( 2*n - 1) * np.sqrt(r0)/r * scspec.hyp2f1(1-n,n,1,r0/r)

targetMolecules = pd.read_csv("Datasets/ChemicalPotentialCoefficientsNoise1-aug26.csv")
targetSurfaces = pd.read_csv("Datasets/SurfacePotentialCoefficientsNoise-1-aug30.csv")
targetE0Val = 50

uniqueMaterials = targetSurfaces['SurfID'].unique().tolist()
for materialName in uniqueMaterials:
    os.makedirs( "predicted_pmfs/"+targetModel+"/allchemicals/"+materialName,exist_ok=True)
    os.makedirs( "predicted_pmfs/"+targetModel+"/UA-surface-predicted/"+materialName,exist_ok=True)
os.makedirs("predicted_pmfs/"+targetModel+"/PMFFigures", exist_ok=True)
#loadedModel = tf.keras.models.load_model("checkpoints/"+targetModel)

 

uniquetargetMolecules = targetMolecules.drop_duplicates( subset=['ChemID']  ,keep='first')
uniquetargetSurfaces = targetSurfaces.drop_duplicates( subset=['SurfID']  ,keep='first')
surfaceParamSet = targetSurfaces.columns.tolist()
chemParamSet = targetMolecules.columns.tolist()   
print(targetSurfaces)

#build the placeholder database with every molecule x surface pair
combinedDataset = pd.merge(uniquetargetMolecules,uniquetargetSurfaces,how="cross")

combinedDataset["PMFName"] = combinedDataset["SurfID"]+"_"+combinedDataset["ChemID"]

#datasetAll=datasetAll.drop( datasetAll[( datasetAll["fittedE0"] ) > maxE0].index )


#combinedDataset = pd.merge(uniquetargetMolecules,uniquetargetSurfaces,how="cross")
#combinedDataset = combinedDataset.drop(       combinedDataset[     combinedDataset["ChemCProbeR0"] != combinedDataset["SurfCProbeR0"]       ].index                ) 





#combinedDataset = combinedDataset.sort_values( by=["ChemCProbeR0"] ,ascending=False)



#combinedDataset["fittedE0"] = targetE0Val
#combinedDataset["rEMin"] = 0.5
#combinedDataset["r0original"] = combinedDataset["r0"]
#combinedDataset["r0"] = 
#combinedDataset["rmin"] = combinedDataset["fljr0"] + 0.01
#combinedDataset["rmin"].clip(lower = 0.2, inplace=True)
#print(combinedDataset)

#placeholder values for the parameters used in model training
combinedDataset["rEMin"] = 0.5
combinedDataset["EMin"] = -1 
combinedDataset["fittedE0"] = 20
combinedDataset["r0"] = 0.2 + combinedDataset["SurfAlignDist"]

def logSumMax( x):
    return np.log(  np.sum(np.exp(x)))
#combinedDataset["r0"] = np.log(  np.exp(combinedDataset["ChemLJR0"] )+ np.exp(combinedDataset["SurfLJR0"]))


#Generate PMFs by stepwise prediction moving either forwards or backwards through the value of rmin in a specified interval, stopping when E(RMin) = targetE0.
def recurrentPredict(model, dataset, direction=-1, r0Min = 0.05, r0Max=0.9, targetE0 = 20):
    workingDataset = dataset.copy()
    if direction==-1:
        currentr0 = r0Max - 0.01
    else:
        currentr0 = r0Min + 0.01 
    workingDataset["r0"] = currentr0
    workingDataset["lastE0"] =direction*targetE0*2
    #nonconvergedSet = workingDataset[   workingDataset["lastE0"] > targetE0]

    if direction == -1:
        nonconvergedMask = workingDataset["lastE0"] < targetE0
    else:
        nonconvergedMask = workingDataset["lastE0"] > targetE0
    #print(len(workingDataset[nonconvergedMask]))
    while len( workingDataset[nonconvergedMask] ) > 0 and currentr0 < r0Max and currentr0 > r0Min:
    
   
        print(  currentr0)

        
        workingDataset.loc[nonconvergedMask,"r0"] = currentr0
        workingDataset.loc[nonconvergedMask, "lastE0" ] = 0
        #sort target molecules by the distance of r0 from currentr0, drop duplicates
        #update workingDataset by target molecule names
        #repeat for surfaces        
        
        
        predictionSet = ( np.array( model.predict([ workingDataset.loc[ nonconvergedMask,  aaVarSet] ])[0]   )[:,:,0] ).T
        for i in range(len(outputVarset)):
            workingDataset.loc[nonconvergedMask,  outputVarset[i]+"_regpredict" ] =predictionSet[:,i].flatten()  
            if i<16:
                workingDataset.loc[ nonconvergedMask,   "lastE0" ] = workingDataset.loc[ nonconvergedMask,   "lastE0" ] + np.sqrt(2*(i+1) - 1) * (predictionSet[:,i].flatten() ) /np.sqrt(currentr0)
        #print( workingDataset.loc[nonconvergedMask,   "lastE0"] , predictionSet[:,-1])
        #workingDataset.update( nonconvergedSet )   
        if direction == -1:
            nonconvergedMask = workingDataset["lastE0"] < targetE0 
        else:     
            nonconvergedMask = workingDataset["lastE0"] > targetE0 
        currentr0 = currentr0+   direction*0.01
    return workingDataset
    
def singleStepPredict(model,dataset):
    predictionSet = ( np.array( model.predict(  [ dataset[aaVarSet]   ]   )[0])[:,:,0] ).T
    for i in range(len(outputVarset)):
        dataset[ outputVarset[i]+"_regpredict" ] =predictionSet[:,i].flatten()  
    return dataset



                
def recurrentPredictV2(model, dataset, direction=-1, r0Min = 0.05, r0Max=0.9, targetE0 = 20):

    moleculeSetWorking = targetMolecules.copy()
    surfaceSetWorking = targetSurfaces.copy()
    if direction==-1:
        currentr0 = r0Max - 0.01
    else:
        currentr0 = r0Min + 0.01 
    #nonconvergedSet = workingDataset[   workingDataset["lastE0"] > targetE0]

    allPMFs = dataset["PMFName"].values.tolist()
    completedPMFs = []
    completedPMFSet = dataset.iloc[:0,:].copy() 
    #print(allPMFs)
    #print(len(workingDataset[nonconvergedMask]))
    while len(completedPMFs) < len(allPMFs) and currentr0 < r0Max and currentr0 > r0Min:
        print(  currentr0)
        #sort all molecule and surface lines to find the ones with the closest set of values of r0
        moleculeSetWorking["R0Dist"] = np.sqrt((moleculeSetWorking["ChemCProbeR0"].to_numpy() - currentr0)**2)
        moleculeSetWorking.sort_values( by=["R0Dist"] ,ascending=True, inplace=True)
        moleculeBestMatches = moleculeSetWorking.drop_duplicates( subset=['ChemID']  ,keep='first')       
        surfaceSetWorking["R0Dist"] = np.sqrt((surfaceSetWorking["SurfCProbeR0"].to_numpy() - currentr0)**2)
        surfaceSetWorking.sort_values( by=["R0Dist"] ,ascending=True, inplace=True)
        #print(targetSurfaces)
        surfaceBestMatches = surfaceSetWorking.drop_duplicates( subset=['SurfID']  ,keep='first')  
        #print(moleculeBestMatches)             
        #print(surfaceBestMatches )
        combinedDatasetAtR0 = pd.merge(moleculeBestMatches,surfaceBestMatches,how="cross")
        combinedDatasetAtR0["PMFName"] = combinedDatasetAtR0["SurfID"]+"_"+combinedDatasetAtR0["ChemID"]
        combinedDatasetAtR0.sort_values( by=["PMFName"],inplace=True,ascending=True)
        combinedDatasetAtR0.drop( combinedDatasetAtR0[combinedDatasetAtR0["PMFName"].isin( completedPMFs)].index,inplace=True )
        combinedDatasetAtR0["r0"] = currentr0
        combinedDatasetAtR0["EMin"] = -1
        combinedDatasetAtR0["rEMin"] = 0.5
        combinedDatasetAtR0[  "lastE0" ] = 0
        combinedDatasetAtR0[  "fittedE0" ] = targetE0
        combinedDatasetAtR0["resolution"] = 0.01
        predictionSet = ( np.array( model.predict([ combinedDatasetAtR0[ aaVarSet]      ])[0]   )[:,:,0] ).T
        for i in range(len(outputVarset)):
            combinedDatasetAtR0[ outputVarset[i]+"_regpredict" ] =predictionSet[:,i].flatten()  
            if i<16:
                combinedDatasetAtR0[   "lastE0" ] = combinedDatasetAtR0[  "lastE0" ] + np.sqrt(2*(i+1) - 1) * (predictionSet[:,i].flatten() ) /np.sqrt(currentr0)
        #print( workingDataset.loc[nonconvergedMask,   "lastE0"] , predictionSet[:,-1])
        #workingDataset.update( nonconvergedSet )   
        if direction == -1:
            newFinishedNames = combinedDatasetAtR0.loc[   combinedDatasetAtR0["lastE0"] > targetE0  ,"PMFName"  ].values.tolist()
            completedPMFSet = completedPMFSet.append(  combinedDatasetAtR0[   combinedDatasetAtR0["lastE0"] > targetE0  ] )

        else:     
            newFinishedNames = combinedDatasetAtR0.loc[   combinedDatasetAtR0["lastE0"] < targetE0  ,"PMFName"  ].values.tolist()
            completedPMFSet = completedPMFSet.append( combinedDatasetAtR0[   combinedDatasetAtR0["lastE0"] < targetE0  ]  )
        completedPMFs = completedPMFs + newFinishedNames    
        print(completedPMFSet)
        currentr0 = currentr0+   direction*0.01
    return completedPMFSet



#Generate PMFs for each specified value of r0, construct a weighted average with weight w = exp( - (r - r0) ).Theta(r - r0) , i.e. 0 for r< r0, exponentially decaying for larger
#coeffSet should be a list of the form [ [r01,A11,A12] ... [r0N,A1N,A2N] ] where N is the number of predictions made for a specific chem-surface pair
def mergePMFPredictions(rRange, coeffSet):
    r0Set = coeffSet[:,0]
    rmesh,r0mesh = np.meshgrid(rRange,r0Set)
    weightMesh = np.where( rmesh > r0mesh, np.exp( -(rmesh-r0mesh) ), 0.0 )
    pmfMesh = np.zeros_like(rmesh)
    for i in range(1,20):
        rmesh2, coeffmesh = np.meshgrid(rRange, coeffSet[:,i])
        pmfMesh = pmfMesh + coeffmesh * HGEFunc(rmesh, r0mesh, i)
    pmfMesh = pmfMesh* weightMesh
    pmfRow = np.sum(pmfMesh,axis=-1)/np.sum( weightMesh,axis=-1)
    
    
outputFig = plt.figure()
numberOutputPMFs = 4
axSet = []
for i in range(2*numberOutputPMFs):
    axSet.append(plt.subplot(numberOutputPMFs,2,i+1))

def stepwisePredict(model, dataset,  rValRange,direction=1,r0Min = 0.05, r0Max=0.9, targetE0 = 20):
    deltaR0 = 0.05
    moleculeSetWorking = targetMolecules.copy()
    surfaceSetWorking = targetSurfaces.copy()
    if direction==-1:
        currentr0 = r0Max - deltaR0
    else:
        currentr0 = r0Min  
    #nonconvergedSet = workingDataset[   workingDataset["lastE0"] > targetE0]

    allPMFs = dataset["PMFName"].values.tolist()
    completedPMFs = []
    completedPMFSet = dataset.iloc[:0,:].copy() 
    #print(allPMFs)
    #print(len(workingDataset[nonconvergedMask]))
    pmfCalcSet = np.zeros( ( len(allPMFs ), len(rValRange) ) )
    pmfPointWeights = np.zeros_like( rValRange)
    while   currentr0 < r0Max and currentr0 >= r0Min:
        print(  currentr0)
        #sort all molecule and surface lines to find the ones with the closest set of values of r0
        moleculeSetWorking["R0Dist"] = np.sqrt((moleculeSetWorking["ChemCProbeR0"].to_numpy() - currentr0)**2)
        moleculeSetWorking.sort_values( by=["R0Dist"] ,ascending=True, inplace=True)
        moleculeBestMatches = moleculeSetWorking.drop_duplicates( subset=['ChemID']  ,keep='first')       
        surfaceSetWorking["R0Dist"] = np.sqrt((surfaceSetWorking["SurfCProbeR0"].to_numpy() - currentr0)**2)
        surfaceSetWorking.sort_values( by=["R0Dist"] ,ascending=True, inplace=True)
        #print(targetSurfaces)
        surfaceBestMatches = surfaceSetWorking.drop_duplicates( subset=['SurfID']  ,keep='first')  
        #print(moleculeBestMatches)             
        #print(surfaceBestMatches )
        combinedDatasetAtR0 = pd.merge(moleculeBestMatches,surfaceBestMatches,how="cross")
        combinedDatasetAtR0["PMFName"] = combinedDatasetAtR0["SurfID"]+"_"+combinedDatasetAtR0["ChemID"]
        combinedDatasetAtR0.sort_values( by=["PMFName"],inplace=True,ascending=True)
        combinedDatasetAtR0.drop( combinedDatasetAtR0[combinedDatasetAtR0["PMFName"].isin( completedPMFs)].index,inplace=True )
        combinedDatasetAtR0["r0"] = currentr0
        combinedDatasetAtR0["EMin"] = -1
        combinedDatasetAtR0["rEMin"] = 0.5
        combinedDatasetAtR0[  "lastE0" ] = 0
        combinedDatasetAtR0[  "fittedE0" ] = targetE0
        combinedDatasetAtR0["resolution"] = 0.01
        predictionSet = ( np.array( model.predict([ combinedDatasetAtR0[ aaVarSet]      ])[0]   )[:,:,0] ).T
        

        r0Mask = np.where( rValRange >= currentr0, 1.0, 0.0)
        #print(r0Mask)
        r0Weight = np.exp( -10.0* ((rValRange - (currentr0 + 3* deltaR0 ) )**2 )/(deltaR0**2)   )*r0Mask
        pmfR0Contribution0 = np.zeros( (len(allPMFs),len(rValRange) ))
        for i in range(20 ):
            basisFuncVals = HGEFunc(rValRange, currentr0, i+1) * r0Mask   
            coeffISet =predictionSet[:,i].flatten()  
            pmfR0Contribution0 += np.outer( coeffISet,basisFuncVals) # numPMFs x numRPoints
        pmfR0Contribution = pmfR0Contribution0 * r0Weight #possibly transpose first to match broadcasting if needed, or manually tile
        pmfCalcSet = pmfCalcSet + pmfR0Contribution
        pmfPointWeights = pmfPointWeights + r0Weight
        #print(pmfPointWeights)
        currentr0 = currentr0+   direction*deltaR0
        currentPMFs = pmfCalcSet / pmfPointWeights
        for j in range(0,numberOutputPMFs):
            axSet[2*j].clear()
            axSet[2*j].scatter( rValRange, currentPMFs[j] )
            axSet[2*j + 1].scatter( rValRange, pmfR0Contribution0[j] ,alpha=0.1)
        plt.pause(0.05)
    return currentPMFs  
    
    
    
    
    
    
    
    

    
    
    
    
rRange = np.arange( 0.05, 1.5, 0.01)
#combinedDataset = recurrentPredictV2( loadedModel, combinedDataset, direction=1, r0Min = 0.15, r0Max = 0.8, targetE0=40)
combinedDataset = stepwisePredict(loadedModel,combinedDataset,rRange)

plt.show()    

'''

predictionSetSingle =  ( np.array( loadedModel.predict([    combinedDataset[aaVarSet] ]  ,verbose=True ))[:,:,0] ).T

outputVarset = ["r0","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict" ]
for i in range(len(outputVarset)):
    combinedDataset[ outputVarset[i]+"_predicted" ] =predictionSetSingle[:,i].flatten()
'''


'''
combinedDataset.to_csv("predicted_pmfs/"+targetModel+"/predicted_coeffs_allpairs.csv")

offsetData = np.genfromtxt("Datasets/SurfaceOffsetDataManual.csv",skip_header=1,delimiter=",", dtype=str)
offsetDict = {}
for materialOff in offsetData:
    offsetDict[ materialOff[0] ] = float( materialOff[3] )
 

makePlots = 1
outputPlot = plt.figure() 
 
 
seenPMFs = [] 
for index,row in combinedDataset.iterrows():
    materialName = row["SurfID"]
    chemName = row["ChemID"]
    r0Target = row["r0"]
    if materialName+"_"+chemName in seenPMFs:
        continue
    rRange = np.arange( r0Target, 1.5, 0.001)
    pmf = np.zeros_like(rRange)
    for i in range(1,20):
        pmf = pmf + row["A"+str(i)+"_regpredict"] * HGEFunc(rRange, r0Target+0.001, i)
    finalPMF = np.stack((rRange,pmf),axis=-1)
    finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
    firstVal = finalPMF[0,1]
    if firstVal < 20 :
        continue
    seenPMFs.append( materialName+"_"+chemName )
    finalPMFOffset = np.copy(finalPMF)
    finalPMFOffset[:,0] = finalPMFOffset[:,0] + offsetDict[ materialName]
    np.savetxt("predicted_pmfs/"+targetModel+"/allchemicals/"+materialName+"/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    if chemName in UnitedAtomNames:
        np.savetxt("predicted_pmfs/"+targetModel+"/UA-surface-predicted/"+materialName+"/"+UnitedAtomNames[chemName]+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
        if makePlots == 1:
             originalPMFData = loadPMF("AllPMFs/"+ materialName+"_"+chemName+".dat"   )
             outputPlot.clear()
             plt.plot( originalPMFData[::5,0], originalPMFData[::5,1], 'bx')
             plt.plot( finalPMF[:,0], finalPMF[:,1], 'k:')
             plt.plot( finalPMFOffset[:,0], finalPMFOffset[:,1], 'k-')
             plt.xlabel("r [nm]")
             plt.ylabel( "U(r) [kJ/mol]")
             plt.xlim(0,1.5)
             plt.savefig("predicted_pmfs/"+targetModel+"/PMFFigures/"+ materialName+"_"+chemName+".png")
             
             
readmeFile = open("predicted_pmfs/"+targetModel+"/README","w")
readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using model variant " + targetModel)
readmeFile.close()
'''
        
