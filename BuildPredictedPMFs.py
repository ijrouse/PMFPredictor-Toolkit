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
from tensorflow.keras.utils import plot_model
import scipy.special as scspec
import datetime
import HGEFuncs
import argparse

parser = argparse.ArgumentParser(description="Parameters for BuildPredictedPMFs")
parser.add_argument("-m","--match", type=int,default=0, help="If zero, use default parameters for SSD, source, methane offset. Else use predefined.")
parser.add_argument("-b","--bootstrap", type=int,default=1, help="If zero, use cluster results. Else bootstrapping")
parser.add_argument("-a","--averageonly",type=int,default=0,help="If non-zero, skip recalculation and just average")

args = parser.parse_args()

skipRecalc  = False
if args.averageonly == 1:
    skipRecalc = True

matchSource = False
matchString = "_canonical"
if args.match!=0:
    matchSource = True
    matchString = "_matched"
#datasetAll= pd.read_csv("HGE16-MaterialChemicalCoefficientDescriptors_R0E0_may26.csv")

#Prepare the set of surfaces and chemicals
targetMolecules = pd.read_csv("Datasets/ChemicalPotentialCoefficients-oct10.csv")
targetSurfaces = pd.read_csv("Datasets/SurfacePotentialCoefficients-oct12.csv")
r0TargetVal = 0.2
kbTVal = 2.48
energyTarget = 100

materialResolutionDict = {}
surfaceDefFile = open("Structures/SurfaceDefinitions.csv","r")
for line in surfaceDefFile:
    if line[0] == "#":
        continue
    lineTerms = line.split(",")
    materialResolutionDict[ lineTerms[0] ] = float(lineTerms[4])
surfaceDefFile.close()


#targetMolecules["CR0Dist"] = np.sqrt( (targetMolecules["ChemCProbeR0"] - r0TargetVal  )**2 )
#targetMolecules.sort_values( by=["CR0Dist"] ,ascending=True, inplace=True)
#targetMolecules.drop_duplicates( subset=['ChemID'] , inplace=True,keep='first')

#set the target r0 value to the point at which the methane potential is closest to the target energy
# np.array( [np.sqrt(2*i - 1) for i in range(1,1+numCoeffs) ] )
#targetSurfaces["EMethaneAtr0"] = 0
#for i in range(1,11):
#    targetSurfaces["EMethaneAtr0"] += targetSurfaces["SurfMethaneProbeC"+str(i)] * np.sqrt(2 * i - 1)
#targetSurfaces["EMethaneAtr0"] = targetSurfaces["EMethaneAtr0"] /np.sqrt(  targetSurfaces["SurfMethaneProbeR0"])
#print(targetSurfaces["EMethaneAtr0"].head(20))
#targetSurfaces = targetSurfaces.drop( targetSurfaces[  targetSurfaces["EMethaneAtr0"] > energyTarget    ].index   )
#print(targetSurfaces)
#targetSurfaces["EMethaneDistance"] =  np.sqrt( (targetSurfaces["EMethaneAtr0"] - energyTarget)**2 )

#"ChemMethaneProbeEAtR0"
#targetSurfaces["SurfR0Dist"] =  np.sqrt( (targetSurfaces["SurfMethaneProbeR0"] - r0TargetVal)**2 )
selectByEnergy = False

#Select the lowest value of r0 for each surface with an energy less than the target. 
if selectByEnergy == True:
    targetSurfaces["SurfMethaneProbeEAtR0"] - energyTarget
    targetSurfaces=targetSurfaces.drop(targetSurfaces[ targetSurfaces["SurfMethaneProbeEAtR0"] > energyTarget   ].index   )
    targetSurfaces.sort_values( by=["SurfMethaneProbeR0"] ,ascending=True, inplace=True)
    targetSurfaces.drop_duplicates( subset=['SurfID'] , inplace=True,keep='first')
    targetSurfaces["targetR0"] = targetSurfaces["SurfMethaneProbeR0"]
    print(targetSurfaces["SurfMethaneProbeEAtR0"])
else:
    targetSurfaces["targetR0"] = r0TargetVal 
    targetSurfaces["surfaceR0Dist"] = np.sqrt( (targetSurfaces["SurfMethaneProbeR0"]-targetSurfaces["targetR0"])**2)
    targetSurfaces.sort_values( by=["surfaceR0Dist"] ,ascending=True, inplace=True)
    targetSurfaces.drop_duplicates( subset=['SurfID'] , inplace=True,keep='first')
    targetSurfaces["resolution"] = targetSurfaces['SurfID'].replace(materialResolutionDict)
'''
#Generate a dataframe of surfaces at their "natural" R0 values
targetSurfaces["SR0Dist"] = np.sqrt( (targetSurfaces["SurfCProbeR0"] - targetSurfaces["targetR0"] )**2 )
targetSurfaces.sort_values( by=["SR0Dist"] ,ascending=True, inplace=True)
targetSurfaces.drop_duplicates( subset=['SurfID'] , inplace=True,keep='first')
'''
print(targetSurfaces)


uniqueMaterials = targetSurfaces['SurfID'].unique().tolist()

#minNeededR0 = np.amin( targetSurfaces["SR0Dist"].to_numpy() ) + r0TargetVal
#maxNeededR0 = np.amax( targetSurfaces["SR0Dist"].to_numpy() ) +r0TargetVal
print(uniqueMaterials)
datasubsets = []
for uniqueMaterial in uniqueMaterials:
    print(uniqueMaterial)
    targetSurface = targetSurfaces[ targetSurfaces[ "SurfID"] == uniqueMaterial].head(1)
    chemR0Target =  (( targetSurface["targetR0"]).to_numpy())[0]
    moleculeWorking = targetMolecules.copy()
    moleculeWorking["CR0Dist"] = np.sqrt( (moleculeWorking["ChemCProbeR0"] - (chemR0Target )  )**2 ) 
    moleculeWorking.sort_values( by=["CR0Dist"] ,ascending=True, inplace=True)
    moleculeWorking.drop_duplicates( subset=['ChemID'],inplace=True,keep='first')
    datasubsets.append( pd.merge(targetSurface,moleculeWorking,how="cross"))

    
combinedDataset = pd.concat(datasubsets)
print(combinedDataset)
combinedDataset.to_csv("testout.csv")


#Add in default values for the parameters used to help compensate for PMF differences during training
#unless we're specifically trying to match the training data
if matchSource == False:
    combinedDataset['source'] = 1
    combinedDataset["ssdType"] = 0
    combinedDataset["SSDRefDist"] = 0
    combinedDataset["MethaneOffset"] = 0
    combinedDataset["PMFMethaneOffset"] = 0
    combinedDataset["r0"] = combinedDataset["targetR0"]
    combinedDataset["resolution"] = 0.002
else: 
    combinedDataset["PMFMethaneOffset"] = combinedDataset["MethaneOffset"]
    combinedDataset['source'] = combinedDataset['source'].clip( 0, 3)  
    combinedDataset["r0"] = combinedDataset["targetR0"] + combinedDataset["MethaneOffset"]
    combinedDataset["r0"] = combinedDataset["r0"].clip(0.1,1)


#These parameters are unused in prediction mode so can be set to arbitrary values.
#If changing these does change predictions, this is a bug and should be reported.
combinedDataset["EMin"] = -1
combinedDataset["rEMin"] = 0.3
combinedDataset["fittedE0"] = 50

def scaledMSE(scaleVal):
    def loss(y_true,y_pred):
        return tf.keras.losses.mean_squared_error( (y_true + 0.0 * y_true * scaleVal)/scaleVal, (y_pred + 0.0 * y_pred * scaleVal)/scaleVal) 
    return loss
#define a dummy function for the KL loss as this isn't actually used for prediction
def potentialKLLoss(y_true,y_pred):
    scaleVal = 1
    return tf.keras.losses.mean_squared_error( (y_true + 0.0 * y_true * scaleVal)/scaleVal, (y_pred + 0.0 * y_pred * scaleVal)/scaleVal) 

#Define the canonical IDs to get the set of PMFs for UnitedAtom input
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
 "EST-AC":"EST",
 "CYMSCA-AC":"CYM",
 "GANSCA-AC":"GAN"
}


outputVarset = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16", "A17","A18","A19","A20" ,"e0predict","Emin","e0predictfrompmf"]


targetModelsCluster = [
"PMFPredictor-oct13-clustersplit-ensemble1",
"PMFPredictor-oct13-clustersplit-ensemble2",
"PMFPredictor-oct13-clustersplit-ensemble3",
"PMFPredictor-oct13-clustersplit-ensemble4",
"PMFPredictor-oct13-clustersplit-ensemble5"
]

targetModelsBootstrap = [
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble2",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble3",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble4",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble5",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble6",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble7",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble8",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble9",
"PMFPredictor-oct13-simplesplit-bootstrapped-ensemble10"
]

if args.bootstrap == 0:
    useBootstrap = False
else:
    useBootstrap = True


if useBootstrap == True:
    targetModels = targetModelsBootstrap
    targetModelAverages = targetModelsBootstrap
else:
    targetModels = targetModelsCluster
    targetModelAverages = ["PMFPredictor-oct13-clustersplit-ensemble1","PMFPredictor-oct13-clustersplit-ensemble2","PMFPredictor-oct13-clustersplit-ensemble5"]

generatedPMFs = []
for index,row in combinedDataset.iterrows():
    materialName = row["SurfID"]
    chemName = row["ChemID"]
    r0Actual = float(row["r0"])
    generatedPMFs.append( [materialName,chemName, r0Actual])


#generatedPMFs = (combinedDataset["SurfID"] + "_" + combinedDataset["ChemID"]).tolist()
resString = ""
if skipRecalc  == False:
    for targetModel in targetModels:
        modelString = targetModel+matchString+resString
        for materialName in uniqueMaterials:
            os.makedirs( "predicted_pmfs/"+modelString+"/"+materialName,exist_ok=True)
        print("Beginning predictions for model "+targetModel, flush=True)
        varsetFile = open("models/"+targetModel+"/varset.txt","r")
        aaVarSet = varsetFile.readline().strip().split(",")
        varsetFile.close()
        loadedModel = tf.keras.models.load_model("models/"+targetModel+"/checkpoints/checkpoint-train"    , custom_objects={ 'loss': scaledMSE(1) ,'aaVarSet':aaVarSet, 'potentialKLLoss':potentialKLLoss },compile=False)
        print("Model loaded", flush=True)
        modelPredictOut = loadedModel.predict(  [ combinedDataset[aaVarSet]   ]   )
        predictionSetSingle =  ( np.array( modelPredictOut[0] )[:,:,0] ).T
        for i in range(len(outputVarset)):
            combinedDataset[ outputVarset[i]+"_predicted" ] =predictionSetSingle[:,i].flatten()
        combinedDataset.to_csv("models/"+targetModel+"/predictedBuild"+matchString+resString+".csv")
        for index,row in combinedDataset.iterrows():
            materialName = row["SurfID"]
            chemName = row["ChemID"]
            r0Actual = float(row["r0"])
            try:
                rRange = np.arange( r0Actual, 1.5, 0.001)
            except:
                print(materialName, chemName, r0Actual)
                continue
            pmf = np.zeros_like(rRange)
            for i in range(1,20):
                pmf = pmf + row["A"+str(i)+"_predicted"] * HGEFuncs.HGEFunc(rRange, r0Actual, i)
            finalPMF = np.stack((rRange,pmf),axis=-1)
            finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
            np.savetxt("predicted_pmfs/"+modelString+"/"+materialName+"/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",", header=materialName+"_"+chemName+"\nh[nm],U(h)[kJ/mol]")
            #generatedPMFs.append( [materialName,chemName, r0Actual])
        readmeFile = open("predicted_pmfs/"+modelString+"/README","w")
        readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using model variant " + modelString)
        readmeFile.close()
        print("Completed PMF generation for this model", flush=True)
        del loadedModel
        tf.keras.backend.clear_session()

#generatedPMFs = list(set(generatedPMFs))

combinedDataset.drop(combinedDataset.index, inplace=True)

               
#Next build composites and plot figures
print("Averaging PMFs",flush=True)
outputFig = plt.figure()
modelString = targetModels[0]+matchString+resString
for generatedPMF in generatedPMFs:
    #generatedPMF = generatedPMFName.split("_")
    materialName = generatedPMF[0]
    chemName = generatedPMF[1]
    r0Actual = generatedPMF[2]
    plt.clf()
    #materialName = row["SurfID"]
    os.makedirs("predicted_avg_pmfs/"+modelString+"/"+materialName+"_simple",exist_ok=True)
    #os.makedirs("predicted_avg_pmfs/"+modelString+"/"+materialName+"_log",exist_ok=True)
    os.makedirs("predicted_avg_pmfs/"+modelString+"/"+materialName+"_figs",exist_ok=True)
    os.makedirs("predicted_avg_pmfs/"+modelString+"/UA/"+materialName+"-pmfp",exist_ok=True)
    #chemName = row["ChemID"]
    #r0Actual = float(row["r0"])
    #print(r0Actual)
    rRange = np.arange( r0Actual, 1.5, 0.001)
    simpleAvgPMF = np.zeros_like(rRange)
    logAvgPMF = np.zeros_like(rRange)
    for targetModel in targetModelAverages:
        pmfCandidate = np.genfromtxt("predicted_pmfs/"+targetModel+matchString+resString+"/"+materialName+"/"+chemName+".dat", delimiter=",")
        plt.plot(pmfCandidate[:,0],pmfCandidate[:,1],'k:',alpha=0.2)
        simpleAvgPMF = simpleAvgPMF + pmfCandidate[:,1]
        pmfProbNorm = 1.0/np.trapz(  np.exp( -pmfCandidate[:,1] / kbTVal )  , pmfCandidate[:,0]        ) 
        logAvgPMF += pmfProbNorm * np.exp(-pmfCandidate[:,1]/kbTVal)
    simpleAvgPMF = simpleAvgPMF / len(targetModelAverages)
    logAvgPMF = - kbTVal * np.log(  logAvgPMF/len(targetModelAverages) )
    finalPMF = np.stack((rRange,simpleAvgPMF),axis=-1)
    finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
    np.savetxt("predicted_avg_pmfs/"+modelString+"/"+materialName+"_simple/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",", header=materialName+"_"+chemName+"\nh[nm],U(h)[kJ/mol]")
    if chemName in UnitedAtomNames:
        np.savetxt("predicted_avg_pmfs/"+modelString+"/UA/"+materialName+"-pmfp/"+UnitedAtomNames[chemName]+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",", header=materialName+"_"+UnitedAtomNames[chemName]+"\nh[nm],U(h)[kJ/mol]")
    finalPMFLog = np.stack((rRange,logAvgPMF),axis=-1)
    finalPMFLog[:,1] = finalPMFLog[:,1] - finalPMFLog[-1,1]
    #np.savetxt("predicted_avg_pmfs/"+modelString+"/"+materialName+"_log/"+chemName+".dat" ,finalPMFLog,fmt='%.18f' ,delimiter=",")

    plt.plot(finalPMF[:,0],finalPMF[:,1],'b-')
    #plt.plot(finalPMFLog[:,0],finalPMFLog[:,1],'r-')
    knownPMF = HGEFuncs.loadPMF("AllPMFs/"+materialName+"_"+chemName+".dat")
    if len(knownPMF) > 2:
        plt.plot( knownPMF[:,0], knownPMF[:,1], 'g-')
    plt.xlabel("h [nm]")
    plt.ylabel(r"U(h) $[\mathrm{kJ}\cdot\mathrm{mol}^{-1}]$")
    plt.tight_layout()
    plt.savefig("predicted_avg_pmfs/"+modelString+"/"+materialName+"_figs/"+chemName+".png" )
    print("Done "+materialName+"_"+chemName+"\n")
plt.close()


readmeFile = open("predicted_avg_pmfs/"+modelString+"/README","w")
readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using models: \n")
for modelName in targetModels:
    readmeFile.write(modelName+"\n")
if matchSource == True:
    readmeFile.write("PMFs match sources given in SurfaceDefinitions at time of generation.\n")
else:
    readmeFile.write("Source matching has been applied to source =  SU-ions, SSD class 1, surface defined by U_C(h=0.2) = 35 convention. Recommended LJ cutoff 1.0 nm.")
readmeFile.write("\nFigures: Blue gives the simple average of contributing PMFs (black lines). Green, where existent, is the reference metadynamics PMF.\n")
readmeFile.close()

readmeFile = open("predicted_avg_pmfs/"+modelString+"/UA/README","w")
readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using models: \n")
for modelName in targetModels:
    readmeFile.write(modelName+"\n")
if matchSource == True:
    readmeFile.write("PMFs match sources given in SurfaceDefinitions at time of generation.\n")
else:
    readmeFile.write("Source matching has been applied to source =  SU-ions, SSD class 1, surface defined by U_C(h=0.2) = 35 convention. Recommended LJ cutoff 1.0 nm.")
readmeFile.close()
