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
args = parser.parse_args()


matchSource = False
if args.match!=0:
    matchSource = True

#datasetAll= pd.read_csv("HGE16-MaterialChemicalCoefficientDescriptors_R0E0_may26.csv")

#Prepare the set of surfaces and chemicals
targetMolecules = pd.read_csv("Datasets/ChemicalPotentialCoefficients-sep15.csv")
targetSurfaces = pd.read_csv("Datasets/SurfacePotentialCoefficients-sep27.csv")
r0TargetVal = 0.2
kbTVal = 2.48
energyTarget = 35

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


#Select the lowest value of r0 for each surface with an energy less than the target. 
targetSurfaces["SurfMethaneProbeEAtR0"] - energyTarget
targetSurfaces=targetSurfaces.drop(targetSurfaces[ targetSurfaces["SurfMethaneProbeEAtR0"] > energyTarget   ].index   )
targetSurfaces.sort_values( by=["SurfMethaneProbeR0"] ,ascending=True, inplace=True)
targetSurfaces.drop_duplicates( subset=['SurfID'] , inplace=True,keep='first')
targetSurfaces["targetR0"] = targetSurfaces["SurfMethaneProbeR0"]
print(targetSurfaces["SurfMethaneProbeEAtR0"])
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

'''
combinedDataset = pd.merge(targetMolecules,targetSurfaces,how="cross")
combinedDataset["CR0Dist"] = np.sqrt( (targetMolecules["ChemCProbeR0"] - (r0TargetVal + targetSurfaces["SurfAlignDist"])  )**2 ) 
combinedDataset["PMFName"] = combinedDataset['SurfID']+"_"+combinedDataset['ChemID']
combinedDataset.sort_values( by=["CR0Dist"] ,ascending=True, inplace=True)
combinedDataset.drop_duplicates( subset=['PMFName'], inplace=True,keep='first')
'''
#Add in default values for the extra 
if matchSource == False:
    combinedDataset['source'] = 1
    combinedDataset["ssdType"] = 0
    combinedDataset["MethaneOffset"] = 0
    combinedDataset["PMFMethaneOffset"] = 0
combinedDataset["r0"] = combinedDataset["targetR0"] #+ combinedDataset["SurfAlignDist"]
combinedDataset["resolution"] = 0.005
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



outputVarset = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16", "A17","A18","A19","A20" ,"e0predict","Emin","e0predictfrompmf"]
targetModels = ["PMFPredictor-oct03-simplesplit-ensemble1"]#,"PMFPredictor-sep15-simplesplit-bootstrapped-ensemble2","PMFPredictor-sep15-simplesplit-bootstrapped-ensemble3","PMFPredictor-sep15-simplesplit-bootstrapped-ensemble4","PMFPredictor-sep15-simplesplit-bootstrapped-ensemble5"]
for targetModel in targetModels:
    for materialName in uniqueMaterials:
        os.makedirs( "predicted_pmfs/"+targetModel+"/"+materialName,exist_ok=True)
    print("Beginning predictions for model "+targetModel, flush=True)
    varsetFile = open("models/"+targetModel+"/varset.txt","r")
    aaVarSet = varsetFile.readline().strip().split(",")
    varsetFile.close()
    loadedModel = tf.keras.models.load_model("models/"+targetModel+"/checkpoints/checkpoint-train"    , custom_objects={ 'loss': scaledMSE(1) ,'aaVarSet':aaVarSet, 'potentialKLLoss':potentialKLLoss },compile=False)
    modelPredictOut = loadedModel.predict(  [ combinedDataset[aaVarSet]   ]   )
    predictionSetSingle =  ( np.array( modelPredictOut[0] )[:,:,0] ).T
    for i in range(len(outputVarset)):
        combinedDataset[ outputVarset[i]+"_predicted" ] =predictionSetSingle[:,i].flatten()
    combinedDataset.to_csv("models/"+targetModel+"/predictedBuild.csv")
    for index,row in combinedDataset.iterrows():
        materialName = row["SurfID"]
        chemName = row["ChemID"]
        r0Actual = row["r0"]
        rRange = np.arange( r0Actual, 1.5, 0.001)
        pmf = np.zeros_like(rRange)
        for i in range(1,20):
            pmf = pmf + row["A"+str(i)+"_predicted"] * HGEFuncs.HGEFunc(rRange, r0Actual, i)
        finalPMF = np.stack((rRange,pmf),axis=-1)
        finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
        np.savetxt("predicted_pmfs/"+targetModel+"/"+materialName+"/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    readmeFile = open("predicted_pmfs/"+targetModel+"/README","w")
    readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using model variant " + targetModel)
    readmeFile.close()
        



               
#Next build composites and plot figures
print("Averaging PMFs",flush=True)
outputFig = plt.figure()
for index,row in combinedDataset.iterrows():
    plt.clf()
    materialName = row["SurfID"]
    os.makedirs("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_simple",exist_ok=True)
    os.makedirs("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_log",exist_ok=True)
    os.makedirs("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_figs",exist_ok=True)
    chemName = row["ChemID"]
    r0Actual = row["r0"]
    rRange = np.arange( r0Actual, 1.5, 0.001)
    simpleAvgPMF = np.zeros_like(rRange)
    logAvgPMF = np.zeros_like(rRange)
    for targetModel in targetModels:
        pmfCandidate = np.genfromtxt("predicted_pmfs/"+targetModel+"/"+materialName+"/"+chemName+".dat", delimiter=",")
        plt.plot(pmfCandidate[:,0],pmfCandidate[:,1],'k:',alpha=0.2)
        simpleAvgPMF = simpleAvgPMF + pmfCandidate[:,1]
        pmfProbNorm = 1.0/np.trapz(  np.exp( -pmfCandidate[:,1] / kbTVal )  , pmfCandidate[:,0]        ) 
        logAvgPMF += pmfProbNorm * np.exp(-pmfCandidate[:,1]/kbTVal)
    simpleAvgPMF = simpleAvgPMF / len(targetModels)
    logAvgPMF = - kbTVal * np.log(  logAvgPMF/len(targetModels) )
    finalPMF = np.stack((rRange,simpleAvgPMF),axis=-1)
    finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
    np.savetxt("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_simple/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    finalPMFLog = np.stack((rRange,logAvgPMF),axis=-1)
    finalPMFLog[:,1] = finalPMFLog[:,1] - finalPMFLog[-1,1]
    np.savetxt("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_log/"+chemName+".dat" ,finalPMFLog,fmt='%.18f' ,delimiter=",")

    plt.plot(finalPMF[:,0],finalPMF[:,1],'b-')
    plt.plot(finalPMFLog[:,0],finalPMFLog[:,1],'r-')
    knownPMF = HGEFuncs.loadPMF("AllPMFs/"+materialName+"_"+chemName+".dat")
    if len(knownPMF) > 2:
        plt.plot( knownPMF[:,0], knownPMF[:,1], 'g-')
    plt.xlabel("r [nm]")
    plt.ylabel("U(r) [kJ/mol]")
    plt.tight_layout()
    plt.savefig("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_figs/"+chemName+".png" )
