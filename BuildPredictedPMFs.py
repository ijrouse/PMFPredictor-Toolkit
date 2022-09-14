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



#datasetAll= pd.read_csv("HGE16-MaterialChemicalCoefficientDescriptors_R0E0_may26.csv")

#Prepare the set of surfaces and chemicals
targetMolecules = pd.read_csv("Datasets/ChemicalPotentialCoefficients-aug26.csv")
targetSurfaces = pd.read_csv("Datasets/SurfacePotentialCoefficients-sep07.csv")
r0TargetVal = 0.2
kbTVal = 1
matchSource = True

targetMolecules["CR0Dist"] = np.sqrt( (targetMolecules["ChemCProbeR0"] - r0TargetVal  )**2 )
targetMolecules.sort_values( by=["CR0Dist"] ,ascending=True, inplace=True)
targetMolecules.drop_duplicates( subset=['ChemID'] , inplace=True,keep='first')
targetSurfaces["SR0Dist"] = np.sqrt( (targetSurfaces["SurfCProbeR0"] - r0TargetVal  )**2 )
targetSurfaces.sort_values( by=["SR0Dist"] ,ascending=True, inplace=True)
targetSurfaces.drop_duplicates( subset=['SurfID'] , inplace=True,keep='first')


uniqueMaterials = targetSurfaces['SurfID'].unique().tolist()


combinedDataset = pd.merge(targetMolecules,targetSurfaces,how="cross")
#Add in default values for the extra 
if matchSource == False:
    combinedDataset['source'] = 1
combinedDataset["r0"] = r0TargetVal
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


rRange = np.arange( r0TargetVal, 1.5, 0.001)
outputVarset = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16", "A17","A18","A19","A20" ,"e0predict","Emin","e0predictfrompmf"]
targetModels = ["PMFPredictor-sep09-simplesplit-bootstrapped-ensemble1","PMFPredictor-sep09-simplesplit-bootstrapped-ensemble2","PMFPredictor-sep09-simplesplit-bootstrapped-ensemble3"]
for targetModel in targetModels:
    for materialName in uniqueMaterials:
        os.makedirs( "predicted_pmfs/"+targetModel+"/"+materialName,exist_ok=True)
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

        pmf = np.zeros_like(rRange)
        for i in range(1,20):
            pmf = pmf + row["A"+str(i)+"_predicted"] * HGEFuncs.HGEFunc(rRange, r0TargetVal, i)
        finalPMF = np.stack((rRange,pmf),axis=-1)
        finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
        np.savetxt("predicted_pmfs/"+targetModel+"/"+materialName+"/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    readmeFile = open("predicted_pmfs/"+targetModel+"/README","w")
    readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using model variant " + targetModel)
    readmeFile.close()
        


def loadPMF(target):
    PMFData = []
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
        return PMFData
    except: 
        return [-1]
               
#Next build composites and plot figures

outputFig = plt.figure()
for index,row in combinedDataset.iterrows():
    plt.clf()
    materialName = row["SurfID"]
    os.makedirs("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_simple",exist_ok=True)
    os.makedirs("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_log",exist_ok=True)
    os.makedirs("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_figs",exist_ok=True)
    chemName = row["ChemID"]
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
    knownPMF = loadPMF("AllPMFs/"+materialName+"_"+chemName+".dat")
    if len(knownPMF) > 2:
        plt.plot( knownPMF[:,0], knownPMF[:,1], 'g-')
    plt.xlabel("r [nm]")
    plt.ylabel("U(r) [kJ/mol]")
    plt.tight_layout()
    plt.savefig("predicted_avg_pmfs/"+targetModels[0]+"/"+materialName+"_figs/"+chemName+".png" )