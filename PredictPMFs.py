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
from tensorflow.strings import bytes_split
from tensorflow.keras.utils import plot_model
import scipy.special as scspec
import datetime



targetModel ="pmfpredict-july12-potential2x32-dense3x32"
#targetModel = "june14-roughonly-predictr0"


def scaledMSE(scaleVal):
    def loss(y_true,y_pred):
        return tf.keras.losses.mean_squared_error( y_true/scaleVal, y_pred/scaleVal) 
    return loss
datasetAll= pd.read_csv("Datasets/TrainingData.csv")
loadedModel = tf.keras.models.load_model("models/"+targetModel+"/checkpoints/checkpoint-train"    , custom_objects={ 'loss': scaledMSE(1) })
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
outputVarset = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict"]
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

targetMolecules = pd.read_csv("Datasets/ChemicalPotentialCoefficients.csv")
targetSurfaces = pd.read_csv("Datasets/SurfacePotentialCoefficients.csv")
targetE0Val = 10

uniqueMaterials = targetSurfaces['SurfID'].unique().tolist()
for materialName in uniqueMaterials:
    os.makedirs( "predicted_pmfs/"+targetModel+"/allchemicals/"+materialName,exist_ok=True)
    os.makedirs( "predicted_pmfs/"+targetModel+"/UA-surface-predicted/"+materialName,exist_ok=True)

#loadedModel = tf.keras.models.load_model("checkpoints/"+targetModel)

 



combinedDataset = pd.merge(targetMolecules,targetSurfaces,how="cross")
combinedDataset["fittedE0"] = targetE0Val

#combinedDataset["r0original"] = combinedDataset["r0"]
#combinedDataset["r0"] = 
#combinedDataset["rmin"] = combinedDataset["fljr0"] + 0.01
#combinedDataset["rmin"].clip(lower = 0.2, inplace=True)
#print(combinedDataset)

def logSumMax( x):
    return np.log(  np.sum(np.exp(x)))
#combinedDataset["r0"] = np.log(  np.exp(combinedDataset["ChemLJR0"] )+ np.exp(combinedDataset["SurfLJR0"]))


def recurrentPredict(model, dataset, initialR0 = 0.18, targetE0 = 25):
    workingDataset = dataset.copy()
    currentr0 = initialR0
    workingDataset["r0"] = currentr0
    workingDataset["lastE0"] =targetE0*2
    #nonconvergedSet = workingDataset[   workingDataset["lastE0"] > targetE0]
    nonconvergedMask = workingDataset["lastE0"] > targetE0
    print(len(workingDataset[nonconvergedMask]))
    while len( workingDataset[nonconvergedMask] ) > 0 and currentr0 < 1:
        print(  currentr0)
        workingDataset.loc[nonconvergedMask,"r0"] = currentr0
        workingDataset.loc[nonconvergedMask, "lastE0" ] = 0
        predictionSet = ( np.array( model.predict([ workingDataset.loc[ nonconvergedMask,  aaVarSet] ]))[:,:,0] ).T
        for i in range(len(outputVarset)):
            workingDataset.loc[nonconvergedMask,  outputVarset[i]+"_regpredict" ] =predictionSet[:,i].flatten()  
            if i<16:
                workingDataset.loc[ nonconvergedMask,   "lastE0" ] = workingDataset.loc[ nonconvergedMask,   "lastE0" ] + np.sqrt(2*(i+1) - 1) * (predictionSet[:,i].flatten() ) /np.sqrt(currentr0)
        print( workingDataset.loc[nonconvergedMask,   "lastE0"] , predictionSet[:,-1])
        #workingDataset.update( nonconvergedSet )        
        nonconvergedMask = workingDataset["lastE0"] > targetE0
        currentr0 = currentr0+0.01
    return workingDataset

combinedDataset = recurrentPredict( loadedModel, combinedDataset)

'''

predictionSetSingle =  ( np.array( loadedModel.predict([    combinedDataset[aaVarSet] ]  ,verbose=True ))[:,:,0] ).T

outputVarset = ["r0","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict" ]
for i in range(len(outputVarset)):
    combinedDataset[ outputVarset[i]+"_predicted" ] =predictionSetSingle[:,i].flatten()
'''



combinedDataset.to_csv("predicted_pmfs/"+targetModel+"/predicted_coeffs_allpairs.csv")

offsetData = np.genfromtxt("Datasets/SurfaceOffsetData.csv",skip_header=1,delimiter=",", dtype=str)
offsetDict = {}
for materialOff in offsetData:
    offsetDict[ materialOff[0] ] = float( materialOff[3] )
 
 
for index,row in combinedDataset.iterrows():
    materialName = row["SurfID"]
    chemName = row["ChemID"]
    r0Target = row["r0"]
    rRange = np.arange( r0Target, 1.5, 0.001)
    pmf = np.zeros_like(rRange)
    for i in range(1,16):
        pmf = pmf + row["A"+str(i)+"_regpredict"] * HGEFunc(rRange, r0Target, i)
    finalPMF = np.stack((rRange,pmf),axis=-1)
    finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
    finalPMF[:,0] = finalPMF[:,0] - offsetDict[ materialName]
    np.savetxt("predicted_pmfs/"+targetModel+"/allchemicals/"+materialName+"/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    if chemName in UnitedAtomNames:
        np.savetxt("predicted_pmfs/"+targetModel+"/UA-surface-predicted/"+materialName+"/"+UnitedAtomNames[chemName]+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    
    
readmeFile = open("predicted_pmfs/"+targetModel+"/README","w")
readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using model variant " + targetModel)
readmeFile.close()
        
