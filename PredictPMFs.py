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


targetModel = "june14-roughonly-predictr0"



datasetAll= pd.read_csv("Datasets/TrainingData.csv")
loadedModel = tf.keras.models.load_model("checkpoints/"+targetModel)
os.makedirs("predicted_pmfs/"+targetModel,exist_ok=True)

UnitedAtomNames = {
"ALASCA":"ALA",
"ARGSCA":"ARG",
"LYSSCA":"LYS",
"HIDSCA":"HID",
"HIESCA":"HIE",
"ASPSCA":"ASP",
"GLUSCA":"GLU",
"SERSCA":"SER",
"THRSCA":"THR",
"ASNSCA":"ASN",
"GLNSCA":"GLN",
"CYSSCA":"CYS",
 "GLY":"GLY",
 "PRO":"PRO",
 "VALSCA":"VAL",
 "ILESCA":"ILE",
 "LEUSCA":"LEU",
 "METSCA":"MET",
 "PHESCA":"PHE",
 "TRYSCA":"TYR",
 "TRPSCA":"TRP",
 "HIPSCA":"HIP",
 "ETA":"ETA",
 "PHO":"PHO",
 "CHL":"CHL",
 "DGL":"DGL",
 "EST":"EST"
}



#Write the training set predictions out

varsetFile = open(targetModel+"_varset.txt","r")
aaVarSet = varsetFile.readline().strip().split(",")
varsetFile.close()

predictionSetAll =  ( np.array( loadedModel.predict([    datasetAll[aaVarSet] ]))[:,:,0] ).T

outputVarset = ["r0","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict"]
for i in range(len(outputVarset)):
    datasetAll[ outputVarset[i]+"_regpredict" ] =predictionSetAll[:,i].flatten()

datasetAll.to_csv("predicted_pmfs/"+targetModel+"/checkpointpredictallCoeffs.csv")

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

loadedModel = tf.keras.models.load_model("checkpoints/"+targetModel)

inputVariableFile = open(targetModel+"_varset.txt")
inputVarSet = inputVariableFile.read().strip().split(",")
inputVariableFile.close()



combinedDataset = pd.merge(targetMolecules,targetSurfaces,how="cross")
combinedDataset["fittedE0"] = targetE0Val
#combinedDataset["rmin"] = combinedDataset["fljr0"] + 0.01
#combinedDataset["rmin"].clip(lower = 0.2, inplace=True)
#print(combinedDataset)


aaVarSet =  inputVarSet
predictionSetSingle =  ( np.array( loadedModel.predict([    combinedDataset[aaVarSet] ]  ,verbose=True ))[:,:,0] ).T

outputVarset = ["r0","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict" ]
for i in range(len(outputVarset)):
    combinedDataset[ outputVarset[i]+"_predicted" ] =predictionSetSingle[:,i].flatten()

combinedDataset.to_csv("predicted_pmfs/"+targetModel+"/predicted_coeffs_allpairs.csv")

offsetData = np.genfromtxt("Datasets/SurfaceOffsetData.csv",skip_header=1,delimiter=",", dtype=str)
offsetDict = {}
for materialOff in offsetData:
    offsetDict[ materialOff[0] ] = float( materialOff[3] )
 
 
for index,row in combinedDataset.iterrows():
    materialName = row["SurfID"]
    chemName = row["ChemID"]
    r0Target = row["r0_predicted"]
    rRange = np.arange( r0Target, 1.5, 0.001)
    pmf = np.zeros_like(rRange)
    for i in range(1,16):
        pmf = pmf + row["A"+str(i)+"_predicted"] * HGEFunc(rRange, r0Target, i)
    finalPMF = np.stack((rRange,pmf),axis=-1)
    finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
    finalPMF[:,0] = finalPMF[:,0] - offsetDict[ materialName]
    np.savetxt("predicted_pmfs/"+targetModel+"/allchemicals/"+materialName+"/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    if chemName in UnitedAtomNames:
        np.savetxt("predicted_pmfs/"+targetModel+"/UA-surface-predicted/"+materialName+"/"+UnitedAtomNames[chemName]+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    
    
readmeFile = open("predicted_pmfs/"+targetModel+"/README","w")
readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using model variant " + targetModel)
readmeFile.close()
        
