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

def HGEFunc(r, r0, n):
    return (-1)**(1+n) * np.sqrt( 2*n - 1) * np.sqrt(r0)/r * scspec.hyp2f1(1-n,n,1,r0/r)

targetModel = "potentialcoeffs-may31-preprocess256-256-64-predictE0"

#datasetAll= pd.read_csv("HGE16-MaterialChemicalCoefficientDescriptors_R0E0_may26.csv")

targetMolecules = pd.read_csv("prediction_templates/ChemicalPotentialCoeffs.csv")
targetSurfaces = pd.read_csv("prediction_templates/material_potentials_coeffs_freeenergy_water.csv")


uniqueMaterials = targetSurfaces['Material'].unique().tolist()
for materialName in uniqueMaterials:
    os.makedirs( "predicted_pmfs/"+targetModel+"/"+materialName,exist_ok=True)


loadedModel = tf.keras.models.load_model("checkpoints/"+targetModel)

inputVariableFile = open(targetModel+"_varset.txt")
inputVarSet = inputVariableFile.read().strip().split(",")
inputVariableFile.close()


#print(targetMolecules)
#print(targetSurfaces)

combinedDataset = pd.merge(targetMolecules,targetSurfaces,how="cross")
combinedDataset["rmin"] = combinedDataset["fljr0"] + 0.01
combinedDataset["rmin"].clip(lower = 0.2, inplace=True)
print(combinedDataset)


aaVarSet =  inputVarSet
predictionSetSingle =  ( np.array( loadedModel.predict([    combinedDataset[aaVarSet] ]  ,verbose=True ))[:,:,0] ).T

outputVarset = ["rmin","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict" ]
for i in range(len(outputVarset)):
    combinedDataset[ outputVarset[i]+"_predicted" ] =predictionSetSingle[:,i].flatten()

combinedDataset.to_csv(targetModel+"_predictedBuild.csv")

 
for index,row in combinedDataset.iterrows():
    r0Target = row["rmin"]
    materialName = row["Material"]
    chemName = row["Chemical"]
    rRange = np.arange( r0Target, 1.5, 0.001)
    pmf = np.zeros_like(rRange)
    for i in range(1,16):
        pmf = pmf + row["A"+str(i)+"_predicted"] * HGEFunc(rRange, r0Target, i)
    finalPMF = np.stack((rRange,pmf),axis=-1)
    finalPMF[:,1] = finalPMF[:,1] - finalPMF[-1,1]
    np.savetxt("predicted_pmfs/"+targetModel+"/"+materialName+"/"+chemName+".dat" ,finalPMF,fmt='%.18f' ,delimiter=",")
    
    
readmeFile = open("predicted_pmfs/"+targetModel+"/README","w")
readmeFile.write("PMFs contained here were generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " using model variant " + targetModel)
readmeFile.close()
        
