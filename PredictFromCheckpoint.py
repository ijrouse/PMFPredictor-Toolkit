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


targetModel = "oddpotentialcoeffs11-june09-32c-predr0testing-predictr0"
datasetAll= pd.read_csv("HGE16-MaterialChemicalCoefficientDescriptors_R0E0_june06.csv")
loadedModel = tf.keras.models.load_model("checkpoints/"+targetModel)

varsetFile = open(targetModel+"_varset.txt","r")
aaVarSet = varsetFile.readline().strip().split(",")
varsetFile.close()

predictionSetAll =  ( np.array( loadedModel.predict([    datasetAll[aaVarSet] ]))[:,:,0] ).T

outputVarset = ["rmin","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict" ,"roughA1","roughA2","roughA3","roughA4","roughA5","roughA6","roughA7","roughA8","roughA9","roughA10","roughA11","roughA12","roughA13","roughA14","roughA15","roughA16"]
for i in range(len(outputVarset)):
    datasetAll[ outputVarset[i]+"_regpredict" ] =predictionSetAll[:,i].flatten()

datasetAll.to_csv(targetModel+"_checkpointpredictall.csv")


targetE0=10
datasetSingle = pd.read_csv(targetModel+"_uniquepredictions.csv")
datasetSingle["fittedE0"]=targetE0
predictionSetSingle =  ( np.array( loadedModel.predict([    datasetSingle[aaVarSet] ]))[:,:,0] ).T
for i in range(len(outputVarset)):
    datasetSingle[ outputVarset[i]+"_regpredict" ] =predictionSetSingle[:,i].flatten()

datasetSingle.to_csv(targetModel+"_checkpointpredict_targetinput_"+str(targetE0)+".csv")

'''



uniquedataset = datasetAll.copy()
uniquedataset.drop_duplicates( subset=['Material','Chemical'] , inplace=True,keep='last')


#uniquedataset['ChemValidation'] = 0
#uniquedataset.loc[  uniquedataset['Chemical'].isin(validationAA)  ,'ChemValidation'] = 1
#uniquedataset['MaterialValidation'] = 0
#uniquedataset.loc[  uniquedataset['Material'].isin(validationMaterials) , 'MaterialValidation' ] = 1
uniquedataset["fittedE0"]=7.5

chemicalVars =  ["MW", "TotalNegative","TotalPositive", "nHeavyAtom","Diameter", "GeomDiameter", "MOMI-X", "MOMI-Y", "nRing", "ATSC1c","MIC1",  "ATSC1i", "TIC1", "GGI1", "n3ARing","nBase", "nAcid", "IC1", "LabuteASA"]
pmfVars = ["source","fittedE0", "electroAtLJZeroCrossing", "LJ(LJMin)", "dLJMin", "electroAtLJMin", "ljzeroCrossing", "shape", "ljr0"]



ljCoeffs = ["LJC1","LJC2","LJC3","LJC4"   ,  "LJC5","LJC6", "LJC7", "LJC8", "LJC9", "LJC10",  "LJC11", "LJC12"]


aaVarSet =  chemicalVars + pmfVars + ljCoeffs

outputVarset = ["rmin","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"e0predict" ]

loadedModel = tf.keras.models.load_model("saved_models/potentialcoeffs-may06-sparse-fittedE0-512")
predictionSetSingle =  ( np.array( loadedModel.predict([ uniquedataset["Material"], uniquedataset["Chemical"],  uniquedataset[aaVarSet] ]))[:,:,0] ).T


for i in range(len(outputVarset)):
    uniquedataset[ outputVarset[i]+"_regpredict" ] =predictionSetSingle[:,i].flatten()
uniquedataset.to_csv("Finalpredicts_E0"+str(7.5)+".csv")

'''
