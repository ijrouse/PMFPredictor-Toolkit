import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
import os
import shutil
import sklearn.cluster as skcluster
import sklearn.preprocessing as skpreproc
import keras_nlp


@tf.keras.utils.register_keras_serializable()
def activationXLog(x):
    alphaVal = 0.25
    absx = tf.math.abs(x*alphaVal)
    return tf.math.sign(x) * tf.math.log1p( absx )/alphaVal

@tf.keras.utils.register_keras_serializable()
def activationXSqLog(x):
    absx = tf.math.square(x)
    return tf.math.sign(x) * tf.math.log1p( absx )

def activationSoftPlusSmall(x):
    return   tf.math.softplus(2*x)/2
    
    
@tf.keras.utils.register_keras_serializable()
def activationIdentity(x):
    return x


@tf.keras.utils.register_keras_serializable()
def activationDesatLog(x):
    alpha=0.25
    return tf.math.sign(x) * ( tf.exp(alpha* x*tf.math.sign(x)) - 1)/alpha


'''
    def __init__(self, flipProb,**kwargs):
        super(BoolFlip, self).__init__(**kwargs)
        self.flipProb = flipProb
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'flipProb': self.flipProb,
        })
        return config

'''

class TrainableDesat4(keras.layers.Layer):
    def __init__(self, targetMean=0.0,targetScale=1.0):
        super(TrainableDesat, self).__init__()
        self.targetMean = targetMean
        self.targetScale = targetScale
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'targetMean': self.targetMean,
            'targetScale': self.targetScale,
        })
        return config
    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1], ),
            initializer=tf.keras.initializers.Constant(0.1),
            name="desatalpha",
            trainable=True,
        )

        self.alpha2 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.Constant(self.targetScale), 
            name="desatalpha2", trainable=True,
        )

        self.b1 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.Constant(0.1), 
            name="desatb1", trainable=True,
        )
        self.b2 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.Constant(self.targetMean), 
            name="desatb2", trainable=True,
        )
    def call(self, inputs):
        shiftedInputs = self.b1 + inputs
        return  self.b2 + self.alpha2 * tf.math.sign( shiftedInputs  ) *    tf.math.expm1(self.alpha*  tf.math.abs(shiftedInputs )) 

class TrainableDesat(keras.layers.Layer):
    def __init__(self, targetMean=0.0,targetScale=1.0):
        super(TrainableDesat, self).__init__()
        self.targetMean = targetMean
        self.targetScale = targetScale
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'targetMean': self.targetMean,
            'targetScale': self.targetScale,
        })
        return config
    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1], ),
            initializer=tf.keras.initializers.Constant(0.1),
            name="desatalpha",
            trainable=True,
        )


 
        self.b1 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.Constant(0.1), 
            name="desatb1", trainable=True,
        )
        self.b2 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.Constant(self.targetMean), 
            name="desatb2", trainable=True,
        )
    def call(self, inputs):
        shiftedInputs = self.b1 + inputs
        return  self.b2  +    tf.math.sign( shiftedInputs  ) *   tf.math.divide_no_nan( tf.math.expm1(self.alpha*  tf.math.abs(shiftedInputs ))  , self.alpha)



class TrainableXLog4(keras.layers.Layer):
    def __init__(self):
        super(TrainableXLog, self).__init__()

    #previous tf.keras.initializers.Constant(2.0)
    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1], ),
            initializer=tf.keras.initializers.GlorotNormal(seed=None),
            name="xlogalpha",
            trainable=True,
        ) 

        self.alpha2 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.GlorotNormal(seed=None), 
            name="xlogalpha2", trainable=True,
        )

        self.b1 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.GlorotNormal(seed=None), 
            name="xlogb1", trainable=True,
        )

        self.b2 = self.add_weight( shape=(input_shape[-1], ), 
            initializer=tf.keras.initializers.GlorotNormal(seed=None), 
            name="xlogb2", trainable=True,
        )
    def call(self, inputs):
        shiftedInputs = self.b1 + inputs
        return  self.b2 + self.alpha2*tf.math.sign(shiftedInputs) *  tf.math.multiply_no_nan(  tf.math.log1p( tf.math.abs( tf.math.divide_no_nan( shiftedInputs,self.alpha))) ,self.alpha)
        #return  tf.math.sign(inputs) * tf.math.log1p( tf.math.abs(inputs*self.alpha))/self.alpha   



class TrainableXLog(keras.layers.Layer):
    def __init__(self):
        super(TrainableXLog, self).__init__()

    #previous tf.keras.initializers.Constant(2.0)
    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1], ),
            initializer=tf.keras.initializers.GlorotNormal(seed=None),
            name="xlogalpha",
            trainable=True,
        )


 


        self.b1 = self.add_weight( shape=(input_shape[-1], ),
            initializer=tf.keras.initializers.Constant(0.01),
            name="xlogb1", trainable=True,
        )

        self.b2 = self.add_weight( shape=(input_shape[-1], ),
            initializer=tf.keras.initializers.Constant(0.01),
            name="xlogb2", trainable=True,
        )

    def call(self, inputs):
        shiftedInputs = self.b1 + inputs
        return  self.b2 +   tf.math.sign(shiftedInputs) *   tf.math.multiply_no_nan(  tf.math.log1p( tf.math.abs( tf.math.divide_no_nan( shiftedInputs,self.alpha))) ,self.alpha)
        #return  tf.math.sign(inputs) * tf.math.log1p( tf.math.abs(inputs*self.alpha))/self.alpha


datasetAll= pd.read_csv("Datasets/TrainingData-r0matched-sep07.csv")
#datasetAll = datasetAll.head(1000)

#datasetExtra= pd.read_csv("Datasets/TrainingData_PMFn9xCombon2-r0matched-aug02-negativeE0-partial.csv")
#datasetAll=datasetAll.append(  datasetExtra )
#datasetAll = pd.read_csv("Datasets/TrainingData_PMFn2xCombon5-r0matched-july27.csv")
basedir = "models"


numDeconvResidCorrections =8
#filetag = "pmfpredict-aug16-mix1-deconv8x8-mix2-c6-pmfclusterfix-resolution-mixing-addwholepotentials-directdecode-extrawater-klonly-loweps"
#filetag = "pmfpredict-aug21-mix6-deconv-mix6-desatweightspreset-withpost-KLloss-nodesat-allscaled-batchnorms-oldxlogwithshift"
filetag = "pmfpredict-sep07-mixtrainval-v2-moredropout-normedclusters-localnormxlogpotential-oldclusters"
workingDir = basedir+"/"+filetag

datasetAll['pmfname'] = datasetAll['Material'] + "_" + datasetAll['Chemical']
parser = argparse.ArgumentParser(description="Parameters for PMFPredictor")
parser.add_argument("-i","--ensembleID", type=int,default=0, help="ID number for this ensemble member, set to 0 for development split")
parser.add_argument("-b","--bootstrap", type=int,default=0,help="If non-zero switches the sampling to bootstrapping")

args = parser.parse_args()
numChemClusters = 20

E0TargetVal = 10 #1 kbT = 2.5 kJ mol
numEpochs = 2000
random.seed(1648113195 + args.ensembleID) #epoch time at development start with the ensembleID used as an offset - this way each member gets a different split, but can be recreated.

#filter out inputs which are likely to cause problems during fitting
maxE0 = 500
absErrorThreshold = 10
relErrThreshold = 2

initialSamples = len(datasetAll)
datasetAll=datasetAll.drop( datasetAll[   np.abs(datasetAll["TargetE0"] - datasetAll["fittedE0"] ) > absErrorThreshold].index )
firstReduction = len(datasetAll)

secondReductionVals=np.abs( datasetAll["TargetE0"] - datasetAll["fittedE0"])/(np.abs(datasetAll["TargetE0"]) + np.abs(datasetAll["fittedE0"]) + 0.001) > relErrThreshold
print(secondReductionVals)
datasetAll=datasetAll.drop( datasetAll[ secondReductionVals].index)
secondReduction = len(datasetAll)

datasetAll=datasetAll.drop( datasetAll[( datasetAll["fittedE0"] ) > maxE0].index )
thirdReduction = len(datasetAll)
print("Initial: ", initialSamples, " absolute filter: ", firstReduction, "relative filter", secondReduction, "E0 max", thirdReduction)


#remove the Al PMFs 
#datasetAll = datasetAll.drop(  datasetAll[   datasetAll["Material"] == "AlFCC100UCD"       ].index   )
#datasetAll = datasetAll.drop(  datasetAll[   datasetAll["Material"] == "AlFCC110UCD"       ].index   )
datasetAll = datasetAll.drop(  datasetAll[   datasetAll["Material"] == "AlFCC111UCD"       ].index   )


suspiciousPMFs= [
 ["AuFCC100UCD","TRPSCA-JS"],
 ["AuFCC100UCD","PHESCA-JS"],
 ["AuFCC110UCD","ALASCA-JS"],
  ["AuFCC110UCD","CYSSCA-JS"],
 ["Ag100","GLUSCA-JS"],
 ["Ag100","TYRSCA-JS"]
 ]
for pmfToDrop in suspiciousPMFs:
    datasetAll = datasetAll.drop(  datasetAll[   (datasetAll["Material"] == pmfToDrop[0] )   &  (datasetAll["Chemical"] == pmfToDrop[1] )      ].index   )


#re-assign the Al PMFs to source=2 because these are different to all the others
#datasetAll.loc[ datasetAll["Material"] == "AlFCC100UCD" ,   "source" ] = 2
#datasetAll.loc[ datasetAll["Material"] == "AlFCC110UCD" ,   "source" ] = 2

#and re-assign CdSe, AuSU because these don't have ions
#datasetAll.loc[ datasetAll["Material"] == "AuFCC100" ,   "source" ] = 3
#datasetAll.loc[ datasetAll["Material"] == "CdSeWurtzite2-10" ,   "source" ] = 3


#datasetAll.loc[ datasetAll["Material"] == "AlFCC111UCD" ,   "source" ] = 2
#then drop the al for now


datasetAll.loc[ datasetAll["Material"] == "AuFCC100UCD" ,   "source" ] = 2
datasetAll.loc[ datasetAll["Material"] == "AuFCC110UCD" ,   "source" ] = 2
datasetAll.loc[ datasetAll["Material"] == "AuFCC111UCD" ,   "source" ] = 2
datasetAll = datasetAll.drop( datasetAll[datasetAll["source"]==3].index )




if args.bootstrap == 0:
    dataset = datasetAll
else:
    filetag = filetag+"-bootstrapped"
    pmfSet  =  datasetAll['pmfname'].unique().tolist()
    selectedPMFs = random.choices( pmfSet, len(pmfSet) )
    dataset = pd.DataFrame()
    for selectedPMF in selectedPMFs:
        dataset = pd.concat( [dataset, datasetAll[datasetAll["pmfname"]==selectedPMF] ]   )
    #dataset = datasetAll.sample(frac=1,replace=True,random_state=1+args.ensembleID)

if args.ensembleID>0:
    filetag = filetag+"-ensemble"+str(args.ensembleID)
    numEpochs = 50

#build the folders for logging, outputs, metadata
os.makedirs(basedir, exist_ok=True)
os.makedirs(workingDir, exist_ok=True)
os.makedirs(workingDir+"/checkpoints", exist_ok=True)
os.makedirs(workingDir+"/final_model", exist_ok=True)
os.makedirs(workingDir+"/figures", exist_ok=True)
localName = os.path.basename(__file__)
shutil.copyfile(localName, workingDir+"/"+localName)


#IMPORTANT: r0 has to remain the first variable for the slicing later on to work correctly.
pmfVars = ["r0",  "SurfAlignDist", "SSDRefDist", "source", "numericShape","resolution"]
ssdVar = ["ssdType"]

#coeffOrders = [  1,2,3,4,5,6,7,8,9 ,10,11,12 ,13,14,15,16 ]

nMaxOrder = 20
coeffOrders = range(1,nMaxOrder+1)

#SurfCProbe must remain the first entry here as its used to define the initial state of output potential
surfacePotentialModels =  ["SurfCProbe", "SurfKProbe", "SurfClProbe", "SurfWaterFullProbe",    "SurfC2AProbe",  "SurfC4AProbe", "SurfMethaneProbe", "SurfCarbonRingProbe", "SurfCPlusProbe", "SurfCMinusProbe" , "SurfCMinProbe" ,"SurfCLine3Probe"]
chemicalPotentialModels = ["ChemCProbe", "ChemKProbe", "ChemClProbe",  "ChemWaterUCDProbe",    "ChemC2AProbe", "ChemSlabProbe", "ChemMethaneProbe", "ChemCLineProbe" ,"ChemCPlusProbe","ChemCMinusProbe", "ChemCEps20Probe", "ChemCMinProbe", "ChemCarbRingProbe"]
potentialModels = surfacePotentialModels + chemicalPotentialModels
numCoeffs = len(coeffOrders)
numPotentials = len(potentialModels)
numOutputCoeffs = nMaxOrder
chosenCoeffs = []
chosenCoeffsNorm = []
coeffNormR0Matrix = []

potentialEMins = []
overrideCoeffNorm = False
for potModel in potentialModels:
    #chosenCoeffs.append( potModel+"R0")
    potentialEMins.append(potModel+"EMin")
    potentialEMins.append(potModel+"RightEMin")
    for coeffNum in coeffOrders:
        chosenCoeffs.append(potModel+"C"+str(coeffNum))
        if overrideCoeffNorm == True:
            chosenCoeffsNorm.append(potModel+"C"+str("1")) #override per-variable to per-potential normalisation
        else:
            chosenCoeffsNorm.append(potModel+"C"+str(coeffNum)) 
        coeffNormR0Matrix.append("r0")

numericVars = chosenCoeffs +potentialEMins+ ["EMin", "rEMin" , "fittedE0"]
aaVarSet = pmfVars + ssdVar + numericVars
aaPresetIn =keras.Input( shape=(len(aaVarSet),))

pseudoInput = keras.Input( shape=(1,))

numGenericInputs = len(pmfVars)
totalNumInputs = len(aaVarSet)

#at this point the set of input variables is defined so we write these out to a file

varsetOutputFile=open(workingDir+"/varset.txt","w")
varsetOutputFile.write( ",".join(aaVarSet))
#for inputVar in aaVarSet:
#    varsetOutputFile.write(inputVar+",")
varsetOutputFile.close()


inputs = [aaPresetIn]


allowMixing = 1

if allowMixing == 0:
    #Find the set of materials present in the training set and get their canonical coefficients
    uniqueMaterials = dataset['Material'].unique().tolist()
    #print(dataset)
    canonicalMaterialSet =   pd.read_csv("Datasets/SurfacePotentialCoefficients-sep07.csv")
    canonicalMaterialSet["R0Dist"] = np.sqrt( (canonicalMaterialSet["SurfCProbeR0"] - 0.2 )**2 ) 
    canonicalMaterialSet.sort_values( by=["R0Dist"] ,ascending=True, inplace=True)
    canonicalMaterialSet.drop_duplicates( subset=['SurfID'] , inplace=True,keep='first')
    #print(canonicalMaterialSet)
    trainingMaterialSet = canonicalMaterialSet[canonicalMaterialSet["SurfID"].isin(uniqueMaterials)]
    #print(trainingMaterialSet)
    #override the random assignment to ensure that both Stockholm-style and UCD-style AuFC100 PMFs are in the training set
    #this is so that a) we get the really strongly-binding Stockholm-style gold and b) to provide a baseline for conversion between the two
    #we also add the results of cluster analysis based on the generated potentials
    clusterpotentialModels = surfacePotentialModels
    chosenClusterCoeffs = ["SurfAlignDist","numericShape"] #  ,"SSDRefDist"]
    for potModel in clusterpotentialModels:
        for coeffNum in range(1,9):
            chosenClusterCoeffs.append(potModel+"C"+str(coeffNum))
    #print(materialSet[chosenClusterCoeffs])
    actualClusterNum = min( 15, int(round(len(trainingMaterialSet)*0.5)) )
    agglomCluster = skcluster.AgglomerativeClustering(n_clusters = actualClusterNum ) # None, distance_threshold =5)
    print(trainingMaterialSet[chosenClusterCoeffs].values)
    agglomCluster.fit( skpreproc.normalize( trainingMaterialSet[chosenClusterCoeffs].values)) 
    clusterLabels =  agglomCluster.labels_
    fixedMaterials = ["AuFCC100", "AuFCC100UCD"]
    
    
    clustersOutputFile=open(workingDir+"/outputvarset.txt","w")
    #clustersOutputFile.write( ",".join(outputVarset))

    clustersOutputFile.write("Material clusters")
    for i in range(max(agglomCluster.labels_) + 1):
        clusterMembers = trainingMaterialSet[ clusterLabels == i][ "SurfID" ].values 
        randomMember = random.choice(clusterMembers)
        print("Cluster", i, "assigning", randomMember, "from", clusterMembers)
        fixedMaterials.append(randomMember)
        clustersOutputFile.write( "Cluster " + str(i)+ " assigning " + randomMember + " from " + ",".join(clusterMembers) + "\n")
    fixedMaterials = list(set(fixedMaterials)) #remove any duplicates
    
    fixedSMILES = ["C","Cc1c[nH]c2ccccc12"]
    uniqueChemicals = dataset['Chemical'].unique().tolist()
    canonicalChemicalSet = pd.read_csv("Datasets/ChemicalPotentialCoefficients-aug26.csv")

    canonicalChemicalSet["R0Dist"] = np.sqrt( (canonicalChemicalSet["ChemCProbeR0"] - 0.2 )**2 ) 
    canonicalChemicalSet.sort_values( by=["R0Dist"] ,ascending=True, inplace=True)
    canonicalChemicalSet.drop_duplicates( subset=['ChemID'] , inplace=True,keep='first')


    clustersOutputFile.write("Chemical clustering: \n")
    trainingChemicalSet = canonicalChemicalSet[canonicalChemicalSet["ChemID"].isin(uniqueChemicals)]
    clusterpotentialModels = chemicalPotentialModels
    chosenClusterCoeffs = []
    for potModel in clusterpotentialModels:
        for coeffNum in range(1,9):
            chosenClusterCoeffs.append(potModel+"C"+str(coeffNum))
    #print(materialSet[chosenClusterCoeffs])
    actualClusterNum = min( numChemClusters, int(round(len(trainingChemicalSet)*0.5)) )
    agglomCluster = skcluster.AgglomerativeClustering(n_clusters = actualClusterNum )
    agglomCluster.fit(  skpreproc.normalize( trainingChemicalSet[chosenClusterCoeffs].values) )
    clusterLabels =  agglomCluster.labels_
    for i in range(max(agglomCluster.labels_) + 1):
        clusterMembers = trainingChemicalSet[ clusterLabels == i][ "SMILES" ].values 
        randomMember = random.choice(clusterMembers)
        print("Cluster", i, "assigning", randomMember, "from", clusterMembers)
        fixedSMILES.append(randomMember)
        clustersOutputFile.write( "Cluster " + str(i)+ " assigning " + randomMember + " from " + ",".join(clusterMembers) + "\n")
    fixedSMILES = list(set(fixedSMILES))
    #fixedMaterials = ["AuFCC100", "AuFCC100UCD",  "CNT15-COO--10" ,   "SiO2-Quartz","grapheneoxide","Fe2O3-001O" , "Ag110" , "CNT15-COOH-30", "SiO2-Amorphous", "CdSeWurtzite2-10", "TiO2-ana-100"]
    #fixedSMILES = ["C", "OCC1OC(O)C(O)C(O)C1O", "Cc1c[nH]c2ccccc12"]
    #sample over all AA and materials present in the (possibly bootstrapped) dataset
    #chemicals are selected based on SMILES code to allow for duplicates, the -3 is used to account for the fact 3 SMILES codes are manually assigned to the training set
    uniqueSMILES = dataset['SMILES'].unique().tolist()
    
    targetSMILESValidationNumber = int(len(uniqueSMILES)*0.2)
    targetMaterialValidationNumber = int(len(uniqueMaterials)*0.2)
    uniqueUnusedSMILES = sorted(list(   set(uniqueSMILES) - set(fixedSMILES) ))
    uniqueUnusedMaterials = sorted(list( set(uniqueMaterials) - set(fixedMaterials) ))
    validationSMILES =sorted( random.sample( uniqueUnusedSMILES, targetSMILESValidationNumber) )
    validationMaterials = sorted( random.sample(uniqueUnusedMaterials, targetMaterialValidationNumber) )
    trainingSMILES = sorted( list (  set(uniqueSMILES) - set(validationSMILES) ))
    trainingMaterials = sorted( list(  set(uniqueMaterials) - set(validationMaterials)) )
    #generate the lists of chemicals based on smiles codes
    uniqueAA = dataset['Chemical'].unique().tolist()
    trainingAA = sorted(  (dataset[dataset['SMILES'].isin( trainingSMILES)])['Chemical'].unique().tolist() )
    validationAA = sorted( list( set(uniqueAA) - set(trainingAA) ) )
    print(trainingAA)
    print("Training AA: ", trainingAA)
    print("Validation AA: ", validationAA)
    print("Training Materials: ", trainingMaterials)
    print("Validation Materials: ", validationMaterials)
    dataset['ChemValidation'] = 0
    dataset.loc[  dataset['Chemical'].isin(validationAA)  ,'ChemValidation'] = 1
    dataset['MaterialValidation'] = 0
    dataset.loc[  dataset['Material'].isin(validationMaterials) , 'MaterialValidation' ] = 1
    test_dataset = dataset[ (dataset['Chemical'].isin(validationAA) ) | (dataset['Material'].isin(validationMaterials) )].copy() #if either of the material or AA is in the validation set, it gets assigned to the validation set to prevent leakage
    train_dataset = dataset.drop(test_dataset.index)
    clustersOutputFile.close()
else:
    uniquePMFs =dataset['pmfname'].unique().tolist()
    validationPMFs = sorted( random.sample( sorted(uniquePMFs), int( len(uniquePMFs)*0.3)))
    print(validationPMFs)
    uniqueAA = dataset['Chemical'].unique().tolist()
    uniqueMaterials = dataset['Material'].unique().tolist()
    dataset['ChemValidation']=0
    dataset['MaterialValidation']=0
    dataset.loc[ dataset['pmfname'].isin(validationPMFs) , 'ChemValidation'] = 1
    dataset.loc[ dataset['pmfname'].isin(validationPMFs) , 'MaterialValidation'] = 1
    test_dataset = dataset[  dataset['pmfname'].isin(validationPMFs)     ].copy()
    train_dataset = dataset.drop(test_dataset.index)

activationFuncChoice =  activationXLog
weightInitialiser = tf.keras.initializers.VarianceScaling( scale =  1.5, mode="fan_avg", distribution="truncated_normal", seed = 1648113195 + args.ensembleID)
#print(train_dataset)





aaNormalizer = layers.Normalization()
aaNormalizer.adapt(np.array(train_dataset[aaVarSet]))
aaPresetNorm = aaNormalizer(aaPresetIn)


#numericVars = chosenCoeffs +potentialEMins+ ["EMin", "rEMin" , "fittedE0"]

coeffNormalizer = layers.Normalization()
coeffNormalizer.adapt( np.array(train_dataset[chosenCoeffsNorm]) * np.sqrt(np.array(train_dataset[coeffNormR0Matrix])  )        ) #norm coefficients by the average value of C1 for that potential, weighted by sqrt(r0)
potentialMinNormalizer = layers.Normalization()
potentialMinNormalizer.adapt( np.array(train_dataset[potentialEMins]) )
eminNormalizer = layers.Normalization(axis=None)
eminNormalizer.adapt( np.array(train_dataset["EMin"]))
e0Normalizer = layers.Normalization(axis=None)
e0Normalizer.adapt( np.array(train_dataset["fittedE0"]))





#From StackOverflow 6-168142, user letitgo
class SliceLayer(keras.layers.Layer):
    def __init__(self, begin, size,**kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.begin = begin
        self.size = size
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size ,
        })
        return config
    def call(self, inputs):
        return  inputs[:, self.begin:(self.begin + self.size)] 


#Remaps 0/1 values to -0.5/0.5
class BoolFlip(keras.layers.Layer):
    def __init__(self, flipProb,**kwargs):
        super(BoolFlip, self).__init__(**kwargs)
        self.flipProb = flipProb
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'flipProb': self.flipProb,
        })
        return config
    def call(self, inputs,training=False):
        xnew = inputs - 0.5
        if training:
            drawnProbs = tf.random.uniform( tf.shape(inputs) )
            flipVal =tf.where (drawnProbs  < self.flipProb, -1.0, 1.0) 
            return xnew * flipVal
        else:
            return xnew

#In inference, returns input[0]. in training, randomly returns either input[0] with prob. flipProb or input[1] (prob 1 - flipProb), such that input[1] can be used for teacher-forcing
#StochasticTeacher(0.5)([ predictedVal, trueVal]) gives both with 50-50, StochasticTeacher(0.1) gives the real value 90% of the time, etc.
class StochasticTeacher(keras.layers.Layer):
    def __init__(self, flipProb,**kwargs):
        super(StochasticTeacher, self).__init__(**kwargs)
        self.flipProb = flipProb
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'flipProb': self.flipProb,
        })
        return config
    def call(self, inputs,training=False):
        if training:
            drawnProbs = tf.random.uniform( tf.shape(inputs[0]) )
            return tf.where (drawnProbs  < self.flipProb, inputs[0], inputs[1]) 
        else:
            return inputs[0]

coeffMultiplicativeNoiseSDev = 0.1
coeffMultiNoiseRate = coeffMultiplicativeNoiseSDev**2/(1 + coeffMultiplicativeNoiseSDev**2)
r0InMultiplicativeNoiseSDev = 0.01
r0InMultiNoiseRate = r0InMultiplicativeNoiseSDev**2/(1 + r0InMultiplicativeNoiseSDev**2)


aaPresetInUpranked = layers.Reshape( (-1,1) )(aaPresetIn)
aaPresetNormUpranked = layers.Reshape( (-1,1) )(aaPresetNorm)

#override the normalisation
aaPresetNorm = aaPresetIn

r0Index = aaVarSet.index("r0")
#inputR0 = SliceLayer( aaVarSet.index("r0"), 1 )(aaPresetIn)
#inputR0 = layers.Lambda( lambda x: tf.slice(x ,aaVarSet.index("r0"), 1   ))(aaPresetIn)
inputR0 = layers.Lambda( lambda x: x[:, aaVarSet.index("r0"):(aaVarSet.index("r0")+1)]    )(aaPresetIn)
#inputR0 = layers.GaussianDropout(r0InMultiNoiseRate)(inputR0)
r0Noise = layers.GaussianDropout(r0InMultiNoiseRate)(inputR0)

print(inputR0)


sqrtr0 = layers.Lambda( lambda x: tf.math.pow( tf.math.maximum( x, 0.001) , 0.5) )(inputR0)
invsqrtr0 = layers.Lambda( lambda x: tf.math.pow( tf.math.maximum( x, 0.001) , -0.5) )(inputR0)

#log transform and scale r0 to get a value useful for calculations
logr0Normalizer = layers.Normalization(axis=None)
logr0Normalizer.adapt( np.log(np.array(train_dataset["r0"])))
logr0 = layers.Lambda(lambda x: tf.math.log(x) )(inputR0)
logr0 = logr0Normalizer(logr0)
logr0 = layers.GaussianNoise(0.1)(logr0)

inputres = SliceLayer( aaVarSet.index("resolution") , 1)(aaPresetNorm)
logresNormalizer = layers.Normalization(axis=None)
logresNormalizer.adapt( np.log(np.array(train_dataset["resolution"])))
logres = layers.Lambda(lambda x: tf.math.log(x) )(inputres)
logres = logresNormalizer(logres)
logres = layers.GaussianNoise(0.1)(logres)



#e0Estimate = layers.Multiply(  name="e0predict" )([ invsqrtr0, coeffWeightedSum] )


#coeffNormalizer = layers.Normalization()
#coeffNormalizer.adapt( np.array(datasetAll[chosenCoeffs]))
#potentialMinNormalizer = layers.Normalization()

#inputR0 = layers.Flatten()(inputR0)



#ssdVar = ["ssdType"]
ssdInputVar = SliceLayer( aaVarSet.index("ssdType") , 1)(aaPresetNorm)
ssdInputVar = layers.CategoryEncoding( num_tokens=4,  output_mode="one_hot")(ssdInputVar)
print(ssdInputVar)
#quit()

sourceInputVar = SliceLayer( aaVarSet.index("source") , 1)(aaPresetNorm)
sourceInputVar = layers.CategoryEncoding( num_tokens=4,  output_mode="one_hot")(sourceInputVar)
print(sourceInputVar)


shapeInputVar = SliceLayer( aaVarSet.index("numericShape") , 1)(aaPresetNorm)
shapeInputVar = layers.CategoryEncoding( num_tokens=2,  output_mode="one_hot")(shapeInputVar)

#pmfInputVarLayer = SliceLayer( aaVarSet.index("r0")+2, len(pmfVars)-1)(aaPresetNorm)

#collect all the categorical booleans together and apply some noise
pmfInputVarLayer = layers.Concatenate()([   ssdInputVar,sourceInputVar, shapeInputVar])
pmfInputVarLayer = BoolFlip( 0.1)(pmfInputVarLayer)
pmfInputVarLayer=layers.GaussianNoise(0.2)(pmfInputVarLayer)

offsetVarLayer = SliceLayer( aaVarSet.index("SurfAlignDist"),  1)(aaPresetNorm)
offsetNorm = layers.Normalization(axis=None)
offsetNorm.adapt( np.array(train_dataset["SurfAlignDist"]))
offsetVarLayer = offsetNorm(offsetVarLayer)
offsetVarLayer = layers.GaussianNoise(0.2)(offsetVarLayer)


ssdoffsetVarLayer = SliceLayer( aaVarSet.index("SSDRefDist"),  1)(aaPresetNorm)
ssdoffsetNorm = layers.Normalization(axis=None)
ssdoffsetNorm.adapt( np.array(train_dataset["SSDRefDist"]))
ssdoffsetVarLayer = ssdoffsetNorm(ssdoffsetVarLayer)
ssdoffsetVarLayer = layers.GaussianNoise(0.2)(ssdoffsetVarLayer)

pmfInputVarLayer = layers.Concatenate()([ logr0, pmfInputVarLayer,offsetVarLayer,ssdoffsetVarLayer,logres,r0Noise])
print(pmfInputVarLayer)

pmfInputsEncoded = layers.Dense(16)(pmfInputVarLayer)
pmfInputsEncoded = TrainableXLog()(pmfInputsEncoded)
pmfInputsEncoded2 = layers.Dense(16)(pmfInputsEncoded)
pmfInputsEncoded2 = TrainableXLog()(pmfInputsEncoded2)
pmfInputsEncoded = pmfInputsEncoded + pmfInputsEncoded2

pmfInputVarLayer = layers.Concatenate()([ pmfInputVarLayer, pmfInputsEncoded ])

slabCoeffs = SliceLayer(aaVarSet.index("ChemSlabProbeC1"), len(coeffOrders))(aaPresetIn)
#slabCoeffNorm = layers.Normalization()
#slaboeffNorm.adapt( np.array(datasetAll["Chem



methaneCoeffs = SliceLayer(aaVarSet.index("SurfMethaneProbeC1"), len(coeffOrders))(aaPresetIn)

slabCoeffs = layers.Reshape( (-1,1) )(slabCoeffs)
methaneCoeffs = layers.Reshape( (-1,1))(methaneCoeffs)
pairedStartCoeffs = layers.Concatenate()([slabCoeffs,methaneCoeffs])

print(pairedStartCoeffs)


potentialCoeffsOnlyInit0 = SliceLayer( aaVarSet.index("SurfCProbeC1") ,len(chosenCoeffs) )(aaPresetNorm)




#coeffNormalizer = layers.Normalization()
#coeffNormalizer.adapt( np.array(datasetAll[chosenCoeffs]))
#potentialCoeffsOnly = coeffNormalizer(potentialCoeffsOnly0) #scales and shifts all potential coefficients according to C1 for that potential
potentialCoeffsOnly0 = TrainableXLog()(potentialCoeffsOnlyInit0)

potentialCoeffsOnly = layers.GaussianNoise(0.01)(potentialCoeffsOnly0)
#potentialMinNormalizer = layers.Normalization()



potentialCoeffsOnly = layers.Reshape( (numCoeffs, -1))(potentialCoeffsOnly)

#multiply each potential coefficient by sqrt(2*i - 1) to boost the high-order coeffs 
class GetCoeffScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(GetCoeffScaleLayer, self).__init__()
      self.scale = tf.constant( np.array( [np.sqrt(2*i - 1) for i in range(1,1+numCoeffs) ] )  , dtype=tf.float32)
    def call(self, inputs):
      return    tf.math.multiply( inputs , self.scale)  


class PCoeffSumLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(PCoeffSumLayer, self).__init__()
      self.scale = tf.constant( np.array( [np.sqrt(2*i - 1) for i in range(1,1+numCoeffs) ] )  , dtype=tf.float32)
    def call(self, inputs):
      tiledPrefactors = tf.tile( tf.reshape( self.scale, (numCoeffs,-1)), (1,numPotentials))
      return tf.reduce_sum( tf.math.multiply( tf.math.multiply(inputs[0] ,tiledPrefactors   ) , inputs[1]), axis=1)
      
      
potentialCoeffsUnnormed = layers.Reshape( (numCoeffs,-1) )(potentialCoeffsOnlyInit0)
invsqrtr0P = layers.RepeatVector(numPotentials * numCoeffs)(invsqrtr0)
invsqrtr0P = layers.Reshape((numCoeffs,-1))(invsqrtr0P)
potentialValsAtR0 = PCoeffSumLayer()([potentialCoeffsUnnormed,invsqrtr0P])
potentialValsAtR0 = layers.Flatten()(potentialValsAtR0)

potentialValsAtR0 = layers.Dropout(0.2)(potentialValsAtR0)

potentialValsAtR0 = layers.GaussianDropout(coeffMultiNoiseRate )(potentialValsAtR0)
potentialValsAtR0 = layers.GaussianNoise(0.5 )(potentialValsAtR0)
potentialValsAtR0 = TrainableXLog()(potentialValsAtR0)
print(potentialValsAtR0)
#potentialValsAtR0 = layers.BatchNormalization()(potentialValsAtR0)
#longinvsqrtr0 = layers.RepeatVector(numPotentials)(invsqrtr0)
#potentialValsAtR0 = layers.Multiply(   )([ longinvsqrtr0, potentialValsAtR0] )




potentialCoeffsScalingFactor = GetCoeffScaleLayer()(sqrtr0)
potentialCoeffsScalingFactor = layers.RepeatVector( numPotentials  )(potentialCoeffsScalingFactor)
potentialCoeffsScalingFactor = layers.Permute( (2,1) )(potentialCoeffsScalingFactor)
#potentialCoeffsOnly = layers.Multiply( )([potentialCoeffsOnly, potentialCoeffsScalingFactor])



#apply multiplicative Gaussian noise so that the smaller ones don't vanish entirely
potentialCoeffsOnly = layers.GaussianDropout(coeffMultiNoiseRate )(potentialCoeffsOnly)
potentialCoeffsOnly = layers.Dropout(0.1)(potentialCoeffsOnly)

eminVals = SliceLayer( aaVarSet.index("SurfCProbeEMin") ,len(potentialEMins) )(aaPresetNorm)
#eminVals = layers.GaussianNoise(0.5)(eminVals)

eminVals = potentialMinNormalizer(eminVals)


eminVals = layers.GaussianDropout(coeffMultiNoiseRate)(eminVals)
eminVals = layers.GaussianNoise(0.3)(eminVals)

rminLocValInput = SliceLayer( aaVarSet.index("rEMin") , 1)(aaPresetIn)
trueEMin = SliceLayer( aaVarSet.index("EMin") , 1)(aaPresetIn)
trueE0 = SliceLayer( aaVarSet.index("fittedE0") , 1)(aaPresetIn)


outputVarset = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16" ,"A17","A18","A19","A20" ]
#outputVarset = ["D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16"  ]

eminValProcessed = layers.Dense(32)(eminVals)
eminValProcessed = TrainableXLog()(eminValProcessed)
eminValProcessed = layers.Dense(16)(eminValProcessed)
eminValProcessed = TrainableXLog()(eminValProcessed)

numOutputVars = len(outputVarset)


outputLayers= []

#shapeOnly = layers.Cropping1D( cropping=(1,1))(pmfInputVarLayer)


eadsFFDim = 256
eadsWorkingDim = 256

#estimate the binding energy given the input minimas

#allNonPotentialCoeffs1 = layers.Concatenate()([ pmfInputVarLayer])
allNonPotentialCoeffs1 = pmfInputVarLayer
stackedCoeffs1 = layers.RepeatVector( numCoeffs)(allNonPotentialCoeffs1)
stackedCoeffs1 = layers.Concatenate( )( [stackedCoeffs1, potentialCoeffsOnly])


potentialCoeffsLognormed = layers.GaussianNoise(0.01)(potentialCoeffsOnly) #TrainableXLog()(potentialCoeffsOnly)
potentialCoeffsLognormed = TrainableXLog()(potentialCoeffsLognormed)
potentialCoeffsLognormed2 = layers.ZeroPadding1D( (0,1) )(potentialCoeffsLognormed)
potentialCoeffsLognormed2 = layers.LocallyConnected1D(numPotentials,2)(potentialCoeffsLognormed2)
potentialCoeffsLognormed = potentialCoeffsLognormed + potentialCoeffsLognormed2

coeffEMinEncoded = layers.Flatten()(potentialCoeffsLognormed)

coeffE0Extra = layers.Dense(64)(potentialCoeffsLognormed)
coeffE0Extra = TrainableXLog()(coeffE0Extra)

coeffE0Extra = layers.Dense(16)(coeffE0Extra)
coeffE0Extra = TrainableXLog()(coeffE0Extra)


coeffE0Extra = layers.Flatten()(coeffE0Extra)


coeffEMinEncoded = layers.Concatenate()([ pmfInputVarLayer, coeffEMinEncoded,coeffE0Extra])
#coeffEMinEncoded = layers.BatchNormalization()(coeffEMinEncoded)
#coeffEMinEncoded = layers.Dense(64,activation=activationFuncChoice)(coeffEMinEncoded)
#coeffEMinEncoded = layers.Dense(32,activation=activationFuncChoice)(coeffEMinEncoded)
coeffEMinEncoded = layers.Dense(32 )(coeffEMinEncoded)
coeffEMinEncoded = TrainableXLog()(coeffEMinEncoded)


eadsEstimate = layers.Concatenate()([pmfInputVarLayer, eminVals,coeffEMinEncoded,eminValProcessed,coeffE0Extra,potentialValsAtR0])
eadsEstimate = layers.Dense(eadsWorkingDim)(eadsEstimate)
eadsEstimate = TrainableXLog()(eadsEstimate)
eadsEstimate = layers.Dense(eadsWorkingDim)(eadsEstimate)

#eadsEstimate = layers.BatchNormalization()(eadsEstimate)
for i in range(6):
    eadsEstimateFF = layers.Dense(eadsFFDim )(eadsEstimate)
    eadsEstimateFF = TrainableXLog()(eadsEstimateFF)
    #eadsEstimateFF = layers.BatchNormalization()(eadsEstimateFF)
    eadsEstimateFF = layers.GaussianNoise(0.5)(eadsEstimateFF)
    eadsEstimateFF = layers.Dropout(0.5)(eadsEstimateFF)
    eadsEstimateFF = layers.Dense(eadsWorkingDim)(eadsEstimateFF)
    eadsEstimate = eadsEstimate + eadsEstimateFF
#eadsEstimate = layers.Dense(64,activation=activationFuncChoice)(eadsEstimate)
eMinRoughEstimate = layers.Dense(1,name="EadsFirstEstimate")(eadsEstimate) #   DesatLog)(eadsEstimate)



eminSet=np.array(train_dataset["EMin"])



#eMinRoughEstimate = TrainableDesat(np.mean(eminSet), np.std(eminSet)  )(eMinRoughEstimate)


#eMinRoughEstimate = TrainableXLog()(eMinRoughEstimate)




eMinRoughEstimateForward =   layers.Lambda(   lambda x : tf.stop_gradient(x)    )(eMinRoughEstimate)
eMinRoughEstimateForward =   StochasticTeacher(0.5)([ eMinRoughEstimateForward, trueEMin])
eMinRoughEstimateForward = TrainableXLog()(eMinRoughEstimateForward)


eMinStateEstimateForward = layers.Lambda( lambda x: tf.stop_gradient(x) )(eadsEstimate)
eMinStateEstimateForward = layers.Dense(8)(eMinStateEstimateForward)
eMinStateEstimateForward = TrainableXLog()(eMinStateEstimateForward)

#eMinRoughEstimateForward = eminNormalizer(eMinRoughEstimateForward)

eMinRoughEstimateForward = layers.GaussianDropout(coeffMultiNoiseRate )(eMinRoughEstimateForward)
eMinRoughEstimateForward = layers.GaussianNoise(1)(eMinRoughEstimateForward ) #Make sure the model doesn't overfit to this exact value - it should just be a guideline for "strongly-binding" vs "non-binding"
eMinRoughEstimateForward = layers.Dropout(0.2)(eMinRoughEstimateForward)
#eMinRoughEstimateForward = eminNormalizer(eMinRoughEstimateForward)





coeffE0Encoded = layers.Flatten()(potentialCoeffsLognormed)
coeffE0Extra = layers.Dense(16)(potentialCoeffsLognormed)
coeffE0Extra = layers.Flatten()(coeffE0Extra)
coeffE0EncodedIn = layers.Concatenate()([ pmfInputVarLayer, coeffE0Encoded,potentialValsAtR0])
#coeffE0Encoded = layers.BatchNormalization()(coeffE0Encoded)
#coeffE0Encoded = layers.Dense(64,activation=activationFuncChoice)(coeffE0Encoded)
#coeffE0Encoded = layers.Dense(32,activation=activationFuncChoice)(coeffE0Encoded)
coeffE0Encoded = layers.Dense(32)(coeffE0EncodedIn)
coeffE0Encoded = layers.Dropout(0.2)(coeffE0Encoded)
coeffE0Encoded=TrainableXLog()(coeffE0Encoded)
coeffE0Encoded = layers.Dense(8)(coeffE0Encoded)
coeffE0Encoded=TrainableXLog()(coeffE0Encoded)
e0Estimate = layers.Concatenate()([coeffE0EncodedIn,coeffE0Encoded])
e0Estimate = layers.Dense(eadsWorkingDim)(e0Estimate)
e0Estimate = TrainableXLog()(e0Estimate)
e0Estimate = layers.Dropout(0.5)(e0Estimate)
e0Estimate = layers.Dense(eadsWorkingDim)(e0Estimate)

#eadsEstimate = layers.BatchNormalization()(eadsEstimate)
for i in range(6):
    e0EstimateFF = layers.Dense(eadsFFDim)(e0Estimate)
    e0EstimateFF = TrainableXLog()(e0EstimateFF)
    #eadsEstimateFF = layers.BatchNormalization()(eadsEstimateFF)
    e0EstimateFF = layers.GaussianNoise(0.5)(e0EstimateFF)
    e0EstimateFF = layers.Dropout(0.5)(e0EstimateFF)
    e0EstimateFF = layers.Dense(eadsWorkingDim)(e0EstimateFF)
    e0EstimateFF = layers.Dropout(0.25)(e0EstimateFF)
    e0Estimate = e0Estimate + e0EstimateFF


#e0Estimate = layers.Dense(4)(e0Estimate)
e0EstimateState = layers.GaussianNoise(0.01)(e0Estimate)
e0Estimate = layers.Dense(1,name="E0FirstEstimate")(e0Estimate) #   DesatLog)(eadsEstimate)

#e0Estimate = TrainableXLog()(e0Estimate)

e0Set = np.array(train_dataset["fittedE0"])
#e0Estimate =     TrainableDesat( np.mean(e0Set), np.std(e0Set))(e0Estimate)



e0RoughEstimateForward =   layers.Lambda(   lambda x : tf.stop_gradient(x)    )(e0Estimate)
e0RoughEstimateForward =   StochasticTeacher(0.5)([ e0RoughEstimateForward, trueE0])
e0RoughEstimateForward = TrainableXLog()(e0RoughEstimateForward)


e0StateEstimateForward = layers.Lambda( lambda x: tf.stop_gradient(x) )(e0Estimate)
e0StateEstimateForward = layers.Dense(8)(e0StateEstimateForward)
e0StateEstimateForward = TrainableXLog()(e0StateEstimateForward)






pseudoInput = e0Normalizer(pseudoInput)
#e0RoughEstimateForward = e0Normalizer(e0RoughEstimateForward)

e0RoughEstimateForward = layers.GaussianDropout(coeffMultiNoiseRate )(e0RoughEstimateForward)
e0RoughEstimateForward = layers.GaussianNoise(1)(e0RoughEstimateForward )
e0RoughEstimateForward = layers.Dropout(0.3)(e0RoughEstimateForward)
#e0RoughEstimateForward = e0Normalizer(e0RoughEstimateForward)


#e0Estimate = e0RoughEstimate

#e0Estimate


class GetValLayer(tf.keras.layers.Layer):
    def __init__(self,targetLoc):
        super(GetValLayer,self).__init__()
        #self.target = r0InputLoc
        targetArr = np.zeros( nMaxOrder)
        targetArr[targetLoc] = 1
        self.chooseVector =  tf.constant( targetArr  , dtype=tf.float32)
    def call(self, inputs):
        #split1 =  tf.unstack( inputs, axis=1)
        return tf.reduce_sum( tf.math.multiply(inputs, self.chooseVector),axis=1,keepdims=True)


class TrainingSwitch(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(TrainingSwitch,self).__init__(**kwargs)
    def call(self, inputs,training=None):
        if training:
            return inputs[0]
        return inputs[1]

'''
class PotentialAtPoint(tf.keras.layers.Layer):
    def __init__(self):
        super(PotentialAtPoint,self).__init__()
        self.scale = tf.constant( np.array( [ (-1)**(1+i) * np.sqrt(2*i - 1) for i in range(1,1+numOutputVars) ] )  , dtype=tf.float32)
        self.hgeA = tf.constant( np.array( [ 1-i for i in range(1,1+numOutputVars) ] )  , dtype=tf.float32)
        self.hgeB = tf.constant( np.array( [ i for i in range(1,1+numOutputVars) ] )  , dtype=tf.float32)
        self.hgeC = tf.constant( np.array( [ 1 for i in range(1,1+numOutputVars) ] )  , dtype=tf.float32)
    def call(self, inputs): #coefficients in inputs[1], r in inputs[0]
        #total = tf.math.multiply( inputs[1], 0 )
        #for n in range(1,17):
        #    total = total + inputs[0,n-1] * (-1)**(1+n) * tf.math.sqrt( 2*n - 1) * sqrt(0.2)/r * tfphyp.hyp2f1_small_argument( 1- n, n, 1, 0.2/r)
        # rmin/r * self.scale * tfphpy(
        weightedCoeffs = tf.math.multiply(inputs[1] , self.scale)
        argRatio1 = tf.math.divide( 0.2, inputs[0])
        argRatio2 = tf.math.divide(tf.math.sqrt( 0.2), inputs[0])
        funcVals =  tf.math.multiply( argRatio2, tfp.math.hypergeometric.hyp2f1_small_argument( self.hgeA, self.hgeB, self.hgeC, argRatio1) ) 
        return tf.reduce_sum( tf.math.multiply(weightedCoeffs  , funcVals) , axis=1,keepdims=True) 

'''
@tf.keras.utils.register_keras_serializable()
def hgBasisFuncV2(x, order):
    order = tf.convert_to_tensor(order,dtype=tf.float64)    
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    indexSet = tf.range(1.0,order +1.0,dtype=tf.float64)
    prefactors = tf.math.pow( x, indexSet - 1.0) * tf.math.pow( tf.cast( -1.0, tf.float64), indexSet-1.0) *  tf.exp(tf.math.lgamma( (order + indexSet - 1.0)   ) - tf.math.lgamma( 1.0 - indexSet + order) - 2* tf.math.lgamma( indexSet)) 
    y = tf.math.reduce_sum(prefactors)
    return y


#calculates 2F1(1-n,n,1,x)
@tf.keras.utils.register_keras_serializable()
def getHGVal(x):
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    indexSet = tf.range(1,nMaxOrder +1,dtype=tf.float64)
    applyBasis = lambda i: hgBasisFuncV2(x,i)
    funcVals= tf.map_fn( applyBasis, indexSet) 
    return funcVals

@tf.keras.utils.register_keras_serializable()
def calcHGE(r,r0,coeffs):
    r = tf.convert_to_tensor(r,dtype=tf.float64)
    r0 = tf.convert_to_tensor(r0,dtype=tf.float64)
    coeffs = tf.convert_to_tensor(coeffs,dtype=tf.float64)
    indexSet  = tf.range(1, nMaxOrder +1,dtype=tf.float64)
    hg2F1Set = getHGVal( tf.math.divide(r0,r) )
    prefactors = tf.math.pow(  tf.cast( -1.0, tf.float64), indexSet+1.0) * tf.math.sqrt( 2*indexSet - 1.0) * hg2F1Set * tf.math.divide( tf.math.sqrt(r0) , r)
    #print(hg2F1Set)
    #print(prefactors)
    #print( tf.math.multiply(prefactors,hg2F1Set))
    return tf.cast( tf.math.reduce_sum( tf.math.multiply(prefactors,coeffs)  ,axis=1,keepdims=True  ), tf.float32)
    
class PotentialAtPoint(tf.keras.layers.Layer):
    def __init__(self):
        super(PotentialAtPoint,self).__init__()
    def call(self, inputs): #coefficients in inputs[1], r in inputs[0]
        return calcHGE(tf.cast(inputs[0],tf.float64),   tf.cast(inputs[1],tf.float64)   ,tf.cast(inputs[2],tf.float64)) 



eminValsProcessed2  = layers.Dense(128 )(eminVals)
eminValsProcessed2  = TrainableXLog()(eminValsProcessed2 )
eminValsProcessed2  = layers.Dense(16 )(eminValsProcessed2)
eminValsProcessed2  = TrainableXLog()(eminValsProcessed2 )


preprocessingState = layers.Concatenate()([e0StateEstimateForward,eMinStateEstimateForward])
preprocessingState = layers.GaussianNoise(0.5)(preprocessingState)
#preprocessingState = layers.Dense(8)(preprocessingState)
#preprocessingState = TrainableXLog()(preprocessingState)

allNonPotentialCoeffs = layers.Concatenate()([ pmfInputVarLayer,eMinRoughEstimateForward,eminValsProcessed2, e0RoughEstimateForward,potentialValsAtR0,preprocessingState])

#allNonPotentialCoeffs =  pmfInputVarLayer

#shapeNonPotentialCoeffs = shapeOnly#layers.Concatenate()([ shapeOnly])

#define the input to the encoder. 
stackedCoeffs = layers.RepeatVector( numCoeffs)(allNonPotentialCoeffs)
stackedCoeffs = layers.Concatenate( )( [stackedCoeffs, potentialCoeffsLognormed])
#stackedCoeffs = layers.Dropout(0.5)(stackedCoeffs)


directEncode = layers.Dense(64)(stackedCoeffs)
directEncode = TrainableXLog()(directEncode)
directEncode = layers.Flatten()(directEncode)
directEncode = layers.Dropout(0.3)(directEncode)
directEncode = layers.Dense(32)(directEncode)
directEncode = TrainableXLog()(directEncode)


directEncodeFF = layers.Concatenate()([directEncode, allNonPotentialCoeffs])
directEncodeFF = layers.Dense(32)(directEncode)
directEncodeFF = TrainableXLog()(directEncode)
directEncode = directEncode + directEncodeFF
#directEncode = layers.Dense(16,activation=activationFuncChoice)(directEncode)

potentialDim = 64

'''
encodeConv0 = layers.Conv1D(potentialDim,3,padding="same" )(stackedCoeffs)
encodeConv0 = TrainableXLog()(encodeConv0)
encodeConv0 = layers.Conv1D(potentialDim,3,padding="same" )(stackedCoeffs)
encodeConv0 = TrainableXLog()(encodeConv0)
encodeConv0 = layers.ZeroPadding1D( (0,4))(encodeConv0)
for i in range(3):
    pmfEstFF = layers.Conv1D(potentialDim,2,strides=2 )(encodeConv0)
    pmfEstFF = TrainableXLog()(pmfEstFF)
    pmfEstFF = layers.BatchNormalization()(pmfEstFF)
    pmfEstFF = layers.Conv1D(potentialDim,2,padding="same")(pmfEstFF)
    encodeConvDS = layers.AveragePooling1D(2)(encodeConv0)
    encodeConv0 = pmfEstFF + encodeConvDS
'''
stackedCoeffsEncoder = layers.ZeroPadding1D( (0,2) )(stackedCoeffs)
encodeConv0 = layers.LocallyConnected1D(potentialDim,3)(stackedCoeffsEncoder)
encodeConv0 = TrainableXLog()(encodeConv0)
#20 -> 10 -> 5
for i in range(2):
    convEncFF = layers.ZeroPadding1D( (0,1) )(encodeConv0)
    convEncFF = layers.LocallyConnected1D(potentialDim,2,strides=2)(convEncFF)
    convEncFF = TrainableXLog()(convEncFF)
    convEncFF = layers.Dropout(0.1)(convEncFF)
    #convEncFF =  ZeroPadding1D( (0,1) )(convEncFF)
    convEncFF = layers.LocallyConnected1D(potentialDim,1)(convEncFF)
    convEncDS = layers.AveragePooling1D(2)(encodeConv0)
    encodeConv0 = convEncDS + convEncFF

encodeConv0Max = layers.GlobalMaxPooling1D()(encodeConv0)
encodeConv0 = layers.Flatten()(encodeConv0)

#for i in range(2):
#    encodeConv0 = layers.Dense(32,activation=activationFuncChoice)(encodeConv0)
#encode to a 4D space
encodeConv0 = layers.Dense(8)(encodeConv0)
encodeConv0  = TrainableXLog()(encodeConv0 )
encodeConv0 = layers.Concatenate()([encodeConv0, encodeConv0Max])
#eMinRoughEstimateForward 



encodedState = layers.Concatenate()([encodeConv0,directEncode,allNonPotentialCoeffs])
encodedState = layers.BatchNormalization()(encodedState)
encodedState = layers.GaussianNoise(0.25)(encodedState)


'''
pmfFirstEstimate = layers.LocallyConnected1D(64,1,activation=activationFuncChoice)( stackedCoeffs )
pmfFirstEstimate = layers.Dropout(0.5)(pmfFirstEstimate)
pmfFirstEstimate = layers.LocallyConnected1D(32,1,activation=activationFuncChoice)( pmfFirstEstimate )
pmfFirstEstimate = layers.Dropout(0.25)(pmfFirstEstimate)
pmfFirstEstimate = layers.LocallyConnected1D(1,1)( pmfFirstEstimate )
'''


class MatMul(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(MatMul,self).__init__(**kwargs)
    def call(self, inputs):
        return tf.linalg.matmul( inputs[0],inputs[1],transpose_b = True )

coeffMultiplicativeNoiseSDev = 0.05
coeffMultiNoiseRate = coeffMultiplicativeNoiseSDev**2/(1 + coeffMultiplicativeNoiseSDev**2)






#potentialCoeffsUpdated = layers.GaussianNoise(0.1)(potentialCoeffsOnly)
numDirectPot = 8


directDecodeWorkingDim = 256
directDecodeStateDim = 256
directDecode = layers.Dense(directDecodeWorkingDim)(encodedState)
directDecode = TrainableXLog()(directDecode)
directDecode = layers.Dropout(0.3)(directDecode)
directDecode = layers.Dense( directDecodeStateDim  )(directDecode)
directDecode = TrainableXLog()(directDecode)

for i in range(6):
    directDecodeFF = layers.Dense( directDecodeWorkingDim  )(directDecode)
    directDecodeFF = TrainableXLog()(directDecodeFF)
    directDecodeFF = layers.Dropout(0.5)(directDecodeFF)
    directDecodeFF = layers.Dense( directDecodeStateDim)(directDecodeFF)
    directDecodeFF = layers.Dropout(0.25)(directDecodeFF)
    #directDecodeFF = layers.BatchNormalization()(directDecodeFF)
    #directDecodeFF = layers.Dense(128)(directDecodeFF)
    directDecode = directDecode + directDecodeFF


directDecode = layers.Dense(20*numDirectPot)(directDecode)
directDecode = TrainableXLog()(directDecode)
#directDecode = layers.Reshape( (-1,1) )(directDecode)
#directDecode = layers.LocallyConnected1D( 1,1)(directDecode)
directDecode = layers.Reshape( (-1,numDirectPot) )(directDecode)
 


potentialCoeffsOnly = layers.Concatenate()([ potentialCoeffsOnly, directDecode])


numMixedPotentialsFirst = 4
#encodedStateMixing

#potentialCoeffsOnly = layers.Concatenate()([ potentialCoeffsOnly, pmfCorrect])
for i in range(numMixedPotentialsFirst):
    for j in range(3):
        if j == 0:
            mixingMatrix = layers.Dense(128)( encodedState )
            mixingMatrix = TrainableXLog()(mixingMatrix)
        mixingMatrixFF = layers.Dense(128)(mixingMatrix)
        mixingMatrixFF = TrainableXLog()(mixingMatrixFF)
        mixingMatrixFF = layers.Dropout(0.3)(mixingMatrixFF)
        mixingMatrix = mixingMatrix + mixingMatrixFF
    '''
    mixingMatrix = layers.Dense(128)( encodedState )
    mixingMatrix = TrainableXLog()(mixingMatrix)
    mixingMatrix = layers.Dropout(0.25)(mixingMatrix)

    mixingMatrix = layers.Dense(128)(mixingMatrix )
    mixingMatrix = TrainableXLog()(mixingMatrix)
    mixingMatrix = layers.Dropout(0.2)(mixingMatrix)

    mixingMatrix = layers.Dense(64)(mixingMatrix )
    mixingMatrix = TrainableXLog()(mixingMatrix)
    mixingMatrix = layers.Dropout(0.2)(mixingMatrix)
    '''
    mixingMatrix = layers.Dense( 1 * (numDirectPot+numPotentials+i))(mixingMatrix)
    mixingMatrix = layers.GaussianDropout( coeffMultiNoiseRate)(mixingMatrix)
    #mixingMatrix = layers.Dropout(0.5)(mixingMatrix)
    mixingMatrix = TrainableXLog()(mixingMatrix)
    mixingMatrix = layers.Dropout(0.1)(mixingMatrix)
    mixingMatrix = layers.RepeatVector( 20)(mixingMatrix)
    #mixingMatrix = layers.Dropout(0.3)(mixingMatrix)
    #mixingMatrix = layers.Reshape( (20,-1) )(mixingMatrix)
    potentialCoeffsOnlyDrop = layers.Dropout(0.2)(potentialCoeffsOnly)
    newCoeffSet = layers.Multiply()([mixingMatrix,potentialCoeffsOnlyDrop])
    newCoeffSet = layers.Lambda( lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(newCoeffSet)
    newCoeffSetSideMix  = layers.ZeroPadding1D( (0,1) )(newCoeffSet)
    newCoeffSetSideMix  = layers.LocallyConnected1D(1,2)(newCoeffSetSideMix)
    newCoeffSetSideMix = TrainableXLog()(newCoeffSetSideMix)
    newCoeffSetSideMix = layers.Dropout(0.2)(newCoeffSetSideMix)
    newCoeffSet = newCoeffSet + newCoeffSetSideMix
    #newCoeffSet = MatMul()( [mixingMatrix, potentialCoeffsOnly] )
    #newCoeffSet = layers.Dense(1)(newCoeffSet)
    #newCoeffSet = TrainableXLog()(newCoeffSet)
    newCoeffSet = TrainableXLog()(newCoeffSet)
    #newCoeffSet = layers.LocallyConnected1D(1,1)(newCoeffSet)
    newCoeffSet = layers.Flatten()(newCoeffSet)
    newCoeffSet = layers.BatchNormalization()(newCoeffSet)

    newCoeffSet = layers.Reshape( (-1,1))(newCoeffSet)
    potentialCoeffsOnly = layers.Concatenate()([ potentialCoeffsOnly, newCoeffSet])



#directDecode = layers.Dense(16)(encodedState)
#directDecode = TrainableXLog()(directDecode)
#directDecode = layers.Dense(20)(directDecode)
#directDecode = TrainableXLog()(directDecode)


set16 = [2,2,2] #starting from 2 go to 16
set20 = [2,2,2]  #starting from 3 go to 6 - 12 - 24 and then trim 4 after
numDeconvPotentials = 8
deconvPotentialDim = 64
potentialCoeffsOnlyDrop = layers.Dropout(0.2)(potentialCoeffsOnly)
pmfFirstEstimate = layers.LocallyConnected1D(numDeconvPotentials,1)(potentialCoeffsOnlyDrop)
pmfFirstEstimate = TrainableXLog()(pmfFirstEstimate)
#pmfFirstEstimate = layers.LocallyConnected1D( 1,1)(pmfFirstEstimate)

for i in range(numDeconvResidCorrections):
    if i> -1:
        #pmf1stFlat = layers.LocallyConnected1D(2,1)(pmfFirstEstimate)
        #pmf1stFlat = layers.Flatten()(pmf1stFlat)
        #stateAppend = layers.Dense(2 )(pmf1stFlat)
        #stateAppend = TrainableXLog()(stateAppend)
        stateAppend = layers.Flatten()(pmfFirstEstimate)
        decodeInitial = layers.Concatenate()([encodedState , stateAppend])
    else:
        decodeInitial = layers.GaussianNoise(0.05)(encodedState)
    #decodeInitial = layers.Dense(16 )(decodeInitial)
    decodeInitial = layers.Dense(64)(decodeInitial)
    decodeInitial = TrainableXLog()(decodeInitial)
    decodeInitial = layers.Dropout(0.2)(decodeInitial)
    decodeInitial = layers.Dense(64)(decodeInitial)
    decodeInitial = TrainableXLog()(decodeInitial)
    decodeInitial = layers.Dropout(0.2)(decodeInitial)
    decodeConv1 = layers.Dense(deconvPotentialDim *3  )(decodeInitial)
    decodeConv1 = TrainableXLog()(decodeConv1)
    decodeConv1 = layers.Reshape( (3,-1) )(decodeConv1)
    decodeConv1 = layers.GaussianNoise(0.1)(decodeConv1)
    for num in set20:
        decodeConv1T = layers.Conv1DTranspose(deconvPotentialDim,num,strides=num )(decodeConv1)
        decodeConv1T = TrainableXLog()(decodeConv1T)
        #decodeConv1T = layers.ZeroPadding1D( (0,1) )(decodeConv1T)
        decodeConv1T = layers.BatchNormalization()(decodeConv1T)
        decodeConv1T = layers.Dropout(0.2)(decodeConv1T)
        decodeConv1T = layers.ZeroPadding1D( (0, 1) )(decodeConv1T)
        decodeConv1T = layers.LocallyConnected1D(deconvPotentialDim,2)(decodeConv1T)
        decodeConv1 = layers.UpSampling1D(num)(decodeConv1)
        decodeConv1T = layers.Dropout(0.25)(decodeConv1T)
        #decodeConv1 = layers.LocallyConnected1D(potentialDim,1)(decodeConv1)
        decodeConv1 = decodeConv1 + decodeConv1T
        #decodeConv1 = layers.Conv1DTranspose(potentialDim,2,padding="same",activation=activationFuncChoice)(decodeConv1)
    decodeConv1 = layers.Cropping1D( (0,3))(decodeConv1)
    #pmfCorrect = layers.Dense(32,activation=activationFuncChoice)(decodeConv1)
    pmfCorrect = layers.BatchNormalization()(decodeConv1)
    pmfCorrect = layers.Dropout(0.1)(pmfCorrect)
    pmfCorrect =layers.LocallyConnected1D(numDeconvPotentials, 2 )(pmfCorrect)
    #pmfCorrect = layers.BatchNormalization()(pmfCorrect)
    pmfCorrect = TrainableXLog()(pmfCorrect)
    pmfCorrect = layers.Dropout(0.05)(pmfCorrect)
    #pmfCorrect = layers.LocallyConnected1D(1,1 )(pmfCorrect)
    #pmfCorrect = layers.Flatten()(pmfCorrect)
    pmfFirstEstimate = pmfFirstEstimate + pmfCorrect


numMixedPotentials = 7
potentialCoeffsOnly = layers.Concatenate()([ potentialCoeffsOnly, pmfFirstEstimate])
for i in range(numMixedPotentials):
    for j in range(3):
        if j == 0:
            mixingMatrix = layers.Dense(128)( encodedState )
            mixingMatrix = TrainableXLog()(mixingMatrix)
            mixingMatrix = layers.Dropout(0.2)(mixingMatrix)
        mixingMatrixFF = layers.Dense(128)(mixingMatrix)
        mixingMatrixFF = TrainableXLog()(mixingMatrixFF)
        mixingMatrixFF = layers.Dropout(0.3)(mixingMatrixFF)
        mixingMatrix = mixingMatrix + mixingMatrixFF
    '''
    mixingMatrix = layers.Dense(64)(mixingMatrix )
    mixingMatrix = TrainableXLog()(mixingMatrix)
    mixingMatrix = layers.Dropout(0.2)(mixingMatrix)
    
    mixingMatrix = layers.Dense(32)(mixingMatrix )
    mixingMatrix = TrainableXLog()(mixingMatrix)
    mixingMatrix = layers.Dropout(0.2)(mixingMatrix)
    '''
    mixingMatrix = layers.Dense( 1 * (numDirectPot+numPotentials+numMixedPotentialsFirst+numDeconvPotentials+i))(mixingMatrix)
    mixingMatrix = layers.GaussianDropout( coeffMultiNoiseRate)(mixingMatrix)
    mixingMatrix = layers.Dropout(0.25)(mixingMatrix)
    mixingMatrix = TrainableXLog()(mixingMatrix)
    #mixingMatrix = layers.Reshape( (20,-1) )(mixingMatrix)
    mixingMatrix = layers.RepeatVector( 20)(mixingMatrix)
    potentialCoeffsOnlyDrop = layers.Dropout(0.1)(potentialCoeffsOnly)
    newCoeffSet = layers.Multiply()([mixingMatrix,potentialCoeffsOnlyDrop])
    newCoeffSet = layers.Lambda( lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(newCoeffSet)
    
    #newCoeffSetSideMix  = layers.ZeroPadding1D( (0,1) )(newCoeffSet)
    #newCoeffSetSideMix  = layers.LocallyConnected1D(1,2)(newCoeffSetSideMix)
    #newCoeffSetSideMix = TrainableXLog()(newCoeffSetSideMix)
    #newCoeffSet = newCoeffSet + newCoeffSetSideMix    
    
    
    
    #newCoeffSet = MatMul()( [mixingMatrix, potentialCoeffsOnly] )
    #newCoeffSet = layers.Dense(1)(newCoeffSet)
    newCoeffSet = TrainableXLog()(newCoeffSet)

    #newC1 =  SliceLayer( 0, 1)(newCoeffSet)
    #newCoeffSet = layers.BatchNormalization()(newCoeffSet)
    newCoeffSet = layers.Flatten()(newCoeffSet)
    newCoeffSet = layers.BatchNormalization()(newCoeffSet)
    #newCoeffSet = TrainableXLog()(newCoeffSet)
    newCoeffSet = layers.Reshape( (-1,1))(newCoeffSet)
    #newCoeffSet = layers.LocallyConnected1D( 1,1)(newCoeffSet)
    potentialCoeffsOnly = layers.Concatenate()([ potentialCoeffsOnly, newCoeffSet])


#encodedSelect = layers.Dense(16)(encodedState)
#encodedSelect = TrainableXLog()(encodedSelect)
#encodedSelect = layers.Dense(4)(encodedSelect)
#encodedSelect = layers.RepeatVector( numPotentials + numDeconvPotentials + numMixedPotentials  )(encodedSelect)

#potentialCoeffsOnly = layers.Permute( (2,1) )(potentialCoeffsOnly)



#construct the PMF from all the generated potentials
'''
potentialCoeffsOnly = layers.ZeroPadding1D( (0,1) )(potentialCoeffsOnly)
pmfFinalCorrect = layers.LocallyConnected1D(numDeconvPotentials, 2)(potentialCoeffsOnly)
pmfFinalCorrect = TrainableXLog()(pmfFinalCorrect)
pmfFinalCorrect = layers.LocallyConnected1D(numDeconvPotentials, 1)(pmfFinalCorrect)
pmfFinalCorrect = TrainableXLog()(pmfFinalCorrect)
#pmfFirstEstimate = pmfFirstEstimate + pmfFinalCorrect
pmfFirstEstimate = layers.LocallyConnected1D(1,1)(pmfFinalCorrect)
'''

pmfFirstEstimate = layers.LocallyConnected1D(1,1)(potentialCoeffsOnly)
#pmfFirstEstimate = newCoeffSet

#pmfFirstEstimate = layers.LocallyConnected1D(1,2)(potentialCoeffsOnly)
#pmfFirstEstimate = layers.Dense(1,activation=activationFuncChoice)(pmfFirstEstimate)
#pmfFirstEstimate = layers.Flatten()(pmfFirstEstimate)






#pmfEstFinal = layers.Dense(1)(potentialCoeffsUpdated)
#print(pmfEstFinal)
pmfEstFinal = layers.Flatten()(pmfFirstEstimate)
#pmfEstFinal = pmfEstFinal #+ directDecode

#pmfEstFinal = layers.LocallyConnected1D(1,1)(pmfFirstEstimate)
#pmfEstFinal =layers.Flatten()(pmfFirstEstimate)







#pmfEst = layers.GaussianDropout(0.005)(pmfEst)
#pmfEst = layers.GaussianNoise(0.05)(pmfEst)


#tf.matmul(inputs, self.w) + self.b  


#pmfEstFinal = layers.Activation(activationDesatLog)(pmfEstFinal)

#pmfEstFinal = TrainableDesat()(pmfEstFinal)


#tf.math.sign(x) * ( tf.exp(alpha* x*tf.math.sign(x)) - 1)/alpha






#multiply each potential coefficient by 1/sqrt(2*i - 1) to deboost the high-order coeffs, then by 1/sqrt(r0) to bring this back in.
class PMFScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(PMFScaleLayer, self).__init__()
      self.scale = tf.constant( np.array( [np.sqrt(2*i - 1) for i in range(1,1+numCoeffs) ] )  , dtype=tf.float32)
    def call(self, inputs):
      return  tf.math.divide( tf.math.divide(inputs[0] , self.scale) , inputs[1])
pmfEstFinal = PMFScaleLayer()( [pmfEstFinal,sqrtr0])



#final coefficient mixing

finalCoeffMixing = layers.Dense(64)(pmfInputVarLayer)
finalCoeffMixing = TrainableXLog()(finalCoeffMixing)
finalCoeffMixing = layers.Dropout(0.2)(finalCoeffMixing)
finalCoeffMixing = layers.Dense(32)(finalCoeffMixing)
finalCoeffMixing = TrainableXLog()(finalCoeffMixing)
finalCoeffMixing = layers.Dropout(0.5)(finalCoeffMixing)
finalCoeffMixing = layers.Dense(20*20 )(finalCoeffMixing)
finalCoeffMixing = TrainableXLog()(finalCoeffMixing)

finalCoeffMixing = layers.Reshape( (20,20) )(finalCoeffMixing)
pmfEstFinalAdjust = layers.Dot(axes=1)([finalCoeffMixing,pmfEstFinal])



pmfEstFinalAdjust = TrainableXLog()(pmfEstFinalAdjust)
pmfEstFinal = TrainableXLog()(pmfEstFinal)
pmfEstFinalAdjust = layers.Reshape( (-1, 1) )(pmfEstFinalAdjust)
pmfEstFinal = layers.Reshape( (-1,1) )(pmfEstFinal)
pmfEstFinal = layers.Concatenate()([pmfEstFinal,pmfEstFinalAdjust])
pmfEstFinal = layers.LocallyConnected1D( 1,1)(pmfEstFinal)
pmfEstFinal = layers.Flatten()(pmfEstFinal)

#pmfEstFinal =pmfEstFinalAdjust #+ pmfEstFinalAdjust




#potentialCoeffsOnly = CoeffScaleLayer()([potentialCoeffsOnly, sqrtr0])


finalValsNoiseSDev = 0.05
finalValsNoiseRate = finalValsNoiseSDev**2/(1 + finalValsNoiseSDev**2)



#pmfEst = layers.Activation(activationIdentity)(pmfEst)
pmfEstFinal = layers.GaussianDropout(finalValsNoiseRate)(pmfEstFinal)
pmfEstFinal = layers.GaussianNoise(0.05)(pmfEstFinal)


outputValSet = []
for i in range(nMaxOrder):
    finalCoeffOutput = GetValLayer(i)(pmfEstFinal)
    #finalCoeffOutput = layers.Dense(1,  name="A"+str(i+1)+"out" )(finalCoeffOutput)
    finalCoeffOutput = layers.Activation(activationIdentity,  name="A"+str(i+1)+"out" )(finalCoeffOutput)
    outputValSet.append( finalCoeffOutput)

coeffVector = layers.Concatenate()( outputValSet[:] )
#there is an additional constraint on the outputs: sum_i A_i u_i(r = r_0) -> sum_i A_i sqrt(2 i - 1)/sqrt(r_0) should be equal to fittedE0
coeffScaleTerm = tf.constant( np.array( [np.sqrt(2*i - 1) for i in range(1,1+numOutputVars) ] )  , dtype=tf.float32)
class CoeffSumLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(CoeffSumLayer, self).__init__()
      self.scale = tf.constant( np.array( [np.sqrt(2*i - 1) for i in range(1,1+numOutputVars) ] )  , dtype=tf.float32)
    def call(self, inputs):
      return tf.reduce_sum( tf.math.multiply(inputs , self.scale) , axis=1) 


coeffWeightedSum = CoeffSumLayer()( coeffVector)


invsqrtr0 = layers.Lambda( lambda x: tf.math.pow( tf.math.maximum( x, 0.005) , -0.5) )(inputR0)

e0PMFEstimate = layers.Multiply(  name="e0PMFEstimate" )([ invsqrtr0, coeffWeightedSum] )
e0PMFEstimate = layers.Flatten()(e0PMFEstimate)


finalValsNoiseSDev = 0.1 #= fraction relative error
finalValsNoiseRate = finalValsNoiseSDev**2/(1 + finalValsNoiseSDev**2)


e0Estimate = layers.GaussianDropout(finalValsNoiseRate)(e0Estimate)
e0PMFEstimate = layers.GaussianDropout(finalValsNoiseRate)(e0PMFEstimate)
eMinRoughEstimate = layers.GaussianDropout(finalValsNoiseRate)(eMinRoughEstimate)
e0Estimate = layers.GaussianNoise(5,name="E0NoiseOut")(e0Estimate)
eMinRoughEstimate = layers.GaussianNoise(5,name="EMinNoiseOut")(eMinRoughEstimate)
e0PMFEstimate = layers.GaussianNoise(5,name="E0PMFNoiseOut")(e0PMFEstimate)



outputValSet.append(e0Estimate)
outputValSet.append(eMinRoughEstimate )
outputValSet.append(e0PMFEstimate)

#eMinEstimate = PotentialAtPoint()( [rminLocValInput, inputR0,pmfEst])
#outputValSet.append(eMinEstimate)


pmfEstWithR0 = layers.Concatenate()([inputR0,pmfEstFinal])
pmfEstWithR0 = layers.Activation(activationIdentity,name="PMFOut")(pmfEstWithR0)
print( outputValSet )

ann_model = keras.Model(inputs = inputs, outputs=[outputValSet,pmfEstWithR0 ])

ann_model.summary()

summaryFile=open(workingDir+"/summary.txt","w")
ann_model.summary(print_fn = lambda x: summaryFile.write( x + "\n"))
summaryFile.close()
#quit()


initialRate = 0.001

initialRate = 0.0001 #for cosine loss
huberdelta=1
logdir="logs"


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=workingDir+"-tblog",histogram_freq=5)
lrReduceCallbackL = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.5,patience=10,verbose=0,cooldown=20, min_lr = 1e-6, min_delta =0.01 )
initial_learning_rate = 0.1
decay_steps = 1.0
decay_rate = 0.5
learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)

def rminRMSE(y_true,y_pred):
    meanSet=tf.sqrt( tf.reduce_mean(  tf.square( y_true[:,0] - y_pred[:,0]) , axis=-1 ))
    return meanSet
#print(aaVocab)


def a1coeffRMSE(y_true,y_pred):
    meanSet=tf.sqrt( tf.reduce_mean(  tf.square( y_true[:,1] - y_pred[:,1]) , axis=-1 ))
    return meanSet

def coeffRMSE(y_true,y_pred):
    meanSet= tf.reduce_mean(tf.sqrt(tf.reduce_mean(  tf.square( y_true[:,2:] - y_pred[:,2:]) , axis=0 )),axis=0)
    return meanSet


def scaledRMSE(y_true,y_pred):
    return tf.reduce_mean(tf.divide( tf.reduce_mean( tf.square(y_true - y_pred), axis=0) ,  tf.square( tf.add(  tf.math.reduce_std( y_true,axis=0)  ,0.0000001)   ) ))



class BrownianWeights(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        currentWeights = self.model.get_weights()
        for i in range(5,len(currentWeights)):
            #print(currentWeights[i])
            currentWeights[i] = np.multiply(currentWeights[i],0.99) + np.random.normal(0, 0.000005, size = (currentWeights[i].shape) )
            #print(currentWeights[i].shape)
        #currentWeights = np.multiply(0.9 , currentWeights)
        self.model.set_weights(currentWeights)
        
        
class BrownianMultiWeights(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        currentWeights = self.model.get_weights()
        for i in range(5,len(currentWeights)):
            #print(currentWeights[i])
            currentWeights[i] = np.multiply(currentWeights[i],  np.random.normal(0.99, 0.01, size = (currentWeights[i].shape) ) ) + np.random.normal(0, 0.001, size = (currentWeights[i].shape) )
            #print(currentWeights[i].shape)
        #currentWeights = np.multiply(0.9 , currentWeights)
        self.model.set_weights(currentWeights)
        

#plt.ion()
outputFig = plt.figure()
numberOutputPlots = 20
axSet = []
for i in range(numberOutputPlots):
    axSet.append(plt.subplot(5,4,i+1))
#ax2 = plt.subplot(332)


    
class PlotPredictionsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            trainData = [ uniqueDataTrain[aaVarSet]   ]
            testData = [ uniqueDataTest[aaVarSet]   ]
            trainingUOutput = uniqueDataTrain[outputVarset].to_numpy()
            testingUOutput = uniqueDataTest[outputVarset].to_numpy()
            trainPredictions = ( np.array( self.model.predict(trainData)[0]  )[:,:,0] ).T
            testPredictions = ( np.array( self.model.predict(testData)[0]  )[:,:,0] ).T
            trainTotalMSErr = 0
            valTotalMSErr = 0
            for i in range(numberOutputPlots):
                axSet[i].clear()
                a1TrainReal = trainingUOutput[:,i]
                a1TrainPred = trainPredictions[:,i].flatten()
                a1TestReal = testingUOutput[:,i]
                a1TestPred = testPredictions[:,i].flatten()
                trainTotalMSErr += np.sum( ( a1TrainReal - a1TrainPred)**2)
                valTotalMSErr += np.sum( ( a1TestReal - a1TestPred)**2)                
                axSet[i].scatter( a1TrainReal,a1TrainPred)
                axSet[i].scatter( a1TestReal,a1TestPred)
                a1min=min( np.amin(a1TrainReal), np.amin(a1TestReal) ) - 2
                a1max=max( np.amax(a1TrainReal), np.amax(a1TestReal) ) + 2
                a1line = np.linspace( a1min, a1max, 5)
                axSet[i].plot(a1line,a1line,"k:")
                axSet[i].set_xlim(a1min,a1max)
                axSet[i].set_ylim(a1min,a1max)   
                axSet[i].set_xlabel("A"+str(i+1))
                axSet[i].set_ylabel("predA"+str(i+1))
            #plt.pause(0.05)
            plt.tight_layout()
            plt.savefig(workingDir+"/figures/epoch"+str(epoch)+".png" )
            #print("Epoch ", epoch, "Train function RMS: ", np.sqrt(trainTotalMSErr), " val function RMS: ", np.sqrt(valTotalMSErr) )

uniquedataset = dataset.copy()
uniquedataset['ChemValidation'] = 0
uniquedataset['MaterialValidation'] = 0
uniquedataset["R0Dist"] = np.sqrt( (uniquedataset["r0"] - (0.2  + uniquedataset["SurfAlignDist"] ))**2 )
uniquedataset["PMFName"] = uniquedataset["Material"]+"_"+uniquedataset["Chemical"]

uniquedataset.sort_values( by=["R0Dist"] ,ascending=True, inplace=True)
    
if allowMixing == 0:
    uniquedataset.drop_duplicates( subset=['PMFName'] , inplace=True,keep='first')
    uniquedataset.loc[  uniquedataset['Chemical'].isin(validationAA)  ,'ChemValidation'] = 1
    uniquedataset.loc[  uniquedataset['Material'].isin(validationMaterials) , 'MaterialValidation' ] = 1
else:
    uniquedataset.drop_duplicates( subset=['PMFName'] , inplace=True,keep='first')
    uniquedataset.loc[ uniquedataset['pmfname'].isin(validationPMFs) , 'ChemValidation'] = 1
    uniquedataset.loc[ uniquedataset['pmfname'].isin(validationPMFs) , 'MaterialValidation'] = 1

uniqueDataTest = uniquedataset[ (uniquedataset['ChemValidation']==1) | (uniquedataset['MaterialValidation']==1) ]
uniqueDataTrain = uniquedataset[ (uniquedataset['ChemValidation']==0) & (uniquedataset['MaterialValidation']==0) ]
print("Unique dataset values: ", len(uniquedataset))
#print("Generating regularised set (E0 = "+str(E0TargetVal)+")")
#uniquedataset["fittedE0"] = E0TargetVal


lseParam = 100.0
uniquedataset["originalr0"] = uniquedataset["r0"]
#uniquedataset["r0"] = 1/lseParam * np.log( np.exp(lseParam*uniquedataset["ChemLJR0"]) + np.exp(lseParam*uniquedataset["SurfCProbeR0"]))

    
    


def singleStepPredict(model,inputdataset):
    modelPredictOut = model.predict(  [ inputdataset[aaVarSet]   ]   )
    print(len(modelPredictOut))
    print(len(modelPredictOut[0]))
    print(len(modelPredictOut[0][1]))
    predictionSet = ( np.array( modelPredictOut[0] )[:,:,0] ).T
    for i in range(len(outputVarset)):
        inputdataset[ outputVarset[i]+"_regpredict" ] =predictionSet[:,i].flatten()  
    return inputdataset
    

def recurrentPredict(model, dataset, initialR0 = 0.1, targetE0 = 25):
    workingDataset = dataset.copy()
    currentr0 = initialR0
    workingDataset["r0"] = currentr0
    workingDataset["lastE0"] =targetE0*2
    #nonconvergedSet = workingDataset[   workingDataset["lastE0"] > targetE0]
    nonconvergedMask = workingDataset["lastE0"] > targetE0
    print(len(workingDataset[nonconvergedMask]))
    while len( workingDataset[nonconvergedMask] ) > 0 and currentr0 < 1:
        #print(  currentr0)
        workingDataset.loc[nonconvergedMask,"r0"] = currentr0
        workingDataset.loc[nonconvergedMask, "lastE0" ] = 0
        predictionSet = ( np.array( model.predict([ workingDataset.loc[ nonconvergedMask,  aaVarSet] ]))[:,:,0] ).T
        for i in range(len(outputVarset)):
            workingDataset.loc[nonconvergedMask,  outputVarset[i]+"_regpredict" ] =predictionSet[:,i].flatten()  
            if i<nMaxOrder:
                workingDataset.loc[ nonconvergedMask,   "lastE0" ] = workingDataset.loc[ nonconvergedMask,   "lastE0" ] + np.sqrt(2*(i+1) - 1) * (predictionSet[:,i].flatten() ) /np.sqrt(currentr0)
        #print( workingDataset.loc[nonconvergedMask,   "lastE0"] , predictionSet[:,-1])
        #workingDataset.update( nonconvergedSet )        
        nonconvergedMask = workingDataset["lastE0"] > targetE0
        currentr0 = currentr0+0.01
    return workingDataset


#uniquepredictions = ( np.array( ann_model.predict([ uniquedataset[aaVarSet] ]))[:,:,0] ).T
#for i in range(len(outputVarset)):
#    uniquedataset[ outputVarset[i]+"_regpredict" ] =uniquepredictions[:,i].flatten()
uniquedatasetPredict = singleStepPredict( ann_model, uniquedataset)
uniquedatasetPredict.to_csv(workingDir+"/uniquepredictions.csv")

class SavePredictionsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            #trainData = [ train_dataset[aaVarSet]   ]
            #testData = [ test_dataset[aaVarSet]   ]
            #trainPredictions = ( np.array( self.model.predict(trainData))[:,:,0] ).T
            #testPredictions = ( np.array( self.model.predict(testData))[:,:,0] ).T
            #uniquepredictions = ( np.array( ann_model.predict([ uniquedataset[aaVarSet] ]))[:,:,0] ).T
            #for i in range(len(outputVarset)):
            #    uniquedataset[ outputVarset[i]+"_regpredict" ] =uniquepredictions[:,i].flatten()
            uniquedatasetPredict = singleStepPredict( ann_model, uniquedataset)
            uniquedatasetPredict.to_csv(workingDir+"/uniquepredictions_"+str(epoch)+".csv")


def sawtoothLR(epoch, lr):
    if epoch < 50:
        return lr
    else:
        newLR = lr
        if epoch % 50 == 0:
            newLR = lr * 0.75
        if newLR < 1e-5:
            newLR = 5e-4
        return newLR

def expexpLR(epoch, lr):
    return 5e-4 * np.exp( np.exp( np.sin( epoch/25.0 ) ) )/50.0



plotPredCallback = PlotPredictionsCallback()

sawtoothLRSchedule = tf.keras.callbacks.LearningRateScheduler(sawtoothLR)
expexpsinLRSchedule = tf.keras.callbacks.LearningRateScheduler(expexpLR)

savePredCallback = SavePredictionsCallback()


bwCallback = BrownianMultiWeights()
csv_logger = keras.callbacks.CSVLogger("../Dropbox/PMFLogs/"+filetag+'.log')
csv_loggerLocal = keras.callbacks.CSVLogger(workingDir+'/training.log')

cosineLoss = keras.losses.CosineSimilarity(axis=-1)

checkpointCallback = keras.callbacks.ModelCheckpoint(workingDir+"/checkpoints/checkpoint-val", save_best_only=True)
checkpointCallbackTrain = keras.callbacks.ModelCheckpoint(workingDir+"/checkpoints/checkpoint-train", monitor="loss",  save_best_only=True)


huberLoss = tf.keras.losses.Huber(delta=1)

huberLossSet = []



numOutputVars += 1
outputVarset.append("fittedE0")



numOutputVars += 1
outputVarset.append("EMin")


#estimate of E0 from PMF
numOutputVars += 1
outputVarset.append("fittedE0")



trainingOutput = train_dataset[outputVarset].to_numpy()
valOutputData = test_dataset[outputVarset].to_numpy()


print(outputVarset)
print(numOutputVars)
outputDataSets = []
outputValSets = []



@tf.keras.utils.register_keras_serializable()
def KLgetHGValv2(x):
    x = tf.convert_to_tensor(  tf.cast( x ,dtype=tf.float64)   , dtype=tf.float64)
    indexSet = tf.range(1,nMaxOrder +1,dtype=tf.float64)
    xm,hgFuncNum,hgInnerIndex = tf.meshgrid(x, indexSet,indexSet)
    mask =  tf.cast( tf.where( hgInnerIndex <= hgFuncNum, 1.0, 0.0 ) ,dtype=tf.float64) 
    hgInnerIndex = tf.cast( tf.where( hgInnerIndex <= hgFuncNum, hgInnerIndex, 1.0) ,dtype=tf.float64) 
    prefactors2 = tf.math.pow( tf.cast( -1.0, tf.float64), hgInnerIndex-1.0) *  tf.exp(tf.math.lgamma( (hgFuncNum + hgInnerIndex - 1.0)   ) - tf.math.lgamma( 1.0 - hgInnerIndex + hgFuncNum) - 2* tf.math.lgamma(hgInnerIndex)) 
    prefactors1 =  tf.math.pow( xm, hgInnerIndex - 1.0) 
    funcVals = tf.reduce_sum( mask*prefactors1*prefactors2, axis=-1)
    return funcVals


@tf.keras.utils.register_keras_serializable()
def KLcalcHGERange(r, expansionDef):
    r = tf.convert_to_tensor(   tf.cast(r ,dtype=tf.float64)   ,dtype=tf.float64)
    expansionDef = tf.convert_to_tensor(tf.cast( expansionDef ,dtype=tf.float64) ,dtype=tf.float64) 
    r0 = expansionDef[0]
    coeffs = expansionDef[1:]   
    indexSet  = tf.range(1, nMaxOrder +1,dtype=tf.float64)
    #scaledR = tf.math.divide_no_nan(r0,r)
    scaledR = tf.where( r > r0, tf.math.divide_no_nan(r0,r), 0.99)
    #scaledR = tf.where( scaledR  < 1, scaledR, 0.99)
    hg2F1Set = tf.transpose( KLgetHGValv2( scaledR)  )
    prefactorTerm1= tf.math.pow(  tf.cast( -1.0, tf.float64), indexSet+1.0) * tf.math.sqrt( 2*indexSet - 1.0) #has size nmaxorder
    scaledR2=tf.math.divide( tf.math.sqrt(r0) , r) #has size num_positions
    pf1Boost, sR2Boost = tf.meshgrid( prefactorTerm1, scaledR2)
    prefactors = pf1Boost * hg2F1Set * sR2Boost
    funcVals =  tf.math.reduce_sum( tf.math.multiply_no_nan(prefactors,coeffs)  ,axis=1 )
    #funcVals = tf.where(tf.math.is_nan(funcVals), 50.0, funcVals)
    rmaskedFunc = tf.where(r >= r0, funcVals, 50.0)
    return tf.cast( rmaskedFunc, tf.float64)
    
@tf.keras.utils.register_keras_serializable()
def KLgetHGValv3(x):
    x = tf.convert_to_tensor(  tf.cast( x ,dtype=tf.float64)   , dtype=tf.float64)
    indexSet = tf.range(1,nMaxOrder +1,dtype=tf.float64)
    xm,hgFuncNum,hgInnerIndex = tf.meshgrid(x, indexSet,indexSet)
    mask =  tf.cast( tf.where( hgInnerIndex <= hgFuncNum, 1.0, 0.0 ) ,dtype=tf.float64) 
    hgInnerIndex = tf.cast( tf.where( hgInnerIndex <= hgFuncNum, hgInnerIndex, 1.0) ,dtype=tf.float64) 
    prefactors2 = tf.math.pow( tf.cast( -1.0, tf.float64), hgInnerIndex-1.0) *  tf.exp(tf.math.lgamma( (hgFuncNum + hgInnerIndex - 1.0)   ) - tf.math.lgamma( 1.0 - hgInnerIndex + hgFuncNum) - 2* tf.math.lgamma(hgInnerIndex)) 
    prefactors1 =  tf.math.pow( xm, hgInnerIndex - 1.0) 
    funcVals =  tf.reduce_sum( mask*prefactors1*prefactors2, axis=-1)
    return   funcVals  


@tf.keras.utils.register_keras_serializable()
def KLcalcHGERangeMulti(r, expansionDefs):
    r = tf.convert_to_tensor(   tf.cast(r ,dtype=tf.float64)   ,dtype=tf.float64) #1.5 / delta r 
    expansionDefs = tf.reshape(  tf.convert_to_tensor(tf.cast( expansionDefs ,dtype=tf.float64) ,dtype=tf.float64) , (-1, nMaxOrder + 1) )
    r0 = expansionDefs[:,0] # batch size  
    coeffs = expansionDefs[:,1:]   #batch size x expansion size
    indexSet  = tf.range(1, nMaxOrder +1,dtype=tf.float64) #expansion size
    r0RangeSet, rRangeSet   = tf.meshgrid( r0, r ) 
    #rRangeSet2, r0RangeSet2, indexSet2   = tf.meshgrid(  r,r0,indexSet ) #    batchSize x numPoints x expansion order    
    #tf.print(r0RangeSet2.shape)
    #scaledR = tf.math.divide_no_nan(r0,r)
    scaledRSet = tf.where( rRangeSet > r0RangeSet, tf.math.divide_no_nan(r0RangeSet,rRangeSet), 0.999)
    scaledRSetShape = tf.shape(scaledRSet)
    hgXIn = tf.reshape( tf.transpose( scaledRSet ),(-1,))
    #scaledR = tf.where( scaledR  < 1, scaledR, 0.99)
    #hg2F1Set = tf.transpose( KLgetHGValv3( scaledRSet)  ) #batch_size x 
    
    hg2F1Set =KLgetHGValv3( hgXIn ) #this gives HGEOrder on the outer dimension, batchi-xj on the inner dimension
    hg2F1Set =  tf.transpose( hg2F1Set) #now we have batchi:xj on the outer, HGE order on the inner
    hg2F1Set = tf.reshape(hg2F1Set , (-1,  len(r) ,nMaxOrder) ) #batch examples x number points x expansion size
    rRangeSet2, r0RangeSet2, indexSet2   = tf.meshgrid(  r,r0,indexSet ) #    batchSize x numPoints x expansion order  
    prefactorTerm1= tf.math.pow(  tf.cast( -1.0, tf.float64), indexSet2 +1.0) * tf.math.sqrt( 2*indexSet2 - 1.0) #has size nmaxorder
    scaledR2=tf.math.divide( tf.math.sqrt(r0RangeSet2) , rRangeSet2) #has size num_positions

    prefactors = prefactorTerm1 * hg2F1Set * scaledR2
    coeffsExpand =tf.expand_dims(coeffs, 1)
    #pf1Boost, sR2Boost = tf.meshgrid( prefactorTerm1, scaledR2)
    #prefactors = pf1Boost * hg2F1Set * sR2Boost
    funcVals =  tf.math.reduce_sum( tf.math.multiply_no_nan(prefactors,coeffsExpand)  ,axis=-1 )
    #funcVals = tf.where(tf.math.is_nan(funcVals), 50.0, funcVals)
    rmaskedFunc = tf.transpose(tf.where(rRangeSet >= r0RangeSet, tf.transpose(funcVals), 50.0))
    return tf.cast( rmaskedFunc, tf.float64)


@tf.keras.utils.register_keras_serializable()
def potentialKLLoss(y_true,y_pred):
    y_true = tf.cast( tf.reshape( tf.convert_to_tensor(y_true) , (-1, nMaxOrder+1) )  ,dtype=tf.float64)
    y_pred = tf.cast( tf.reshape( tf.convert_to_tensor(y_pred), (-1, nMaxOrder+1) ),dtype=tf.float64)
    deltar = 0.005
    rrange = tf.cast( tf.range( 0.1, 1.5, deltar)   ,dtype=tf.float64)
    #r0set = y_true[:, 0]
    #rrangeBoost, r0MaskBoost = tf.meshgrid(rrange, r0set)
    #maskVals = tf.where(rrangeBoost > r0MaskBoost, 1.0,0.0)

    calcHGESet = lambda c: KLcalcHGERange(rrange,c)
     #    * maskVals
    #hgePotsTrue = KLcalcHGERangeMulti( rrange, y_true)
    #hgePotsPred = KLcalcHGERangeMulti(rrange, y_pred)
    #hgePotsTrue = tf.vectorized_map( calcHGESet, y_true)
    #tf.print(hgePotsTrue)
    # * maskVals
    #hgePotsPred = tf.vectorized_map( calcHGESet, y_pred)
    hgePotsPred = tf.map_fn( calcHGESet, y_pred) 
    hgePotsTrue = tf.map_fn( calcHGESet, y_true)
    #tf.print(hgePotsPred)
    #quit()
    rmaxVal =  tf.cast(1.5, tf.float64)
    kbTVal = tf.cast(1.0, tf.float64)
    maxPot = tf.cast(50.0, tf.float64)
    minPot = tf.cast(-75.0, tf.float64)   
    minWeightVal = tf.cast(1e-6, tf.float64) 
    hgePotsTrue = tf.where( tf.math.is_nan(hgePotsTrue), maxPot, hgePotsTrue)
    hgePotsPred = tf.where( tf.math.is_nan(hgePotsPred), maxPot, hgePotsPred)
    hgePotsTrue = tf.where( hgePotsTrue > maxPot, maxPot, hgePotsTrue)
    hgePotsTrue = tf.where( hgePotsTrue < minPot,minPot, hgePotsTrue)
    hgePotsPred = tf.where( hgePotsPred > maxPot,maxPot, hgePotsPred)
    hgePotsPred = tf.where( hgePotsPred < minPot, minPot, hgePotsPred)
    eadsTrueSafe = tf.reduce_min(hgePotsTrue, keepdims=True, axis=-1)
    eadsPredSafe = tf.reduce_min(hgePotsPred, keepdims=True, axis= -1)
    eadsTrue = - kbTVal * tf.math.log( tf.math.reduce_sum( deltar* tf.math.exp(- hgePotsTrue/kbTVal) ,axis = -1, keepdims=True )  /rmaxVal )
    eadsPred = - kbTVal * tf.math.log( tf.math.reduce_sum(deltar*  tf.math.exp(- hgePotsPred/kbTVal) ,axis = -1, keepdims=True )  /rmaxVal )    
    eadsTrue =    tf.where(  tf.math.is_finite( eadsTrue), eadsTrue, eadsTrueSafe)
    eadsPred = tf.where (tf.math.is_finite(eadsPred), eadsPred, eadsPredSafe)
    hgePotsTrueShift = eadsTrue - hgePotsTrue
    hgePotsPredShift = eadsPred - hgePotsPred
    
    weightFunc =  tf.math.exp( (hgePotsTrueShift)/kbTVal)/(kbTVal*rmaxVal)
    weightFunc = tf.where( tf.math.is_finite(weightFunc), weightFunc,1e-6) 
    #eadsTrue  = tf.where( eadsTrue < -75, -75, eadsTrue)
    #eadsPred = tf.where( eadsPred < -75, -75, eadsPred)
    #eadsTrue = tf.where( eadsTrue > 50, 50, eadsTrue)
    #eadsPred = tf.where(eadsPred > 50, 50, eadsPred)
    #weightFunc = tf.where(tf.math.is_nan(weightFunc), 1e-12, weightFunc)
    weightFunc = tf.where(weightFunc < minWeightVal, minWeightVal, weightFunc)
    klLossTerms = tf.reduce_sum( deltar* weightFunc  * (hgePotsTrueShift  - hgePotsPredShift), axis=-1)
    #klLossTerms = eadsTrue - eadsPred
    klLoss = tf.cast( tf.reduce_mean(tf.math.pow(klLossTerms,2)     +tf.reduce_mean( tf.math.pow( eadsTrue - eadsPred  ,2)    )        ) ,dtype=tf.float32)
    return klLoss




#noisevectorout = np.copy(vectorOut[:5])
#noisevectorout[:,1:] = noisevectorout[:,1:]*1.1
#vectorOut[:5,1:] 
#potentialKLLoss( vectorOut[:5] ,noisevectorout)




@tf.keras.utils.register_keras_serializable()
def scaledMAE(scaleVal):
    def loss(y_true,y_pred):
        return tf.keras.losses.mean_absolute_error( (y_true + 0.0 * y_true * scaleVal)/scaleVal, (  y_pred + 0.0 * y_pred * scaleVal)/scaleVal) 
    return loss


@tf.keras.utils.register_keras_serializable()
def scaledMSE(scaleVal):
    def loss(y_true,y_pred):
        return tf.keras.losses.mean_squared_error( (y_true + 0.0 * y_true * scaleVal)/scaleVal, (y_pred + 0.0 * y_pred * scaleVal)/scaleVal) 
    return loss
    
def scaledMSESaturating(scaleVal):
    def loss(y_true,y_pred):
        return tf.math.log1p(     tf.keras.losses.mean_squared_error( y_true/scaleVal, y_pred/scaleVal)   )
    return loss
    

def scaledMSESDev(scaleVal):
    def loss(y_true,y_pred):
        return tf.keras.losses.mean_squared_error( y_true/scaleVal, y_pred/scaleVal) + tf.math.squared_difference( tf.math.reduce_std(y_true/scaleVal) , tf.math.reduce_std(y_pred/scaleVal))
    return loss


def scaledMSEBlend(scaleVal,alpha):
    def loss(y_true,y_pred):
        return (1-alpha)*tf.keras.losses.mean_squared_error( y_true/scaleVal, y_pred/scaleVal) + alpha*tf.keras.losses.mean_squared_error( y_true, y_pred )
    return loss

'''
for i in range(numOutputVars):
    print( outputVarset[i],   i+1,  np.std(trainingOutput[:,i] ))
    scaleVal = np.std(trainingOutput[:,0]) 
    if outputVarset[i] == "fittedE0":
        #scaleVal = scaleVal*5
        huberLossSet.append( scaledMSE(np.std(trainingOutput[:,i]) ) )
    elif outputVarset[i]=="EMin":
        huberLossSet.append( scaledMSE(   np.std(trainingOutput[:,i])     ) )
    else:
        huberLossSet.append(  scaledMAE( 1 ) )
    #huberLossSet.append(  tf.keras.losses.Huber(delta = scaleVal  )  )
    #huberLossSet.append(  tf.keras.losses.MeanSquaredError(  )    )
    outputDataSets.append( trainingOutput[:,i] )
    outputValSets.append( valOutputData[:,i] )
'''

#trainingOutput = train_dataset[outputVarset].to_numpy()
#valOutputData = test_dataset[outputVarset].to_numpy()


#print(huberLossSet)
lossWeightList = []
for i in range(numOutputVars):
    print( outputVarset[i],   i+1,  np.std(trainingOutput[:,i] ))
    if i < 20:
        scaleVal = np.std(trainingOutput[:,i] )
        huberLossSet.append(  scaledMSE(scaleVal)  )
        lossWeightList.append( 0.001)
    else:
        huberLossSet.append(  scaledMSE( np.std(trainingOutput[:,i] ) ) )
        lossWeightList.append(1.0)
    #huberLossSet.append(  tf.keras.losses.Huber(delta = scaleVal  )  )
    #huberLossSet.append(  tf.keras.losses.MeanSquaredError(  )    )
    outputDataSets.append( trainingOutput[:,i] )
    outputValSets.append( valOutputData[:,i] )


vectorOut = []
vectorValOut  = []
vectorOut.append( train_dataset["r0"].to_numpy()   )
vectorValOut.append( test_dataset["r0"].to_numpy() )
#vectorOut = train_dataset["r0"].to_numpy()
#vectorValOut = test_dataset["r0"].to_numpy()

for i in range(nMaxOrder):
    vectorOut.append ( trainingOutput[:,i] )
    vectorValOut.append( valOutputData[:,i] )
    #vectorOut = np.concatenate( (vectorOut, trainingOutput[:,i]),axis=-1)
    #vectorValOut = np.concatenate( (vectorValOut, valOutputData[:,i]),axis=-1)
#this produces 1+20 rows with NTrainingPoints columns, remap to training points x (1+20) then to num examples x 1 x (1+20)
vectorOut = np.reshape( np.transpose( vectorOut ), (-1,  1 + nMaxOrder))
vectorValOut = np.reshape( np.transpose( vectorValOut), (-1, 1+ nMaxOrder))

lossWeightList.append(10.0)
lossWeights =np.array(lossWeightList)

if allowMixing == 0:
    print("Training AA: ", trainingAA)
    print("Validation AA: ", validationAA)
    print("Training Materials: ", trainingMaterials)
    print("Validation Materials: ", validationMaterials)




outvarsetOutputFile=open(workingDir+"/outputvarset.txt","w")
outvarsetOutputFile.write( ",".join(outputVarset))
#for inputVar in aaVarSet:
#    varsetOutputFile.write(inputVar+",")
outvarsetOutputFile.close()
'''

#sample weighting based on A1

a1vals= train_dataset["A1"].to_numpy()
binDensities,binEdges = np.histogram( a1vals ,20,density=True) #generate bins and densities
bindexs =  np.digitize(a1vals, binEdges[:-1]) # assign each training value to a bin
truncatedBindex=np.where( bindexs < 20, bindexs, 19)
probs=binDensities[truncatedBindex]
weightsUnnormed = 1.0/(0.03 + probs) #setting the parameter to higher values reduces the weighting of extreme A1 values, restoring uniform weights for large numbers.
weightsNormed = (weightsUnnormed / np.sum(weightsUnnormed))
weightsNormed = weightsNormed/np.amin(weightsNormed)
'''



#sample weighting by coefficient clustering and then multiplicatively by the ssd type, source, shape

pmfAgglomCluster = skcluster.AgglomerativeClustering(n_clusters =30)
pmfClusterVars =   outputVarset[:11] + outputVarset[21:] + ["r0","resolution"]



totalPMFs = len( train_dataset )
#weightsUnnormed = np.ones_like( train_dataset["A1"].to_numpy() )


pmfAgglomCluster.fit( skpreproc.normalize( uniquedataset[ pmfClusterVars  ].values))
pmfClusterLabels =  pmfAgglomCluster.labels_

train_dataset["PMFName"] = train_dataset["Material"]+"_"+train_dataset["Chemical"]
train_dataset["clusterSize"] = 0

pmfClusterFile = open(workingDir+"/pmfclusters.txt","w")
for i in range(max(pmfAgglomCluster.labels_) + 1):
    clusterMembers = uniquedataset[ pmfClusterLabels == i]
    #train_dataset["PMFName"] = train_dataset["Material"]+"_"+train_dataset["Chemical"]
    clusterPMFNames = uniquedataset.loc[ pmfClusterLabels == i, "PMFName"]
    #outline = str(i) + ",".join(clusterPMFNames)
    pmfClusterFile.write(str(i)+"\n")
    for pmfName in clusterPMFNames:
        pmfClusterFile.write(pmfName+"\n")
    train_dataset.loc[   train_dataset["PMFName"].isin(clusterPMFNames)  , "clusterSize"] = len(clusterMembers)
    #randomMember = random.choice(clusterMembers)
    print("Cluster", i, " size: ", len(clusterMembers))
weightsUnnormed = 1.0/(1 + train_dataset["clusterSize"].to_numpy() )
weightsNormed = weightsUnnormed / np.sum(weightsUnnormed)

for boolVar in ["source", "numericShape", "ssdType"]:
    uniqueBoolVarVals = list(set(uniquedataset[boolVar].values))
    print(boolVar, uniqueBoolVarVals)
    for ubvv in uniqueBoolVarVals:
        groupMemberMask =  uniquedataset[boolVar] == ubvv    
        groupMembers = uniquedataset.loc[ groupMemberMask, "PMFName"]
        train_dataset.loc[ train_dataset["PMFName"].isin(groupMembers) , boolVar+"size"] = len(groupMembers)
        print( boolVar," == ", ubvv, len(groupMembers) , "counts")
    weightFactor = 1.0/(1 + train_dataset[boolVar+"size"].to_numpy() )
    weightFactor = weightFactor / np.sum(weightFactor)
    weightsNormed = weightsNormed * weightFactor

#target variable, scale, power
targetVarWeighting = [ ["EMin" , 1.0, 1.0] , ["fittedE0", 0.1, 0.5] ]
for targetVar in targetVarWeighting:
    numericTargetVar = targetVar[0]
    varVals = train_dataset[numericTargetVar].to_numpy()
    varmean=np.mean(varVals)
    varstdev=np.std(varVals)
    vardist =  1 +  targetVar[1] *(  (((varVals - varmean)/varstdev)**2)   **  targetVar[2] )
    weightsNormed = weightsNormed * vardist
    print(numericTargetVar, varmean, varstdev)

#final normalisation such that the smallest weight is scaled to 1
weightsNormed =  ( weightsNormed/np.amin(weightsNormed) )**(1.0/3.0)
#weightsNormed[:] = 1
pmfClusterFile.close()







'''



normedSqDists = np.zeros_like( train_dataset["A1"].to_numpy() )
coeffA1TrainingVals = train_dataset["A1"].to_numpy()
coeffA1Scale = np.std(coeffA1TrainingVals)
numWeightParams = 0

for i in range(1,17):
    coeffTrainingVals = train_dataset["A"+str(i)].to_numpy()
    coeffScale = np.std(coeffTrainingVals)
    normedSqDist = ((coeffTrainingVals - np.mean(coeffTrainingVals) )/ coeffScale)**2
    coeffWeightFactor = 1 #  np.sqrt(coeffA1Scale/coeffScale)
    print(i, coeffScale, coeffWeightFactor)
    normedSqDists = normedSqDists + normedSqDist * coeffWeightFactor
    numWeightParams +=1


for inputParam in  aaVarSet:
    coeffTrainingVals = train_dataset[inputParam].to_numpy()
    coeffScale = np.std(coeffTrainingVals)
    normedSqDist = ((coeffTrainingVals - np.mean(coeffTrainingVals) )/ coeffScale)**2
    normedSqDists += normedSqDist
    numWeightParams+=1

normedDist = np.sqrt(normedSqDists)/numWeightParams



binDensities,binEdges = np.histogram( normedDist ,20,density=True) #generate bins and densities
bindexs =  np.digitize(normedDist, binEdges[:-1]) # assign each training value to a bin
truncatedBindex=np.where( bindexs < 20, bindexs, 19)
probs=binDensities[truncatedBindex]
weightsUnnormed = 1.0/(0.01 + probs) #setting the parameter to higher values reduces the weighting of extreme A1 values, restoring uniform weights for large numbers.
weightsNormed = (weightsUnnormed / np.sum(weightsUnnormed))
weightsNormed = weightsNormed/np.amin(weightsNormed)

weightsNormed = 1 +  normedDist #normalise by number of parameters sampled
'''
#weightsNormed = weightsNormed + train_dataset["source"]*5.0

print(weightsNormed)
#print(a1vals)
print("Max weight: ", np.amax(weightsNormed))
#was 0.002 for may11 config

'''
if convToOutput == 1:
    #preprocessLayer = layers.Lambda(   lambda x : tf.stop_gradient(x)    )(finalConvLayer)
    huberLossSet = huberLossSet + huberLossSet
    lossWeights = lossWeights + lossWeights
'''
#for scaled MSE LR around 1e-4 -- 5e-4 seems to work ok
#unscaled MSE will have much larger weights so reduce this proportioniately

#was cosineLoss

#all callbacks:
#[reduce_lr,savePredCallback,checkpointCallback ,checkpointCallbackTrain, tensorboard_callback, csv_logger,plotPredCallback,csv_loggerLocal] 
print("Starting: " , filetag )
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, cooldown=10, min_delta = 1)
with tf.device('/cpu:0'):
    ann_model.compile( optimizer = tf.optimizers.Adam(learning_rate=7e-4, epsilon=1e-3 ), loss=[huberLossSet, potentialKLLoss],   loss_weights=lossWeights, metrics=[
        tf.keras.metrics.RootMeanSquaredError() ] )
    history = ann_model.fit([ train_dataset[aaVarSet]   ], [outputDataSets, vectorOut     ], epochs=numEpochs,verbose=1,   sample_weight = weightsNormed,batch_size=64,
        validation_data=([ test_dataset[aaVarSet]], [outputValSets, vectorValOut] )  , callbacks = [  reduce_lr,savePredCallback,checkpointCallback ,checkpointCallbackTrain, tensorboard_callback, csv_logger,plotPredCallback,csv_loggerLocal] 
    )




#save final model

ann_model.save(workingDir+"/final_model/")

'''
fulldataset = datasetAll.copy()
fullpredictions = ( np.array( ann_model.predict([ fulldataset[aaVarSet] ]))[0,:,:,0] ).T

fulldataset['ChemValidation'] = 0
fulldataset.loc[  fulldataset['Chemical'].isin(validationAA)  ,'ChemValidation'] = 1
fulldataset['MaterialValidation'] = 0
fulldataset.loc[  fulldataset['Material'].isin(validationMaterials) , 'MaterialValidation' ] = 1
for i in range(len(outputVarset)):
    fulldataset[ outputVarset[i]+"_fullpredict" ] =fullpredictions[:,i].flatten()

fulldataset.to_csv(workingDir+"/predictions.csv")

realValues = (fulldataset[outputVarset]).to_numpy()
errorSetFull = np.sqrt( np.mean( (realValues - fullpredictions)**2   , axis=0))
print("Unshuffled: ", errorSetFull)


numShuffleTrials = 2

print("Importance: shuffle one")
'''

