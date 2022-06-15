import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
import os
import shutil

@tf.keras.utils.register_keras_serializable()

def activationXLog(x):
    absx = tf.math.abs(x)
    return tf.math.sign(x) * tf.math.log1p( absx )

def activationSoftPlusSmall(x):
    return   tf.math.softplus(2*x)/2
    
    
@tf.keras.utils.register_keras_serializable()
def activationIdentity(x):
    return x

datasetAll= pd.read_csv("Datasets/TrainingData.csv")
filetag = "pmfpredict-june15-lowerepsilon-moreconvT-lossweights"





#datasetAll["A1CoeffEst"] = -2 * np.sqrt( - datasetAll[ "FLJ(FLJMin)"]* datasetAll["potentialScaleFactor"]) + 5


parser = argparse.ArgumentParser(description="Parameters for PMFPredictor")
parser.add_argument("-i","--ensembleID", type=int,default=0, help="ID number for this ensemble member, set to 0 for development split")
parser.add_argument("-b","--bootstrap", type=int,default=0,help="If non-zero switches the sampling to bootstrapping")
parser.add_argument("-r","--predr", type=int,default=1,help="If non-zero trains a network which predicts r0 given an initial energy, if zero r0 is used as an input")
args = parser.parse_args()
E0TargetVal = 10 #1 kbT = 2.5 kJ mol
numEpochs = 2000
random.seed(1648113195 + args.ensembleID) #epoch time at development start with the ensembleID used as an offset - this way each member gets a different split, but can be recreated.

#filter out inputs which are likely to cause problems during fitting
absErrorThreshold = 10
relErrThreshold = 0.2

initialSamples = len(datasetAll)
datasetAll=datasetAll.drop( datasetAll[(datasetAll["TargetE0"] - datasetAll["fittedE0"] ) > absErrorThreshold].index )
firstReduction = len(datasetAll)

secondReductionVals=np.abs( datasetAll["TargetE0"] - datasetAll["fittedE0"])/(np.abs(datasetAll["TargetE0"]) + np.abs(datasetAll["fittedE0"]) + 0.001) > relErrThreshold
print(secondReductionVals)
datasetAll=datasetAll.drop( datasetAll[ secondReductionVals].index)
secondReduction = len(datasetAll)
print("Initial: ", initialSamples, " absolute filter: ", firstReduction, "relative filter", secondReduction)


for i in range(1,17):
    datasetAll["roughA"+str(i)] = datasetAll["A"+str(i)]



if args.predr != 0:
    filetag = filetag+"-predictr0"
else:
    filetag = filetag+"-predictE0"
    print("Predicting E0 from r0")





if args.bootstrap == 0:
    dataset = datasetAll.copy()
else:
    filetag = filetag+"-bootstrapped"
    dataset = datasetAll.sample(frac=1,replace=True,random_state=1+args.ensembleID)

if args.ensembleID>0:
    filetag = filetag+"-"+str(args.ensembleID)
    numEpochs = 200

#build the folders for logging, outputs, metadata
os.makedirs(filetag, exist_ok=True)
os.makedirs(filetag+"/checkpoints", exist_ok=True)
os.makedirs(filetag+"/final_model", exist_ok=True)
os.makedirs(filetag+"/figures", exist_ok=True)
localName = os.path.basename(__file__)
shutil.copyfile(localName, filetag+"/"+localName)

'''
SurfLJR0,SurfLJC1,SurfLJC2,SurfLJC3,SurfLJC4,SurfLJC5,SurfLJC6,SurfLJC7,SurfLJC8,SurfLJC9,SurfLJC10,SurfLJC11,SurfLJC12,SurfLJC13,SurfLJC14,SurfLJC15,SurfLJC16,SurfLJC17,SurfLJC18,
SurfElR0,SurfElC1,SurfElC2,SurfElC3,SurfElC4,SurfElC5,SurfElC6,SurfElC7,SurfElC8,SurfElC9,SurfElC10,SurfElC11,SurfElC12,SurfElC13,SurfElC14,SurfElC15,SurfElC16,SurfElC17,SurfElC18,
SurfWaterR0,SurfWaterC1,SurfWaterC2,SurfWaterC3,SurfWaterC4,SurfWaterC5,SurfWaterC6,SurfWaterC7,SurfWaterC8,SurfWaterC9,SurfWaterC10,SurfWaterC11,SurfWaterC12,SurfWaterC13,SurfWaterC14,SurfWaterC15,SurfWaterC16,SurfWaterC17,SurfWaterC18,
ChemLJR0,ChemLJC1,ChemLJC2,ChemLJC3,ChemLJC4,ChemLJC5,ChemLJC6,ChemLJC7,ChemLJC8,ChemLJC9,ChemLJC10,ChemLJC11,ChemLJC12,ChemLJC13,ChemLJC14,ChemLJC15,ChemLJC16,ChemLJC17,ChemLJC18,
ChemElR0,ChemElC1,ChemElC2,ChemElC3,ChemElC4,ChemElC5,ChemElC6,ChemElC7,ChemElC8,ChemElC9,ChemElC10,ChemElC11,ChemElC12,ChemElC13,ChemElC14,ChemElC15,ChemElC16,ChemElC17,ChemElC18,
ChemWaterR0,ChemWaterC1,ChemWaterC2,ChemWaterC3,ChemWaterC4,ChemWaterC5,ChemWaterC6,ChemWaterC7,ChemWaterC8,ChemWaterC9,ChemWaterC10,ChemWaterC11,ChemWaterC12,ChemWaterC13,ChemWaterC14,ChemWaterC15,ChemWaterC16,ChemWaterC17,ChemWaterC18,
ChemSlabR0,ChemSlabC1,ChemSlabC2,ChemSlabC3,ChemSlabC4,ChemSlabC5,ChemSlabC6,ChemSlabC7,ChemSlabC8,ChemSlabC9,ChemSlabC10,ChemSlabC11,ChemSlabC12,ChemSlabC13,ChemSlabC14,ChemSlabC15,ChemSlabC16,ChemSlabC17,ChemSlabC18
'''

pmfVars = ["source", "numericShape"]


coeffOrders = [ 1,2,3,4,5,6,7,8,9]
potentialModels = ["SurfLJ", "SurfEl", "SurfWater", "ChemLJ", "ChemEl", "ChemWater", "ChemSlab"]

numCoeffs = len(coeffOrders)


chosenCoeffs = []

for potModel in potentialModels:
    chosenCoeffs.append( potModel+"R0")
    for coeffNum in coeffOrders:
        chosenCoeffs.append(potModel+"C"+str(coeffNum))

    
aaVarSet = pmfVars + chosenCoeffs

#fittedE0VarLoc = aaVarSet.index("fittedE0")

if args.predr == 0:
    r0InputLoc = aaVarSet.index("r0")

#"A1CoeffEst"
#a1CoeffEstLoc=aaVarSet.index("A1CoeffEst")

aaPresetIn =keras.Input( shape=(len(aaVarSet),))

numGenericInputs = len(pmfVars)
totalNumInputs = len(aaVarSet)

#at this point the set of input variables is defined so we write these out to a file

varsetOutputFile=open(filetag+"/varset.txt","w")
varsetOutputFile.write( ",".join(aaVarSet))
#for inputVar in aaVarSet:
#    varsetOutputFile.write(inputVar+",")
varsetOutputFile.close()


inputs = [aaPresetIn]


#override the random assignment to ensure that both Stockholm-style and UCD-style AuFC100 PMFs are in the training set
#this is so that a) we get the really strongly-binding Stockholm-style gold and b) to provide a baseline for conversion between the two
#we also add CdSe, Amorphous carbon and amorphous silica because these are otherwise all outliers
fixedMaterials = ["AuFCC100", "AuFCC100UCD","CdSeWurtzite2-10", "SiO2-Amorphous", "C-amorph-2"]
fixedSMILES = ["C", "OCC1OC(O)C(O)C(O)C1O", "Cc1c[nH]c2ccccc12"]

#sample over all AA and materials present in the (possibly bootstrapped) dataset
#chemicals are selected based on SMILES code to allow for duplicates, the -3 is used to account for the fact 3 SMILES codes are manually assigned to the training set
uniqueSMILES = dataset['SMILES'].unique().tolist()
uniqueMaterials = dataset['Material'].unique().tolist()

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


#train_dataset = dataset.sample(frac=0.7, random_state = 0)
#test_dataset = dataset.drop(train_dataset.index)

test_dataset = dataset[ (dataset['Chemical'].isin(validationAA) ) | (dataset['Material'].isin(validationMaterials) )].copy() #if either of the material or AA is in the validation set, it gets assigned to the validation set to prevent leakage
train_dataset = dataset.drop(test_dataset.index)
#print(train_dataset)

moveAla = 0
if moveAla == 1:
    alaValSubset = test_dataset[   (test_dataset['Chemical'] == "ALASCA") & (test_dataset['Material'].isin(validationMaterials)) ] 
    print(alaValSubset)
    train_dataset = pd.concat([train_dataset, alaValSubset])
    test_dataset = test_dataset.drop(alaValSubset.index)
print(train_dataset)



aaNormalizer = layers.Normalization()
aaNormalizer.adapt(np.array(datasetAll[aaVarSet]))
aaPresetNorm = aaNormalizer(aaPresetIn)

#numGenericInputs = len(pmfVars)
#totalNumInputs = len(aaVarSet)
#numCoeffs
aaPresetBoosted =  layers.Reshape( (-1,1)  )(aaPresetNorm)

pmfInputVarLayer = layers.Cropping1D( cropping = (0, totalNumInputs - numGenericInputs) )(aaPresetBoosted)
pmfInputVarLayer = layers.Flatten()(pmfInputVarLayer)

potentialsOnly = layers.Cropping1D( cropping=( numGenericInputs,0 ) )(aaPresetBoosted)
potentialsOnly = layers.Reshape(  (numCoeffs+1, -1 ) )(potentialsOnly)
potentialCoeffsOnly = layers.Cropping1D ( cropping=( 1, 0) )(potentialsOnly)
r0sOnly = layers.Cropping1D( cropping = (0, numCoeffs) )(potentialsOnly)
r0sOnly = layers.Flatten()(r0sOnly)
#r0sOnlyStack = layers.RepeatVector( numCoeffs)(r0sOnly)

#r0Val = layers.Slice(aaPresetIn,"r0")

outputVarset = ["r0","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"  ]

numOutputVars = len(outputVarset)


outputLayers= []






#the dumbest possible way of slicing out one value
class GetR0FromInputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GetR0FromInputLayer,self).__init__()
        #self.target = r0InputLoc
        targetArr = np.zeros( len(aaVarSet) )
        targetArr[r0InputLoc] = 1
        self.chooseVector =  tf.constant( targetArr  , dtype=tf.float32)
    def call(self, inputs):
        #split1 =  tf.unstack( inputs, axis=1)
        return tf.reduce_sum( tf.math.multiply(inputs, self.chooseVector),axis=1,keepdims=True)





#seriously find a better way to do this
'''
class GetA1EstFromInputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GetA1EstFromInputLayer,self).__init__()
        #self.target = r0InputLoc
        targetArr = np.zeros( len(aaVarSet) )
        targetArr[ aaVarSet.index("A1CoeffEst")  ] = 1
        self.chooseVector =  tf.constant( targetArr  , dtype=tf.float32)
    def call(self, inputs):
        #split1 =  tf.unstack( inputs, axis=1)
        return tf.reduce_sum( tf.math.multiply(inputs, self.chooseVector),axis=1,keepdims=True)
'''



if args.predr == 0:
    r0Output = GetR0FromInputLayer()(aaPresetIn)
else:    
    basicInput = aaPresetNorm
    basicInput = layers.GaussianNoise(0.01)(basicInput)
    r0Output = layers.Dense(1,activation=activationXLog)(basicInput)
    for i in range(2):
        #basicInput = r0FF # layers.Concatenate()([ aaPresetNorm, r0FF])
        basicInput = layers.GaussianNoise(0.1)(basicInput)
        r0FF  = layers.Dense(32,activation=activationXLog)(basicInput)
        #r0FF = layers.BatchNormalization()(r0FF)
        r0FF = layers.GaussianNoise(0.15)(r0FF)
        r0FF = layers.Dropout(0.25)(r0FF)
        r0Correction = layers.Dense(1)(r0FF)
        r0Output = layers.Add()([ r0Output, r0Correction])
        basicInput = r0FF
        #basicInput = layers.Concatenate()([aaPresetNorm, r0FF ,r0Output] )
        #basicInput = layers.Concatenate()( [aaPresetNorm, r0FF])
    #r0Output = layers.Dense(1)(r0FF)
    r0Output = layers.Dense(1,activation="exponential",name="rmin_out")(r0Output)
'''
r0Width = 128
r0Skip = layers.Dense(r0Width)(basicInput)
for i in range(2):
    r0FF = layers.Dense(r0Width,activation=activationXLog)(r0Skip)
    r0FF = layers.Dropout(0.5)(r0FF)
    r0FF = layers.Dense(r0Width)(r0FF)
    r0Skip = layers.Add()([r0FF,r0Skip])
    r0Skip = layers.Activation(activationXLog)(r0Skip)

outputValR0 = layers.Activation("softplus",name="rmin_out2")(r0Skip)
'''

outputValR0 = r0Output
#r0Output = layers.Dense(1, activation=activationXLog)(r0Output)
#outputValR0 = layers.Activation("softplus",name="rmin_out2")(r0Output)
#outputValR0 = layers.Activation("exponential", name="rmin_out2")(r0Output)




#generate an "input-like" version of r0 for predicting the other variables
#rmincopy = layers.GaussianNoise(0.0001)( outputValR0 )
#rmincopy = layers.Activation("softplus")(rmincopy)
rmincopy = layers.Lambda( lambda x : tf.stop_gradient(x)  )(outputValR0)
rminnormcopy = layers.BatchNormalization()(rmincopy)

r0sOnly = layers.Concatenate()([ r0sOnly, rminnormcopy])
#r0sOnly = layers.Reshape((1,-1) )(r0sOnly)
#potentialCoeffsOnly = layers.Cropping1D ( cropping=( 1, 0) )(potentialsOnly)

r0sOnlyStack = layers.RepeatVector( numCoeffs)(r0sOnly)
stackedCoeffs = layers.Concatenate( axis=2)( [potentialCoeffsOnly, r0sOnlyStack])



preprocessLayer = layers.Dense(16,activation=activationXLog)(stackedCoeffs)
preprocessLayer = layers.Conv1D( 16,2, strides=2,activation=activationXLog)(preprocessLayer) 
preprocessLayer = layers.Conv1D( 16,2, strides=1,activation=activationXLog)(preprocessLayer)
preprocessLayer = layers.Conv1D( 16,2, strides=2,activation=activationXLog)(preprocessLayer)
#preprocessLayers = layers.Conv1D( 16, strides=2,activation=activationXLog)(preprocessLayers)
preprocessLayer = layers.Flatten()(preprocessLayer)
preprocessLayer  = layers.Concatenate()([ rminnormcopy,  preprocessLayer, pmfInputVarLayer   ])
'''
preprocessThroughConv = 1

initialCombinedLayer = layers.Concatenate()([ aaPresetNorm, rmincopy])
combinedLayer = layers.GaussianNoise(0.05)(initialCombinedLayer)
#outputVal = layers.Dense(numOutputVars)(combinedLayer)

preprocessLayer = layers.Dense( 128, activation=activationXLog)(combinedLayer)
preprocessLayer = layers.Dense(32, activation=activationXLog)(preprocessLayer)

preprocessLayer = layers.GaussianNoise(0.1)(preprocessLayer)
'''





addConv = 1
convToOutput = 1

numChannels = 32
roughOutputs = []



#the dumbest possible way of slicing out one value
class GetValLayer(tf.keras.layers.Layer):
    def __init__(self,targetLoc):
        super(GetValLayer,self).__init__()
        #self.target = r0InputLoc
        targetArr = np.zeros( 16 )
        targetArr[targetLoc] = 1
        self.chooseVector =  tf.constant( targetArr  , dtype=tf.float32)
    def call(self, inputs):
        #split1 =  tf.unstack( inputs, axis=1)
        return tf.reduce_sum( tf.math.multiply(inputs, self.chooseVector),axis=1,keepdims=True)






if addConv == 1:
    preprocessLayer = layers.Dense(numChannels*2, activation=activationXLog)(preprocessLayer)
    preprocessLayer = layers.GaussianNoise(0.1)(preprocessLayer)
    processingLayer = layers.Reshape( (2,-1) )(preprocessLayer)
    currentLength = 2
    #construct the sequence
    layerSet = [2,2,2]
    for i in layerSet:
        upscaledLayer = layers.UpSampling1D( 2 )( processingLayer)
        currentLength = currentLength*2
        transposeUp = layers.Conv1DTranspose( numChannels*2,2, strides=layerSet[i], padding="same",activation=activationXLog)(processingLayer)
        transposeUp = layers.SpatialDropout1D(0.2)(transposeUp)
        #transposeUp = layers.Conv1D( numChannels*4,2,strides=1,padding="same",activation=activationXLog)(transposeUp)
        transposeUp = layers.Conv1DTranspose( numChannels, 2,strides=1,padding="same" )(transposeUp)
        transposeUp = layers.SpatialDropout1D(0.2)(transposeUp)
        #transposeUp = layers.Conv1D( numChannels*4,2,strides=1,padding="same",activation=activationXLog)(transposeUp)
        transposeUp = layers.Conv1DTranspose( numChannels, 2,strides=1,padding="same" )(transposeUp)
        #global mixing stage
        transposeUpF = layers.Flatten()(transposeUp)
        transposeUpF = layers.Dense(4,activation=activationXLog)(transposeUpF)
        transposeUpF = layers.GaussianNoise(0.2)(transposeUpF)
        transposeUpF = layers.Dropout(0.3)(transposeUpF)
        transposeUpF = layers.Dense(numChannels*currentLength)(transposeUpF)
        transposeUpF = layers.Reshape(  (currentLength,-1)   )(transposeUpF)
        transposeUp = layers.Add()([ transposeUp, transposeUpF])

        #transposeUp = layers.GaussianNoise(0.1)(transposeUp)
        #applying dense directly applies the same local filter to each 
        #transposeUp = layers.LocallyConnected1D(numChannels, kernel_size=1)(transposeUp)
        processingLayer = layers.Add()( [upscaledLayer,transposeUp])
    logitLayer = layers.Dense(1)(processingLayer)
    finalConvLayer = layers.Flatten()(logitLayer)
    if convToOutput == 1:
        preprocessLayer = layers.Lambda(   lambda x : tf.stop_gradient(x)    )(finalConvLayer)
        for i in range(16):
            roughOutput = GetValLayer(i)(finalConvLayer)
            roughOutput = layers.Activation(activationIdentity,  name="A"+str(i+1)+"out" )(roughOutput)
            #roughOutput = layers.Lambda( lambda x: tf.slice(x,i,1) )(finalConvLayer)
            roughOutputs.append( roughOutput)
    else:
        preprocessLayer = layers.GaussianNoise(0.01)(finalConvLayer)
    preprocessLayer = layers.Dropout(0.1)(preprocessLayer)
else:
    preprocessLayer = layers.Dense(16)(preprocessLayer)



outputValSet = []


outputValSet.append(outputValR0)
outputValSet = outputValSet + roughOutputs

    

coeffVector = layers.Concatenate()( outputValSet[1:] )
#there is an additional constraint on the outputs: sum_i A_i u_i(r = r_0) -> sum_i A_i sqrt(2 i - 1)/sqrt(r_0) should be equal to fittedE0



coeffScaleTerm = tf.constant( np.array( [np.sqrt(2*i - 1) for i in range(1,numOutputVars) ] )  , dtype=tf.float32)

print(coeffScaleTerm)
#quit()

#coeffWeightedSum = layers.Lambda( lambda x:  tf.reduce_sum( tf.linalg.matvec( x, coeffScaleTerm) , axis = 1))(coeffVector)



class CoeffSumLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(CoeffSumLayer, self).__init__()
      self.scale = tf.constant( np.array( [np.sqrt(2*i - 1) for i in range(1,numOutputVars) ] )  , dtype=tf.float32)

    def call(self, inputs):
      return tf.reduce_sum( tf.math.multiply(inputs , self.scale) , axis=1) 


coeffWeightedSum = CoeffSumLayer()( coeffVector)


#outputValR0
'''
def smoothMax(x):
    return (x*tf.math.exp(x) + 0.1 * tf.math.exp(0.1))/( tf.math.exp(x) + tf.math.exp(0.1) )

@tf.keras.utils.register_keras_serializable()
def lse(x,xmin,alpha):
    return 1/alpha * tf.math.log(  tf.exp(alpha * x) + tf.exp(alpha*xmin)     )
'''
invsqrtr0 = layers.Lambda( lambda x: tf.math.pow( x , -0.5) )(rmincopy)

#invsqrtr0 = layers.Lambda( lambda x: tf.math.pow( x , -0.5) )(outputValR0)

e0Estimate = layers.Multiply(  name="e0predict" )([ invsqrtr0, coeffWeightedSum] )

#unnormalised copy of the actual value of E0 fitted via the PMF
#fittedE0In

addE0Est = 1

if addE0Est == 1:
    outputValSet.append(e0Estimate)

'''
if convToOutput == 1:
    #preprocessLayer = layers.Lambda(   lambda x : tf.stop_gradient(x)    )(finalConvLayer)
    outputValSet = outputValSet + roughOutputs
'''

print( outputValSet )

ann_model = keras.Model(inputs = inputs, outputs=outputValSet)

ann_model.summary()




initialRate = 0.001

initialRate = 0.0001 #for cosine loss
huberdelta=1
logdir="logs"


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
lrReduceCallbackL = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.5,patience=10,verbose=0,cooldown=3, min_lr = 1e-7, min_delta =0.01 )
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
            currentWeights[i] = np.multiply(currentWeights[i],0.99) + np.random.normal(0, 0.01, size = (currentWeights[i].shape) )
            #print(currentWeights[i].shape)
        #currentWeights = np.multiply(0.9 , currentWeights)
        self.model.set_weights(currentWeights)

#plt.ion()
outputFig = plt.figure()
numberOutputPlots = 16
axSet = []
for i in range(numberOutputPlots):
    axSet.append(plt.subplot(4,4,i+1))
#ax2 = plt.subplot(332)

    
class PlotPredictionsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            trainData = [ train_dataset[aaVarSet]   ]
            testData = [ test_dataset[aaVarSet]   ]
            trainPredictions = ( np.array( self.model.predict(trainData))[:,:,0] ).T
            testPredictions = ( np.array( self.model.predict(testData))[:,:,0] ).T
            for i in range(numberOutputPlots):
                axSet[i].clear()
                a1TrainReal = trainingOutput[:,1+i]
                a1TrainPred = trainPredictions[:,1+i].flatten()
                a1TestReal = valOutputData[:,1+i]
                a1TestPred = testPredictions[:,1+i].flatten()
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
            plt.savefig(filetag+"/figures/epoch"+str(epoch)+".png" )



uniquedataset = datasetAll.copy()
uniquedataset.drop_duplicates( subset=['Material','Chemical'] , inplace=True,keep='first')
uniquedataset['ChemValidation'] = 0
uniquedataset.loc[  uniquedataset['Chemical'].isin(validationAA)  ,'ChemValidation'] = 1
uniquedataset['MaterialValidation'] = 0
uniquedataset.loc[  uniquedataset['Material'].isin(validationMaterials) , 'MaterialValidation' ] = 1

#print("Generating regularised set (E0 = "+str(E0TargetVal)+")")
#uniquedataset["fittedE0"] = E0TargetVal

if args.predr == 0:
    uniquedataset["r0"] = uniquedataset["ChemLJR0"]
else:
    uniquedataset["fittedE0"] = E0TargetVal    
    
    
uniquepredictions = ( np.array( ann_model.predict([ uniquedataset[aaVarSet] ]))[:,:,0] ).T
for i in range(len(outputVarset)):
    uniquedataset[ outputVarset[i]+"_regpredict" ] =uniquepredictions[:,i].flatten()
uniquedataset.to_csv(filetag+"/uniquepredictions.csv")


class SavePredictionsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            #trainData = [ train_dataset[aaVarSet]   ]
            #testData = [ test_dataset[aaVarSet]   ]
            #trainPredictions = ( np.array( self.model.predict(trainData))[:,:,0] ).T
            #testPredictions = ( np.array( self.model.predict(testData))[:,:,0] ).T
            uniquepredictions = ( np.array( ann_model.predict([ uniquedataset[aaVarSet] ]))[:,:,0] ).T
            for i in range(len(outputVarset)):
                uniquedataset[ outputVarset[i]+"_regpredict" ] =uniquepredictions[:,i].flatten()
            uniquedataset.to_csv(filetag+"/uniquepredictions_"+str(epoch)+".csv")






'''
class ResetLowLR(keras.callbacks.Callback):
   def on_epoch_start(self, epoch, logs=None):
        if tf.keras.backend.get_value(self.model.optimizer.lr) < 2e-7:
            tf.keras.backend.set_value(self.model.optimizer.lr, 5e-5) 
'''

def sawtoothLR(epoch, lr):
    if epoch < 10:
        return lr
    else:
        newLR = lr
        if epoch % 10 == 0:
            newLR = lr * 0.75
        if newLR < 1e-6:
            newLR = 1e-4
        return newLR


#filetag = "potentialcoeffs-may05-3-sparse-fittedE0"

plotPredCallback = PlotPredictionsCallback()
#resetLRCallback = ResetLowLR()
sawtoothLRSchedule = tf.keras.callbacks.LearningRateScheduler(sawtoothLR)
savePredCallback = SavePredictionsCallback()


bwCallback = BrownianWeights()
csv_logger = keras.callbacks.CSVLogger("../Dropbox/PMFLogs/"+filetag+'.log')
csv_loggerLocal = keras.callbacks.CSVLogger(filetag+'/training.log')

cosineLoss = keras.losses.CosineSimilarity()

checkpointCallback = keras.callbacks.ModelCheckpoint(filetag+"/checkpoints/checkpoint-val", save_best_only=True)
checkpointCallbackTrain = keras.callbacks.ModelCheckpoint(filetag+"/checkpoints/checkpoint-train", monitor="loss",  save_best_only=True)


huberLoss = tf.keras.losses.Huber(delta=1)

huberLossSet = []



if addE0Est == 1:
    outputVarset.append("fittedE0")
    numOutputVars += 1
    
    '''
if convToOutput == 1:
    #preprocessLayer = layers.Lambda(   lambda x : tf.stop_gradient(x)    )(finalConvLayer)
    for i in range(1,17):
        outputVarset.append("roughA"+str(i))
        numOutputVars += 1
'''
trainingOutput = train_dataset[outputVarset].to_numpy()
valOutputData = test_dataset[outputVarset].to_numpy()


print(outputVarset)
print(numOutputVars)
outputDataSets = []
outputValSets = []
for i in range(numOutputVars):
    #huberLossSet.append(  tf.keras.losses.Huber(delta = 0.5* np.std( trainingOutput[:,i]  )  )  )
    huberLossSet.append(  tf.keras.losses.MeanSquaredError(  )    )
    outputDataSets.append( trainingOutput[:,i] )
    outputValSets.append( valOutputData[:,i] )
#print(huberLossSet)

#if addE0Est == 1:
#    huberLossSet[-1] = tf.keras.losses.Huber(delta = 0.5) 

lossWeights = 1.0/(0.01+ np.std( train_dataset[outputVarset].to_numpy()  , axis=0 ) )
#lossWeights[10:] = lossWeights[10:] / 2

#lossWeights[0] = lossWeights[0] *0.5 #boost the weight for r0 to make sure this gets predicted accurately


lossWeights = np.sqrt(np.sqrt(lossWeights / lossWeights[1]))
print(lossWeights)
if addE0Est == 1:
    lossWeights[17] = 0.01

print("Training AA: ", trainingAA)
print("Validation AA: ", validationAA)
print("Training Materials: ", trainingMaterials)
print("Validation Materials: ", validationMaterials)

#sample weighting based on A1
a1vals= train_dataset["A1"].to_numpy()
binDensities,binEdges = np.histogram( a1vals ,20,density=True) #generate bins and densities
bindexs =  np.digitize(a1vals, binEdges[:-1]) # assign each training value to a bin
truncatedBindex=np.where( bindexs < 20, bindexs, 19)
probs=binDensities[truncatedBindex]
weightsUnnormed = 1.0/(0.03 + probs) #setting the parameter to higher values reduces the weighting of extreme A1 values, restoring uniform weights for large numbers.
weightsNormed = (weightsUnnormed / np.sum(weightsUnnormed))
weightsNormed = weightsNormed/np.amin(weightsNormed)




normedSqDists = np.zeros_like( train_dataset["A1"].to_numpy() )
coeffA1TrainingVals = train_dataset["A1"].to_numpy()
coeffA1Scale = np.std(coeffA1TrainingVals)


for i in range(1,17):
    coeffTrainingVals = train_dataset["A"+str(i)].to_numpy()
    coeffScale = np.std(coeffTrainingVals)

    normedSqDist = ((coeffTrainingVals - np.mean(coeffTrainingVals) )/ coeffScale)**2
    coeffWeightFactor = np.sqrt(coeffA1Scale/coeffScale)
    print(i, coeffScale, coeffWeightFactor)
    normedSqDists = normedSqDists + normedSqDist * coeffWeightFactor
normedDist = np.sqrt(normedSqDists)
weightsNormed = 1 + 0.5* normedDist

print(weightsNormed)
print(a1vals)
print("Max weight: ", np.amax(weightsNormed))
#was 0.002 for may11 config

'''
if convToOutput == 1:
    #preprocessLayer = layers.Lambda(   lambda x : tf.stop_gradient(x)    )(finalConvLayer)
    huberLossSet = huberLossSet + huberLossSet
    lossWeights = lossWeights + lossWeights
'''



with tf.device('/cpu:0'):
    ann_model.compile( optimizer = tf.optimizers.Adam(learning_rate=5e-4, epsilon=0.001 ), loss=huberLossSet, loss_weights = lossWeights,metrics=[
        tf.keras.metrics.RootMeanSquaredError() ] )
    history = ann_model.fit([ train_dataset[aaVarSet]   ], outputDataSets, epochs=numEpochs,verbose=1, sample_weight = weightsNormed,batch_size=32,
        validation_data=([ test_dataset[aaVarSet]], outputValSets )  , callbacks = [savePredCallback,sawtoothLRSchedule,checkpointCallback ,checkpointCallbackTrain, tensorboard_callback, csv_logger,plotPredCallback,csv_loggerLocal] 
    )




#save final model

ann_model.save(filetag+"/final_model/")


fulldataset = datasetAll.copy()
fullpredictions = ( np.array( ann_model.predict([ fulldataset[aaVarSet] ]))[:,:,0] ).T


#filetag
fulldataset['ChemValidation'] = 0
fulldataset.loc[  fulldataset['Chemical'].isin(validationAA)  ,'ChemValidation'] = 1
fulldataset['MaterialValidation'] = 0
fulldataset.loc[  fulldataset['Material'].isin(validationMaterials) , 'MaterialValidation' ] = 1
for i in range(len(outputVarset)):
    fulldataset[ outputVarset[i]+"_fullpredict" ] =fullpredictions[:,i].flatten()

fulldataset.to_csv(filetag+"/predictions.csv")

realValues = (fulldataset[outputVarset]).to_numpy()
errorSetFull = np.sqrt( np.mean( (realValues - fullpredictions)**2   , axis=0))
print("Unshuffled: ", errorSetFull)


numShuffleTrials = 2

print("Importance: shuffle one")

