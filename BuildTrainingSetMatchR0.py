import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import scipy.special as scspec
import datetime
import random

'''
chemicalCoefficientFile = open("Datasets/ChemicalPotentialCoefficients.csv","r")
chemHeader=chemicalCoefficientFile.readline().strip().split(",")
chemDict = {}
knownChems = chemicalCoefficientFile.readlines()
for chemLine in knownChems:
    chemLineTerms = chemLine.strip().split(",")
    chemDict[chemLineTerms[0]] =chemLineTerms
#print(chemDict)
chemicalCoefficientFile.close()


surfaceCoefficientFile = open("Datasets/SurfacePotentialCoefficients.csv","r")
surfaceHeader=surfaceCoefficientFile.readline().strip().split(",")
surfaceDict = {}
knownSurfaces = surfaceCoefficientFile.readlines()
for surfaceLine in knownSurfaces:
    surfaceLineTerms = surfaceLine.strip().split(",")
    surfaceDict[surfaceLineTerms[0]] = surfaceLineTerms
#print(surfaceDict)
surfaceCoefficientFile.close()
'''
numReplicates = 1

if numReplicates==1:
    chemicalCoefficientFile = open("Datasets/ChemicalPotentialCoefficients-aug26.csv","r")
else:
    chemicalCoefficientFile = open("Datasets/ChemicalPotentialCoefficientsNoise10.csv","r")
chemHeader=chemicalCoefficientFile.readline().strip().split(",")
chemSet= []
knownChems = chemicalCoefficientFile.readlines()
for chemLine in knownChems:
    chemLineTerms = chemLine.strip().split(",")
    chemSet.append(chemLineTerms)
#print(chemDict)
chemicalCoefficientFile.close()

if numReplicates==1:
    surfaceCoefficientFile = open("Datasets/SurfacePotentialCoefficients-sep01.csv","r")
else:
    surfaceCoefficientFile = open("Datasets/SurfacePotentialCoefficientsNoise-10.csv","r")
surfaceHeader=surfaceCoefficientFile.readline().strip().split(",")
surfaceSet = []
knownSurfaces = surfaceCoefficientFile.readlines()
surfaceIDs = []

for surfaceLine in knownSurfaces:
    surfaceLineTerms = surfaceLine.strip().split(",")
    surfaceSet.append( surfaceLineTerms)
    surfaceIDs.append( surfaceLineTerms[0] )
    #print(surfaceLineTerms)
#print(surfaceDict)
surfaceCoefficientFile.close()




pmfCoefficientFile = open("Datasets/PMFCoefficientsDiffs-ManualN1nooffset-aug09.csv","r")
pmfHeader=pmfCoefficientFile.readline().strip().split(",")
headerSet= [pmfHeader[0]]+[pmfHeader[1]]+  surfaceHeader[1:]+chemHeader[1:]+ pmfHeader[2:]

pmfCoefficientFileLines = []
for line in pmfCoefficientFile:
    pmfCoefficientFileLines.append(line)
pmfCoefficientFile.close()
targetE0Index = headerSet.index("TargetE0")
fittedE0Index = headerSet.index("fittedE0")
nmaxIndex = headerSet.index("NMaxBest")

print(pmfCoefficientFileLines[-1])
extraPMFFiles = ["Datasets/PMFCoefficientsDiffs-ManualN4nooffset_noise-aug09.csv"]
for extraPMFFilename in extraPMFFiles:
    extraPMFs = 0
    extraFile  = open( extraPMFFilename,"r")
    spareHeader = extraFile.readline()
    for line in extraFile:
        pmfCoefficientFileLines.append(line)
        extraPMFs +=1
    print("Loaded ", extraPMFs, "extra")
    extraFile.close()


outputFile = open("Datasets/TrainingData-r0matched-aug30.csv","w")

#print(pmfCoefficientFileLines[-1])
print(",".join(headerSet))
outputFile.write( ",".join(headerSet) + "\n")

knownAbsent = []

#numPMFs = len(pmfCoefficientFile)

for line in pmfCoefficientFileLines:

    lineTerms = line.strip().split(",")
    r0Val = float(lineTerms[ 6 ])
    surfaceAllR0 = [ surfaceLine for surfaceLine in surfaceSet if (surfaceLine[0] == lineTerms[0] )]
    chemAllR0 = [ chemLine for chemLine in chemSet if (chemLine[0] == lineTerms[1] )] 
    #print(surfaceLine[0])
    #print(surfaceAllR0)

    for i in range(numReplicates):
        #print(line)
        #if i % 100 == 0:
        #    print(i, "/", numPMFs)
        if 1==1:
            surfaceChoices = [ surfaceLine for surfaceLine in surfaceAllR0 if   (float(surfaceLine[6]) - r0Val)**2 < (0.01)**2 ]
            #surfaceData = surfaceDict[lineTerms[0]]
            #print(lineTerms[0], len(surfaceChoices) )
            if len(surfaceChoices) == 0:
                if not lineTerms[0] in knownAbsent:
                    print("Nothing found for "+lineTerms[0], "r0 = ", r0Val, "chem", lineTerms[1])
                    print(line)
                    knownAbsent.append(lineTerms[0])
                continue
            surfaceData = random.choice(surfaceChoices)
            chemChoices = [ chemLine for chemLine in chemAllR0 if (  (float(chemLine[2]) - r0Val)**2 < (0.01)**2             )]
            if len(chemChoices) == 0:
                if not lineTerms[1] in knownAbsent:
                    print("Nothing found for ", lineTerms[1])
                    knownAbsent.append(lineTerms[1])
                continue
            #chemData = chemDict[ lineTerms[1] ]
            chemData = random.choice(chemChoices)
        else:
            print("An error occured! ", line)#  , lineTerms[0], lineTerms[1] , "Number of candidates chem,surface:", len(chemChoices), len(surfaceChoices) )
            #print( lineTerms[0], lineTerms[1], r0Val)
            #continue
        shape = surfaceData[1]
        numericShape = 0
        if shape=="cylinder":
            numericShape = 1
        resSet = [lineTerms[0], lineTerms[1]]+   surfaceData[1:] + chemData[1:] + lineTerms[2:]
        #print(resSet)
        fittedE0 = float(resSet[fittedE0Index])
        targetE0 = float(resSet[targetE0Index])
        
        nmaxBest = float(resSet[nmaxIndex])
        if np.sqrt( (fittedE0-targetE0)**2 ) < 5 and np.sqrt( (fittedE0-targetE0)**2 ) <  np.abs( 5*targetE0 ) and nmaxBest > 15:
            #print(   ",".join(resSet) )
            outputFile.write( ",".join(resSet) + "\n")
        #else:
        #    #print("large error", lineTerms[0] ,lineTerms[1] , fittedE0,targetE0,nmaxBest)


outputFile.close()
