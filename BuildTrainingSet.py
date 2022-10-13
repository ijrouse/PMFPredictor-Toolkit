import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import scipy.special as scspec
import datetime
import random


numReplicates = 1

if numReplicates==1:
    chemicalCoefficientFile = open("Datasets/ChemicalPotentialCoefficients-oct10.csv","r")
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
    surfaceCoefficientFile = open("Datasets/SurfacePotentialCoefficients-oct12.csv","r")
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


#PMFCoefficients-N1_noalign-oct12.csv
#PMFCoefficients-N4_noise-oct12.csv


pmfCoefficientFile = open("Datasets/PMFCoefficients-N1_noalign-oct12.csv","r")
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
extraPMFFiles = ["Datasets/PMFCoefficients-N4_noise-oct12.csv"]
#extraPMFFiles = []
for extraPMFFilename in extraPMFFiles:
    extraPMFs = 0
    extraFile  = open( extraPMFFilename,"r")
    spareHeader = extraFile.readline()
    for line in extraFile: 
        if len(line) < 5:
            continue
        lineterms = line.strip().split(",")
        if len(lineterms) < 5:
            print(line)
            continue
        pmfCoefficientFileLines.append(line)
        extraPMFs +=1
    print("Loaded ", extraPMFs, "extra")
    extraFile.close()


outputFile = open("Datasets/TrainingData-oct13.csv","w")

#print(pmfCoefficientFileLines[-1])
print(",".join(headerSet))
outputFile.write( ",".join(headerSet) + "\n")

knownAbsent = []
seenPMFs = []
pmfOnceOnly = 0

#numPMFs = len(pmfCoefficientFile)
failFile = open("failed.txt","w")
for line in pmfCoefficientFileLines:
    #print(line)
    lineTerms = line.strip().split(",")
    if len(lineTerms) < 3:
        continue
    try:
        r0Val = float(lineTerms[ 6 ])
    except:
        print(lineTerms[0], lineTerms[1])
        failFile.write(line)
        continue
    pmfName = lineTerms[0]+"_"+lineTerms[1]
    if pmfOnceOnly == 1 and pmfName in seenPMFs:
        continue
    seenPMFs.append(pmfName)
    surfaceAllR0 = [ surfaceLine for surfaceLine in surfaceSet if (surfaceLine[0] == lineTerms[0] )]
    chemAllR0 = [ chemLine for chemLine in chemSet if (chemLine[0] == lineTerms[1] )] 
    #print(surfaceLine[0])
    #print(surfaceAllR0)

    for i in range(numReplicates):
        #print(line)
        #if i % 100 == 0:
        #    print(i, "/", numPMFs)
        if 1==1:
            surfaceTargetR0 = 0.2
            surfaceChoices = [ surfaceLine for surfaceLine in surfaceAllR0 if   (float(surfaceLine[8]) - surfaceTargetR0)**2 < (0.01)**2 ]
            #surfaceData = surfaceDict[lineTerms[0]]
            #print(lineTerms[0], len(surfaceChoices) )
            if len(surfaceChoices) == 0:
                if not lineTerms[0] in knownAbsent:
                    print("Nothing found for "+lineTerms[0], "r0 = ", r0Val, "chem", lineTerms[1])
                    print(line)
                    knownAbsent.append(lineTerms[0])
                continue
            surfaceData = random.choice(surfaceChoices)
            chemTargetR0 = 0.2
            chemChoices = [ chemLine for chemLine in chemAllR0 if (  (float(chemLine[2]) - chemTargetR0)**2 < (0.01)**2             )]
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
