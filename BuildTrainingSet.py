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

chemicalCoefficientFile = open("Datasets/ChemicalPotentialCoefficientsNoise.csv","r")
chemHeader=chemicalCoefficientFile.readline().strip().split(",")
chemSet= []
knownChems = chemicalCoefficientFile.readlines()
for chemLine in knownChems:
    chemLineTerms = chemLine.strip().split(",")
    chemSet.append(chemLineTerms)
#print(chemDict)
chemicalCoefficientFile.close()


surfaceCoefficientFile = open("Datasets/SurfacePotentialCoefficientsNoise.csv","r")
surfaceHeader=surfaceCoefficientFile.readline().strip().split(",")
surfaceSet = []
knownSurfaces = surfaceCoefficientFile.readlines()
for surfaceLine in knownSurfaces:
    surfaceLineTerms = surfaceLine.strip().split(",")
    surfaceSet.append( surfaceLineTerms)
#print(surfaceDict)
surfaceCoefficientFile.close()



pmfCoefficientFile = open("Datasets/PMFCoefficients-v2.csv","r")
pmfHeader=pmfCoefficientFile.readline().strip().split(",")
headerSet= [pmfHeader[0]]+[pmfHeader[1]]+  surfaceHeader[1:]+chemHeader[1:]+ pmfHeader[2:]

targetE0Index = headerSet.index("TargetE0")
fittedE0Index = headerSet.index("fittedE0")
nmaxIndex = headerSet.index("NMaxBest")

outputFile = open("Datasets/TrainingData.csv","w")


print(",".join(headerSet))
outputFile.write( ",".join(headerSet) + "\n")



for line in pmfCoefficientFile:
    for i in range(3):
        #print(line)
        lineTerms = line.strip().split(",")
        try:
            surfaceChoices = [ surfaceLine for surfaceLine in surfaceSet if surfaceLine[0] == lineTerms[0] ]
            #surfaceData = surfaceDict[lineTerms[0]]
            surfaceData = random.choice(surfaceChoices)
            #print(surfaceData)
            chemChoices = [ chemLine for chemLine in chemSet if chemLine[0] == lineTerms[1] ]
            #chemData = chemDict[ lineTerms[1] ]
            chemData = random.choice(chemChoices)
        except:
            continue
        shape = surfaceData[1]
        numericShape = 0
        if shape=="cylinder":
            numericShape = 1
        resSet = [lineTerms[0], lineTerms[1]]+   surfaceData[1:] + chemData[1:] + lineTerms[2:]
        fittedE0 = float(resSet[fittedE0Index])
        targetE0 = float(resSet[targetE0Index])
        nmaxBest = float(resSet[nmaxIndex])
        if np.sqrt( (fittedE0-targetE0)**2 ) < 2 and np.sqrt( (fittedE0-targetE0)**2 ) < 0.1 * targetE0 and nmaxBest > 15:
            #print(   ",".join(resSet) )
            outputFile.write( ",".join(resSet) + "\n")


pmfCoefficientFile.close()
outputFile.close()
