import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import scipy.special as scspec
import datetime



chemicalCoefficientFile = open("Datasets/ChemicalPotentialCoefficients.csv","r")
chemHeader=chemicalCoefficientFile.readline().strip().split(",")
chemDict = {}
knownChems = chemicalCoefficientFile.readlines()
for chemLine in knownChems:
    chemLineTerms = chemLine.strip().split(",")
    chemDict[chemLineTerms[0]] =chemLineTerms
print(chemDict)
chemicalCoefficientFile.close()

surfaceCoefficientFile = open("Datasets/SurfacePotentialCoefficients.csv","r")
surfaceHeader=surfaceCoefficientFile.readline().strip().split(",")
surfaceDict = {}
knownSurfaces = surfaceCoefficientFile.readlines()
for surfaceLine in knownSurfaces:
    surfaceLineTerms = surfaceLine.strip().split(",")
    surfaceDict[surfaceLineTerms[0]] = surfaceLineTerms
print(surfaceDict)
surfaceCoefficientFile.close()


pmfCoefficientFile = open("Datasets/PMFCoefficients.csv","r")
pmfHeader=pmfCoefficientFile.readline().strip().split(",")
headerSet= [pmfHeader[0]]+[pmfHeader[1]]+  surfaceHeader[1:]+chemHeader[1:]+ pmfHeader[2:]

targetE0Index = headerSet.index("TargetE0")
fittedE0Index = headerSet.index("fittedE0")


outputFile = open("Datasets/TrainingData.csv","w")


print(",".join(headerSet))
outputFile.write( ",".join(headerSet) + "\n")



for line in pmfCoefficientFile:
    #print(line)
    lineTerms = line.strip().split(",")
    try:
        surfaceData = surfaceDict[lineTerms[0]]
        chemData = chemDict[ lineTerms[1] ]
    except:
        continue
    shape = surfaceData[1]
    numericShape = 0
    if shape=="cylinder":
        numericShape = 1
    resSet = [lineTerms[0], lineTerms[1]]+   surfaceData[1:] + chemData[1:] + lineTerms[2:]
    fittedE0 = float(resSet[fittedE0Index])
    targetE0 = float(resSet[targetE0Index])
    if np.sqrt( (fittedE0-targetE0)**2 ) < 10 and np.sqrt( (fittedE0-targetE0)**2 ) < 0.1 * targetE0 :
        #print(   ",".join(resSet) )
        outputFile.write( ",".join(resSet) + "\n")


pmfCoefficientFile.close()
outputFile.close()
