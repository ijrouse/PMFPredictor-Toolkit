#!/bin/bash
#Completely resets the datasets, including recalculating all chemical and surface potentials and regenerating the training sets.
#Recalculate potentials
python3 GenerateChemicalPotentials.py -f 1
python3 GenerateSurfacePotentials.py -f 1
#Rebuild the HGExpansions
python3 HGExpandChemicalPotential.py -f 1
python3 HGExpandSurfacePotential.py -f 1
python3 HGExpandPMFs.py -f 1
#Generate the training set
python3 BuildTrainingSet.py
