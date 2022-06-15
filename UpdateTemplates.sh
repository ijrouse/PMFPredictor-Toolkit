#!/bin/bash
#Calculate potentials but only for chemicals and surfaces which do not already possess them.
python3 GenerateChemicalPotentials.py -f 0
python3 GenerateSurfacePotentials.py -f 0
#Rebuild the HGExpansions
python3 HGExpandChemicalPotential.py
python3 HGExpandSurfacePotential.py
