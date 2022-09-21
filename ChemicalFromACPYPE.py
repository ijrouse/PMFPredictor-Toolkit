#Generation of chemical structure files in CSV format via ACPYPE
import os
import argparse


parser = argparse.ArgumentParser(description="Parameters for ChemicalFromACPYPE")
parser.add_argument("-s","--scanstructures", type=int,default=0, help="If non-zero, scans Structures/ChemicalDefinitions for any missing entries and constructs these")
parser.add_argument("-v","--verbose", type=int,default=0, help="If non-zero, prints debugging messages")
args = parser.parse_args()



scanStructures = 0
if args.scanstructures!=0:
    scanStructures = 1
#An example of the structure required for targets: a short descriptive name without underscores, the SMILES code, and the total charge.
targetList = [
["ETHANE-AC","CC","0"]
]
acpypeOutputFolder = "ACPYPE2CSV"

def GenerateChemStructure( chemName, chemSmiles, chemCharge):
    currentDir = os.getcwd()
    os.chdir(currentDir+"/"+acpypeOutputFolder)
    os.system("acpype -i '"+chemSmiles+"' -b "+chemName+" -n "+chemCharge)
    #os.system("mv " + chemName+".acpype" + " " +  acpypeOutputFolder)
    #os.system("mv " + chemName+".mol2" + " " +  acpypeOutputFolder)
    os.chdir(currentDir)
    gmxITP = open(acpypeOutputFolder+"/"+chemName+".acpype/"+chemName+"_GMX.itp", "r")
    gmxGRO = open(acpypeOutputFolder+"/"+chemName+".acpype/"+chemName+"_GMX.gro","r")
    outputfile = open( "Structures/Chemicals/"+chemName+"_combined.csv","w")
    lastITP = ""
    while lastITP != "[ atomtypes ]":
        lastITP = gmxITP.readline().strip()
    gmxITP.readline() #skip the header
    atomTypes  = {}
    while lastITP != "[ moleculetype ]":
        lastITP = gmxITP.readline().strip()
        itpTerms = lastITP.split()
        if len(itpTerms)>1 and itpTerms[0]!="[":
            print(itpTerms)
            atomTypes[itpTerms[0]] =  [ itpTerms[5], itpTerms[6] ]
    print(atomTypes)

    while lastITP != "[ atoms ]":
        lastITP = gmxITP.readline().strip()
        #print(lastITP)
    gmxITP.readline() #skip the header
    headerLine = gmxGRO.readline()
    numAtoms = int(gmxGRO.readline())
    outputfile.write("AtomID,AtomName,AtomType,x[nm],y[nm],z[nm],charge[e],mass[AMU],sigma[nm],epsilon[kjmol]\n")
    for i in range(numAtoms):
        groLine = gmxGRO.readline().strip().split()
        itpLine = gmxITP.readline().strip().split()
        atomParams = atomTypes[ itpLine[1] ]
        #type, x , y , z ,charge,mass,sigma,epsilon
        #print(len(groLine))
        #print(len(itpLine))
        #print(itpLine)
        resLine = ",".join( [  itpLine[0], itpLine[1]+str(itpLine[0]),itpLine[1], groLine[4], groLine[5], groLine[6]  , itpLine[6], itpLine[7], atomParams[0], atomParams[1]        ] )
        print(resLine)
        outputfile.write(resLine+"\n")
        #print( itpLine[1] , atomTypes[itpLine[1]]) #sigma, eps
    gmxGRO.close()
    gmxITP.close()
    outputfile.close()

#Append targets found in the ChemicalDefinitions if asked
if scanStructures == 1:
    chemDefFile = open("Structures/ChemicalDefinitions.csv","r")
    for line in chemDefFile:
        if line[0]=="#":
            continue
        lineTerms = line.strip().split(",")
        if os.path.exists("Structures/Chemicals/"+lineTerms[0]+"_combined.csv"):
            continue
        else:
            smilesCode = lineTerms[1].replace("<COMMA>",",").replace("<HASH>","#")
            print("Generating structure for", lineTerms[0] , smilesCode, lineTerms[2])
            targetList.append([lineTerms[0], smilesCode, lineTerms[2]]  )

for target in targetList:
    GenerateChemStructure( target[0], target[1], target[2])