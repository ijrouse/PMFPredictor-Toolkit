import os

targetList = [
["ETHANE-AC","CC","0"],
["PROPANE-AC","CCC","0"],
["ETHENE-AC","C=C","0"],
["PROPENE-AC","C=CC","0"],
["BUTENE1-AC","C=CCC","0"],
["BUTENE2-AC","CC=CC","0"],
["BUTENE13-AC","C=CC=C","0"],
["ALASCA-AC","C","0"],
["ARGSCA-AC","CCCNC(N)=[NH2+]","1"],
["ASNSCA-AC","CC(N)=O","0"],
["ASPSCA-AC","CC(=O)[O-]","-1"],
["CYSSCA-AC","CS","0"],
["GLNSCA-AC","CCC(N)=O","0"],
["GLUSCA-AC","CCC(=O)[O-]","-1"],
["HIDSCA-AC","Cc1cnc[nH]1","0"],
["HIESCA-AC","Cc1c[nH]cn1","0"],
["ILESCA-AC","CCCC","0"],
["LEUSCA-AC","CC(C)C","0"],
["LYSSCA-AC","CCCC[NH3+]","1"],
["METSCA-AC","CCSC","0"],
["PHESCA-AC","Cc1ccccc1","0"],
["SERSCA-AC","CO","0"],
["THRSCA-AC","CCO","0"],
["TRPSCA-AC","Cc1c[nH]c2ccccc12","0"],
["TYRSCA-AC","Cc1ccc(O)cc1","0"],
["VALSCA-AC","CCC","0"],
["HIPSCA-AC","Cc1c[nH]c[nH+]1","1"],
["CHL-AC","C[N+](C)(C)C","1"],
["ETA-AC","C[NH3+]","1"],
["PHO-AC","COP(=O)([O-])OC","-1"],
["EST-AC","COC(C)=O","0"],
["DGL-AC","OCC1OC(O)C(O)C(O)C1O","0"],
["PROSCA-AC","C1CC1","0"],
["CYMSCA-AC","C[S-]","-1"],
["DMEP-AC","COP(=O)([O-])OC","-1"],
["NC4-AC","C[N+](C)(C)C","1"],
["MAS-AC","COC(C)=O","0"],
["HSPSCA-AC","Cc1c[nH]c[nH+]1","1"],
["MAMM-AC","C[NH3+]","1"],
["GLY-AC","NCC(=O)O","0"],
["PRO-AC","O=C(O)C1CCCN1","0"],
["GANSCA-AC","CCC(=O)O","0"],
["GLUPSCA-AC","CCC(=O)O","0"],
["ASPPSCA-AC","CC(=O)O","0"],
["AFUC-AC","CC1OC(O)C(O)C(O)C1O","0"],
["BGLCNA-AC","CC(=O)NC1C(O)OC(CO)C(O)C1O","0"],
["AMAN-AC","OCC1OC(O)C(O)C(O)C1O","0"],
["BGLC-AC","OCC1OC(O)C(O)C(O)C1O","0"],
["CAFF-AC", "Cn1cnc2n(C)c(=O)n(C)c(=O)c12","0"]













]
#CS-3-AC,CC(CN)O,0,1-amino-2-propanol,
#CS-173-AC,CC(=N)O,0,Acetamide,
#CS-210-AC,CC(=O)CN,0,Aminoacetone,

targetFile = open("csexport.csv","r")
for line in targetFile:
    lineTerms = line.strip().split(",")
    chemID = lineTerms[0]
    smiles = lineTerms[1]
    netcharge = lineTerms[2]
    #targetList.append( [chemID,smiles,netcharge])

print(targetList)



for target in targetList:
    os.system("acpype -i '"+target[1]+"' -b "+target[0]+" -n "+target[2])
    gmxITP = open(target[0]+".acpype/"+target[0]+"_GMX.itp", "r")
    gmxGRO = open(target[0]+".acpype/"+target[0]+"_GMX.gro","r")
    outputfile = open( "../Structures/Chemicals/"+target[0]+"_combined.csv","w")
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
