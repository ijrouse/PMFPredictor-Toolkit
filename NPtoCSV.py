import os

targetList = [
["GoldBrush","../Downloads/gold_brush/gold_brush_charmmgui"]
]

print("Generating structures for target CHARMM-GUI downloads")
for target in targetList:
    print("Processing ", target[0], " working folder ", target[1])
    #os.system("acpype -i "+target[1]+" -b "+target[0])
    gmxFF = open(target[1]+"/gromacs/toppar/forcefield.itp","r")
    gmxPSF = open(target[1]+"/gromacs/step3_input.psf","r")
    gmxGRO = open(target[1]+"/gromacs/step3_input.gro","r")
    outputfile = open( "Structures/Surfaces/"+target[0]+"_combined.csv","w")
    lastITP = ""
    while lastITP != "[ atomtypes ]":
        lastITP = gmxFF.readline().strip()
    gmxFF.readline() #skip the header
    atomTypes  = {}
    while lastITP != "" and lastITP != "[ pairtypes ]":
        lastITP = gmxFF.readline().strip()
        itpTerms = lastITP.split()
        if len(itpTerms)>1 and itpTerms[0]!="[" and itpTerms[0]!=";":
            print(itpTerms)
            atomTypes[itpTerms[0]] =  [ itpTerms[5], itpTerms[6] ]
    print(atomTypes)

    #while lastITP != "[ atoms ]":
    #    lastITP = gmxITP.readline().strip()
    #    #print(lastITP)
    #gmxITP.readline() #skip the header

    #parse the .psf file
    psfStarted = 0
    while psfStarted == 0:
        psfTerms = gmxPSF.readline().strip().split()
        if len(psfTerms)>0:
            if psfTerms[-1] == "!NATOM":
                psfStarted = 1


    headerLine = gmxGRO.readline()
    numAtoms = int(gmxGRO.readline())
    resultSet = []
    outputfile.write("AtomID,AtomName,AtomType,x[nm],y[nm],z[nm],charge[e],mass[AMU],sigma[nm],epsilon[kjmol]\n")
    for i in range(numAtoms):
        groLine = gmxGRO.readline().strip().split()
        #itpLine = gmxITP.readline().strip().split()
        psfLine = gmxPSF.readline().strip().split()
        atomType = psfLine[5]
        atomNum = psfLine[0]
        atomParams = atomTypes[ atomType ]
        atomCharge = psfLine[6]
        atomMass = psfLine[7]
        #type, x , y , z ,charge,mass,sigma,epsilon
        #print(len(groLine))
        #print(len(itpLine))
        #print(itpLine)
        #print(groLine)
        resultSet.append([  str(atomNum), atomType+str(atomNum),atomType, groLine[3], groLine[4], groLine[5]  ,atomCharge,atomMass, atomParams[0], atomParams[1]        ] )
        resLine = ",".join( [  str(atomNum), atomType+str(atomNum),atomType, groLine[3], groLine[4], groLine[5]  ,atomCharge,atomMass, atomParams[0], atomParams[1]        ] )
        #print(resLine)
        #outputfile.write(resLine+"\n")
        #print( itpLine[1] , atomTypes[itpLine[1]]) #sigma, eps
    gmxGRO.close()
    gmxFF.close()
    gmxPSF.close()
    xc,yc,zc,mc=0,0,0,0
    zmax = -9000
    for line in resultSet:
        x = float(line[3])
        y = float(line[4])
        z = float(line[5])    
        m = float(line[7])
        xc += x*m
        yc += y*m
        zc += z*m
        zmax = max(z,zmax)
        mc += m
    xc = xc/mc
    yc = yc/mc
    zc = zc/mc
    for line in resultSet:    
        resLine = ",".join([ line[0],line[1],line[2], str(float(line[3])-xc), str(float(line[4])-yc), str(float(line[5])-zmax) , line[6],line[7],line[8],line[9] ])
        outputfile.write(resLine+"\n")
    outputfile.close()
print("All structures generated. Remember to update SurfaceDefinitions.csv")
