import os

targetList = [
["AuFCC110UCD","../Downloads/CharmmGUINPs/Au110CG/Au110CG"],
["AuFCC111UCD","../Downloads/CharmmGUINPs/Au111CG/Au111CG"],
["CaO001","../Downloads/CharmmGUINPs/CaO-001-charmmgui"],
["Pt001","../Downloads/CharmmGUINPs/charmm-gui-platinum/charmm-gui-5634115124"],
["AlFCC100UCD","../Downloads/CharmmGUINPs/Al001-Charmm/Al001"],
["AlFCC110UCD","../Downloads/CharmmGUINPs/Al110-Charmm/Al110"],
["AlFCC111UCD","../Downloads/CharmmGUINPs/Al111-Charmm/Al111"],
["TricalciumSilicate001","../Downloads/CharmmGUINPs/TricalciumSilicate001/TricalciumSilicate001"],
["Au-001-PE","../Downloads/CharmmGUINPs/Au-001-PE/charmm-gui-6238260236"],
["Au-001-PEG","../Downloads/CharmmGUINPs/Au-001-PEG/charmm-gui-6238233452"],
["Al2O3-001","../Downloads/CharmmGUINPs/Al2O3-001/charmm-gui-6238142642"],
["Cr2O3-001","../Downloads/CharmmGUINPs/Cr2O3001/charmm-gui-5659838040"],
["Ce-001","../Downloads/CharmmGUINPs/Ce001/charmm-gui-5659812945"],
["Hydroxyapatite-001"    ,   "../Downloads/CharmmGUINPs/Hydroxyapatite-001/charmm-gui-6306913381"]     ,
["Fe001","../Downloads/CharmmGUINPs/Iron-001/charmm-gui-6306819483"],
["Fe110", "../Downloads/CharmmGUINPs/Iron-110/charmm-gui-6306837287"],
["Fe111","../Downloads/CharmmGUINPs/Iron-111/charmm-gui-6306849347"],
["Cu001","../Downloads/CharmmGUINPs/Cu-100/charmm-gui-6306869904"],
["Cu110","../Downloads/CharmmGUINPs/Cu-110/charmm-gui-6306887761"],
["Cu111","../Downloads/CharmmGUINPs/Cu-111/charmm-gui-6306898000"]
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
