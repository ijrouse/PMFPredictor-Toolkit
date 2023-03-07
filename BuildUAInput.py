'''Prepare the set of figures based on the UnitedAtom selections and prepare the directory for inclusion in UA, including generation of average histidine PMFs.
If generateAll is set to True, then beads are generated for all generated PMFs. Else, only those with an entry in UABeadCodes.csv are generated.
If all beads are generated, three-letter codes are automatically assigned to any without a manual assignment in UABeadCodes. This supports up to 36^2 = 1200 beads and can be extended to more.

'''

import numpy as np
import matplotlib.pyplot as plt
import os
import HGEFuncs
import shutil
from scipy.interpolate import interp1d
import datetime

generateAll = True
makeNullHamaker = True
makeAllHamaker = True
aww = 3.7e-20 #Hamaker constant for water, used for estimating chemical-material Hamaker constants
kb=1.38064852e-23
na = 6.022e23
kT = kb * 300

baseDir = "UAInputFiles"
os.makedirs(baseDir,exist_ok=True)
dateString =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
shortDate = datetime.datetime.now().strftime("%d-%b-%y")

def nameFromNum(counter):
    letterSet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F", "G", "H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    letter1 = "X"
    letter2 = letterSet[int(counter / len(letterSet) )]
    letter3 = letterSet[ counter % len(letterSet) ]
    return letter1+letter2+letter3




#Build a lookup table to get UA three-letter code from the full name. First load in any pre-defined ones.
uaTargetFile = open("UABeadCodes.csv","r")


chemDefFile = open("Structures/ChemicalDefinitions.csv","r")
uaTargets = {}
usedNames = []
for line in uaTargetFile:
    if line[0] == "#":
        continue
    uaTarget = line.strip().split(",")
    if uaTarget[0] == "HISTIDINE-AUTO" or uaTarget[1]=="HIS":
        continue
    
    uaTargets[ uaTarget[0] ] =   uaTarget[1] 
    usedNames.append( uaTarget[1] )



#Load in the definitions for all chemicals and get an estimate of the effective radius from the volume, then use this to estimate the chemical self-Hamaker constant 
chemicalDefinitions = {}
chemDefFile = open("Structures/ChemicalDefinitions.csv","r")
for line in chemDefFile:
    if line[0]=="#":
        continue
    lineTerms = line.strip().split(",")
    try:
        structureDef = np.genfromtxt("Structures/Chemicals/"+lineTerms[0]+"_combined.csv", skip_header=1,delimiter=",",dtype=str) 
        radiusVals = structureDef[:,8].astype(float)/2.0
        beadVolumes = 4.0*np.pi/3.0 * radiusVals**3
        totalVolume = np.sum(beadVolumes)
        estRadius    =  ( totalVolume*3.0/(4.0*np.pi) )**(1.0/3.0)
    except:
        print("Structure not found for "+lineTerms[0])
        estRadius = 0.1
    try:
        structNumeric =  structureDef[:, 3:].astype(float)
        structNumeric[:,0] = structNumeric[:,0] - np.sum(structNumeric[:,0] * structNumeric[:,4])/np.sum(structNumeric[:,4])
        structNumeric[:,1] = structNumeric[:,1] - np.sum(structNumeric[:,1] * structNumeric[:,4])/np.sum(structNumeric[:,4])
        structNumeric[:,2] = structNumeric[:,2] - np.sum(structNumeric[:,2] * structNumeric[:,4])/np.sum(structNumeric[:,4])
        struct1 = np.copy(structNumeric)
        struct2 = np.copy(structNumeric)
        zwidth = np.amax(struct2[:,2]) - np.amin(struct2[:,2])
        ccd = (2.0 + zwidth)
        struct2[:,2] += ccd
        sigmaGrid = 0.5* np.add.outer( struct1[:,5], struct2[:,5])
        epsGrid = np.sqrt( np.multiply.outer(struct1[:,6], struct2[:,6]) )
        #print(sigmaGrid)
        #print(epsGrid)
        distGrid = np.sqrt(  (np.add.outer(struct1[:,0] , -1*struct2[:,0] ))**2 +  (np.add.outer(struct1[:,1] , -1*struct2[:,1] ))**2 +  (np.add.outer(struct1[:,2] , -1*struct2[:,2] ))**2)
        totalLJ = np.sum(4*epsGrid*(   (sigmaGrid/distGrid)**12 - (sigmaGrid/distGrid)**6  ))
        #print(totalLJ)
        radiusEstSq = estRadius**2
        hamakerAtCCD = -1.0/(6.0) * (   (2*radiusEstSq)/(ccd**2 ) + (2*radiusEstSq)/(ccd**2 - 4.0 * (radiusEstSq)) + np.log(  (ccd**2 - 4.0 * (radiusEstSq))/(ccd**2)  ) )
        accEstkjmol = totalLJ/hamakerAtCCD
        accEst = (accEstkjmol *1000.0)/(na)
    except:
        print("Failed to compute estimated Hamaker")
        accEst = aww
    chemicalDefinitions[ lineTerms[0] ] =  [  lineTerms[1], lineTerms[2] ,estRadius,accEst]
    print("Loaded: ", lineTerms[0] ,   lineTerms[1], lineTerms[2]  ,estRadius,accEst)

#Manually append the special averaged histidine
#Define and normalise the fractions of HID,HIE,HIP used for generating an average HIS PMF
targetPH = 7
hispKa = 6.0
hidFraction = 0.2 #HIE (epsilon-protonated) is favoured over HID (delta-protonated) by a ratio of 4:1 in solution
deprotonatedHisFrac = np.exp(targetPH)/( np.exp(targetPH) + np.exp(  hispKa ))
protonatedHisFrac = 1 - deprotonatedHisFrac
hisFactor = 1-hidFraction
hidFrac = deprotonatedHisFrac*hidFraction
hieFrac = deprotonatedHisFrac*(1-hidFraction)
hipFrac = protonatedHisFrac
hidFrac = hidFrac/(hidFrac+hieFrac+hipFrac)
hieFrac = hieFrac/(hidFrac+hieFrac+hipFrac)
hipFrac = hipFrac/(hidFrac+hieFrac+hipFrac)
hisString = "HID "+str(hidFrac)+" HIE "+str(hieFrac)+" HIP "+str(hipFrac)
hisRadiusEst = hidFrac*chemicalDefinitions["HIDSCA-AC"][2] + hieFrac*chemicalDefinitions["HIESCA-AC"][2]+ hipFrac*chemicalDefinitions["HIPSCA-AC"][2]
hisACCEst = hidFrac*chemicalDefinitions["HIDSCA-AC"][3] + hieFrac*chemicalDefinitions["HIESCA-AC"][3]+ hipFrac*chemicalDefinitions["HIPSCA-AC"][3]
uaTargets["HISTIDINE-AUTO"] = "HIS"
chemicalDefinitions[ "HISTIDINE-AUTO" ] = [hisString,str(hipFrac),hisRadiusEst,hisACCEst]

#Next automatically generate tags for any chemicals which don't have a pre-determined one

if generateAll == True:
    nameCounter = 0
    for chem in chemicalDefinitions.keys():
        if chem in uaTargets.keys():
            continue
        nextName = nameFromNum(nameCounter)
        nameCounter += 1
        while nextName in usedNames:
            nextName = nameFromNum(nameCounter)
            nameCounter += 1
        usedNames.append(nextName)
        uaTargets[chem ] = nextName
print(uaTargets)


beadMapOut = open(baseDir+"/beadmaps.txt","w")
beadMapOut.write("#Full name, UA Code\n")
for chem in uaTargets:
    beadMapOut.write(chem+","+uaTargets[chem]+"\n")
beadMapOut.close()

#Write out the null Hamaker file for use as a default

if makeNullHamaker == True:
    os.makedirs(baseDir+"/hamaker-pmfp",exist_ok=True)
    handle = open(baseDir+"/hamaker-pmfp/Null-PFMP.dat", 'w')
    #print("#{:<6}{:<10}{:<14}{:<10}".format("Name", "kT", "Joules", "kJ/mol"))   
    handle.write("#{:<6}{:<10}{:<14}{:<10}\n".format("Name", "kT", "Joules", "kJ/mol"))
    for chem in uaTargets:
        hamakerConstant = 0
        aminoAcid = uaTargets[ chem ]
        handle.write("{:<7}{:<10.3f}{:<14.3E}{:<10.3f}\n".format(aminoAcid, 0 / kT, 0, 0 / kT / 0.4))
    handle.close()




os.makedirs(baseDir+"/pmfp-beads",exist_ok=True)
for chem in uaTargets:
    beadOut = open(baseDir+"/pmfp-beads/"+chem+".pdb","w")
    beadOut.write("HEADER "+ '{:40.40}'.format(chem)+shortDate+"    X"+uaTargets[chem]+"\n")
    #beadOut.write("REMARK Bead ID:" + uaTargets[chem]+"\n")
    beadOut.write("TITLE   "+chem+" to bead "+uaTargets[chem]+"\n")
    beadOut.write("REMARK SMILES: "+chemicalDefinitions[ chem ][0]+"\n")
    beadOut.write("REMARK Generated: "+dateString+"\n")
    beadOut.write("ATOM      1  CA  "+uaTargets[chem]+" A   1       0.000   0.000   0.000  1.00  0.00\n")
    beadOut.write("END\n")
    beadOut.close()


os.makedirs(baseDir+"/pmfp-beadsetdef",exist_ok=True)
beadDef = open(baseDir+"/pmfp-beadsetdef/PMFP-BeadSet.csv","w")
beadDef.write("#List of beads to include in UA autogenerated config files, providing the three-letter code, charge and radius to write to config files.\n#BeadID,Charge[e],Radius[nm]\n")
for chem in uaTargets:
    beadDef.write(uaTargets[chem]+","+chemicalDefinitions[ chem ][1]+","+str(chemicalDefinitions[ chem ][2])+"\n")
beadDef.close()



basePath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_canonical"
matchPath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_matched"
plt.rcParams.update({
    "text.usetex": True,
})

outputFolder=baseDir+"/surface-pmfp"




pmfsPerLine = 4
os.makedirs(baseDir+"/UAPMFFigs", exist_ok =True)



os.makedirs(outputFolder, exist_ok=True)
shutil.copy2( basePath+"/README",  outputFolder+"/README"  )



updatedReadme = open(outputFolder+"/README","a")
updatedReadme.write("\nHIS averaging: "+hisString+"\n")
updatedReadme.write("It is recommended that HIS charge in UA is set to "+str(hipFrac)+" for consistency\n")
updatedReadme.close()
materialSetOutFile = open(outputFolder+"/MaterialSetPMFP.csv","w")
materialSetOutFile.write("#Material name, surface folder, Hamaker file, default shape\n")

knownMaterialFile = open("Structures/SurfaceDefinitions.csv","r")
materialDict = {}
for line in knownMaterialFile:
    if line[0]=="#":
        continue
    lineTerms = line.strip().split(",")
    materialDict[lineTerms[0]] = lineTerms[1:]
knownMaterialFile.close()
    
for material in materialDict:
    print("PMF Copy material: ", material)
    os.makedirs(outputFolder+"/"+material+"-pmfp", exist_ok = True)
    foundMaterial = False
    if material in materialDict:
        foundMaterial  = True
    shapeVal = 1
    if foundMaterial == True:
        if materialDict[material] == "cylinder":
            shapeVal = 4
            
    #Produce very rough estimates of Hamaker constants based on the combining relations. These are known to be bad in water but because the UA-Hamaker contribution is small the error is small.

    if makeAllHamaker == True:
        os.makedirs(baseDir+"/hamaker-pmfp",exist_ok=True)
        handle = open(baseDir+"/hamaker-pmfp/"+material+"-PFMP.dat", 'w')
        #use the methane probe at long distance as an approximation for a spherical Hamaker like interaction
        structurePotentialFile = np.genfromtxt("SurfacePotentials/"+material+"_methanefe.dat",delimiter=",",skip_header=1)
        cprobeData= structurePotentialFile[:,2:4]
        rp = 0.2392389367213202
        app = 4.2615615043644403e-20
        targetCCD = 1.0 + rp
        probeEnergy = (cprobeData[cprobeData[:,0] > targetCCD   ])[0,1]
        #approximate the probe as a sphere of radius sigma/2
        if shapeVal == 4:
            distanceFactor = -1.0/6.0 *( ( rp/(rp+targetCCD) + rp/(targetCCD-rp)   ) + np.log( (targetCCD-rp)/(targetCCD+rp)  )    )
        else:
            distanceFactor = -1.0/6.0 *(( rp/(rp+targetCCD) + rp/(targetCCD-rp)   ) + np.log( (targetCCD-rp)/(targetCCD+rp)  )    )
        amp = (probeEnergy/distanceFactor)  *(1000.0)/(na)
        
        amm = amp**2/app
        #print(probeEnergy, distanceFactor, amp, amm)
        #print("#{:<6}{:<10}{:<14}{:<10}".format("Name", "kT", "Joules", "kJ/mol"))   
        handle.write("#{:<6}{:<10}{:<14}{:<10}\n".format("Name", "kT", "Joules", "kJ/mol"))
        for chem in uaTargets:
            
            hamakerConstant = ( np.sqrt(chemicalDefinitions[ chem ][3]) - np.sqrt(aww) )*(np.sqrt(amm) - np.sqrt(aww) )
            kT = 1.38064852e-23 * 300
            aminoAcid = uaTargets[ chem ]
            handle.write("{:<7}{:<10.3f}{:<14.3E}{:<10.3f}\n".format(aminoAcid, hamakerConstant / kT, hamakerConstant, hamakerConstant / kT / 0.4))
        handle.close()        
        materialSetOutFile.write(material+"-pmfp,surface-pmfp/"+material+"-pmfp,hamaker-pmfp/"+material+"-PFMP.dat,"+str(shapeVal)+"\n")            
    else:         
        materialSetOutFile.write(material+"-pmfp,surface-pmfp/"+material+"-pmfp,hamaker-pmfp/Null-PFMP.dat,"+str(shapeVal)+"\n")
    #Produce linear-average HIS PMFs from HID, HIE and HIP to use when the protein isn't being manually propka'd.
    hipPMF = np.genfromtxt( basePath+"/"+material+"_simple/HIPSCA-AC.dat",delimiter="," )
    hiePMF = np.genfromtxt( basePath+"/"+material+"_simple/HIESCA-AC.dat",delimiter="," )
    hidPMF = np.genfromtxt( basePath+"/"+material+"_simple/HIDSCA-AC.dat",delimiter="," )
    rmin = max(hipPMF[0,0],hiePMF[0,0],hidPMF[0,0])
    rmax = min(hipPMF[-1,0],hiePMF[-1,0],hidPMF[-1,0])
    rrange = hiePMF[ (  hiePMF[:,0] >= rmin  ) & ( hiePMF[:,0] <= rmax)   ,0]
    hipInterp = interp1d( hipPMF[:,0],hipPMF[:,1])
    hieInterp = interp1d( hiePMF[:,0],hiePMF[:,1])
    hidInterp = interp1d( hidPMF[:,0],hidPMF[:,1])
    hisPMF = np.stack( (rrange, hipFrac*hipInterp(rrange) + hieFrac*hieInterp(rrange) + hidFrac*hidInterp(rrange) ),axis=-1)
    np.savetxt(outputFolder+"/"+material+"-pmfp/HIS.dat" ,hisPMF,fmt='%.18f' ,delimiter=",", header=material+"_HIS: "+hisString+"\nh[nm],U(h)[kJ/mol]")
    for chem in uaTargets:
        if uaTargets[chem]=="HIS":
            continue
        #print("copy: ",  basePath+"/"+material+"/"+chem+".dat", "to ", outputFolder+"/"+material+"_pmfp/"+uaTargets[chem]+".dat" )
        shutil.copy2( basePath+"/"+material+"_simple/"+chem+".dat",  outputFolder+"/"+material+"-pmfp/"+uaTargets[chem]+".dat"  )
    
    
quit()

pmfPlotSet =["ALA.dat","ARG.dat","ASN.dat","ASP.dat",
"CHL.dat","CYM.dat","CYS.dat","DGL.dat",
"EST.dat","ETA.dat","GAN.dat","GLN.dat",
"GLU.dat","GLY.dat","HIE.dat","HIP.dat",
"ILE.dat","LEU.dat","LYS.dat","MET.dat",
"PHE.dat","PHO.dat","PRO.dat","SER.dat",
"THR.dat","TRP.dat","TYR.dat","VAL.dat"]
for material in materialSet:
    try:
        pmfSet = os.listdir(basePath+"/"+material)
    except:
        continue
    if len(pmfSet) < 5:
        continue
    print("PMF Figures material: ", material)
    pmfSet = pmfPlotSet    
    #Generate figures
    pmfSet.sort()
    numPMFs = len(pmfSet)
    numRows = int(numPMFs/pmfsPerLine)
    if numPMFs % pmfsPerLine != 0:
        numRows += 1
    #plt.figure()
    #plt.suptitle(material)
    fig,axs=plt.subplots( numRows , pmfsPerLine,figsize=(12,14),dpi=600 )
    plt.suptitle(material)
    minEnergy = 0
    for i in range(numPMFs):
        column = i % pmfsPerLine
        row = int(i/pmfsPerLine)
        chemName =  pmfSet[i][:-4]
        #print(i,row,column, pmfSet[i])
        pmfData = np.genfromtxt( outputFolder+"/"+material+"/"+pmfSet[i],delimiter="," )
        minEnergy =  np.amin( pmfData[:,1]) 
        axs[row,column].plot(pmfData[:,0],pmfData[:,1],"b-")
        axs[row,column].plot(pmfData[:,0],0*pmfData[:,1],"k:")
        try:
            matchPMF = np.genfromtxt( matchPath+"/"+material+"/"+pmfSet[i],delimiter="," )
            axs[row,column].plot(matchPMF[:,0],matchPMF[:,1],"r--")
            minEnergy = min( minEnergy, np.amin(matchPMF[:,1]) )
        except:
            minEnergy = minEnergy
        #chemName
        #print("Trying path  AllPMFs/"+material[:-5]+"_"+chemName+"-AC.dat")
        if os.path.exists("AllPMFs/"+material[:-5]+"_"+chemName+"-AC.dat"):
            knownPMF = HGEFuncs.loadPMF("AllPMFs/"+material[:-5]+"_"+chemName+"-AC.dat",False)
        elif os.path.exists("AllPMFs/"+material[:-5]+"_"+chemName+"SCA-AC.dat"):
            knownPMF = HGEFuncs.loadPMF("AllPMFs/"+material[:-5]+"_"+chemName+"SCA-AC.dat",False)
        elif os.path.exists("AllPMFs/"+material[:-5]+"_"+chemName+"SCA-JS.dat"):
            knownPMF = HGEFuncs.loadPMF("AllPMFs/"+material[:-5]+"_"+chemName+"SCA-JS.dat",False)
        elif os.path.exists("AllPMFs/"+material[:-5]+"_"+chemName+"-JS.dat"):
            knownPMF = HGEFuncs.loadPMF("AllPMFs/"+material[:-5]+"_"+chemName+"-JS.dat",False)
        else:
            knownPMF = [-1]
        if len(knownPMF) > 2:
            axs[row,column].plot( knownPMF[:,0], knownPMF[:,1], 'g-')
            minEnergy = min( minEnergy, np.amin( knownPMF[:,1] ) )
        
        axs[row,column].set_xlim(0,1.5)
        axs[row,column].set_ylim( minEnergy - 5  , 25)
        axs[row,column].set_xlabel("h [nm]" )
        axs[row,column].set_ylabel(r"U(h) $[\mathrm{kJ}\cdot\mathrm{mol}^{-1}]$")
        axs[row,column].set_title( pmfSet[i][:-4] )
    #plt.show()
    #print(pmfSet)
    plt.tight_layout()
    plt.savefig( "UAPMFFigs/"+material+".png" )
materialSetOutFile.close()
#print(os.listdir(basePath))
