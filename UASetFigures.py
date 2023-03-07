'''Prepare the set of figures based on the UnitedAtom selections and prepare the directory for inclusion in UA, including generation of average histidine PMFs'''

import numpy as np
import matplotlib.pyplot as plt
import os
import HGEFuncs
import shutil
from scipy.interpolate import interp1d

#Build a lookup table to get UA three-letter code from the full name. First load in any pre-defined ones.
uaTargetFile = np.genfromtxt("UABindings.csv",dtype=str)
uaTargets = {}
for uaTarget in uaTargetFile
    uaTargets[ uaTarget[0] ] = uaTargets[ uaTarget[1] ]
    

#Next automatically generate tags for any missing chemicals


#Write out the null Hamaker file for use as a default
makeHamaker == True:
if makeHamaker == True:
    os.makedirs("hamaker-pmfp",exist_ok=True")
    handle = open('hamaker-pmfp/Null-PFMP.dat', 'w')
    print("#{:<6}{:<10}{:<14}{:<10}".format("Name", "kT", "Joules", "kJ/mol"))   
    handle.write("#{:<6}{:<10}{:<14}{:<10}\n".format("Name", "kT", "Joules", "kJ/mol"))
    for chem in uaTargets:
        hamakerConstant = 0
        kT = 1
        aminoAcid = uaTargets[ chem ]
        handle.write("{:<7}{:<10.3f}{:<14.3E}{:<10.3f}\n".format(aminoAcid, 0 / kT, 0, 0 / kT / 0.4))
    handle.close()




basePath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_canonical/UA"
matchPath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_matched/UA"
plt.rcParams.update({
    "text.usetex": True,
})

outputFolder="surface_pmfp"

materialSet = os.listdir(basePath)
materialSet.sort()
pmfsPerLine = 4
os.makedirs("UAPMFFigs", exist_ok =True)
os.makedirs(outputFolder, exist_ok=True)
shutil.copy2( basePath+"/README",  outputFolder+"/README"  )

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
    materialDict[lineTerms[0]+"_pmfp"] = lineTerms[1:]
knownMaterialFile.close()
    
for material in materialSet:
    print("PMF Copy material: ", material)
    try:
        pmfSet = os.listdir(basePath+"/"+material)
    except:
        continue
    os.makedirs(outputFolder+"/"+material, exist_ok = True)
    foundMaterial = False
    if material in materialDict:
        foundMaterial  = True
    shapeVal = 1
    if foundMaterial == True:
        if materialDict[material] == "cylinder":
            shapeVal = 4
    materialSetOutFile.write(material+","+outputFolder+"/"+material+",hamaker/Null-PFMP.dat,"+str(shapeVal)+"\n")
    #Produce linear-average HIS PMFs from HID, HIE and HIP to use when the protein isn't being manually propka'd.
    hipPMF = np.genfromtxt( basePath+"/"+material+"/HIP.dat",delimiter="," )
    hiePMF = np.genfromtxt( basePath+"/"+material+"/HIE.dat",delimiter="," )
    hidPMF = np.genfromtxt( basePath+"/"+material+"/HID.dat",delimiter="," )
    rmin = max(hipPMF[0,0],hiePMF[0,0],hidPMF[0,0])
    rmax = min(hipPMF[-1,0],hiePMF[-1,0],hidPMF[-1,0])
    rrange = hiePMF[ (  hiePMF[:,0] >= rmin  ) & ( hiePMF[:,0] <= rmax)   ,0]
    hipInterp = interp1d( hipPMF[:,0],hipPMF[:,1])
    hieInterp = interp1d( hiePMF[:,0],hiePMF[:,1])
    hidInterp = interp1d( hidPMF[:,0],hidPMF[:,1])
    hisPMF = np.stack( (rrange, hipFrac*hipInterp(rrange) + hieFrac*hieInterp(rrange) + hidFrac*hidInterp(rrange) ),axis=-1)
    np.savetxt(outputFolder+"/"+material+"/HIS.dat" ,hisPMF,fmt='%.18f' ,delimiter=",", header=material+"_HIS: "+hisString+"\nh[nm],U(h)[kJ/mol]")
    pmfSet.sort()
    numPMFs = len(pmfSet)
    for i in range(numPMFs):
        shutil.copy2( basePath+"/"+material+"/"+pmfSet[i],  outputFolder+"/"+material+"/"+pmfSet[i]  )
    

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
