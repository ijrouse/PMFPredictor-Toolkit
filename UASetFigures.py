'''Prepare the set of figures based on the UnitedAtom selections and prepare the directory for inclusion in UA, including generation of average histidine PMFs'''

import numpy as np
import matplotlib.pyplot as plt
import os
import HGEFuncs
import shutil
from scipy.interpolate import interp1d


basePath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_canonical/UA"
matchPath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_matched/UA"
plt.rcParams.update({
    "text.usetex": True,
})
materialSet = os.listdir(basePath)
pmfsPerLine = 4
os.makedirs("UAPMFFigs", exist_ok =True)
os.makedirs("surface_pmfp", exist_ok=True)
shutil.copy2( basePath+"/README",  "surface_pmfp/README"  )

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
updatedReadme = open("surface_pmfp/README","a")
updatedReadme.write("\nHIS averaging: "+hisString+"\n")
updatedReadme.write("It is recommended that HIS charge in UA is set to "+str(hipFrac)+" for consistency\n")
updatedReadme.close()


for material in materialSet:
    print("Start material: ", material)
    try:
        pmfSet = os.listdir(basePath+"/"+material)
    except:
        continue
    os.makedirs("surface_pmfp/"+material, exist_ok = True)
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
    np.savetxt("surface_pmfp/"+material+"/HIS.dat" ,hisPMF,fmt='%.18f' ,delimiter=",", header=material+"_HIS: "+hisString+"\nh[nm],U(h)[kJ/mol]")
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
        shutil.copy2( basePath+"/"+material+"/"+pmfSet[i],  "surface_pmfp/"+material+"/"+pmfSet[i]  )
        pmfData = np.genfromtxt( basePath+"/"+material+"/"+pmfSet[i],delimiter="," )
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

#print(os.listdir(basePath))
