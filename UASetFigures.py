import numpy as np
import matplotlib.pyplot as plt
import os
import HGEFuncs

basePath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_canonical/UA"
matchPath = "predicted_avg_pmfs/PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_matched/UA"
plt.rcParams.update({
    "text.usetex": True,
})
materialSet = os.listdir(basePath)
pmfsPerLine = 4
os.makedirs("UAPMFFigs", exist_ok =True)

for material in materialSet:
    print("Start material: ", material)
    try:
        pmfSet = os.listdir(basePath+"/"+material)
    except:
        continue
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
        axs[row,column].set_xlabel("r [nm]" )
        axs[row,column].set_ylabel(r"U(r) $[\mathrm{kJ} \mathrm{mol}^{-1}]$")
        axs[row,column].set_title( pmfSet[i][:-4] )
    #plt.show()
    #print(pmfSet)
    plt.tight_layout()
    plt.savefig( "UAPMFFigs/"+material+".png" )

#print(os.listdir(basePath))

