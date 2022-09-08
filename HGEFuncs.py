import numpy as np
import scipy.special as scspec
import scipy.integrate
import scipy.interpolate


def HGEFunc(r, r0, n):
    '''Return the HGE basis function of order n, parameter r0'''
    return (-1)**(1+n) * np.sqrt( 2*n - 1) * np.sqrt(r0)/r * scspec.hyp2f1(1-n,n,1,r0/r)

def HGECoeffs( inputPotential, r0Val, nmax):
    '''Generates the HGE expansion for an input potential of shape [ [r,U(r) ] ] up to order nmax using parameter r0'''
    r0Actual = max(np.amin(inputPotential[:,0]), r0Val)
    hgeCoeffRes = [r0Actual]
    for n in range(1,nmax+1):
        hgeCoeff =  scipy.integrate.simpson( inputPotential[:,1]*HGEFunc( inputPotential[:,0] ,r0Actual, n),  inputPotential[:,0] )
        hgeCoeffRes.append(hgeCoeff)
    return hgeCoeffRes

def estimateValueLocation( potential, target):
    firstIndex =   np.nonzero( potential[:,1] < target)[0][0] 
    if firstIndex < 1:
        return (potential[firstIndex,0],potential[firstIndex,0])
    pointa = potential[firstIndex - 1]
    pointb = potential[firstIndex]
    mEst = (pointb[1] - pointa[1])/(pointb[0] - pointa[0])
    cEst = -( ( pointb[0]*pointa[1] - pointa[0]*pointb[1]  )/(  pointa[0] - pointb[0] )  )
    crossingEst = (target - cEst) / mEst
    return (crossingEst,target)

def BuildHGEFromCoeffs(r , coeffSet):
    r0Val = coeffSet[0]
    validRegion = r > r0Val
    funcVal = np.zeros_like(r[validRegion])
    for i in range(2,len(coeffSet)):
        funcVal += HGEFunc(r[validRegion], r0Val, i-1) * coeffSet[i]
    return funcVal


def getValidRegion(potential,rmin=0.05):
    MaskStart =  np.where(  np.logical_and(  potential[:,0] >= rmin  ,np.logical_and(np.logical_and(    np.isfinite( potential[:,1] )     , potential[:,1] > -1000)  , potential[:,1] < 1000     ) ))[0][0]
    MaskEnd = np.where(  potential[:,0] > 1.5)[0][0]
    return potential[ MaskStart:MaskEnd ]

def applyNoise(freeEnergySet,deltarsd=0.01,alphasd=0.1,epssd=0.2):
    #translate with probability 0.5
    freeEnergySet[:,0] = freeEnergySet[:,0] + np.random.normal( 0, deltarsd) 
    freeEnergySet[:,1] = freeEnergySet[:,1] * np.random.normal( 1, alphasd) + np.random.normal( 0, epssd, len(freeEnergySet))
    return freeEnergySet

def HGECoeffsInterpolate( inputPotential, r0Val, nmax):
    r0Actual = r0Val
    potentialInterpolated = scipy.interpolate.interp1d(inputPotential[:,0],inputPotential[:,1],  bounds_error=False,fill_value="extrapolate")
    #start from  just before r0 or the second recorded point, whichever is higher and to ensure the gradients are still somewhat sensible
    rminVal =  max( r0Actual, inputPotential[1,0])
    rmaxVal = inputPotential[-1,0]
    #rRange = np.arange( max( r0Actual, inputPotential[0,0]), inputPotential[-1,0], 0.000001)
    #potentialUpscaled = potentialInterpolated(rRange)
    
    #inputRange = inputPotential [ np.logical_and( inputPotential[:,0] > r0Actual ,inputPotential[:,0] <= 1.5 ) ]
    #print("Integrating over ", rRange[0] , " to ", rRange[-1])
    hgeCoeffRes = [r0Actual]
    for n in range(1,nmax+1):
        integrand = lambda x: potentialInterpolated(x) * HGEFunc(x, r0Actual, n)
        #hgeCoeff =  scipy.integrate.trapz( potentialUpscaled*HGEFunc( rRange,r0Actual, n),  rRange )
        hgeCoeffGaussian = scipy.integrate.quadrature( integrand, rminVal, rmaxVal, maxiter=100)
        hgeCoeffRes.append(hgeCoeffGaussian[0])
    #print(hgeCoeffRes)
    return hgeCoeffRes