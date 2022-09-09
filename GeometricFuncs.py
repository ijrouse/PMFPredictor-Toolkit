import numpy as np
import numpy.linalg as npla

def UARotateMatrix(theta,phi):
    '''
    Generates the rotation matrix corresponding to rotation around the z axis by phi, followed by rotation around the y axis by theta.
    '''
    return np.array([ [ np.cos(theta) * np.cos(phi) , -1 * np.cos(theta) * np.sin(phi) , np.sin(theta) ],
      [  np.sin(phi)                 ,   np.cos(phi)                    , np.zeros_like(phi) ],
      [ -1 * np.sin(theta) * np.cos(phi) ,   np.sin(theta) * np.sin(phi) , np.cos(theta) ]
    ])

def StandardiseAxesTransformMatrix(coordArr):
    '''
    Returns the matrix used to rotate the input coordArr to a standardised form: axis of greatest extent aligned to the z-axis, second longest to y.
    '''
    ixx = np.sum(coordArr[:,1]**2 + coordArr[:,2]**2)
    ixy = np.sum(- coordArr[:,0]*coordArr[:,1])
    ixz = np.sum(-coordArr[:,0]*coordArr[:,2])
    iyy = np.sum(coordArr[:,0]**2 + coordArr[:,2]**2)
    iyx = ixy
    iyz = np.sum(-coordArr[:,1]*coordArr[:,2])
    izz = np.sum(coordArr[:,0]**2 + coordArr[:,1]**2)
    izx = ixz
    izy = iyz
    inertialArray = np.array([ [ixx, ixy,ixz],[iyx,iyy,iyz],[izx,izy,izz] ])
    eigvals,eigvecs = npla.eig(inertialArray)
    #invarr = npla.inv(eigvecs)
    sortIndex = eigvals.argsort()[::-1]
    invarr = npla.inv(eigvecs[:,sortIndex])
    return invarr
