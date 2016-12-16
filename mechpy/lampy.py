# -*- coding: utf-8 -*-
"""
code to compute composite properties, applied mechanical and thermal loads
and stress and strain

general outline for computing elastic properties of composites

1) Determine engineering properties of unidirectional laminate. E1, E2, nu12, G12
2) Calculate ply stiffnesses Q11, Q22, Q12, Q66 in the principal/local coordinate system
3) Determine Fiber orientation of each ply
4) Calculate the transformed stiffness Qxy in the global coordinate system
5) Determine the through-thicknesses of each ply
6) Determine the laminate stiffness Matrix (ABD)
7) Calculate the laminate compliance matrix by inverting the ABD matrix
8) Calculate the laminate engineering properties

TODO -    create a function to determine the max load or strain to failure

# Stress Strain Relationship for a laminate, with Q=reduced stiffness matrix
|sx | |Qbar11 Qbar12 Qbar16| |ex +z*kx |
|sy |=|Qbar12 Qbar22 Qbar26|=|ey +z*ky |
|sxy| |Qbar16 Qbar26 Qbar66| |exy+z*kxy|

# Herakovich pg 84
Qbar =  inv(T1) @ Q @ T2 == solve(T1, Q) @ T2

transformation reminders - see Herakovich for details
sig1 = T1*sigx
sigx = inv(T1)*sig1
eps1 = T2*epsx
epsx = inv(T2)*epsx
sigx = inv(T1)*Q*T2*epsx
Qbar = inv(T1)*Q*T2
Sbar = inv(T2)*inv(Q)*T2




References
Hyer-Stress Analysis of Fiber-Reinforced Composite Materials
Herakovich-Mechanics of Fibrous Composites
Daniel-Engineering Mechanics of Composite Materials


"""

#==============================================================================
# Import Modules
#==============================================================================
from __future__ import print_function, division

__author__ = 'Neal Gordon <nealagordon@gmail.com>'
__date__ =   '2016-12-16'

from copy import copy

from numpy import pi, zeros, ones, linspace, arange, array, sin, cos
from numpy.linalg import solve, inv

#from scipy import linalg
import numpy as np
#np.set_printoptions(suppress=False,precision=2)   # suppress scientific notation
np.set_printoptions(precision=4, linewidth=150)

import pandas as pd

import sympy as sp
from sympy import Function, dsolve, Eq, Derivative, symbols, pprint
from sympy.plotting import plot3d  

#from sympy import cos, sin
#sp.init_printing(use_latex='mathjax')
#sp.init_printing(wrap_line=False, pretty_print=True)

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,figure,xlim,ylim,title,legend, \
grid, show, xlabel,ylabel, tight_layout
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (8,5)
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14

# inline plotting
from IPython import get_ipython
#get_ipython().magic('matplotlib inline')

###disable inline plotting
get_ipython().magic('matplotlib inline')    
    
from IPython.display import display


class laminate(object):
    """
    laminate object for composite material analysis
    """
    
    # constructor
    def __init__(self, plyangle, matindex, matname):
        # run when laminate is instantiated
        
        # loads materials used
        self.plyangle = plyangle
        self.matindex = matindex
        self.matname = matname
        
        
        self.__mat = self.__import_matprops(matname)
        
        # create a simple function to handle CTE properties
        def __alphaf(self, mat):
            return array([[mat.alpha1], [mat.alpha2], [0]])
    
        self.laminatethk = array([self.__mat[matname[i]].plythk for i in matindex ])
        
        self.nply = len(self.laminatethk) # number of plies
        self.H =   np.sum(self.laminatethk) # plate thickness
        #    area = W*H
        z = zeros(self.nply+1)
        zmid = zeros(self.nply)
        z[0] = -self.H/2
        for i in range(self.nply):
            z[i+1] = z[i] + self.laminatethk[i]
            zmid[i] = z[i] + self.laminatethk[i]/2    
        self.z = z
        self.zmid = zmid   
        
        self.__abdmatrix()

    def __Qf(self, E1,E2,nu12,G12):
        '''transversly isptropic compliance matrix. pg 58 herakovich
        G12 = E1/(2*(1+nu12))  if isotropic'''
        nu21 = E2*nu12/E1
        Q = array([[E1/(1-nu12*nu21),    E2*nu12/(1-nu12*nu21), 0],
                   [ E2*nu12/(1-nu12*nu21), E2/(1-nu12*nu21),    0],
                   [0,        0,       G12]])
        return Q         
        
        
    def __T1(self, th):
        '''Stress Transform for Plane Stress 
        th=ply angle in degrees
        voight notation for stress tranform. sigma1 = T1 @ sigmax
        recall T1(th)**-1 == T1(-th)'''
        n = sin(th*pi/180)
        m = cos(th*pi/180)
        T1 = array( [[m**2, n**2, 2*m*n],  
                     [n**2, m**2,-2*m*n],  
                     [-m*n, m*n,(m**2-n**2)]])    
        return T1
        
    def __T2(self, th):
        '''Strain Transform for Plane Stress
        th=ply angle in degrees
        voight notation for strain transform. epsilon1 = T2 @ epsilonx'''
        n = sin(th*pi/180)
        m = cos(th*pi/180)
        T2 = array( [[m**2, n**2, m*n],  
                     [n**2, m**2,-m*n],  
                     [-2*m*n, 2*m*n,  (m**2-n**2)]])     
        return T2        
        
    # private method
    def __abdmatrix(self):
        '''used within the object but not accessible outside'''
        #==========================================================================
        # ABD Matrix Compute
        #==========================================================================
        # Reduced stiffness matrix for a plane stress ply in principal coordinates
        # calcluating Q from the Compliance matrix may cause cancE1ation errors
    
        A = zeros((3,3)); B = zeros((3,3)); D = zeros((3,3))  
        for i in range(self.nply):  # = nply
            Q = self.__Qf(self.__mat[self.matname[self.matindex[i]]].E1, 
                   self.__mat[self.matname[self.matindex[i]]].E2, 
                   self.__mat[self.matname[self.matindex[i]]].nu12, 
                   self.__mat[self.matname[self.matindex[i]]].G12 )
            
            Qbar = inv(self.__T1(self.plyangle[i])) @ Q @ self.__T2(self.plyangle[i]) # solve(T1(plyangle[i]), Q) @ T2(plyangle[i])
            A += Qbar*(self.z[i+1]-self.z[i])
            # coupling  stiffness
            B += (1/2)*Qbar*(self.z[i+1]**2-self.z[i]**2)
            # bending or flexural laminate stiffness relating moments to curvatures
            D += (1/3)*Qbar*(self.z[i+1]**3-self.z[i]**3)      
        
        # laminate stiffness matrix
        ABD = zeros((6,6))
        ABD[0:3,0:3] = A
        ABD[0:3,3:6] = B
        ABD[3:6,0:3] = B
        ABD[3:6,3:6] = D
        self.ABD = ABD

    
    # method
    def available_materials(self):
        '''show the materials available in the library'''
        matprops = pd.read_csv('./compositematerials.csv', index_col=0)
        print('---available materials---')
        for k in matprops.columns.tolist():
            print(k)
        print('-------------------------')
 
    # private method to be used internally
    def __import_matprops(self, mymaterial=['T300_5208','AL_7075']):
        '''
        import material properties
        '''
    
        matprops = pd.read_csv('./compositematerials.csv', index_col=0)
    
        if mymaterial==[] or mymaterial=='':
            print(matprops.columns.tolist())
                
        mat = matprops[mymaterial]
        #mat.applymap(lambda x:np.float(x))
        mat = mat.applymap(lambda x:pd.to_numeric(x, errors='ignore'))
        return mat    
      

        
if __name__ == '__main__':
    
    plyangle = [0,45]
    matindex = [0,0]
    matname = ['Carbon_cloth_AGP3705H', 'HRH-10-0.125-3.0']
    
    
    lam1 = laminate(plyangle, matindex, matname)
    
    

    

    
  
       