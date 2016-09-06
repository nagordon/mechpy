# coding: utf-8

'''
Module for composite material analysis

Hyer-Stress Analysis of Fiber-Reinforced Composite Materials
Herakovich-Mechanics of Fibrous Composites
Daniel-Engineering Mechanics of Composite Materials
'''

#==============================================================================
# Import Modules
#==============================================================================
from __future__ import print_function, division

__author__ = 'Neal Gordon <nealagordon@gmail.com>'
__date__ =   '2016-09-06'

from copy import copy

from numpy import pi, zeros, ones, linspace, arange, array, sin, cos
from numpy.linalg import solve, inv
#from scipy import linalg
import numpy as np
#np.set_printoptions(suppress=False,precision=2)   # suppress scientific notation
np.set_printoptions(precision=4, linewidth=150)

import pandas as pd

import sympy as sp
from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols, pprint
from sympy.plotting import plot3d  

#from sympy import cos, sin
#sp.init_printing(use_latex='mathjax')
sp.init_printing(wrap_line=False, pretty_print=True)

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

#==============================================================================
# Functions
#==============================================================================


def import_matprops(mymaterial=['T300_5208','AL_7075']):
    '''
    import material properties
    '''

    matprops = pd.read_csv('./compositematerials.csv', index_col=0)

    if mymaterial==[] or mymaterial=='':
        print(matprops.columns.tolist())
            
    mat = matprops[mymaterial]
    return mat


def Sf(E1,E2,nu12,G12):
    '''transversely isptropic compliance matrix. pg 58 herakovich'''
    nu21 = E2*nu12/E1
    S = array([[1/E1,    -nu21/E2, 0],
               [-nu12/E1, 1/E2,    0],
               [0,        0,       1/G12]])
    return S    

def S6f(E1,E2,nu12,nu23,G12):
    '''
    transversely isotropic compliance matrix. 
    For transversly isotropic
    E2=E3, nu12=nu13,G12=G13,G23=E2/(2(1+nu23))
    '''
    G23=E2/(2*(1+nu23))
    S6 = array( [[    1/E1, -nu12/E1, -nu12/E1,     0,     0,       0],  
                 [-nu12/E1,     1/E2, -nu23/E2,     0,     0,       0],  
                 [-nu12/E1, -nu23/E2,     1/E2,     0,     0,       0],
                 [     0,        0,        0,      1/G23,  0,       0],
                 [     0,        0,        0,       0,    1/G12,    0],
                 [     0,        0,        0,       0,     0,   1/G12]])    
    return S6
    
def Qf(E1,E2,nu12,G12):
    '''transversly isptropic compliance matrix. pg 58 herakovich
    G12 = E1/(2*(1+nu12))  if isotropic'''
    nu21 = E2*nu12/E1
    Q = array([[E1/(1-nu12*nu21),    E2*nu12/(1-nu12*nu21), 0],
               [ E2*nu12/(1-nu12*nu21), E2/(1-nu12*nu21),    0],
               [0,        0,       G12]])
    return Q  

def T61(th):
    '''Stress
    th=ply angle in degrees
    voight notation for stress tranform. sigma1 = T1 @ sigmax'''
    n = sin(th*pi/180)
    m = cos(th*pi/180)
    T1 = array( [[m**2, n**2, 0, 0, 0, 2*m*n],  
                 [n**2, m**2, 0, 0, 0,-2*m*n],  
                 [0,    0,    1, 0, 0, 0],
                 [0,    0,    0, m,-n, 0],
                 [0,    0,    0, n, m, 0],
                 [-m*n, m*n,  0, 0, 0,(m**2-n**2)]])    
    return T1
    
def T62(th):
    '''Strain
    voight notation for strain transform. epsilon1 = T2 @ epsilonx
    th=ply angle in degrees
    '''
    n = sin(th*pi/180)
    m = cos(th*pi/180)
    T2 = array( [[m**2, n**2, 0, 0, 0, m*n],  
                 [n**2, m**2, 0, 0, 0,-m*n],  
                 [0,    0,    1, 0, 0, 0],
                 [0,    0,    0, m,-n, 0],
                 [0,    0,    0, n, m, 0],
                 [-2*m*n, 2*m*n,  0, 0, 0,(m**2-n**2)]])     
    return T2


def T1(th):
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
    
def T2(th):
    '''Strain Transform for Plane Stress
    th=ply angle in degrees
    voight notation for strain transform. epsilon1 = T2 @ epsilonx'''
    n = sin(th*pi/180)
    m = cos(th*pi/180)
    T2 = array( [[m**2, n**2, m*n],  
                 [n**2, m**2,-m*n],  
                 [-2*m*n, 2*m*n,  (m**2-n**2)]])     
    return T2

def failure_envelope():
    # failure envelopes

    # max stress criteria
    # 1 direction in first row
    # 2 direction in second row
    
    # failure strength in compression
    #Fc = matrix([[-1250.0, -600.0],         
    #            [-200.0,  -120.0]]) # ksi
    #            
    ##failure strength in tension        
    #Ft =  matrix([[1500, 1000]
    #              [50,     30]]) # ksi
    #              
    ##Failure strength in shear              
    #Fs = matrix( [100,    70] ) # Shear
    
    Fc1 = [-1250, -600] # Compression 1 direction
    Fc2 = [-200,  -120] # Compression 2 direction
    Ft1 = [1500, 1000]  # Tension 1 direction
    Ft2 = [50,     30]  # Tension 2 direction
    Fs =  [100,    70]   # Shear
    
    # F1 = Ft(1);
    # F2 = Ft(1);
    # F6 = Fs(1);
    
    for c in range(2):# mattype
        factor = 1.9
        # right
        plot( [Ft1[c], Ft1[c]] , [Fc2[c], Ft2[c]])
        
        # left
        plot( [Fc1[c], Fc1[c]] , [Fc2[c], Ft2[c]])
        # top
        plot( [Fc1[c], Ft1[c]] , [Ft2[c], Ft2[c]])
        # bottom
        plot( [Fc1[c], Ft1[c]]  , [Fc2[c], Fc2[c]])
        # center horizontal
        plot( [Fc1[c], Ft1[c]]  , [0, 0])
        # center vertical
        plot( [0, 0]            , [Fc2[c], Ft2[c]])
        
        #xlim([min(Fc1) max(Ft1)]*factor)
        #ylim([min(Fc2) max(Ft2)]*factor)
        xlabel('$\sigma_1,ksi$')
        ylabel('$\sigma_2,ksi$')
        title('failure envelope with Max-Stress Criteria')   

def material_plots():
    '''
    plotting composite properties
    '''
#    plt.rcParams['figure.figsize'] = (10, 8)
#    plt.rcParams['font.size'] = 14
#    plt.rcParams['legend.fontsize'] = 14
    
    get_ipython().magic('matplotlib inline') 
    
    plt.close('all')
    
    mat = import_matprops('T300_5208')
    S = Sf6(mat.E1,mat.E2,mat.nu12,mat.nu23,mat.G12 )    
    C = inv(S)
    plyangle = arange(-90, 90.1, 0.1) 
    
    C11 = [(inv(T61(th)) @ C @ T62(th))[0,0] for th in plyangle]
    C22 = [(inv(T61(th)) @ C @ T62(th))[1,1] for th in plyangle]
    C44 = [(inv(T61(th)) @ C @ T62(th))[3,3] for th in plyangle]
    C66 = [(inv(T61(th)) @ C @ T62(th))[5,5] for th in plyangle]
    
    Exbar = zeros(len(plyangle))
    Eybar = zeros(len(plyangle))
    Gxybar = zeros(len(plyangle))

    h = 1      # lamina thickness
    Q = Qf(mat.E1,mat.E2,mat.nu12,mat.G12)

    Qbar = zeros((len(plyangle),3,3))
    for i,th in enumerate(plyangle):
        Qbar[i] = solve(T1(th), Q) @ T2(th)
    #Qbar = [solve(T1(th),Q) @ T2(th) for th in plyangle]

    Qbar11 = Qbar[:,0,0]
    Qbar22 = Qbar[:,1,1]
    Qbar66 = Qbar[:,2,2]
    Qbar12 = Qbar[:,0,1]
    Qbar16 = Qbar[:,0,2]
    Qbar26 = Qbar[:,1,2]  

    Aij = Qbar*h

    # laminate Stiffness
    #     | Exbar    Eybar    Gxybar   |
    # A = | vxybar   vyxbar   etasxbar |
    #     | etaxsbar etaysbar etasybar | 

    # laminate Comnpliance
    aij = zeros((len(plyangle),3,3))
    for i, _Aij in enumerate(Aij):
        aij[i] = inv(_Aij)   

    # material properties for whole laminate (Daniel, pg183)
    Exbar  = [1/(h*_aij[0,0]) for _aij in aij]
    Eybar  = [1/(h*_aij[1,1]) for _aij in aij]
    Gxybar = [1/(h*_aij[2,2]) for _aij in aij]    
    
    # Global Stress
    s_xy = array([[100], 
                  [10], 
                  [5]])
    
    # local ply stress
    s_12 = np.zeros((3,len(plyangle)))
    for i,th in enumerate(plyangle):
        #s_12[:,i] = np.transpose(T1(th) @ s_xy)[0]   # local stresses
        s_12[:,[i]] = T1(th) @ s_xy   
    
    
    # Plotting
    figure()#, figsize=(10,8))
    plot(plyangle, C11, plyangle, C22, plyangle, C44, plyangle, C66)
    legend(['$\overline{C}_{11}$','$\overline{C}_{22}$', '$\overline{C}_{44}$', '$\overline{C}_{66}$'])
    title('Transversly Isotropic Stiffness properties of carbon fiber T300_5208')
    xlabel("$\Theta$")
    ylabel('$\overline{C}_{ii}$, ksi')
    grid()

    figure()#, figsize=(10,8))
    plot(plyangle, Exbar, label = r"Modulus: $E_x$")
    plot(plyangle, Eybar, label = r"Modulus: $E_y$")
    plot(plyangle, Gxybar, label = r"Modulus: $G_{xy}$")
    title("Constitutive Properties in various angles")
    xlabel("$\Theta$")
    ylabel("modulus, GPa")
    legend()
    grid()
    
    figure()#,figsize=(10,8))
    plot(plyangle, s_12[0,:], label = '$\sigma_{11},ksi$' )
    plot(plyangle, s_12[1,:], label = '$\sigma_{22},ksi$' )
    plot(plyangle, s_12[2,:], label = '$\sigma_{12},ksi$' )
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel("Stress, ksi")
    grid()

    # plot plyangle as a function of time
    figure()#,figsize=(10,8))
    plot(plyangle,Qbar11, label = "Qbar11")
    plot(plyangle,Qbar22, label = "Qbar22")
    plot(plyangle,Qbar66, label = "Qbar66")
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel('Q')
    grid()

    # plot plyangle as a function of time
    figure()#,figsize=(10,8))
    plot(plyangle,Qbar12, label = "Qbar12")
    plot(plyangle,Qbar16, label = "Qbar16")
    plot(plyangle,Qbar26, label = "Qbar26")
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel('Q')
    grid()
    
    show()

   
def laminate_gen(lamthk=1.5, symang=[45,0,90], plyratio=2.0, matrixlayers=False, balancedsymmetric=True):
    '''
    ## function created to quickly create laminates based on given parameters
    lamthk=1.5    # total #thickness of laminate
    symang = [45,0,90, 30]  #symmertic ply angle
    plyratio=2.0  # lamina/matrix ratio
    matrixlayers=False  # add matrix layers between lamina plys
    nonsym=False    # symmetric
    mat = material type, as in different plies, matrix layer, uni tapes, etc
    #ply ratio can be used to vary the ratio of thickness between a matrix ply
         and lamina ply. if the same thickness is desired, plyratio = 1, 
         if lamina is 2x as thick as matrix plyratio = 2
    '''
    if matrixlayers:
        nply = (len(symang)*2+1)*2
        nm = nply-len(symang)*2
        nf = len(symang)*2
        tm = lamthk / (plyratio*nf + nm)
        tf = tm*plyratio
        plyangle = zeros(nply//2)
        mat = 2*ones(nply//2)  #  orthotropic fiber and matrix = 1, isotropic matrix=2, 
        mat[1:-1:2] = 1   #  [2 if x%2 else 1 for x in range(nply//2) ]
        plyangle[1:-1:2] = symang[:]  # make a copy
        thk = tm*ones(nply//2)
        thk[2:2:-1] = tf
        lamang = list(symang) + list(symang[::-1])
        plyangle = list(plyangle) + list(plyangle[::-1])
        mat = list(mat) + list(mat[::-1])
        thk = list(thk) + list(thk[::-1])  
    else: # no matrix layers, ignore ratio
        if balancedsymmetric:
            nply = len(symang)*2
            mat = list(3*np.ones(nply)) 
            thk = list(lamthk/nply*np.ones(nply))
            lamang = list(symang) + list(symang[::-1])
            plyangle = list(symang) + list(symang[::-1])
        else:            
            nply = len(symang)
            mat =[1]*nply
            thk = list(lamthk/nply*np.ones(nply))
            lamang = symang[:]
            plyangle = symang[:]

    return thk,plyangle,mat,lamang



def laminate():
    '''
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
    '''
    
    #==========================================================================
    # Initialize
    #==========================================================================
    get_ipython().magic('matplotlib') 
    plt.close('all')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 13
    #plt.rcParams['legend.fontsize'] = 14
    
    #==========================================================================
    # Import Material Properties
    #==========================================================================
    	

    matindex = ['AS4_3501-6']
    mat  = import_matprops(matindex)  
    #mat  = import_matprops('T300_5208')  # Herakovich
    alphaf = lambda mat: array([[mat.alpha1], [mat.alpha2], [0]])
    
    ''' to get ply material info, use as follows
    alpha = alphaf(mat[matindex[plymat[i]]])
    
    mat[matindex[1]].E2
    
    '''
    W =   0.25  # plate width
    L =  .125           # laminate length  
    plyangle = [0,90,90,0]  # angle for each ply
    plymat =   [0,0,0,0]  # material for each ply
    
    laminatethk = array([mat[matindex[i]].plythk for i in plymat ])
    
    nply = len(laminatethk) # number of plies
    H =   sum(laminatethk) # plate thickness
    #    area = W*H
    z = zeros(nply+1)
    zmid = zeros(nply)
    z[0] = -H/2
    for i in range(nply):
        z[i+1] = z[i] + laminatethk[i]
        zmid[i] = z[i] + laminatethk[i]/2
    
    #==========================================================================
    # ABD Matrix Compute
    #==========================================================================
    # Reduced stiffness matrix for a plane stress ply in principal coordinates
    # calcluating Q from the Compliance matrix may cause cancellation errors

    A = zeros((3,3)); B = zeros((3,3)); D = zeros((3,3))  
    for i in range(nply):  # = nply
        Q = Qf(mat[matindex[plymat[i]]].E1, mat[matindex[plymat[i]]].E2, mat[matindex[plymat[i]]].nu12, mat[matindex[plymat[i]]].G12 )
        
        Qbar = inv(T1(plyangle[i])) @ Q   @ T2(plyangle[i]) # solve(T1(plyangle[i]), Q) @ T2(plyangle[i])
        A += Qbar*(z[i+1]-z[i])
        # coupling  stiffness
        B += (1/2)*Qbar*(z[i+1]**2-z[i]**2)
        # bending or flexural laminate stiffness relating moments to curvatures
        D += (1/3)*Qbar*(z[i+1]**3-z[i]**3)      
    
    # laminate stiffness matrix
    ABD = zeros((6,6))
    ABD[0:3,0:3] = A
    ABD[0:3,3:6] = B
    ABD[3:6,0:3] = B
    ABD[3:6,3:6] = D
 
    # laminatee compliance
    abcd = inv(ABD)
    a = abcd[0:3,0:3]
    
    #==========================================================================
    # Laminate Properties    
    #==========================================================================
    
    # effective laminate shear coupling coefficients
    etasxbar = a[0,2]/a[2,2]
    etasybar = a[1,2]/a[2,2]
    etaxsbar = a[2,0]/a[0,0]
    etaysbar = a[2,1]/a[1,1]
    
    # laminate engineer properties
    Exbar  = 1 / (H*a[0,0])
    Eybar  = 1 / (H*a[1,1])
    Gxybar = 1 / (H*a[2,2])
    nuxybar = -a[0,1]/a[0,0]
    nuyxbar = -a[0,1]/a[1,1]
    

    # --------------------- Double Check ---------------------
#    # Laminate compliance matrix
#    LamComp = array([ [1/Exbar,       -nuyxbar/Eybar,  etasxbar/Gxybar],
#                      [-nuxybar/Exbar,  1/Eybar ,       etasybar/Gxybar],
#                      [etaxsbar/Exbar, etaysbar/Eybar, 1/Gxybar]] ) 
#    # Daniel pg 183
#    # combines applied loads and applied strains
#    strain_laminate = LamComp @ Nxyzapplied[:3]/H + strainxyzapplied[:3]
#    Nxyz = A @ strain_laminate
#    stress_laminate = Nxyz/H
    # --------------------------------------------------------
    

    #==========================================================================
    #  Applied Loads 
    #==========================================================================
    # either apply strains or loads 
            #               Nx Ny  Nxy  Mx  My Mxy 
    NMbarapp =      array([[0],[0],[0],[0],[0],[0]])
    #                       ex ey exy  kx  ky kxy
    epsilonbarapp = array([[5e-3],[0],[0],[0],[0],[0]]) 
    
    NMbarapptotal = NMbarapp + ABD@epsilonbarapp
    #==========================================================================
    # Thermal Loads  
    #==========================================================================
    '''
    if the material is isotropic and unconstrained, then no thermal stresses 
        will be experienced. If there are constraints, then the material will experience
        thermally induced stresses. As with orthotropic materials, various directions will have 
        different stresses, and when stacked in various orientations, stresses can be 
        unintuitive and complicated. Global Thermal strains are subtracted from applied strains 
    # 1) determine the free unrestrained thermal strains in each layer, alphabar
    '''
    
    Ti = 0   # initial temperature (C)
    Tf = 0 # final temperature (C)
    dT = Tf-Ti 
    
    Nhatth= zeros((3,1))  # unit thermal force in global CS
    Mhatth = zeros((3,1)) # unit thermal moment in global CS
    alphabar = zeros((3,nply))    # global ply CTE 
    for i in range(nply):  # = nply
        Q = Qf(mat[matindex[plymat[i]]].E1, mat[matindex[plymat[i]]].E2, mat[matindex[plymat[i]]].nu12, mat[matindex[plymat[i]]].G12 )
        alpha = alphaf(mat[matindex[plymat[i]]])
        Qbar = inv(T1(plyangle[i])) @ Q   @ T2(plyangle[i])
        alphabar[:,[i]] = solve(T2(plyangle[i]),  alpha)
        #alphabar[:,[i]] = inv(T2(plyangle[i])) @ alpha # Convert to global CS    
        Nhatth += Qbar @ (alphabar[:,[i]])*(z[i+1] - z[i]) # Hyer method for calculating thermal unit loads
        Mhatth += 0.5*Qbar@(alphabar[:,[i]])*(z[i+1]**2-z[i]**2)   
        
    NMhatth = np.vstack((Nhatth,Mhatth))
    NMbarth = NMhatth*dT # resultant thermal loads    
    
    # Laminate CTE
    epsilonhatth = abcd@NMhatth # laminate CTE
    
    # applied loads and thermal loads
    epsilonbarapp = abcd @ NMbarapptotal
    epsilonbarth  = abcd @ NMbarth  # resultant thermal strains
    epsilonbartotal = epsilonbarapp + epsilonbarth
    
    # Composite respone from applied mechanical loads and strains. Average
    # properties only. Used to compare results from tensile test.
    #epsilon_laminate = abcd@NMbarapptotal
    #sigma_laminate = ABD@epsilon_laminate/H    
    
    epsilon_laminate = epsilonbartotal[:]
    sigma_laminate = ABD@epsilonbartotal/H    
    alpha_laminate = a@Nhatth

    # determine thermal load and applied loads or strains Hyer pg 435,452
    Nx = NMbarapptotal[0,0]*W # units kiloNewtons, total load as would be applied in a tensile test
    Ny = NMbarapptotal[1,0]*L # units kN 

    #==========================================================================
    # Thermal and mechanical local and global stresses at the ply interface
    #==========================================================================
    # Declare variables for plotting
    epsilon_app         = zeros((3,2*nply))    
    sigma_app           = zeros((3,2*nply))    
    epsilonbar_app      = zeros((3,2*nply))
    sigmabar_app        = zeros((3,2*nply))
    epsilon_th          = zeros((3,2*nply))
    sigma_th            = zeros((3,2*nply))    
    epsilonbar_th       = zeros((3,2*nply))
    sigmabar_th         = zeros((3,2*nply))
    epsilon             = zeros((3,2*nply))    
    epsilonbar          = zeros((3,2*nply))
    sigma               = zeros((3,2*nply))    
    sigmabar            = zeros((3,2*nply))
    
    for i,k in enumerate(range(0,2*nply,2)):
        # stress is calcuated at top and bottom of each ply
        Q = Qf(mat[matindex[plymat[i]]].E1, mat[matindex[plymat[i]]].E2, mat[matindex[plymat[i]]].nu12, mat[matindex[plymat[i]]].G12 )
        Qbar = inv(T1(plyangle[i])) @ Q   @ T2(plyangle[i])

         # Global stresses and strains, applied load only
        epsbarapp1 = epsilonbarapp[0:3] + z[i]*epsilonbarapp[3:7]
        epsbarapp2 = epsilonbarapp[0:3] + z[i+1]*epsilonbarapp[3:7]
        sigbarapp1 = Qbar @ epsbarapp1
        sigbarapp2 = Qbar @ epsbarapp2
        # Local stresses and strains, appplied load only
        epsapp1 = T2(plyangle[i]) @ epsbarapp1
        epsapp2 = T2(plyangle[i]) @ epsbarapp2
        sigapp1 = Q @ epsapp1
        sigapp2 = Q @ epsapp2
        # Interface Stresses and Strains
        epsilon_app[:,k:k+2]    = np.column_stack((epsapp1,epsapp2))
        epsilonbar_app[:,k:k+2] = np.column_stack((epsbarapp1,epsbarapp2))
        sigma_app[:,k:k+2]      = np.column_stack((sigapp1,sigapp2))
        sigmabar_app[:,k:k+2]   = np.column_stack((sigbarapp1,sigbarapp2))

        # Global stress and strains, thermal loading only
        epsbarth1 = epsilonbarth[0:3] + z[i]*epsilonbarth[3:7]   - dT*alphabar[:,[i]]
        epsbarth2 = epsilonbarth[0:3] + z[i+1]*epsilonbarth[3:7] - dT*alphabar[:,[i]]
        sigbarth1 = Qbar @ epsbarth1
        sigbarth2 = Qbar @ epsbarth2   
        
        # Local stress and strains, thermal loading only
        epsth1 = T2(plyangle[i]) @ epsbarth1
        epsth2 = T2(plyangle[i]) @ epsbarth2
        sigth1 = Q @ epsth1
        sigth2 = Q @ epsth2
        
        # Interface Stresses and Strains
        epsilon_th[:,k:k+2]    = np.column_stack((epsth1,epsth2))
        epsilonbar_th[:,k:k+2] = np.column_stack((epsbarth1+dT*alphabar[:,[i]],epsbarth2+dT*alphabar[:,[i]])) # remove the local thermal loads for plotting. only use local thermal strains for calculating stress
        sigma_th[:,k:k+2]      = np.column_stack((sigth1,sigth2))
        sigmabar_th[:,k:k+2]   = np.column_stack((sigbarth1,sigbarth2))       
        
        # TOTAL global stresses and strains, applied and thermal
        epsbar1 = epsbarapp1 + epsbarth1
        epsbar2 = epsbarapp2 + epsbarth2
        sigbar1 = Qbar @ epsbar1
        sigbar2 = Qbar @ epsbar2
        # TOTAL local stresses and strains , applied and thermal
        eps1 = T2(plyangle[i]) @ epsbar1
        eps2 = T2(plyangle[i]) @ epsbar2
        sig1 = Q @ eps1
        sig2 = Q @ eps2
        # Interface Stresses and Strains
        epsilon[:,k:k+2]     = np.column_stack((eps1,eps2))
        epsilonbar[:,k:k+2]  = np.column_stack((epsbar1+dT*alphabar[:,[i]],epsbar2+dT*alphabar[:,[i]])) # remove the local thermal loads for plotting. only use local thermal strains for calculating stress
        sigma[:,k:k+2]       = np.column_stack((sig1,sig2))
        sigmabar[:,k:k+2]    = np.column_stack((sigbar1,sigbar2))   
            
            
    #==========================================================================
    # Failure Calculations            
    #==========================================================================
    
    # Max Stress            
    SR = zeros((3,2*nply)) 
    TS = zeros((nply))
    for i,k in enumerate(range(0,2*nply,2)):
        
        s1 = sigma[0,k]
        s2 = sigma[1,k]
        s12 = sigma[2,k]
                    
        F1 =  mat[matindex[plymat[i]]].F1t  if s1 > 0 else  mat[matindex[plymat[i]]].F1c
        F2 =  mat[matindex[plymat[i]]].F2t  if s2 > 0 else  mat[matindex[plymat[i]]].F2c
        F12 = mat[matindex[plymat[i]]].F12t if s12 > 0 else mat[matindex[plymat[i]]].F12c

        #Tsai Hill
        TS[i] = s1**2/F1**2 + s2**2/F2**2 + s12**2/F12**2 - s1*s2/F2**2
        
        # strength ratio, if < 1, then fail, 
        SR[0,k:k+2] = s1 / F1  
        SR[1,k:k+2] = s2 / F2
        SR[2,k:k+2] = s12 / F12
            
       
    #==========================================================================
    # Printing Results    
    #==========================================================================
    print('--------------- laminate1 Stress analysis of fibers----------')
    print('plyangles'); print(plyangle)
    print('ply layers') ; print(z)
    print('alpha')
    print(alpha)
    print('ABD=');print(ABD)   
    print('Ex=   %0.2E'%Exbar)
    print('Ey=   %0.2E'%Eybar)
    print('nuxy= %0.2E'%nuxybar)
    print('Gxy=  %0.2E'%Gxybar)
    print('alpha_laminate')
    print(alpha_laminate)
    print('epsilon_laminate')
    print(epsilon_laminate)
    print('NMhatth')    
    print(NMhatth)    
    print('sigma_laminate')
    print(sigma_laminate)
    print('epsilon_th')   
    print(epsilon_th) 
    print('epsilonbar_th')
    print(epsilonbar_th)
    print('sigma_th')
    print(sigma_th)
    print('sigmabar_th')       
    print(sigmabar_th)    
    print('epsilon_app')
    print(epsilon_app)
    print('epsilonbar_app')
    print(epsilonbar_app)
    print('NMbarapp')
    print(NMbarapp)
    print('epsilon')   
    print(epsilon) 
    print('epsilonbar')
    print(epsilonbar)
    print('sigma')
    print(sigma)
    print('sigmabar')    
    print(sigmabar)
    print('Stress Ratio')
    print(SR)
    print('Tsai-Hill Failure')
    print(TS)
    #display(sp.Matrix(sigmabar))
    

            
    #==========================================================================
    # Plotting
    #==========================================================================
    zplot = zeros(2*nply)
    for i,k in enumerate(range(0,2*nply,2)):  # = nply
        zplot[k:k+2] = z[i:i+2]  
        
    #legendlab = ['total','thermal','applied','laminate']
    # global stresses and strains
    mylw = 1.5 #linewidth
    # Global Stresses and Strains
    f1, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex='row', sharey=True)
    f1.canvas.set_window_title('Global Stress and Strain of %s laminate' % (plyangle))
    stresslabel = ['$\sigma_x$','$\sigma_y$','$\\tau_{xy}$']
    strainlabel = ['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$']
    
    for i,ax in enumerate([ax1,ax2,ax3]):
        ## the top axes
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(strainlabel[i])
        #ax.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        ax.plot(epsilonbar[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(epsilonbar_th[i,:],  zplot, color='red', lw=mylw, alpha=0.75, linestyle='--',  label='thermal')
        ax.plot(epsilonbar_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied') 
        ax.plot([epsilon_laminate[i], epsilon_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate') 
        ax.grid(True)  
        #ax.set_xticks(linspace( min(ax.get_xticks()) , max(ax.get_xticks()) ,6))           
    
    for i,ax in enumerate([ax4,ax5,ax6]):
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(stresslabel[i])
        #ax.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        ax.plot(sigmabar[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(sigmabar_th[i,:], zplot, color='red', lw=mylw, alpha=0.75,linestyle='--', label='thermal')
        ax.plot(sigmabar_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied')  
        ax.plot([sigma_laminate[i], sigma_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate') 
        ax.grid(True)

    leg = legend(fancybox=True) ; leg.get_frame().set_alpha(0.3)     
    tight_layout() 
    #mngr = plt.get_current_fig_manager() ; mngr.window.setGeometry(50,50,800, 500)
    f1.show()             
    #plt.savefig('global-stresses-strains.png')    
    ### Local Stresses and Strains
    f2, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex='row', sharey=True)
    f2.canvas.set_window_title('Local Stress and Strain of %s laminate' % (plyangle))
    stresslabel = ['$\sigma_1$','$\sigma_2$','$\\tau_{12}$']
    strainlabel = ['$\epsilon_1$','$\epsilon_2$','$\gamma_{12}$']
    
    for i,ax in enumerate([ax1,ax2,ax3]):
        ## the top axes
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(strainlabel[i])
        #ax.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        ax.plot(epsilon[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(epsilon_th[i,:], zplot, color='red', lw=mylw, alpha=0.75,linestyle='--', label='thermal')
        ax.plot(epsilon_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied')   
        ax.plot([epsilon_laminate[i], epsilon_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate')
        ax.grid(True)
                          
    for i,ax in enumerate([ax4,ax5,ax6]):
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(stresslabel[i])
        #ax.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        ax.plot(sigma[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(sigma_th[i,:], zplot, color='red', lw=mylw, alpha=0.75,linestyle='--', label='thermal')
        ax.plot(sigma_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied')   
        ax.plot([sigma_laminate[i], sigma_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate')   
        ax.grid(True)
    

    ### Failure
    f3, ((ax1,ax2,ax3)) = plt.subplots(1,3, sharex=True, sharey=True)
    f3.canvas.set_window_title('Failure Ratios %s laminate' % (plyangle))
    stresslabel = ['$\sigma_1/F_1$','$\sigma_2/F_2$','$\\tau_{12}/F_{12}$']
    for i,ax in enumerate([ax1,ax2,ax3]):
        ## the top axes
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(stresslabel[i])
        #ax.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        ax.plot(SR[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.grid(True)
    leg = legend(fancybox=True) ; leg.get_frame().set_alpha(0.3)   
    tight_layout() 
    #mngr = plt.get_current_fig_manager() ; mngr.window.setGeometry(850,50,800, 500)
    f2.show()
    #plt.savefig('local-stresses-strains.png')
   
    ### warpage
    res = 100
    Xplt,Yplt = np.meshgrid(np.linspace(-W/2,W/2,res), np.linspace(-L/2,L/2,res))
    epsx = epsilon_laminate[0,0]
    epsy = epsilon_laminate[1,0]
    epsxy = epsilon_laminate[2,0]
    kapx = epsilon_laminate[3,0]
    kapy = epsilon_laminate[4,0]
    kapxy = epsilon_laminate[5,0]
    ### dispalcement
    w = -0.5*(kapx*Xplt**2 + kapy*Yplt**2 + kapxy*Xplt*Yplt)
    u = epsx*Xplt  # pg 451 hyer
    fig = plt.figure('plate-warpage')
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xplt, Yplt, w+zmid[0], cmap=mpl.cm.jet, alpha=0.3)
    ###ax.auto_scale_xyz([-(W/2)*1.1, (W/2)*1.1], [(L/2)*1.1, (L/2)*1.1], [-1e10, 1e10])
    ax.set_xlabel('plate width,y-direction,in')
    ax.set_ylabel('plate length,x-direction, in')
    ax.set_zlabel('warpage,in')
    #ax.set_zlim(-0.01, 0.04)
    #mngr = plt.get_current_fig_manager() ; mngr.window.setGeometry(450,550,600, 450)
    plt.show()
    #plt.savefig('plate-warpage')   


def plate():
    '''
    composite plate mechanics
    
    TODO - results need vetted
    '''
      
      
    #==========================================================================
    # Initialize
    #==========================================================================
    get_ipython().magic('matplotlib') 
    plt.close('all')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 13
    #plt.rcParams['legend.fontsize'] = 14
     
    #==========================================================================
    # Import Material Properties
    #==========================================================================
 
    plythk = 0.0025
    plyangle = array([0,90,90,0]) * np.pi/180 # angle for each ply
    nply = len(plyangle) # number of plies
    laminatethk = np.zeros(nply) + plythk
    H =   sum(laminatethk) # plate thickness

    a =   20  # plate width;
    b =   10  # plate height
    q0_ = 5.7 # plate load;
    # Transversly isotropic material properties
    Ell = 150e9
    Ett = 12.1e9
    vlt = 0.248
    Glt = 4.4e9
    vtt = 0.458
    Gtt = Ett / (2*(1+vtt))
    # Failure Strengths
    SLLt =  1500e6
    SLLc = -1250e6
    STTt =  50e6
    STTc = -200e6
    SLTs =  100e6
    Sxzs =  100e6
    Strength = np.array([[SLLt, SLLc],
                            [STTt, STTc],
                            [SLTs, Sxzs]])
                                
    ## Stiffness Matrix
    th = sp.symbols('th')   # th = sp.var('th')
    # tranformation
    Tij6 = sp.Matrix([[cos(th)**2, sin(th)**2, 0, 0, 0, -sin(2*th)],
                      [sin(th)**2, cos(th)**2, 0, 0, 0,  sin(2*th)],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, cos(th), sin(th), 0],
                      [0, 0, 0, -sin(th), cos(th), 0],
                      [cos(th)*sin(th), -cos(th)*sin(th), 0, 0, 0, (cos(th)**2-sin(th)**2)]])
    
    Tij = sp.Matrix([[cos(th)**2, sin(th)**2, 2*sin(th)*cos(th)],
                     [sin(th)**2, cos(th)**2, -2*sin(th)*cos(th)],
                     [-cos(th)*sin(th), sin(th)*cos(th), (cos(th)**2-sin(th)**2)]])
    
    # compliance matrix
    Sij6 = np.array([[1/Ell, -vlt/Ell, -vlt/Ell, 0, 0, 0],
                      [-vlt/Ell, 1/Ett, -vtt/Ett, 0, 0, 0],
                      [-vlt/Ell, -vtt/Ett, 1/Ett, 0, 0, 0],
                      [0, 0, 0, 1/Gtt, 0, 0],
                      [0, 0, 0, 0, 1/Glt, 0],
                      [0, 0, 0, 0, 0, 1/Glt]])
       
    # Stiffnes matrix in material coordinates
    Cijm6 = inv(Sij6)
    
    
    # Stiffness matrix in Structural coordinates
    Cij6 = Tij6 @ Cijm6 @ Tij6.inv()
    
    # reduced stiffness in structural
    Cij = sp.Matrix([[Cij6[0,0], Cij6[0,1], 0],
                     [Cij6[0,1], Cij6[1,1], 0],
                     [0, 0, Cij6[5,5] ]] )

    
    # Create z dimensions of laminate
    z_ = np.linspace(-H/2, H/2, nply+1)
    
    # extensional stiffness
    Aij = sp.Matrix(np.zeros((6,6)))
    for i in range(nply):
        Aij += Cij6.evalf(subs={th:plyangle[i]}) * (z_[i+1]-z_[i])
        
    # coupling  stiffness
    Bij = sp.Matrix(np.zeros((6,6)))
    for i in range(nply):
        Bij += 0.5 * Cij6.evalf(subs={th:plyangle[i]}) * (z_[i+1]**2-z_[i]**2)
    
    # bending or flexural laminate stiffness relating moments to curvatures
    Dij = sp.Matrix(np.zeros((6,6)))
    for i in range(nply):
        Dij += (1/3.0)* Cij6.evalf(subs={th:plyangle[i]}) * (z_[i+1]**3-z_[i]**3)
    
    
    ## Cylindrical Bending of a laminated plate
    
    # displacement in w (z direction)  
    from sympy.abc import x  
    f = Function('f')
    eq = dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x), hint = '1st_homogeneous_coeff_best', simplify=False)
    pprint(eq)
    #==============================================================================
    
    th,x,y,z,q0,C1,C2,C3,C4,C5,C6,C7,A11,B11,D11,A16,B16 = symbols('th x y z q0 C1 C2 C3 C4 C5 C6 C7 A11 B11 D11 A16 B16')
    
    wfun = Function('wfun')
    ufun = Function('ufun')
    
    ## EQ 4.4.1a
    eq1 = A11*ufun(x).diff(x,2) - B11*wfun(x).diff(x,3)
    #eq1   = A11*diff(ufun,x,2) - B11*diff(wfun,x,3); # C5 C1
    
    ## EQ 4.4.1b
    #eq2   = A16*diff(ufun,x,2) - B16*diff(wfun,x,3); # C5 C1
    eq2 = A16*ufun(x).diff(x,2) - B16*wfun(x).diff(x,3)
    
    ## EQ 4.4.1c
    #eq3 = B11*diff(ufun,x,3) - D11*diff(wfun,x,4) + q0;
    eq3 = B11*ufun(x).diff(x,3) - D11*wfun(x).diff(x,4) + q0
    
    ################## python conversion eded here ################################
    
    # solve eq1 eq2 and eq3 to get the w and u functions
    
    # displacement in w (z direction) from eq1,eq2,eq3
    wfun = A11*q0*x**4 / (4*(6*B11**2-6*A11*D11)) + C1 + C2*x + C3*x**2 + C4*x**3 #  C1 C2 C3 C4
    
    # displacement in u (x direction) from eq1,eq2,eq3
    ufun = B11*q0*x**3 / (6*(B11**2-A11*D11)) + C7 + x*C6 + 3*B11*x**2*C5/A11 # C5 C6 C7
    
    # Cij6.evalf(subs={th:plyangle[i]}) * (z_[i+1]**3-z_[i]**3)
    
    # cond1 -> w(0)=0 at x(0), roller
    C1sol = sp.solve(wfun.subs(x,0), C1)[0] # = 0
    # cond2 -> angle at dw/dx at x(0) is 0, cantilever
    C2sol = sp.solve(wfun.diff(x).subs(x,0),C2)[0]  # =  0
    # cond3 -> w(z) = 0 at x(a), roller
    C4sol1 =  sp.solve(wfun.subs({x:a,C1:C1sol,C2:C2sol}),C4)[0] # C3
    # cond4 u = 0 at x = 0
    C7sol = sp.solve(ufun.subs(x,0),C7)[0] #=0
    # u=0 at x = a
    C5sol1 = sp.solve(ufun.subs({x:a, C7:C7sol}),C5)[0] #C6
    # cond 5 EQ 4.4.14a Myy = 0 @ x(a) (Mxx , B11 D11) (Myy, B12 D12) roller no moment
    C6sol1 = sp.solve( ( ((B11*ufun.diff(x)+0.5*wfun.diff(x)**2 ) - D11*wfun.diff(x,2)).subs({x:a, C1:C1sol, C2:C2sol, C4:C4sol1, C5:C5sol1, C7:C7sol})), C6)[0] # C6 C3 
    # EQ 4.4.13a, Nxx = 0 @ x(0) roller has no Nxx
    C6sol2 = sp.solve( ((A11* ufun.diff(x) + 0.5*wfun.diff(x)**2)-B11*wfun.diff(x,2)).subs({x:a, C1:C1sol, C2:C2sol, C4:C4sol1, C5:C5sol1, C7:C7sol}),C6)[0] # C6 C3 
    C3sol = sp.solve(C6sol1 - C6sol2,C3)[0]
    C4sol = C4sol1.subs(C3,C3sol)
    C6sol = sp.simplify(C6sol2.subs(C3,C3sol))
    C5sol = sp.simplify(C5sol1.subs(C6,C6sol))
    # substitute integration constants with actual values( _ is actual number)
    C1_ = copy(C1sol)
    C2_ = copy(C2sol)
    C7_ = copy(C7sol)
    C3_ = C3sol.subs({q0:q0_, A11:Aij[0,0], B11:Bij[0,0], D11:Dij[0,0]})
    C4_ = C4sol.subs({q0:q0_, A11:Aij[0,0], B11:Bij[0,0], D11:Dij[0,0]})
    C5_ = C5sol.subs({q0:q0_, A11:Aij[0,0], B11:Bij[0,0], D11:Dij[0,0]})
    C6_ = C6sol.subs({q0:q0_, A11:Aij[0,0], B11:Bij[0,0], D11:Dij[0,0]})
    
    # function w(x) vertical displacement w along z with actual vaules
    wsol = wfun.subs({q0:q0_, C1:C1_, C2:C2_, C3:C3_, C4:C4_,  A11:Aij[0,0], B11:Bij[0,0], D11:Dij[0,0]}) 
    # function u(x) horizontal displacement u along x with actual vaules
    usol = ufun.subs({q0:q0_, C5:C5_, C6:C6_, C7:C7_,  A11:Aij[0,0], B11:Bij[0,0], D11:Dij[0,0]}) 

    # 3d plots
    plot3d(wsol,(x,0,a), (y,0,b)) 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cylindrical Bending -Displacement of a plate With CLPT')
    
    ## Strain calculation
    # eq 3.3.8 (pg 116 reddy (pdf = 138))
    epstotal = array([[usol.diff(x) + 0.5* wsol.diff(x)**5 - z*wsol.diff(x,2)],[0],[0]])
    epsx = epstotal[0,0]
    ## Calculating and plotting Stress in each layer
    res = 8 # accuracy of finding max and min stress
    xplot = linspace(0,a,res)
    yplot = linspace(0,b,res)
    G0 = sp.symbols('G0')
    Globalminstress = np.zeros((3, nply))
    Globalmaxstress = np.zeros((3, nply))
    
    for kstress in range(3): # stress state s_x, s_y, s_xz
        plt.figure(kstress+1)

        for klay in range(nply): # loop through all layers
            thplot = plyangle[klay]
            zplot = linspace(z_[klay],z_[klay+1],res)
            stressplot = np.zeros((len(zplot),len(xplot)))
            ## Calc Stresses
            if kstress == 2:
                # Shear stresses
                
                G0_ = -sp.integrate(s_stress[0].diff(x),z)+G0
                # solve for shear stresses from s_1
                s_xz = sp.solve(G0_,G0)[0] 
                # out of plane shear S_xz does not need to be transformed ??
                plot3d(s_xz, (x,0, a), (z, z_[klay], z_[klay+1]) ) 
            else:
                # normal stresses
                # Cij = reduced structural stiffness in strictural coordinates 3x3
                # stress in structural coordinates
                s_stress = Cij.subs(th,thplot) @ epstotal
                # stressin material coordinates
                m_stress = Tij.subs(th,thplot) @ s_stress        
                
                #ezsurf(m_stress(kstress),[0,a,z_(klay),z_(klay+1)])
                 
            ## find max stress in each layer
            ii=0
            for i in xplot:
                jj=0
                for j in zplot:
                    if kstress == 2:
                        stressplot[ii,jj] = s_xz.subs({x:i, z:j})
                    else:
                        stressplot[ii,jj] = m_stress[kstress].subs({x:i, z:j})
                    jj+=jj
                ii+=ii
   
            Globalminstress[kstress,klay] = np.min(stressplot)
            Globalmaxstress[kstress,klay] = np.max(stressplot)
            #

        plt.title('\sigma_%i' % kstress)

    ## Plot max stress and failure strength
    plt.figure()
    for i in range(3):

        plt.subplot(1, 3, i+1)
        plt.bar(range(nply), Globalmaxstress[i,:])

        plt.bar(range(nply), Globalminstress[i,:])
        plt.scatter(range(nply),np.ones(nply) * Strength[i,0])
        plt.scatter(range(nply),np.ones(nply) * Strength[i,1])

        plt.xlabel('layer')
        plt.title('\sigma%i' % i)


def plate_navier():
    '''
    composite plate bending with navier solution
    
    TODO - code needs to be converted from matlab
    '''
    
    ## Plate a*b*h simply supported under q = q0 CLPT
    
    pass
    '''
    q0,a,b,m,n,x,y = sp.symbols('q0 a b m n x y')    

    Qmn = 4/(a*b)*sp.integrate( sp.integrate( q0*sp.sin(m*pi*x/a)*sp.sin(n*pi*y/b),(x,0,a)) ,(y,0,b))
    
    dmn = pi**4 / b**4 * (DTij(1,1)*m**4*(b/a)**4 + 2* (DTij(1,2)+2*DTij(6,6)) *m**2*n**2*(b/a)**2 + DTij(2,2)*n**4)
    
    Wmn = Qmn/dmn;
    
    w0 = Wmn * sin(m*pi*x/a) * sin(n*pi*y/b);
    
    w0_ = subs(w0,[q0 a b],[-q0_ a_ b_] );
    
    figure
    w0sum = 0;
    for n_ = 1:10
        for m_ = 1:10
            w0sum = w0sum + subs(w0_,[n m],[n_ m_]);
        end
    end
    w0sum;
    
    % xplot = linspace(0,a_,res);
    % yplot = linspace(0,b_,res);
    
    ii=1;
    for i = xplot
        jj=1;
        for j = yplot
            w0plot(ii,jj) = subs(w0sum,[x y],[i j]);
            jj=jj+1;
        end
        ii=ii+1;
    end
    
    surf(xplot,yplot,w0plot)
    colorbar
    set(gca,'PlotBoxAspectRatio',[2 1 1]);
    xlabel('length a, u(x)')
    ylabel('length b, v(y)')
    zlabel('w(z)')
    '''

if __name__=='__main__':
    
    
    #material_plots()
    #laminate()
    plate()





