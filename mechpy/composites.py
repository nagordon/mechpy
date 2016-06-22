# coding: utf-8

'''
Module to be used for composite material analysis
'''

from __future__ import print_function
from __future__ import division
from numpy import pi, zeros, ones, linspace, arange, array, sin, cos
from numpy.linalg import solve, inv
#from scipy import linalg
import numpy as np
#np.set_printoptions(suppress=False,precision=2)   # suppress scientific notation
np.set_printoptions(precision=3, linewidth=200)

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,figure,xlim,ylim,title,legend, \
grid, show, xlabel,ylabel, tight_layout
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d  

mpl.rcParams['figure.figsize'] = (8,5)
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14


# inline plotting
from IPython import get_ipython
#get_ipython().magic('matplotlib inline')

###disable inline plotting
get_ipython().magic('matplotlib')    
    


def import_matprops(mymaterial='T300_5208'):
    '''
    import material properties
    '''
    matprops = pd.read_csv('./mechpy/compositematerials.csv', index_col=0)
    mat = matprops[mymaterial]
    return mat

def Sf6(E1,E2,nu12,nu23,G12):
    '''transversly isptropic compliance matrix. pg 58 herakovich'''
    S = array([ [1/E1, -nu12/E1, -nu12/E1,0,0,0],
              [-nu12/E1,1/E2,-nu23/E2,0,0,0],
              [-nu12/E1, -nu23/E2, 1/E2,0,0,0],
              [0,0,0,2*(1+nu23)/E2,0,0],
              [0,0,0,0,1/G12,0],
              [0,0,0,0,0,1/G12]])
    return S


def Sf(E1,E2,nu12,G12):
    '''transversly isptropic compliance matrix. pg 58 herakovich'''
    nu21 = E2*nu12/E1
    S = array([[1/E1,    -nu21/E2, 0],
               [-nu12/E1, 1/E2,    0],
               [0,        0,       1/G12]])
    return S    


def Qf(E1,E2,nu12,G12):
    '''transversly isptropic compliance matrix. pg 58 herakovich'''
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
    theta = arange(-90, 90.1, 0.1) 
    
    C11 = [(inv(T61(th)) @ C @ T62(th))[0,0] for th in theta]
    C22 = [(inv(T61(th)) @ C @ T62(th))[1,1] for th in theta]
    C44 = [(inv(T61(th)) @ C @ T62(th))[3,3] for th in theta]
    C66 = [(inv(T61(th)) @ C @ T62(th))[5,5] for th in theta]
    
    Exbar = zeros(len(theta))
    Eybar = zeros(len(theta))
    Gxybar = zeros(len(theta))

    h = 1      # lamina thickness
    Q = Qf(mat.E1,mat.E2,mat.nu12,mat.G12)

    Qbar = zeros((len(theta),3,3))
    for i,th in enumerate(theta):
        Qbar[i] = solve(T1(th), Q) @ T2(th)
    #Qbar = [solve(T1(th),Q) @ T2(th) for th in theta]

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
    aij = zeros((len(theta),3,3))
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
    s_12 = np.zeros((3,len(theta)))
    for i,th in enumerate(theta):
        #s_12[:,i] = np.transpose(T1(th) @ s_xy)[0]   # local stresses
        s_12[:,[i]] = T1(th) @ s_xy   
    
    
    # Plotting
    figure()#, figsize=(10,8))
    plot(theta, C11, theta, C22, theta, C44, theta, C66)
    legend(['$\overline{C}_{11}$','$\overline{C}_{22}$', '$\overline{C}_{44}$', '$\overline{C}_{66}$'])
    title('Transversly Isotropic Stiffness properties of carbon fiber T300_5208')
    xlabel("$\Theta$")
    ylabel('$\overline{C}_{ii}$, ksi')
    grid()

    figure()#, figsize=(10,8))
    plot(theta, Exbar, label = r"Modulus: $E_x$")
    plot(theta, Eybar, label = r"Modulus: $E_y$")
    plot(theta, Gxybar, label = r"Modulus: $G_{xy}$")
    title("Constitutive Properties in various angles")
    xlabel("$\Theta$")
    ylabel("modulus, GPa")
    legend()
    grid()
    
    figure()#,figsize=(10,8))
    plot(theta, s_12[0,:], label = '$\sigma_{11},ksi$' )
    plot(theta, s_12[1,:], label = '$\sigma_{22},ksi$' )
    plot(theta, s_12[2,:], label = '$\sigma_{12},ksi$' )
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel("Stress, ksi")
    grid()

    # plot theta as a function of time
    figure()#,figsize=(10,8))
    plot(theta,Qbar11, label = "Qbar11")
    plot(theta,Qbar22, label = "Qbar22")
    plot(theta,Qbar66, label = "Qbar66")
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel('Q')
    grid()

    # plot theta as a function of time
    figure()#,figsize=(10,8))
    plot(theta,Qbar12, label = "Qbar12")
    plot(theta,Qbar16, label = "Qbar16")
    plot(theta,Qbar26, label = "Qbar26")
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
        nl = (len(symang)*2+1)*2
        nm = nl-len(symang)*2
        nf = len(symang)*2
        tm = lamthk / (plyratio*nf + nm)
        tf = tm*plyratio
        theta = zeros(nl//2)
        mat = 2*ones(nl//2)  #  orthotropic fiber and matrix = 1, isotropic matrix=2, 
        mat[1:-1:2] = 1   #  [2 if x%2 else 1 for x in range(nl//2) ]
        theta[1:-1:2] = symang[:]  # make a copy
        thk = tm*ones(nl//2)
        thk[2:2:-1] = tf
        lamang = list(symang) + list(symang[::-1])
        theta = list(theta) + list(theta[::-1])
        mat = list(mat) + list(mat[::-1])
        thk = list(thk) + list(thk[::-1])  
    else: # no matrix layers, ignore ratio
        if balancedsymmetric:
            nl = len(symang)*2
            mat = list(3*np.ones(nl)) 
            thk = list(lamthk/nl*np.ones(nl))
            lamang = list(symang) + list(symang[::-1])
            theta = list(symang) + list(symang[::-1])
        else:            
            nl = len(symang)
            mat =[1]*nl
            thk = list(lamthk/nl*np.ones(nl))
            lamang = symang[:]
            theta = symang[:]

    return thk,theta,mat,lamang



def laminate1():
    '''
    code to compute composite properties
    
    general outline for computing elastic properties of composites
    
    1) Determine engineering properties of unidirectional laminate. E1, E2, nu12, G12
    2) Calculate ply stiffnesses Q11, Q22, Q12, Q66 in the principal/local coordinate system
    3) Determine Fiber orientation of each ply
    4) Calculate the transformed stiffness Qxy in the global coordinate system
    5) Determine the through-thicknesses of each ply
    6) Determine the laminate stiffness Matrix (ABD)
    7) Calculate the laminate compliance matrix by inverting the ABD matrix
    8) Calculate the laminate engineering properties
    '''
    
    get_ipython().magic('matplotlib') 
    
    plt.close('all')
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 13
#    plt.rcParams['legend.fontsize'] = 14
    
    
    mat  = import_matprops('T300_5208')
    alpha = array([[mat.alpha1], [mat.alpha2], [0]])
    
    W =   10  # plate width
    L =  4           # laminate length  
    plyangle = [0,45,90]# [0, 90, 90, 0]  # [0 90 90 0]# [90 0 90 0 0 90 0 90] # [30 -30 0 0 -30 30]  #
    laminatethk =   zeros(len(plyangle)) + mat.plythk  # ply thicknesses
    nply = len(laminatethk) # number of plies
    H =   mat.plythk*nply # plate thickness
#    area = W*H
    z = linspace(-H/2,H/2,nply+1)  # ply boundary vertical dimensions
    
    ###################  Mechanical loading #######################
    
#    Ti = 100   # initial temperature (C)
#    Tf = 100 # final temperature (C)
#    dT = Tf-Ti 
    
    A = zeros((3,3))
    B = zeros((3,3))
    D = zeros((3,3))
    
    # Engineering contants for a single ply 
    # calcluating Q from the Compliance matrix may cause cancellation errors
    Q = Qf(mat.E1,mat.E2,mat.nu12,mat.G12)

    for i in range(nply):  # = nply
        Qbar = solve(T1(plyangle[i]), Q) @ T2(plyangle[i])
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
    abcd = np.linalg.inv(ABD)
    a = abcd[0:3,0:3]
    
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
    

    # Laminate compliance matrix
    LamComp = array([ [1/Exbar,       -nuyxbar/Eybar,  etasxbar/Gxybar],
                      [-nuxybar/Exbar,  1/Eybar ,       etasybar/Gxybar],
                      [etaxsbar/Exbar, etaysbar/Eybar, 1/Gxybar]] ) 
    
    print('--------------- laminate1 Stress analysis of fibers----------')
    print('ABD=')
    print(ABD)   
    print('Ex=%0.1f'%Exbar)
    print('Ey=%0.1f'%Eybar)
    print('nuxy=%0.1f'%nuxybar)
    print('Gxy=%0.1f'%Gxybar)

    ################# Stresses and Strains #################    
    Nxyzapplied = array([[100],[0],[0]])
    strainxyzapplied = array([[1e-3],[0],[0]]) 
    
    # Daniel pg 183
    # combines applied loads and applied strains
    strain_laminate = LamComp @ Nxyzapplied/H + strainxyzapplied
    Nxyz = A @ strain_laminate
    stress_laminate = Nxyz/H
    
    strainxyz = zeros((3,2*nply))   # global strain 
    stressxyz = zeros((3,2*nply))   # global stress    
    zplot = zeros(2*nply)
    
    strainxyz = strainxyz + strain_laminate[:]

    for i,k in enumerate(range(0,2*nply,2)):  # = nply
        Qbar = solve(T1(plyangle[i]), Q) @ T2(plyangle[i])
        # stress is calcuated at top and bottom of each ply
        stressxyz[:,[k]] = Qbar @ strain_laminate
        stressxyz[:,[k+1]] = Qbar @ strain_laminate
        zplot[k] = z[i]
        zplot[k+1] = z[i+1]

    #==============================================================================
    #     Plotting
    #==============================================================================
    
    f, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex='row', sharey=True)
    f.canvas.set_window_title('Global Stress and Strain of %s laminate' % (plyangle))
    
    ax1.plot(strainxyz[0,:] , zplot)
    ax1.plot([strain_laminate[0],strain_laminate[0]],[zplot[0], zplot[-1]], '--k' )
    ax1.set_xlabel(r'$\epsilon_x$')
    ax1.set_ylabel(r'$t,in$')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.grid(True)
    
    ax2.plot(strainxyz[1,:] , zplot)
    ax2.plot([strain_laminate[1],strain_laminate[1]],[zplot[0], zplot[-1]], '--k' )
    ax2.set_xlabel(r'$\epsilon_y$')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.grid(True)
    
    ax3.plot(strainxyz[2,:] , zplot)
    ax3.plot([strain_laminate[2],strain_laminate[2]],[zplot[0], zplot[-1]], '--k' )
    ax3.set_xlabel(r'$\epsilon_{xy}$')
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.grid(True)
    
    ax4.plot(stressxyz[0,:] , zplot)
    ax4.plot([stress_laminate[0],stress_laminate[0]],[zplot[0], zplot[-1]], '--k' )
    ax4.set_xlabel(r'$\sigma_x$')
    ax4.set_ylabel(r'$t,in$')
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax4.grid(True)
    
    ax5.plot(stressxyz[1,:] , zplot)
    ax5.plot([stress_laminate[1],stress_laminate[1]],[zplot[0], zplot[-1]], '--k' )
    ax5.set_xlabel(r'$\sigma_y$')
    ax5.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax5.grid(True)
    
    ax6.plot(stressxyz[2,:] , zplot)
    ax6.plot([stress_laminate[2],stress_laminate[2]],[zplot[0], zplot[-1]], '--k' )
    ax6.set_xlabel(r'$\tau_{xy}$')
    ax6.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax6.grid(True)
    
    f.show()
    tight_layout()
    
    
if __name__=='__main__':
    
    
    #material_plots()
    laminate1()


'''CODE GRAVEYARD

#### Needs validation, Currently incorrect
def laminate2(): 
    """
    Code to anaylze composite plates using classical laminate plate theory
    
    clpt = classical lamianted plate theory
    
    First written by Neal Gordon, January, 2014
    This program was developed to determine the linear-elastic properties
     of laminate composites
    
    suffix bar refers to global coordinate system(CS), while unspecified refers to
     material CS. Laminate model with plane stress assumption. sigma_3 = tau_23 = tau_13 = 0
    ----References----
    Daniel, I. M., & Ishai, O. (2006). Engineering Mechanics of Composite Materials (2nd ed.). Oxford University Press.
    Hyer, M. W. (1998). Stress Analysis of Fiber-Reinforced Composite Materials.
    Reddy, J. N. (2004). Mechanics of Laminated Composite PLates and Shells: Theory and Analysis (2nd ed.).
    Herakovich, C. T. (n.d.). Mechanics of Fibrous Composites.
    """
    
    get_ipython().magic('matplotlib') 
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.titlesize'] = 10 
    plt.rcParams['font.size'] = 14  
     
    ########################  Laminate properties #############################
    
    # The custom function laminate_gen generates the properties of a laminate
    # ith the following parameters.
    
    epsxapp = 1e-3 #10e-4 #10e-4
    Nxapp = 0
    
    # properties per ply
    theta = [0, 0, 45, 90]
    matthk = [0.005, 0.12]  # inches
    mati = [0,1,0,0]  # material index, starting with 0
    laythk = [matthk[k] for k in mati]
    nlay = len(theta)
    
    W =  10     # laminate width
    L =  4           # laminate length
    H = np.sum(laythk)     # laminate thickness
    area = W*H       # total cross-sectioal area of laminate

     # calculate unique materials used
    matname = [int(k) for k in set(mati)]      
    
    ###################  Mechanical and thermal loading #######################
    
    Ti = 200   # initial temperature (C)
    Tf = 100 # final temperature (C)
    dT = Tf-Ti 
    # Applied loads
    # NMbarapp = Nx*1000/W , Nx is the composite load (kN) 
    # applied load  of laminate. This is used to simulate a load controlled tensile test
    #           Nx   Ny Nxy  Mx My Mxy
    NMbarapp = array([[Nxapp], [0],  [0],  [0],  [0],  [0]   ])*1000/W # load or moments per unit length (kN/m)
    # Applied strains
    # kapx = 1/m , units are in mm so 2.22*m**-1 = 2.22e-3 *m**-1
    # 5e-4 = 0.05# strain
    #                epsx  epsy  epsxy  kapx  kapy kapxy
    # epsxapp = 1e-5
    epsilonbarapp = array([[epsxapp], [0],  [0], [0], [0], [0] ])   #'  # used to simulate strain controlled tensile test
    
    ########################  Material properties #############################
    
    # Material Properties from Hyer pg58
    # # material 1 in column 1, material 2 in colum 2 etc, 
    # # graphite-polymer lamina
    # multiple lamina properties can be given by adding columns
    # # material 1 in column 1, material 2 in colum 2 etc
    E1  =  [19.2e6 , 400e3]
    E2  =  [1.56e6, 400e3]
    G12  = [0.82e6 , 170e3]
    G23  = [0.49e6  , 170e3]
    nu12 = [0.24  ,  0.17]
    nu23 = [0.59, 0.17]
    alpha1 =  [-0.43e-6, -0.3e-6]  # coefficient of thermal expansion in 1
    alpha2 =  [13.6e-6, -0.3e-6]  # coefficient of thermal expansion in 1
    alpha12 = [0, 0] # coefficient of thermal expansion in 12
    F1 = [219.5e3, 0]
    F2 = [6.3e3, 0]
    F12 =[0, 0]
    
    alpha = zeros((len(matname),3,1))
    for i in matname:
        alpha[i] = array([ [alpha1[i]], [alpha2[i]], [alpha12[i]] ])
       
    
    ########################  Material properties #############################
    # lamina bound coordinates
    z = linspace(-H/2,H/2,nlay+1)
    
    # create a 3 dimension matrix access values with Q[3rd Dim, row, column] , Q[0,2,2]

    Q = zeros((len(matname),3,3))
    for i in matname:
        Q[i] = Qf(E1[i],E2[i],nu12[i],G12[i])

    Qbar = zeros((len(theta),3,3))
    for i,th in enumerate(theta):
        Qbar[i] = solve(T1(th), Q[mati[i]]) @ T2(th)    
    
    # mechanical property calc
    A = zeros((3,3)) #extensional stiffness, in-plane laminate moduli
    B = zeros((3,3))  # coupling stiffness
    D = zeros((3,3))  # flexural or bending stiffness
    Nhatth = zeros((3,1))  # unit thermal force in global CS
    Mhatth = zeros((3,1)) # unit thermal moment in global CS
    NMhatth = zeros((6,1))
    alphabar = zeros((3,nlay))
    
    for i in range(nlay): # loop through each ply to calculate the properties
        Qbar = solve(T1(theta[i]), Q[mati[i]]) @ T2(theta[i])
        # Stiffness
        A += Qbar*(z[i+1]-z[i])
        # coupling  stiffness
        B += 0.5*Qbar*(z[i+1]**2-z[i]**2)
        # bending or flexural laminate stiffness relating moments to curvatures
        D += (1/3)*Qbar*(z[i+1]**3-z[i]**3)  
        # stress is calcuated at top and bottom of each ply
        alphabar[:,[i]] = inv(T2(theta[i])) @ alpha[mati[i]]  # Convert to global CS    
        Nhatth += Qbar @ alphabar[:,[i]] * (z[i+1] - z[i]) # Hyer method for calculating thermal unit loads[]
        Mhatth += 0.5*Qbar @ alphabar[:,[i]] * (z[i+1]**2-z[i]**2) 
        
    
    NMhatth[:3] = Nhatth
    NMhatth[3:] = Mhatth
    
    NMbarth = NMhatth*dT # resultant thermal loads
    # laminate stiffness matrix
    ABD = np.zeros((6,6))
    ABD[0:3, 3:6] = B
    ABD[3:6, 0:3] = B
    ABD[0:3, 0:3] = A
    ABD[3:6, 3:6] = D
    
    
    # laminatee compliance
    # abd = [a bc d]
    abcd = np.linalg.inv(ABD)
    a = abcd[0:3,0:3]
    b = abcd[0:3,3:7]
    c = abcd[3:7,0:3]
    d = abcd[3:7,3:7]
    # Average composite properties, pg 183 daniel ishai, pg 325 hyer
    # coefficients of thermal epansion for entire composite
    alpha_composite = a @ Nhatth # laminate CTE
    # effective laminate moduli
    Exbar  = 1 / (H*a[0,0])
    Eybar  = 1 / (H*a[1,1])
    # effective shear modulus
    Gxybar = 1 / (H*a[2,2])
    # effective poisson ratio
    nuxybar = -a[0,1]/a[0,0]
    nuyxbar = -a[0,1]/a[1,1]
    # effective laminate shear coupling coefficients
    etasxbar = a[0,2]/a[2,2]
    etasybar = a[1,2]/a[2,2]
    etaxsbar = a[2,0]/a[0,0]
    etaysbar = a[2,1]/a[1,1]
    # Laminate compliance matrix
    
    LamComp = array([[1/Exbar,          -nuyxbar/Eybar,  etasxbar/Gxybar],
                          [-nuxybar/Exbar,   1/Eybar,        etasybar/Gxybar],
                          [ etaxsbar/Exbar,  etaysbar/Eybar, 1/Gxybar]])
    LamMod = inv(LamComp) # Laminate moduli
    # combines applied loads and applied strains
    NMbarapptotal = NMbarapp + ABD @ epsilonbarapp
    # Composite respone from applied mechanical loads and strains. Average
    # properties only. Used to compare results from tensile test.
    epsilon_composite = abcd @ NMbarapptotal
    sigma_composite = ABD @ epsilon_composite/H
    ## determine thermal load and applied loads or strains Hyer pg 435,452
    Nx = NMbarapptotal[0]*W/1000 # units kiloNewtons, total load as would be applied in a tensile test
    Ny = NMbarapptotal[1]*L/1000 # units kN
    epsilonbarth = abcd @ NMbarth
    epsilonbarapptotal = epsilonbarapp + abcd @ NMbarapp # includes applied loads and strains
    # Note, epsilonbarapptotal == abcd*NMbarapptotal
    
    print('--------------- laminate2 Stress analysis of fibers----------')
    	## OUTPUT
    print('ABD=')
    print(ABD)    
    print('LamMod')
    print(LamMod)
    print('Ex=%0.1f'%Exbar)
    print('Ey=%0.1f'%Eybar)
    print('nuxy=%0.1f'%nuxybar)
    print('Gxy=%0.1f'%Gxybar)

    # create a 3 dimension matrix access values with Q[3rd Dim, row, column] , Q[0,2,2]
    
    epsilonbar 	    = zeros((3,len(z)))
    sigmabar            = zeros((3,len(z)))
    epsilon             = zeros((3,len(z)))
    sigma               = zeros((3,len(z)))
    epsilonbar_app      = zeros((3,len(z)))
    sigmabar_app        = zeros((3,len(z)))
    epsilon_app         = zeros((3,len(z)))
    sigma_app           = zeros((3,len(z)))
    epsilonbar_th       = zeros((3,len(z)))
    sigmabar_th         = zeros((3,len(z)))
    epsilon_th          = zeros((3,len(z)))
    sigma_th            = zeros((3,len(z)))
    
    epsilon_app_plot    = zeros((3,2*nlay))
    epsilonbar_app_plot = zeros((3,2*nlay))
    sigma_app_plot      = zeros((3,2*nlay))
    sigmabar_app_plot   = zeros((3,2*nlay))
    epsilon_th_plot     = zeros((3,2*nlay))
    epsilonbar_th_plot  = zeros((3,2*nlay))
    sigma_th_plot       = zeros((3,2*nlay))
    sigmabar_th_plot    = zeros((3,2*nlay))
    epsilonplot         = zeros((3,2*nlay))
    epsilonbarplot      = zeros((3,2*nlay))
    sigmaplot           = zeros((3,2*nlay))
    sigmabarplot        = zeros((3,2*nlay))
    zplot               = zeros((2*nlay))
    
#    ######################  prepare data for plotting##########################
    
    for i,k in enumerate(range(0,2*nlay,2)):
        
        Qbar = solve(T1(theta[i]), Q[mati[i]]) @ T2(theta[i]) # Convert to global CS 
 
        # stress is calcuated at top and bottom of each ply

        zplot[k] = z[i]
        zplot[k+1] = z[i+1]    
        #zplot[2*k:2*(k+1)] = z[k:(k+2)]
        

         # Global stresses and strains, applied load only
        epsbar1 = epsilonbarapptotal[0:3] + z[i]*epsilonbarapptotal[3:7]
        epsbar2 = epsilonbarapptotal[0:3] + z[i+1]*epsilonbarapptotal[3:7]
        sigbar1 = Qbar @ epsbar1
        sigbar2 = Qbar @ epsbar2
        epsilonbar_app[:,i:i+2] =  np.column_stack((epsbar1,epsbar2))
        sigmabar_app[:,i:i+2] =  np.column_stack((sigbar1,sigbar2))
    
        # Local stresses and strains, appplied load only
        eps1 = T2(theta[i]) @ epsbar1
        eps2 = T2(theta[i]) @ epsbar2
        sig1 = Q[mati[i]] @ eps1
        sig2 = Q[mati[i]] @ eps2
        epsilon_app[:,i:i+2] = np.column_stack((eps1,eps2))
        sigma_app[:,i:i+2] = np.column_stack((sig1,sig2))
       
        epsilon_app_plot[:,k:k+2]    = np.column_stack((eps1,eps2))
        epsilonbar_app_plot[:,k:k+2] = np.column_stack((epsbar1,epsbar2))
        sigma_app_plot[:,k:k+2]      = np.column_stack((sig1,sig2))
        sigmabar_app_plot[:,k:k+2]   = np.column_stack((sigbar1,sigbar2))
        
        # Global stress and strains, thermal loading only
        epsbar1 = (alpha_composite - alphabar[:,[i]])*dT
        sigbar1 = Qbar @ epsbar1
        epsilonbar_th[:,i:i+2] = np.column_stack((epsbar1,epsbar1))
        sigmabar_th[:,i:i+2] = np.column_stack((sigbar1,sigbar1))
        
        # Local stress and strains, thermal loading only
        eps1 = T2(theta[i]) @ epsbar1
        sig1 = Q[mati[i]] @ eps1
        epsilon_th[:,i:i+2] = np.column_stack((eps1,eps1))
        sigma_th[:,i:i+2] = np.column_stack((sig1,sig1))
        
        # Create array for plotting purposes only
        epsilon_th_plot[:,k:k+2] = np.column_stack((eps1,eps1))
        epsilonbar_th_plot[:,k:k+2] = np.column_stack((epsbar1,epsbar1))
        sigma_th_plot[:,k:k+2] = np.column_stack((sig1,sig1))
        sigmabar_th_plot[:,k:k+2] = np.column_stack((sigbar1,sigbar1))  
        
        # global stresses and strains, bar ,xy coord including both applied
        # loads and thermal loads. NET, or relaized stress-strain
        epsbar1 = epsilonbarapptotal[0:3] + z[i]*epsilonbarapptotal[3:7] + (alpha_composite - alphabar[:,[i]])*dT
        epsbar2 = epsilonbarapptotal[0:3] + z[i+1]*epsilonbarapptotal[3:7] + (alpha_composite - alphabar[:,[i]])*dT
        sigbar1 = Qbar @ epsbar1
        sigbar2 = Qbar @ epsbar2
        epsilonbar[:,i:i+2] = np.column_stack((epsbar1,epsbar2))
        sigmabar[:,i:i+2] = np.column_stack((sigbar1,sigbar2))
        
        # local stresses and strains , 12 coord, includes both applied loads
        # and thermal load. NET , or realized stress and strain
        eps1 = T2(theta[i]) @ epsbar1
        eps2 = T2(theta[i]) @ epsbar2
        sig1 = Q[mati[i]] @ eps1
        sig2 = Q[mati[i]] @ eps2
        epsilon[:,i:i+2] = np.column_stack((eps1,eps2))
        sigma[:,i:i+2] = np.column_stack((sig1,sig2))
        
        # Create array for plotting purposes only
        epsilonplot[:,k:k+2] = np.column_stack((eps1,eps2))
        epsilonbarplot[:,k:k+2] = np.column_stack((epsbar1,epsbar2))
        sigmaplot[:,k:k+2] = np.column_stack((sig1,sig2))
        sigmabarplot[:,k:k+2] = np.column_stack((sigbar1,sigbar2))  
       
    ############################ Plotting #####################################   

    legendlab = ['total','thermal','applied','composite']
    # global stresses and strains
    mylw = 1.5 #linewidth

    # Global Stresses and Strains
    f1, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex='row', sharey=True)
    f1.canvas.set_window_title('Global Stress and Strain of %s laminate' % (theta))
    
    stresslabel = ['$\sigma_x,\ ksi$','$\sigma_y,\ ksi$','$\\tau_{xy},\ ksi$']
    strainlabel = ['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$']
    
    for i,ax in enumerate([ax1,ax2,ax3]):
        ## the top axes
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(strainlabel[i])
        ax.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        
        ax.plot(epsilonbarplot[i,:],     zplot, color='blue', lw=mylw)
        ax.plot(epsilonbar_th_plot[i,:], zplot, color='red', lw=mylw)
        ax.plot(epsilonbar_app_plot[i,:], zplot, color='green', lw=mylw)   
        ax.plot([epsilon_composite[i], epsilon_composite[i]],[np.min(z) , np.max(z)], color='black', lw=mylw) 
        ax.grid(True)              
    
    for i,ax in enumerate([ax4,ax5,ax6]):
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(stresslabel[i])
        ax.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        
        ax.plot(sigmabarplot[i,:],     zplot, color='blue', lw=mylw)
        ax.plot(sigmabar_th_plot[i,:], zplot, color='red', lw=mylw)
        ax.plot(sigmabar_app_plot[i,:], zplot, color='green', lw=mylw)   
        ax.plot([sigma_composite[i], sigma_composite[i]],[np.min(z) , np.max(z)], color='black', lw=mylw) 
        ax.grid(True)
    
    legend(legendlab)
    f1.show()
    tight_layout()          
    #plt.savefig('global-stresses-strains.png')
    

    ### Local Stresses and Strains
    f2, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex='row', sharey=True)
    f2.canvas.set_window_title('Local Stress and Strain of %s laminate' % (theta))
    stresslabel = ['$\sigma_1,\ ksi$','$\sigma_2,\ ksi$','$\\tau_{12},\ ksi$']
    strainlabel = ['$\epsilon_1$','$\epsilon_2$','$\gamma_{12}$']
    
    for i,ax in enumerate([ax1,ax2,ax3]):
        ## the top axes
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(strainlabel[i])
        ax.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        
        ax.plot(epsilonplot[i,:],     zplot, color='blue', lw=mylw)
        ax.plot(epsilon_th_plot[i,:], zplot, color='red', lw=mylw)
        ax.plot(epsilon_app_plot[i,:], zplot, color='green', lw=mylw)   
        ax.plot([epsilon_composite[i], epsilon_composite[i]],[np.min(z) , np.max(z)], color='black', lw=mylw) 
        ax.grid(True)
                          
    
    for i,ax in enumerate([ax4,ax5,ax6]):
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(stresslabel[i])
        ax.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        
        ax1.plot(sigmaplot[i,:],     zplot, color='blue', lw=mylw)
        ax1.plot(sigma_th_plot[i,:], zplot, color='red', lw=mylw)
        ax1.plot(sigma_app_plot[i,:], zplot, color='green', lw=mylw)   
        ax1.plot([sigma_composite[i], sigma_composite[i]],[np.min(z) , np.max(z)], color='black', lw=mylw)     
        ax.grid(True)
        
    legend(legendlab)
    f2.show()
    tight_layout()          
    #plt.savefig('local-stresses-strains.png')
               
    ### warpage
    res = 250
    Xplt,Yplt = np.meshgrid(np.linspace(-W/2,W/2,res), np.linspace(-L/2,L/2,res))
    epsx = epsilon_composite[0,0]
    epsy = epsilon_composite[1,0]
    epsxy = epsilon_composite[2,0]
    kapx = epsilon_composite[3,0]
    kapy = epsilon_composite[4,0]
    kapxy = epsilon_composite[5,0]
    ### dispalcement
    w = -0.5*(kapx*Xplt**2 + kapy*Yplt**2 + kapxy*Xplt*Yplt)
    u = epsx*Xplt  # pg 451 hyer
    fig = plt.figure('plate-warpage')
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xplt, Yplt, w, cmap=cm.jet, alpha=0.3)
    ###ax.auto_scale_xyz([-(W/2)*1.1, (W/2)*1.1], [(L/2)*1.1, (L/2)*1.1], [-1e10, 1e10])
    ax.set_xlabel('plate width,y-direction,mm')
    ax.set_ylabel('plate length,x-direction, mm')
    ax.set_zlabel('warpage,mm')
    #ax.set_zlim(-0.01, 0.04)
    plt.show()
    #plt.savefig('plate-warpage')
    
    ########################## DATA display ##################################






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





'''