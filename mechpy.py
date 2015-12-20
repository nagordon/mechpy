# coding: utf-8

'''
Module to be used for calculating mechanical engineering stuff

'''


import sympy as sp
import numpy as np
import matplotlib.pyplot as mp
from pprint import pprint

#np.set_printoptions(edgeitems=3,linewidth=75, precision=5, suppress=False, threshold=1000)
#get_ipython().magic('matplotlib inline')


# %load T3r.py
def T3rot(th):
    from numpy import matrix, pi, cos, sin
    # rotation only about the z axis
    th *= pi/180 # change degrees to radians
    T3 = matrix([ [ cos(th), sin(th), 0],
                  [-sin(th),  cos(th), 0],
                  [ 0,       0 ,       1]])
    return T3


def T6rot(thx=0, thy=0, thz=0):
    '''
     aij = [1 0 0;0 1 0; 0 0 1]
     derivation of the voight notation transformation matrix
     from barberom FEA of composites with abaqus, pg 14a
     accepts a as cosine direction matrix, aij, i new axis, and j is old axisa
     C = 6x6 stiffness matrix
     stress = C*strain
     S = inv(C)
     Cp = T*C*T'
     C = Tbar'Cp*Tbar
     Cp = T*C*T'
     Tbar' = inv(T)
     setup for transformation about axis 3 only
     thx = 45 # rot about x
     thy = 45  # rot about y
     thz = 45  # rot about z
    '''
    
    from numpy import pi, cos, sin, matrix, eye, zeros
    #from numpy.linalg import inv
    ## Method 2 rotation matrices
    thx = thx*pi/180 # convert to radians
    thy = thy*pi/180 # convert to radians
    thz = thz*pi/180 # convert to radians
    # transformation about axis 1
    T1 = matrix([[1,  0,         0],
                 [0,  cos(thx),  sin(thx)],
                 [0, -sin(thx), cos(thx)]])

    # transformation about axis 2
    T2 = matrix([[cos(thy), 0, -sin(thy)],
                 [0,        1,  0],
                 [sin(thy), 0,  cos(thy)]])

    # transformation about axis 3
    T3 = matrix([[cos(thz),   sin(thz), 0],
                 [-sin(thz),  cos(thz), 0],
                 [0,          0,        1]])
    aij = T1*T2*T3
    # reuter matrix
    R = matrix(eye(6,6))
    R[3,3]=2
    R[4,4]=2
    R[5,5]=2
    T = matrix(zeros((6,6)))
    for i in [1,2,3]:
        for j in [1,2,3]:
            alph = j if i==j else 9-i-j 
            for p in [1,2,3]:
                for q in [1,2,3]:
                    beta = p if p==q else 9-p-q    
                    if   alph <= 3 and beta <= 3:
                        T[alph-1,beta-1] = aij[i-1,p-1]*aij[i-1,p-1] 
                    elif alph >  3 and beta <= 3:
                        T[alph-1,beta-1] = aij[i-1,p-1]*aij[j-1,p-1]
                    elif alph <= 3 and beta >  3:
                        T[alph-1,beta-1] = aij[i-1,q-1]*aij[i-1,p-1]+aij[i-1,p-1]*aij[i-1,q-1]
                    elif alph > 3 and beta > 3:
                        T[alph-1,beta-1] = aij[i-1,p-1]*aij[j-1,q-1] + aij[i-1,q-1]*aij[j-1,p-1]
                    else:
                        T[alph-1,beta-1] = 0
                    
    Tbar =  R*T*(R.I)  # ==R*T*inv(R) 
    #print(Tbar)
    return Tbar



def vary_ply_direction_plot():
    '''
    composites calculations
    '''
    from numpy import arange, pi, sin, cos, zeros, matrix
    from numpy.linalg import solve
    import matplotlib.pyplot as mp
    
    th = arange(0, 180.1, 0.1) * (pi/180)
    deg = th*180/pi
    
    Exbar = zeros(len(th))
    Eybar = zeros(len(th))
    Gxybar = zeros(len(th))
    
    for i,theta in enumerate(th):
        
        # lamina thickness
        h = 1
        ## Initial lamina Properties
        E1    = 147.0  # GPa
        E2    = 10.3 # GPa
        G12   = 7.0    # GPa 
        Nu12 =  0.27
        Nu21 =  E2*Nu12/E1
        ## Calculate stiffness
        Q = matrix(zeros((3,3)))
        Q[0,0] = E1 / (1-Nu12*Nu21)
        Q[1,1] = E2 / (1-Nu12*Nu21)
        Q[2,2] = G12
        Q[0,1] = E2*Nu12 / (1-Nu12*Nu21)
        Q[1,0] = E2*Nu12 / (1-Nu12*Nu21)     
        n = sin(theta)
        m = cos(theta)
        R = matrix([[m**2, n**2, m*n],  [n**2, m**2, -m*n],  [-2*m*n, 2*m*n, (m**2-n**2)]])
        T = matrix([[m**2, n**2, 2*m*n],[n**2, m**2, -2*m*n],[-m*n,   m*n,   (m**2-n**2)]])
        Qbar = solve(T,Q)*R
        
        Aij = Qbar*h
        # laminate Stiffness
        #     | Exbar    Eybar    Gxybar   |
        # A = | vxybar   vyxbar   etasxbar |
        #     | etaxsbar etaysbar etasybar | 
    
        # laminate Comnpliance
        aij = Aij.I
        # material properties for whole laminate (Daniel, pg183)
        Exbar[i]  = 1 / (h*aij[0,0])
        Eybar[i] = 1 / (h*aij[1,1])
        Gxybar[i] = 1 / (h*aij[2,2])
    
    
    fig1 = mp.figure(1, figsize=(12,8), frameon=False)
    mp.plot(deg, Exbar, label = r"Modulus: $E_x$")
    mp.plot(deg, Eybar, label = r"Modulus: $E_y$")
    mp.plot(deg, Gxybar, label = r"Modulus: $G_{xy}$")
    mp.title("Constitutive Properties in various angles")
    mp.xlabel("deg")
    mp.ylabel("modulus, GPa")
    mp.legend(loc='best')
    mp.show()


def qbar_transformtion():

    # -*- coding: utf-8 -*-
    """
    Created on Fri Oct 16 08:45:22 2015
    
    @author: ngordon
    
    """
    
    from numpy import linspace, pi, sin, cos, matrix, array
    
    import numpy as np
    import matplotlib.pyplot as mp
    
    theta = linspace(-90,90,100) * pi/180
    
    s_xy = matrix([[100], 
                    [10], 
                    [5]])
    
    s_12 = matrix(np.zeros((3,100)))
    
    for i,th in enumerate(theta):
        n = sin(th)
        m = cos(th)
        T = matrix([[m**2,  n**2,   2*m*n],
                    [n**2,  m**2,  -2*m*n], 
                    [-m*n,  m*n, (m**2-n**2)]])
        s_12[:,i] = T*s_xy
    
    fig = mp.figure(1,figsize=(8,8))
    mp.plot(theta, array(s_12[0,:])[0], label = '$\sigma_{11},MPa$' )
    mp.plot(theta, array(s_12[1,:])[0], label = '$\sigma_{22},MPa$' )
    mp.plot(theta, array(s_12[2,:])[0], label = '$\sigma_{12},MPa$' )
    mp.legend(loc='lower left')
    mp.xlabel("$/theta, rad$") ; mp.ylabel("Stress, MPa")
    mp.show()
    
    Ex = 150e9
    Ey = 12.1e9
    vxy = 0.248
    Gxy = 4.4e9
    vy = 0.458
    Gyx = Ey / (2*(1+vy))
    vyx = Ey*vxy/Ex
    
    Q = matrix(np.zeros((3,3)))
    Qbar = np.zeros((3,3,len(theta)))
    
    Q[0,0] = Ex / (1-vxy*vyx)
    Q[1,1] = Ey / (1-vxy*vyx)
    Q[2,2] = Gyx
    Q[0,1] = Ey*vxy / (1-vxy*vyx)
    Q[1,0] = Q[0,1]
    
    for i in range(len(theta)):
        n = sin(theta[i])
        m = cos(theta[i])
        R = matrix([ [m**2, n**2, m*n]   , [n**2, m**2, -m*n]   , [-2*m*n, 2*m*n, (m**2-n**2)] ])
        T = matrix([ [m**2, n**2, 2*m*n] , [n**2, m**2, -2*m*n] , [-m*n,   m*n,   (m**2-n**2)] ])
        Qbar[:,:,i] = np.linalg.solve(T, Q)*R
    
    Qbar11 = Qbar[0,0,:]
    Qbar22 = Qbar[1,1,:]
    Qbar66 = Qbar[2,2,:]
    Qbar12 = Qbar[0,1,:]
    Qbar16 = Qbar[0,2,:]
    Qbar26 = Qbar[1,2,:]
    
    # plot theta as a function of time
    fig = mp.figure(2,figsize=(8,8))
    ax1 = fig.add_subplot(211)
    ax1.plot(theta,Qbar11, label = "Qbar11")
    ax1.plot(theta,Qbar22, label = "Qbar22")
    ax1.plot(theta,Qbar66, label = "Qbar66")
    mp.legend(loc='lower left')
    ax1.set_xlabel('theta')
    ax1.set_ylabel('Q')
    fig.show()
    
    
    # plot theta as a function of time
    fig = mp.figure(3,figsize=(8,8))
    ax2 = fig.add_subplot(212)
    ax2.plot(theta,Qbar12, label = "Qbar12")
    ax2.plot(theta,Qbar16, label = "Qbar16")
    ax2.plot(theta,Qbar26, label = "Qbar26")
    mp.legend(loc='lower left')
    ax2.set_xlabel('theta')
    ax2.set_ylabel('Q')
    fig.show()


def laminate_gen(lamthk=1.5, symang = [45,0,90], plyratio=2.0, matrixlayers=False, nonsym=False ):
    '''
    ## function created to quickly create laminates based on given parameters
    lamthk=1.5    # total #thickness of laminate
    symang = [45,0,90, 30]  #symmertic ply angle
    plyratio=2.0  # lamina/matrix
    matrixlayers=False  # add matrix layers between lamina plys
    nonsym=False    # symmetric

    #ply ratio can be used to vary the ratio of thickness between a matrix ply
         and lamina ply. if the same thickness is desired, plyratio = 1, 
         if lamina is 2x as thick as matrix plyratio = 2
    '''
    
    import numpy as np
    if matrixlayers:
        nl = (len(symang)*2+1)*2
        nm = nl-len(symang)*2
        nf = len(symang)*2
        tm = lamthk / (plyratio*nf + nm)
        tf = tm*plyratio
        ang = np.zeros(nl/2)
        mat = 2*np.ones(nl/2)  #  orthotropic fiber and matrix = 1, isotropic matrix=2, 
        mat[1:-1:2] = 1   #  [2 if x%2 else 1 for x in range(nl//2) ]
        ang[1:-1:2] = symang[:]
        thk = tm*np.ones(1,nl/2)
        thk[2:2:-1] = tf
        lamang = list(symang) + list(symang[::-1])
        ang = list(ang) + list(ang[::-1])
        mat = list(mat) + list(mat[::-1])
        thk = list(thk) + list(thk[::-1])  
    else: # no matrix layers, ignore ratio
        if nonsym:
            nl = len(symang)
            mat =[1]*nl
            thk = list(lamthk/nl*np.ones(nl))
            lamang = symang[:]
            ang = symang[:]
        else:
            nl = len(symang)*2
            mat = list(3*np.ones(nl)) 
            thk = list(lamthk/nl*np.ones(nl))
            lamang = list(symang) + list(symang[::-1])
            ang = list(symang) + list(symang[::-1])

    return thk,ang,mat,lamang


def composite_plate():
    # -*- coding: utf-8 -*-
    
    import numpy as np
    import numpy.matlib  # needed to create a zeros matrix
    from matplotlib import cm
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=300)
    from mpl_toolkits.mplot3d import axes3d
    
    #from IPython import get_ipython
    #ipython = get_ipython()
    #ipython.magic('matplotlib')
    
    plt.rcParams['figure.figsize'] = (18, 10)
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 8
    #plt.rcParams['title.fontsize'] = 12
    
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
    
    
    ########################  Laminate properties #############################
    
    # The custom function laminate_gen generates the properties of a laminate
    # ith the following parameters.
    
    epsxapp = 10e-4 #10e-4
    Nxapp = 0
    
    # properties per ply
    #ang = [90, 0, 90, 0, 0, 90, 0, 90]  # degrees
    #ang = [90, 0, 90, 0, 0, 90, 0, 90]
    ang = [0, 90,90,0]
    lamang = ang
    
    
    thk = [2]*len(ang)  # mm
    mat = [0]*len(ang)  # material index, starting with 0
    
    W =  30     # laminate width
    L = 100          # laminate length
    nl = len(mat) # number of layers
    H = np.sum(thk)     # laminate thickness
    area = W*H       # total cross-sectioal area of laminate
    
    ###################  Mechanical and thermal loading #######################
    
    Ti = 1000   # initial temperature (C)
    Tf = 100 # final temperature (C)
    dT = Tf-Ti 
    # Applied loads
    # NMbarapp = Nx*1000/W , Nx is the composite load (kN) 
    # applied load  of laminate. This is used to simulate a load controlled tensile test
    #           Nx   Ny Nxy  Mx My Mxy
    NMbarapp = np.matrix([Nxapp, 0,  0,   0,  0,  0   ]).T*1000/W # load or moments per unit length (kN/m)
    # Applied strains
    # kapx = 1/m , units are in mm so 2.22*m**-1 = 2.22e-3 *m**-1
    # 5e-4 = 0.05# strain
    #                epsx  epsy  epsxy  kapx  kapy kapxy
    # epsxapp = 1e-5
    epsilonbarapp = np.matrix([epsxapp,   0,     0,     0,  0,     0,   ]).T   #'  # used to simulate strain controlled tensile test
    
    ########################  Material properties #############################
    
    # Material Properties from Hyer pg58
    # # material 1 in column 1, material 2 in colum 2 etc, 
    # # graphite-polymer lamina
    # multiple lamina properties can be given by adding columns
    # # material 1 in column 1, material 2 in colum 2 etc
    E1  = [155e3 , 400e3]
    E2  = [12.1e3 , 400e3]
    E3  = [12.1e3 , 400e3]
    G12  = [4.4e3 , 170e3]
    G13  = [4.4e3 , 170e3]
    G23  = [3.2e3 , 170e3]
    nu12 = [0.248, 0.17]
    nu13 = [0.248, 0.17]
    nu23 = [0.458, 0.17]
    alpha1 =  [-0.018e-6, -0.3e-6]  # coefficient of thermal expansion in 1
    alpha2 =  [24.3e-6, -0.3e-6]  # coefficient of thermal expansion in 1
    alpha12 = [0, 0] # coefficient of thermal expansion in 12
    
    alpha = [np.matrix([alpha1[k], alpha2[k], alpha12[k]]).T for k in range(len(alpha1))]
    
    ########################  Material properties #############################
    
    # lamina bound coordinates
    z = [0.0]*(nl+1) # lamina coordinates
    zmid = [0.0]*(nl) 
    z[0] = -H/2
    
    for k in range(nl):
        z[k+1] = z[k] + thk[k]
        zmid[k] = z[k] + thk[k]/2
    
     # calculate Compliance and reduced stiffness matrix
    matname = [int(k) for k in set(mat)]                       
    # create a 3 dimension matrix access values with Q[3rd Dim, row, column] , Q[0,2,2]
    Q = [np.matrix(np.zeros((3,3))) for k in set(mat)]
    
    for k in range(len(matname)):
        # compliance matrix
        S6 = np.matrix(np.zeros((6,6)))
        
        S6[0,0] = 1/E1[k]
        S6[1,1] = 1/E2[k]
        S6[2,2] = 1/E3[k]
        S6[3,3] = 1/G23[k]
        S6[4,4] = 1/G13[k]
        S6[5,5] = 1/G12[k]
        S6[0,1] = -nu12[k]/E1[k]
        S6[1,0] = S6[0,1]
        S6[0,2] = -nu13[k]/E1[k]
        S6[2,0] = S6[0,2]
        S6[1,2] = -nu23[k]/E2[k]
        S6[2,1] = S6[1,2]
        # reduced stiffness matrix    
        Q[k][0,0] =  S6[1,1] / ( S6[0,0]*S6[1,1]-S6[0,1]**2)
        Q[k][0,1] = -S6[0,1] / ( S6[0,0]*S6[1,1]-S6[0,1]**2) 
        Q[k][1,1] =  S6[0,0] / ( S6[0,0]*S6[1,1]-S6[0,1]**2) 
        Q[k][2,2] = 1/S6[5,5]  # actually Q66 in many texts
        Q[k][1,0] = Q[k][0,1]    
        # calcluating Q from the Compliance matrix may cause cancellation errors
    
    # mechanical property calc
    A = np.matlib.zeros((3,3)) #extensional stiffness, in-plane laminate moduli
    B = np.matlib.zeros((3,3))  # coupling stiffness
    D = np.matlib.zeros((3,3))  # flexural or bending stiffness
    Nhatth = np.matlib.zeros((3,1))  # unit thermal force in global CS
    Mhatth = np.matlib.zeros((3,1)) # unit thermal moment in global CS
    NMhatth = np.matlib.zeros((6,1))
    alphabar = [np.matlib.zeros((3,1)) for k in range(nl)]
    # transformation reminders - see Herakovich for details
    # sig1 = T*sigx
    # sigx = inv(T)*sig1
    # eps1 = R*epsx
    # epsx = inv(R)*epsx
    # sigx = inv(T)*Q*R*epsx
    # Qbar = inv(T)*Q*R
    # Sbar = inv(R)*inv(Q)*R
    # determine mechanical properties
    
    for k in range(nl): # loop through each ply to calculate the properties
        n = np.sin(np.deg2rad(ang[k]))  
        m = np.cos(np.deg2rad(ang[k]))
        T = np.matrix([[m**2, n**2, 2*m*n], [n**2, m**2, -2*m*n], [-m*n,   m*n,   (m**2-n**2)]])
        R = np.matrix([[m**2, n**2, m*n  ], [n**2, m**2, -m*n],   [-2*m*n, 2*m*n, (m**2-n**2)]])
        Qbar = T.I*Q[mat[k]]*R # Convert to global CS
        A += Qbar * (z[k+1]-z[k])   # coupling  stiffness
        B += 0.5*Qbar * (z[k+1]**2-z[k]**2)   # bending or flexural laminate stiffness relating moments to curvatures
        D += (1/3)*Qbar * (z[k+1]**3-z[k]**3)  
        alphabar[k] = R.I*alpha[mat[k]]  # Convert to global CS    
        Nhatth += Qbar*alphabar[k]*(z[k+1] - z[k]) # Hyer method for calculating thermal unit loads[]
        Mhatth += 0.5*Qbar*alphabar[k]*(z[k+1]**2-z[k]**2) 
    
    NMhatth[:3] = Nhatth
    NMhatth[3:] = Mhatth
    
    NMbarth = NMhatth*dT # resultant thermal loads
    # laminate stiffness matrix
    ABD = np.matlib.zeros((6,6))
    ABD[0:3, 3:6] = B
    ABD[3:6, 0:3] = B
    ABD[0:3, 0:3] = A
    ABD[3:6, 3:6] = D
    
    
    # laminatee compliance
    # abd = [a bc d]
    abcd = ABD.I
    a = abcd[0:3,0:3]
    b = abcd[0:3,3:7]
    c = abcd[3:7,0:3]
    d = abcd[3:7,3:7]
    # Average composite properties, pg 183 daniel ishai, pg 325 hyer
    # coefficients of thermal epansion for entire composite
    alpha_composite = a*Nhatth # laminate CTE
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
    
    LamComp = np.matrix([[1/Exbar,          -nuyxbar/Eybar,  etasxbar/Gxybar],
                          [-nuxybar/Exbar,   1/Eybar,        etasybar/Gxybar],
                          [ etaxsbar/Exbar,  etaysbar/Eybar, 1/Gxybar]])
    LamMod = LamComp.I # Laminate moduli
    # combines applied loads and applied strains
    NMbarapptotal = NMbarapp + ABD*epsilonbarapp
    # Composite respone from applied mechanical loads and strains. Average
    # properties only. Used to compare results from tensile test.
    epsilon_composite = abcd*NMbarapptotal
    sigma_composite = ABD*epsilon_composite/H
    ## determine thermal load and applied loads or strains Hyer pg 435,452
    Nx = NMbarapptotal[0]*W/1000 # units kiloNewtons, total load as would be applied in a tensile test
    Ny = NMbarapptotal[1]*L/1000 # units kN
    epsilonbarth = abcd*NMbarth
    epsilonbarapptotal = epsilonbarapp + abcd*NMbarapp # includes applied loads and strains
    # Note, epsilonbarapptotal == abcd*NMbarapptotal
    
    
    
    ######################  prepare data for plotting##########################
    
    
    zmid = [0.0]*8
                   
                
                   
    # create a 3 dimension matrix access values with Q[3rd Dim, row, column] , Q[0,2,2]
    
    epsilonbar 		    = np.matlib.zeros((3,len(z)))
    sigmabar            = np.matlib.zeros((3,len(z)))
    epsilon             = np.matlib.zeros((3,len(z)))
    sigma               = np.matlib.zeros((3,len(z)))
    epsilonbar_app      = np.matlib.zeros((3,len(z)))
    sigmabar_app        = np.matlib.zeros((3,len(z)))
    epsilon_app         = np.matlib.zeros((3,len(z)))
    sigma_app           = np.matlib.zeros((3,len(z)))
    epsilonbar_th       = np.matlib.zeros((3,len(z)))
    sigmabar_th         = np.matlib.zeros((3,len(z)))
    epsilon_th          = np.matlib.zeros((3,len(z)))
    sigma_th            = np.matlib.zeros((3,len(z)))
    
    epsilon_app_plot    = np.matlib.zeros((3,2*nl))
    epsilonbar_app_plot = np.matlib.zeros((3,2*nl))
    sigma_app_plot      = np.matlib.zeros((3,2*nl))
    sigmabar_app_plot   = np.matlib.zeros((3,2*nl))
    epsilon_th_plot     = np.matlib.zeros((3,2*nl))
    epsilonbar_th_plot  = np.matlib.zeros((3,2*nl))
    sigma_th_plot       = np.matlib.zeros((3,2*nl))
    sigmabar_th_plot    = np.matlib.zeros((3,2*nl))
    epsilonplot         = np.matlib.zeros((3,2*nl))
    epsilonbarplot      = np.matlib.zeros((3,2*nl))
    sigmaplot           = np.matlib.zeros((3,2*nl))
    sigmabarplot        = np.matlib.zeros((3,2*nl))
    zplot               = np.zeros((2*nl))
    
    for k in range(nl):
        n = np.sin(np.deg2rad(ang[k]))  
        m = np.cos(np.deg2rad(ang[k]))
        T = np.matrix([[m**2, n**2, 2*m*n], [n**2, m**2, -2*m*n], [-m*n,   m*n,   (m**2-n**2)]])
        R = np.matrix([[m**2, n**2, m*n  ], [n**2, m**2, -m*n],   [-2*m*n, 2*m*n, (m**2-n**2)]])
        
        zplot[2*k:2*(k+1)] = z[k:(k+2)]
        
        Qbar = T.I*Q[mat[k]]*R # Convert to global CS 
    
         # Global stresses and strains, applied load only
        epsbar1 = epsilonbarapptotal[0:3] + z[k]*epsilonbarapptotal[3:7]
        epsbar2 = epsilonbarapptotal[0:3] + z[k+1]*epsilonbarapptotal[3:7]
        sigbar1 = Qbar*epsbar1
        sigbar2 = Qbar*epsbar2
        epsilonbar_app[:,k:k+2] =  np.column_stack((epsbar1,epsbar2))
        sigmabar_app[:,k:k+2] =  np.column_stack((sigbar1,sigbar2))
    
        # Local stresses and strains, appplied load only
        eps1 = R*epsbar1
        eps2 = R*epsbar2
        sig1 = Q[mat[k]]*eps1
        sig2 = Q[mat[k]]*eps2
        epsilon_app[:,k:k+2] = np.column_stack((eps1,eps2))
        sigma_app[:,k:k+2] = np.column_stack((sig1,sig2))
       
        epsilon_app_plot[:, 2*k:2*(k+1)]    = np.column_stack((eps1,eps2))
        epsilonbar_app_plot[:, 2*k:2*(k+1)] = np.column_stack((epsbar1,epsbar2))
        sigma_app_plot[:, 2*k:2*(k+1)]      = np.column_stack((sig1,sig2))
        sigmabar_app_plot[:, 2*k:2*(k+1)]   = np.column_stack((sigbar1,sigbar2))
        
        # Global stress and strains, thermal loading only
        epsbar1 = (alpha_composite - alphabar[k])*dT
        sigbar1 = Qbar*epsbar1
        epsilonbar_th[:,k:k+2] = np.column_stack((epsbar1,epsbar1))
        sigmabar_th[:,k:k+2] = np.column_stack((sigbar1,sigbar1))
        
        # Local stress and strains, thermal loading only
        eps1 = R*epsbar1
        sig1 = Q[mat[k]]*eps1
        epsilon_th[:,k:k+2] = np.column_stack((eps1,eps1))
        sigma_th[:,k:k+2] = np.column_stack((sig1,sig1))
        
        # Create array for plotting purposes only
        epsilon_th_plot[:, 2*k:2*(k+1)] = np.column_stack((eps1,eps1))
        epsilonbar_th_plot[:, 2*k:2*(k+1)] = np.column_stack((epsbar1,epsbar1))
        sigma_th_plot[:, 2*k:2*(k+1)] = np.column_stack((sig1,sig1))
        sigmabar_th_plot[:, 2*k:2*(k+1)] = np.column_stack((sigbar1,sigbar1))  
        
        # global stresses and strains, bar ,xy coord including both applied
        # loads and thermal loads. NET, or relaized stress-strain
        epsbar1 = epsilonbarapptotal[0:3] + z[k]*epsilonbarapptotal[3:7] + (alpha_composite - alphabar[k])*dT
        epsbar2 = epsilonbarapptotal[0:3] + z[k+1]*epsilonbarapptotal[3:7] + (alpha_composite - alphabar[k])*dT
        sigbar1 = Qbar*epsbar1
        sigbar2 = Qbar*epsbar2
        epsilonbar[:,k:k+2] = np.column_stack((epsbar1,epsbar2))
        sigmabar[:,k:k+2] = np.column_stack((sigbar1,sigbar2))
        
        # local stresses and strains , 12 coord, includes both applied loads
        # and thermal load. NET , or realized stress and strain
        eps1 = R*epsbar1
        eps2 = R*epsbar2
        sig1 = Q[mat[k]]*eps1
        sig2 = Q[mat[k]]*eps2
        epsilon[:,k:k+2] = np.column_stack((eps1,eps2))
        sigma[:,k:k+2] = np.column_stack((sig1,sig2))
        
        # Create array for plotting purposes only
        epsilonplot[:, 2*k:2*(k+1)] = np.column_stack((eps1,eps2))
        epsilonbarplot[:, 2*k:2*(k+1)] = np.column_stack((epsbar1,epsbar2))
        sigmaplot[:, 2*k:2*(k+1)] = np.column_stack((sig1,sig2))
        sigmabarplot[:, 2*k:2*(k+1)] = np.column_stack((sigbar1,sigbar2))  
       
    ############################ Plotting #####################################   
    legendlab = ['composite','total','thermal','applied']
    
    # global stresses and strains
    ## initialize the figure
    fig = plt.figure()
    fig.canvas.set_window_title('global-stresses-strains') 
    mylw = 1.5 #linewidth
    stresslabel = ['$\sigma_x,\ MPa$','$\sigma_y,\ MPa$','$\\tau_{xy},\ MPa$']
    strainlabel = ['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$']
    for k in range(3):
        ## the top axes
        ax1 = fig.add_subplot(2,3,k+1)
        ax1.set_ylabel('thickness,z')
        ax1.set_xlabel(strainlabel[k])
        ax1.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        
        line1 = ax1.plot(np.array(epsilonbarplot[k,:])[0],     zplot, color='blue', lw=mylw)
        line2 = ax1.plot(np.array(epsilonbar_th_plot[k,:])[0], zplot, color='red', lw=mylw)
        line3 = ax1.plot(np.array(epsilonbar_app_plot[k,:])[0], zplot, color='green', lw=mylw)   
        line4 = ax1.plot([np.array(epsilon_composite[k])[0], np.array(epsilon_composite[k])[0]],\
                            [np.min(z) , np.max(z)], color='black', lw=mylw)                       
    
    for k in range(3):
        ## the top axes
        ax1 = fig.add_subplot(2,3,k+4)
        ax1.set_ylabel('thickness,z')
        ax1.set_xlabel(stresslabel[k])
        ax1.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        
        line1 = ax1.plot(np.array(sigmabarplot[k,:])[0],     zplot, color='blue', lw=mylw)
        line2 = ax1.plot(np.array(sigmabar_th_plot[k,:])[0], zplot, color='red', lw=mylw)
        line3 = ax1.plot(np.array(sigmabar_app_plot[k,:])[0], zplot, color='green', lw=mylw)   
        line4 = ax1.plot([np.array(sigma_composite[k])[0], np.array(sigma_composite[k])[0]],\
                            [np.min(z) , np.max(z)], color='black', lw=mylw) 
          
    plt.savefig('global-stresses-strains.png')
    # local stresses and strains
    ## initialize the figure
    fig = plt.figure()
    fig.canvas.set_window_title('local-stresses-strains') 
    mylw = 1.5 #linewidth
    stresslabel = ['$\sigma_1,\ MPa$','$\sigma_2,\ MPa$','$\\tau_{12},\ MPa$']
    strainlabel = ['$\epsilon_1$','$\epsilon_2$','$\gamma_{12}$']
    for k in range(3):
        ## the top axes
        ax1 = fig.add_subplot(2,3,k+1)
        ax1.set_ylabel('thickness,z')
        ax1.set_xlabel(strainlabel[k])
        ax1.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        
        line1 = ax1.plot(np.array(epsilonplot[k,:])[0],     zplot, color='blue', lw=mylw)
        line2 = ax1.plot(np.array(epsilon_th_plot[k,:])[0], zplot, color='red', lw=mylw)
        line3 = ax1.plot(np.array(epsilon_app_plot[k,:])[0], zplot, color='green', lw=mylw)   
        line4 = ax1.plot([np.array(epsilon_composite[k])[0], np.array(epsilon_composite[k])[0]],\
                            [np.min(z) , np.max(z)], color='black', lw=mylw)                       
    
    for k in range(3):
        ## the top axes
        ax1 = fig.add_subplot(2,3,k+4)
        ax1.set_ylabel('thickness,z')
        ax1.set_xlabel(stresslabel[k])
        ax1.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        
        line1 = ax1.plot(np.array(sigmaplot[k,:])[0],     zplot, color='blue', lw=mylw)
        line2 = ax1.plot(np.array(sigma_th_plot[k,:])[0], zplot, color='red', lw=mylw)
        line3 = ax1.plot(np.array(sigma_app_plot[k,:])[0], zplot, color='green', lw=mylw)   
        line4 = ax1.plot([np.array(sigma_composite[k])[0], np.array(sigma_composite[k])[0]],\
                            [np.min(z) , np.max(z)], color='black', lw=mylw) 
               
    plt.savefig('local-stresses-strains.png')
               
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
    plt.show()
    plt.savefig('plate-warpage')
    
    ########################### DATA display ##################################
    
    print('---------------Stress analysis of fibers----------')
    	## OUTPUT
    
    print(Exbar)
    print(Eybar)
    print(Gxybar)
    print(A)
    print(alpha_composite)
    print(nuxybar)
    

def ode1():
    """
    First order ode from Learning Scipy for Numerical and Scientific computing
    """
    import numpy
    from scipy.integrate import ode
    
    f = lambda t,y: -20*y
    
    actual_solution = lambda t:numpy.exp(-20*t)
    dt = 0.1
    
    solver = ode(f).set_integrator('dop853')
    solver.set_initial_value(1,0)
    
    while solver.successful() and solver.t <= 1+dt:
        print(solver.t, solver.y, actual_solution(solver.t))
        solver.integrate(solver.t+dt)

    

if __name__ == '__main__':
    
    T3rot(45)
    
    T6rot(45,45,45)
    
    qbar_transformtion()