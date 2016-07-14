# -*- coding: utf-8 -*-
"""
FEM Code
"""
from numpy import array, matrix, zeros, linspace, arange
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from numpy.linalg import solve, inv
from matplotlib.pyplot import plot, figure, xlim, ylim, title, xlabel, ylabel, show
import numpy as np 
plt.rcParams['figure.figsize'] = (10, 8)  # (width, height)

def cst_fem(structure='4node'):
    '''
    Gusset plate problem using 8 CST elemetnts. Uniform load across top edge
    is modeled with 2 concentrated forces    
    structure = ['truss','4node', '9node']
    '''
    
    ## define variables
    E      = 10e6    # modulus of elasticity
    L      = 20     # length of sketch (see drawing)
    Q      = 1000    # pounds/inch load 
    plotfactor = 1e2  # displacement factor for plot
    poisson     = 0.3    # poisson ratio
    ## Set nodal coordinates and element destination entries. 
    #         u1   u2   u3   u4   u5  u6   u7   u8
    
    #==============================================================================
    if structure == '4node':
        nodexy = array([0, 0, 10, 10, 20, 20, 0, 20])  # global node coordinets (arbitrary)    
        nodeBC = array([1, 1, 0,  0,  0,  0,  1, 1])   # boundary conditions, 1 if u=0
        
        nodex = list(nodexy[0::2])
        nodey = list(nodexy[1::2])
        ####nodexyplot = [nodex, nodey]
        nodexyT = list(zip(nodex, nodey))
        ### list(zip(nodexplot, nodeyplot))
        
        #### node        0  1  2  3
        adj = array([[0, 1, 0, 1],
                     [0, 0, 1, 1], 
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]])
                     
        ####          x  y  x  y  x  y
        ####          u1 u2 u# u# u# u#            
        elnodes = array([[0, 1, 2, 3, 6, 7],
                         [6, 7, 2, 3, 4, 5]])   # 1 element per row, dofs labled CCW (arbitrary)                       
    #==============================================================================
    elif structure == '9node':
        # 9 nodes
        nodexy = array([0, 0, L/4, L/4, L/2, L/2,3*L/4, 3*L/4, L, 
                 L,  L/2,  L,  L/4,  3*L/4,  0,  L,  0,  L/2]) # global node coordinets (arbitrary)    
        #         u1   u2   u3  u4  u5  u6  u7  u8  u9  u10  u11  u12
        nodeBC = array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # is dof fixed?
        #          x  y  x  y  x  y
        #          u1 u2 u# u# u# u#            
        elnodes = array([[ 0,  1,  2,  3, 16, 17],
                         [ 2,  3,  4,  5, 16, 17],
                         [ 4,  5, 12, 13, 16, 17],
                         [ 4,  5, 10, 11, 12, 13],
                         [ 4,  5,  6,  7, 10, 11],
                         [ 6,  7,  8,  9, 10, 11],
                         [12, 13, 10, 11, 14, 15],
                         [16, 17, 12, 13, 14, 15]])  # 1 element per row, dofs labled CCW (arbitrary)
                                                  
        adj = array([[0,1,0,0,0,0,0,0,1],
                     [0,0,1,0,0,0,0,0,1],
                     [0,0,0,1,0,1,1,0,1],
                     [0,0,0,0,1,1,0,0,0],
                     [0,0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,0,1,1,0],
                     [0,0,0,0,0,0,0,1,1],
                     [0,0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,0,0,0]])     
    #==============================================================================                 
    elif structure == 'truss':
        nodexy = array([0,0,1,0,2,0,3,0,4,0,5,0,5,1,4,1,3,1,2,1,1,1,0,1]  )
        nodeBC = array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]    ) 
            
        elnodes = array([  [ 0,  1,  2,  3, 22, 23],
                           [ 2,  3,  4,  5, 20, 21],
                           [ 4,  5,  6,  7, 18, 19],
                           [ 6,  7,  8,  9, 16, 17],
                           [ 8,  9, 10, 11, 14, 15],
                           [10, 11, 12, 13, 14, 15],
                           [ 8,  9, 14, 15, 16, 17],
                           [ 6,  7, 16, 17, 18, 19],
                           [ 4,  5, 18, 19, 20, 21],
                           [ 2,  3, 20, 21, 22, 23]])
            
        nodes = int(len(nodexy)//2)
        adj = np.zeros((nodes,nodes))
        conmat = array([[0,0, 1, 1,1,2,2,2,3,3,3,4,4,4,5,5,6,7,  8,  9, 10],
                        [1,11,2,10,11,3,9,10,4,8,9,5,7,8,6,7,  7,8,  9, 10, 11]])
        conmat = np.transpose(conmat)
        for i in range(len(conmat)):
            adj[conmat[i,0],conmat[i,1]] = 1


    #### Begin calculations
    nodex = list(nodexy[0::2])
    nodey = list(nodexy[1::2])
    ####nodexyplot = [nodex, nodey]
    nodexyT = list(zip(nodex, nodey))
    ### list(zip(nodexplot, nodeyplot))
                 
    elements     = int(len(elnodes))      # Number of elements
    nodes        = int(len(nodexy)//2)      # number of nodes
    doftotal     = int(nodes*2)      # number of total degrees of freedom
    nodexyplot   = zeros((nodes,2))  # global coordinates of nodes for plotting
    nodeplotload = zeros((nodes,2)) # global coordiantes for deflected nodes for plotting 
    P       = zeros((doftotal,1))       # total load vector
    U       = zeros((doftotal,1))        # displacements 
    Ue      = zeros((6,1))           # displacements per element, 6 for CST
    Ke      = zeros((6,6))           # stiffness per element
    K       = zeros((doftotal,doftotal))   #  totral structure stiffness
    B       = zeros((3,6))           # dN/dx , strain = B*u, correct for CST
    D       = zeros((3,3))           # Elasticity Matrix (D), correct for CST
    strain  = zeros((elements,3))       # Element(row) strain per node (column)
    stress  = zeros((elements,3))     # Element(row) stress per node (column)
    pstress = 1 # pstress >0 plane stress pstress = 0 plane strain
    ## Load Vector
    P[9] = -2e6 # 10 kips load
    #P[3] = -20000/2 # 10 kips load
    ## Elasticity Matrix
    D[1,0] = poisson
    D[0,1] = poisson
    
    if pstress == 1:
        print('plane stress condition')
        D[0,0] = 1
        D[1,1] = 1
        D[2,2] = 0.5*(1-poisson)
        D = D*E/(1-poisson*poisson)
    else:
        print('plane strain condition')
        D[0,0] = 1-poisson
        D[1,1] = 1-poisson
        D[2,2] = 0.5*(1-2*poisson)
        D = D*E/((1-2*poisson)*(1+poisson))
    
        
    
    ## loop over each element, build element [B] matrix then element matrix.
    # Assemble element stiffness into structure stiffness by building B matrix
    # B = dN/dx
    # x,y are the local nodal coordinates
    for i in range(elements): # looping through each element and building shape function
        x1 = nodexy[elnodes[i,0]]
        x2 = nodexy[elnodes[i,2]]
        x3 = nodexy[elnodes[i,4]]
        y1 = nodexy[elnodes[i,1]]
        y2 = nodexy[elnodes[i,3]]
        y3 = nodexy[elnodes[i,5]]
        x13 = x1-x3
        x21 = x2-x1
        x32 = x3-x2
        y12 = y1-y2
        y23 = y2-y3
        y31 = y3-y1
        B[0,0] = y23
        B[2,0] = x32
        B[1,1] = x32
        B[2,1] = y23
        B[0,2] = y31
        B[2,2] = x13
        B[1,3] = x13
        B[2,3] = y31
        B[0,4] = y12
        B[2,4] = x21
        B[1,5] = x21
        B[2,5] = y12
        A = 0.5*(x1*y23 + x2*y31 + x3*y12)
        B = B/(2*A)
        Ke = B.T @ D @ B * A  # matrix multiplcation
        
        # assemble elements into structure stiffness
        for kk in range(6):                    # 6 u dof in CST
            ii = elnodes[i,kk]           # u(j) in element i
            for j in range(6):               # 6 v dof in CST
                jj = elnodes[i,j]   # vj in element i
                K[ii,jj] += Ke[kk,j]   # add element to total structure
    
    ## Apply Boundary Conditions via partition matrix method for 0 displacement only
    Ksol = np.copy(K)
    Psol = np.copy(P)
    for i in range(doftotal):
    	if nodeBC[i] == 1:
    		Ksol[i,:] = 0
    		Ksol[:,i] = 0
    		Ksol[i,i] = 1
    		Psol[i] = 0
    
    ## Solve displacements
    #U = Ksol\Psol
    U = solve(Ksol,Psol) 
    
    ## retrieve kru of total structure stiffness matrix and get reactions
    R = K @ U
    ## loop over each element and form [B], get strains then stresses
    for i in range(elements):
        Ue = zeros((6,1))
        x1 = nodexy[elnodes[i,0]]
        x2 = nodexy[elnodes[i,2]]
        x3 = nodexy[elnodes[i,4]]
        y1 = nodexy[elnodes[i,1]]
        y2 = nodexy[elnodes[i,3]]
        y3 = nodexy[elnodes[i,5]]
        x13 = x1-x3
        x21 = x2-x1
        x32 = x3-x2
        y12 = y1-y2
        y23 = y2-y3
        y31 = y3-y1
        B[0,0] = y23
        B[2,0] = x32
        B[1,1] = x32
        B[2,1] = y23
        B[0,2] = y31
        B[2,2] = x13
        B[1,3] = x13
        B[2,3] = y31
        B[0,4] = y12
        B[2,4] = x21
        B[1,5] = x21
        B[2,5] = y12
        A = 0.5*(x1*y23 + x2*y31 + x3*y12)
        B = B*0.5/A
        for j in range(6):
            ii = elnodes[i,j]
            Ue[j] = U[ii]
        
        strain[i,:] = np.transpose(B @ Ue)
        stress[i,:] = np.transpose(D @ B @ Ue)
    
    # plot shape function
    x = linspace(0,20,100)
    y = linspace(0,20,100)
    N1 = (1/(2*A))*( x2*y3-y2*x3 + (y2-y3)*x  +(x3-x2)*y )
    N2 = (1/(2*A))*( x3*y1-y3*x1 + (y3-y1)*x  +(x1-x3)*y )
    N3 = (1/(2*A))*( x1*y2-y1*x2 + (y1-y2)*x  +(x2-x1)*y )
    Nsum = N1 + N2 + N3
    fig1 = figure()
    plot(N1,'-.')
    plot(N2,'--')
    plot(N3,'.')
    plot(Nsum)
    title('CST shape functions')
    show()
    
    ## plotting of FEA structure for visual confirmation of results
    fig2 = figure()
    # transforms global node coordinates nodexy from 1xc vector to nx2
    #                         x             y
    
    #adj += adj.T      
                 
    # nodexy = array([ 0,  0, 10, 10, 20, 20,  0, 20])        
    plotmag = 1
    nodexyu = nodexy+U.T[0]*plotmag
    nodexu = list(nodexyu[0::2])
    nodeyu = list(nodexyu[1::2])
    
    nodexyuT = list(zip(nodexu, nodeyu))
    
    rlen, clen = adj.shape
    x = [] ; y = []
    for r in range(rlen):
        for c in range(clen):
            if adj[r,c] == 1:
                
                x = [nodexyT[r][0], nodexyT[c][0]]
                y = [nodexyT[r][1], nodexyT[c][1]]
                plt.plot(x,y,'b')           
                x = [nodexyuT[r][0], nodexyuT[c][0]]
                y = [nodexyuT[r][1], nodexyuT[c][1]]
                plt.plot(x,y,'b--')  
    
    plt.xlim([np.min(nodex)-3, np.max(nodex)+3])
    plt.ylim([np.min(nodey)-3, np.max(nodey)+3])
    
    for i in range(nodes):
        plt.text(nodex[i]+0.5, nodey[i], '$n_%i$' % (i+1),fontsize=20)
    
    for i in range(elements):
        xtemp = nodexy[elnodes[i][0::2]]
        ytemp = nodexy[elnodes[i][1::2]]
        elnodesx = (max(xtemp) + min(xtemp)) / 2
        elnodesy = (max(ytemp) + min(ytemp)) / 2
        plt.text(elnodesx, elnodesy, '$e_%i$' % (i+1),fontsize=20)
    plt.plot(nodex, nodey, 'o')
    plt.plot(nodexu, nodeyu, '*')

    # plot stress
    elnodeslist = []
    for e in elnodes:
        tmpnodex = list(nodexy[e[0::2]])
        tmpnodey = list(nodexy[e[1::2]])
        elnodeslist.append(list(zip(tmpnodex, tmpnodey)))
    plot_contour(elnodeslist,stress, '$\sigma$')
    #plot_contour(elnodeslist,strain,'$\epsilon$')

def gplot(adj,xy):
    '''
    # for a 2 element triangle
    xy = [(0, 0), (10, 10), (20, 20), (0, 20)]
    adj = [[0, 1, 0, 1],
                 [0, 0, 1, 1], 
                 [0, 0, 0, 1],
                 [0, 0, 0, 0]]
    gplot(adj,xy)
    
    # simple truss
    xy =[(0, 0),
         (1, 0),
         (2, 0),
         (3, 0),
         (4, 0),
         (5, 0),
         (5, 1),
         (4, 1),
         (3, 1),
         (2, 1),
         (1, 1),
         (0, 1)]  
    adj = [[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]
    gplot(adj,xy)
    '''
   
    adj = np.array(adj)    
    rlen, clen = adj.shape
    nodex = [k[0] for k in xy]
    nodey = [k[1] for k in xy]
    
    xtemp = [] ; ytemp = []
    for r in range(rlen):
        for c in range(clen):
            if adj[r,c] == 1:
                x = [xy[r][0], xy[c][0]]
                y = [xy[r][1], xy[c][1]]
                plt.plot(x,y,'b') 
                #plt.plot(x,y,'b--') 
                xlim([np.min(nodex)-3, np.max(nodex)+3])
                ylim([np.min(nodey)-3, np.max(nodey)+3])     
                plt.plot(nodex, nodey, 'o')
        
def plot_contour(xynodes, stress, mytitle='contour plot'):
    """
    Simple demo of the fill function.
    xynodes = [[(0, 0), (10, 10), (0, 20)], [(10, 10), (20, 20), (0, 20)]]
    stress = [[-300,-100,-700],[300,350,-350]]
    plot_stress(xynodes, stress)
    """
  
    
    stress = np.array(stress)
    fig2 = plt.figure()

    fig2 = plt.figure()
    sigma1 = stress[:,0]
    sigma2 = stress[:,1]
    sigma12 = stress[:,2]   
    vonmises = np.sqrt(sigma1**2 - sigma1*sigma2+sigma2**2+3*sigma12**2) 
    prins1 = (sigma1 + sigma2) / 2 + np.sqrt( ((sigma1-sigma2)/2)**2 + sigma12**2 )
    prins2 = (sigma1 + sigma2) / 2 - np.sqrt( ((sigma1-sigma2)/2)**2 + sigma12**2 )
    
    sigma1norm     = sigma1/np.max(np.abs(sigma1))
    sigma2norm     = sigma2/np.max(np.abs(sigma2))
    sigma12norm   = sigma12/np.max(np.abs(sigma12))
    vonmisesnorm = vonmises/np.max(np.abs(vonmises))
    prins1norm   = prins1/np.max(np.abs(prins1))
    prins2norm = prins2/np.max(np.abs(prins2))
    
    pltsigma = sigma1norm
    for i, xy in enumerate(xynodes):
        x = [k[0] for k in xy]
        y = [k[1] for k in xy]       
        # red for tension, blue for compression        
        pltcolor = 'r' if pltsigma[i] >= 0 else 'b'
        sigmavalnorm = pltsigma[i] if pltsigma[i] > 0 else pltsigma[i]*-1
        plt.fill(x,y,pltcolor, alpha=sigmavalnorm)  
        plt.plot(x,y,'k') 

    title(mytitle)
    tmpx = [x[0] for k in xynodes for x in k]
    tmpy = [x[1] for k in xynodes for x in k]
    plt.xlim([np.min(tmpx)-3, np.max(tmpx)+3])
    plt.ylim([np.min(tmpy)-3, np.max(tmpy)+3])          
    plt.show()    


    

if __name__ == '__main__':
    
    # 4node, 9node, truss
    cst_fem('9node')
    #plot_stress()
