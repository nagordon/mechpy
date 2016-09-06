# coding: utf-8

'''
Module for mechnical engineering static analysis
'''

__author__ = 'Neal Gordon <nealagordon@gmail.com>'
__date__ =   '2016-09-06'

import numpy as np
import sympy as sp
import scipy
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D  
    
def simple_support():
    L = 15
    P = 5
    Ploc = 5
    
    plt.rcParams['figure.figsize'] = (10, 8)  # (width, height)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(311)#, aspect='equal')
    
    def add_beam():
        #plt.subplot(3,1,1)
        #ax = plt.gca()
        
        plt.xlim([-1, L+1])
        plt.ylim([-1, P*2])
        
        # add rigid ground
        rectangle = plt.Rectangle((-1, -2), L+2, 2, hatch='//', fill=False)
        ax1.add_patch(rectangle)
        
        # add rigid rollers
        #circle = plt.Circle((0, 5), radius=1, fc='g')
        #ax.add_patch(circle)
        e1 = patches.Ellipse((0, 2), L/20, 4, angle=0, linewidth=2, fill=False, zorder=2)
        ax1.add_patch(e1)
        
        # add triangle
        points = [[L, 4], [L-L/40, 0], [L+L/40,0]]
        polygon = plt.Polygon(points, fill=False)
        ax1.add_patch(polygon)
        
        # add beam
        rectangle = plt.Rectangle((0, 4), L, 4, fill=False)
        ax1.add_patch(rectangle)
    
    def point_load():
        # point load shear
        x = np.linspace(0,L,100)
        y = np.ones(len(x))*P/2
        y[x>Ploc] = y[x>Ploc]-P
        x[0]=0
        x[-1]=0
        
        plt.subplot(3,1,2)
        plt.ylabel('Shear, V')
        plt.title('Shear Diagram')
        plt.fill(x, y, 'b', alpha=0.25)
        plt.grid(True)
        plt.xlim([-1, L+1])
        
        # point load bending
        x = np.linspace(-L/2,L/2,100)
        y = -(x**2)+(np.max(x**2))
        x = np.linspace(0,L,100)
        plt.subplot(3,1,3)
        plt.title('Bending Diagram')
        plt.ylabel('Moment, M')
        plt.fill(x, y, 'b', alpha=0.25)
        plt.grid(True)
        plt.xlim([-1, L+1])    
        
        # add point load
        plt.subplot(3,1,1)
        plt.annotate('P=%i'%P, ha = 'center', va = 'bottom',
                     xytext = (Ploc, 15), xy = (Ploc,7.5),
                    arrowprops = { 'facecolor' : 'black', 'shrink' : 0.05 })    
        plt.title('Free Body Diagram')
        plt.axis('off') # removes axis and labels       
        
        #    # add point load
        #    ax1.arrow(3, 11+L/10, 0, -3, head_width=L*0.02, head_length=L*0.1, fc='k', ec='k')
        #    plt.title('Free Body Diagram')
        #    plt.axis('off') # removes axis and labels
        #    #ax1.set_yticklabels('')            
    
    def dist_load():
        
                # add distributed load
        plt.subplot(3,1,1)
        for k in np.linspace(0,L,20):
            ax1.arrow(k, 11+L/10, 0, -3, head_width=L*0.01, head_length=L*0.1, fc='k', ec='k')
        plt.title('Free Body Diagram')
        plt.axis('off') # removes axis and labels
        #ax1.set_yticklabels('') 
        
        # dist load shear
        x = [0,0,L,L]
        y = [0,5,-5,0]
        plt.subplot(3,1,2)
        plt.ylabel('Shear, V')
        plt.title('Shear Diagram')
        plt.fill(x, y, 'b', alpha=0.25)
        plt.grid(True)
        plt.xlim([-1, L+1])
        
        # dist load bending
        x = np.linspace(-L/2,L/2,100)
        y = -(x**2)+(np.max(x**2))
        x = np.linspace(0,L,100)
        plt.subplot(3,1,3)
        plt.title('Bending Diagram')
        plt.ylabel('Moment, M')
        plt.fill(x, y, 'b', alpha=0.25)
        plt.grid(True)
        plt.xlim([-1, L+1])
        
    add_beam()
    dist_load()
    #point_load()
    plt.tight_layout()
    plt.show()
    
def moment_calc():

    fig = plt.figure()
    
    ax = plt.axes(projection='3d')
    
    # bar
    x=[0,0,4,4]
    y=[0,5,5,5]
    z=[0,0,0,-2]
    
    # Applied Forces
    X=[0,0,4]
    Y=[5,5,5]
    Z=[0,0,-2]
    U=[-60,0 ,80]
    V=[40 ,50,40]
    W=[20 ,0 ,-30]
    
    ax.plot(x, y, z, '-b', linewidth=5)
    ax.view_init(45, 45)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hibbler pg 129 example')
    ax.set_xlim([min(X)-2, max(X) + 2])
    ax.set_ylim([min(Y)-5, max(Y) + 2])
    ax.set_zlim([min(Z)-2, max(Z) + 2])
    
    #plt.tight_layout()
    
    ax.quiver3D(X, Y, Z, U, V, W, pivot='tail');
    
    rA = np.array([0,5,0])  # start of F1 and F2
    rB = np.array([4,5,-2])  # start of F3
    F1 = np.array([-60,40,20])
    F2 = np.array([0,50,0])
    F3 = np.array([80,40,-30])
    M = np.cross(rA,F1) + np.cross(rA,F2) + np.cross(rB,F3)
    print('Total Moment vector') 
    print(M)
    
    print('Total Force Vector about point O')
    print(sum([F1,F2,F3]))
    
    print('unit vector of the moment')
    u = M/np.linalg.norm(M)
    print(u)
    
    print('angles at which the moments react')
    print(np.rad2deg(np.arccos(u)))
    
def point_ss_shear_bending(L,Pin,ain):
    '''
    Shear Bending plot of point loads of a simply supported beam
    L = 4 # total length of beam
    Pin = [5]  # point load
    ain = [2]  # location of point load

    # or more multiple points    
    L = 10
    Pin = [3,15]
    ain = [2,6]
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.arange(0,L,L*0.02)
    V = np.zeros(len(x))
    M = np.zeros(len(x))
    
    for a, P in zip(ain, Pin):
        V[x<=a] += P*(1-a/L)
        V[x>a] += -P*a/L
        M[x<=a] += P*(1-a/L)*x[x<=a]
        M[x>a] += -P*a*(x[x>a]/L-1)    
    
    plt.subplot(2,1,1)    
    plt.stem(x,V)
    plt.ylabel('V,shear')
    plt.subplot(2,1,2)
    plt.stem(x,M)
    plt.ylabel('M,moment')
    
def moment_ss_shear_bending(L,Pin,ain):
    '''
    Shear Bending plot of moment loads of a simply supported beam
    L = 4 # total length of beam
    Pin = [5]  # point moment load
    ain = [2]  # location of point load

    # or more multiple point moments
    L = 10
    Pin = [3,-15]
    ain = [2,6]
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.arange(0,L,L*0.02)
    V = np.zeros(len(x))
    M = np.zeros(len(x))
    
    for a, P in zip(ain, Pin):
        V += -P/L
        M[x<=a] += -P*x[x<=a]/L
        M[x>a] += P*(1-x[x>a]/L)    
    
    plt.figure()
    plt.title('Point Moment Loads')
    plt.subplot(2,1,1)    
    plt.stem(x,V)
    plt.ylabel('V,shear')
    plt.subplot(2,1,2)
    plt.stem(x,M)
    plt.ylabel('M,moment')    

def dist_ss_shear_bending(L, win, ain):
    '''
    Shear Bending plot of distributed loads of a simply supported beam
    L = 10 # total length of beam
    win = [5]  # distributed load
    ain = [[3,4]]  # location of point load

    # or more multiple point moments
    L = 10
    win = [3,6]
    ain = [[0,3],[4,6]]
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    
    
    x = np.arange(0,L,L*0.02)
    V = np.zeros(len(x))
    M = np.zeros(len(x))
    
    for a, w in zip(ain, win):
        #a = ain[0]
        P = w*(a[1]-a[0])  # convert distributed load to point load
        l = (a[1]+a[0])/2
        
        i = [x<a[0]] 
        V[i] += P*(1-l/L)
        M[i] += x[i]*P*(1-l/L)
        
        i = [x>a[1]] 
        V[i] += -P*l/L
        M[i] += x[i]*-P*l/L + P*l         
        
        i = [ (a[0]<=x) & (x<=a[1]) ] 
        V[i] += P*(1-l/L) - w*(x[i]-a[0])
        M[i] += (P*(1-l/L) - w*(x[i]-a[0]))*x[i] + w*(x[i]-a[0])*(a[0]+x[i])/2
        #V[i] += P*(1-l/L)-P*x[i]
        #M[i] += P/2*(L*x[i] - x[i]**2)
        #M[i] += x[i]*P*(1-l/L) - (P*x[i]**2)/2
                

    
    plt.figure()
    plt.title('Point Moment Loads')
    plt.subplot(2,1,1)    
    plt.stem(x,V)
    plt.ylabel('V,shear')
    plt.subplot(2,1,2)
    plt.stem(x,M)
    plt.ylabel('M,moment') 

if __name__ == '__main__':
    # executed when script is run alone
    #moment_calc()
    dist_ss_shear_bending()
