# coding: utf-8

'''
Module to be used for static analysis
'''
import numpy as np
import sympy as sp
import scipy
import matplotlib.pyplot as mp
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D

def simple_support():
    L = 10
    b = 5
    a = 3
    P = 10
    
    mp.rcParams['figure.figsize'] = (12, 8)  # (width, height)
    
    fig1 = mp.figure()
    
    #mp.subplot(3,1,1)
    #ax = mp.gca()
    ax1 = fig1.add_subplot(311)#, aspect='equal')
    
    
    mp.xlim([-1, L+1])
    mp.ylim([-1, P*2])
    
    # add rigid ground
    rectangle = mp.Rectangle((-1, -2), L+2, 2, hatch='//', fill=False)
    ax1.add_patch(rectangle)
    
    # add rigid rollers
    #circle = mp.Circle((0, 5), radius=1, fc='g')
    #ax.add_patch(circle)
    e1 = patches.Ellipse((0, 2), .5, 4, angle=0, linewidth=2, fill=False, zorder=2)
    ax1.add_patch(e1)
    
    points = [[10, 4], [9.75, 0], [10.25,0]]
    polygon = mp.Polygon(points, fill=False)
    ax1.add_patch(polygon)
    
    # add beam
    rectangle = mp.Rectangle((0, 4), L, 4, fill=False)
    ax1.add_patch(rectangle)
    
    # add load
    for k in np.linspace(0,L,20):
        ax1.arrow(k, 12, 0, -3, head_width=L*.01, head_length=L*.1, fc='k', ec='k')
    mp.title('Free Body Diagram')
    mp.axis('off') # removes axis and labels
    #ax1.set_yticklabels('')
    
    x = [0,0,L,L]
    y = [0,5,-5,0]
    mp.subplot(3,1,2)
    mp.ylabel('Shear, V')
    mp.title('Shear Diagram')
    mp.fill(x, y, 'b', alpha=0.25)
    mp.grid(True)
    mp.xlim([-1, 11])
    
    x = np.linspace(-L/2,L/2,100)
    y = -(x**2)+(np.max(x**2))
    x = np.linspace(0,L,100)
    mp.subplot(3,1,3)
    mp.title('Bending Diagram')
    mp.ylabel('Moment, M')
    mp.fill(x, y, 'b', alpha=0.25)
    mp.grid(True)
    mp.xlim([-1, 11])
    
    mp.tight_layout()
    
    mp.show()
    
def moment_calc():

    fig = mp.figure()
    
    ax = mp.axes(projection='3d')
    
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
    
    #mp.tight_layout()
    
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
    
    
if __name__ == '__main__':
    # executed when script is run alone
    moment_calc()