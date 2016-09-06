# -*- coding: utf-8 -*-
'''
scripts and boilerplate code to use for mechanical engineering design tasks
'''

__author__ = 'Neal Gordon <nealagordon@gmail.com>'
__date__ =   '2016-09-06'

import pandas as pd
import numpy as np
from numpy import pi, array
import matplotlib.pyplot as plt

def gear():
    '''
    Plotting of the Involute function applied to gear
    design (adapated from http://www.arc.id.au/GearDrawing.html)    
    Transcendental Parametric function describing the contour of the gear face
    '''    

    th1 = np.pi/4
    th2 = np.pi/3
    thcir = np.linspace(-np.pi,np.pi,100)
    th = np.linspace(th1,th2,100)
    Rb = 0.5
    x = Rb*(np.sin(th)+np.cos(th))
    y = Rb*(np.sin(th)-np.cos(th))
    xcir = np.sin(thcir)
    ycir = np.cos(thcir)
    
    ofst = 0.05
    y = max(ycir)+y
    x = x-min(x)+ofst
    
    plt.plot(x,y)
    plt.plot(-x,y)
    plt.plot([-ofst , ofst],[max(y) , max(y)] )
    
    plt.plot(xcir,ycir,'--')
    plt.show()


def fastened_joint(fx, fy, P, l):
    '''computes stressed in fastened joints with bolts or rivets
    INCOMPLETE
    
    # fastener location
    fx = array([0,1,2,3,0,1,2,3])
    fy = array([0,0,0,0,1,1,1,1])
    # Force(x,y)
    P = array([-300,-500])
    l = [2,1]
    
    '''
    
    fn = range(len(fx))
    
    df = pd.DataFrame()
    
    Pnorm = P/np.max(np.abs(P))  # for plotting
    # Location of Force P, x,y
    
    d = array(5/16*np.ones(len(fn)))
    
    A = array([pi*d1**2/4 for d1 in d])
    
    fn = range(len(fx))
    
    df = pd.DataFrame({ 'Fastener' : fn,
                         'x' : fx,
                         'y' : fy}, index=fn)
                  
    df['x^2'] = df.x**2
    df['y^2'] = df.y**2
    df['xbar'] = np.sum(A*fx)/np.sum(A)
    df['ybar'] = np.sum(A*fy)/np.sum(A)    
    return df


def mohr(s):
    pass



if __name__=='__main__':
    shear_bending()
