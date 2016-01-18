# -*- coding: utf-8 -*-
'''
scripts and boilerplate code to use for mechanical engineering design tasks
'''



def gear():
    '''
    Plotting of the Involute function applied to gear
    design (adapated from http://www.arc.id.au/GearDrawing.html)    
    Transcendental Parametric function describing the contour of the gear face
    '''    
    import numpy as np
    import matplotlib.pylab as plt
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

if __name__=='__main__':
    gear()