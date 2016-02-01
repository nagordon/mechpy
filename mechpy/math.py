# coding: utf-8

'''
Module to be used for mathematical tools for mechanical engineering stuff
'''


import sympy as sp
import numpy as np
import matplotlib.pyplot as mp
from pprint import pprint

#np.set_printoptions(edgeitems=3,linewidth=75, precision=5, suppress=False, threshold=1000)
#get_ipython().magic('matplotlib inline')

def T2rot(th):
    return np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

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


def fft_example():
    from scipy import fft
    from numpy import arange, cos, pi, random
    from matplotlib.pyplot import subplot, plot, ylabel, xlabel, title, grid, xlim, show
    
    N = 2**9
    F = 25
    t = arange(N)/float(N)
    x = cos(2*pi*t*F) + random.rand(len(t))*3
    subplot(2,1,1)
    plot(t,x)
    ylabel('x []')
    xlabel('t [seconds]')
    title('A cosine wave')
    grid()
    
    subplot(2,1,2)
    f = t*N
    xf = fft(x)
    xf = xf[:N/2]
    f = f[:N/2]
    plot(f,abs(xf))
    title('Fourier transform of a cosine wave')
    xlabel('xf []')
    ylabel('xf []')
    xlim([0,N/2])
    grid()
    show()

def fft_example2():
    from numpy import sin, linspace, pi
    from pylab import plot, show, title, xlabel, ylabel, subplot
    from scipy import fft, arange
 
    Fs = 150.0;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = arange(0,1,Ts) # time vector
    
    ff = 5;   # frequency of the signal
    y = sin(2*pi*ff*t)
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    
    subplot(2,1,1)
    plot(t,y)
    xlabel('Time')
    ylabel('Amplitude')
    subplot(2,1,2)
    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')
    
    show()        
  
def fft(y,Fs):
    from numpy import sin, linspace, pi
    from pylab import plot, show, title, xlabel, ylabel, subplot
    from scipy import fft, arange
 
    Ts = 1.0/Fs; # sampling interval
    t = arange(0,1,Ts) # time vector
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    
    subplot(2,1,1)
    plot(t,y)
    xlabel('Time')
    ylabel('Amplitude')
    subplot(2,1,2)
    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')
    
    show()   

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

if __name__ == '__main__':
    

    T3rot(45)
    
    T6rot(45,45,45)
    
    qbar_transformtion()