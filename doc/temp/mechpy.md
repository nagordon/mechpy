
- - - -
# Mechpy
a mechanical engineer's toolbox   

To view this notebook, use the [nbviewer](http://nbviewer.jupyter.org/github/nagordon/mechpy/blob/master/mechpy.ipynb)
- - - -

##  [Getting Started with Engineering Python](#Engineering-Python)  
 * getting started
 * packages
 * math with numpy and sympy
 * plotting
 


### Modules


## 1) [Statics](#Statics)
* [Example 1: A simple supported beam with shear-bending plots](#Statics-Example-1)
* [Example 2: Vector calculation method to calculate 3-D moments](#Statics-Example-2)
* [Distributed Loads Calculations](#Distributed-Loads-Calculations)

## 2) [Materials](#Materials)
* [composite mechanics](#Composite-Mechanics)
* [composite plates](#Composite-Plates)  


## 3) Kinematics
* [double_pendulum](http://matplotlib.org/examples/animation/double_pendulum_animated.html)


## 4) Dynamics 
* [dynamics](#Dynamics)
* [Python Vibration Codes](http://vibrationdata.com/software.htm)
* [Dynamics Notes](#Dynamics-Vibrations-Notes)

## 5) Design
* [Factors of Safety](#(Factors-of-Safety)  


## Appendix A: [Engineering Mathematics with Python](#Engineering-Mathematics-with-Python)
[Differential Equations](#Differential-Equations)  
[Linear Algebra](#Linear-Algebra)  
[Signal Processing](#Signal-Processing)  
[Finite Element Method](#Finite-Element-Method)
* [solids FEM example](#FEM-Example-1)  

[Curve Fitting](#Curve-Fitting)   

[Units](#Units)    


## Appendix B: [Engineering Software APIs](Engineering-Software-APIs)

[Abaqus](Abaqus)  
[CATIA](CATIA)  
[Excel](Excel)  


- - - -



```python
!jupyter nbconvert --to html mechpy.ipynb
```

    [NbConvertApp] Converting notebook mechpy.ipynb to html
    [NbConvertApp] Writing 1893062 bytes to mechpy.html


## Python Initilaization with module imports


```python
# setup 
import numpy as np
import sympy as sp
import scipy
from pprint import pprint
sp.init_printing(use_latex='mathjax')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)  # (width, height)
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 16
from matplotlib import patches

get_ipython().magic('matplotlib')  # seperate window
get_ipython().magic('matplotlib inline') # inline plotting
```

    Using matplotlib backend: Qt4Agg


- - - -
# Statics 
[index](#Mechpy)
- - - -

The sum of the forces is zero
$$
\Sigma F_x =0 , \Sigma F_y =0 , \Sigma F_z =0  
$$
The sum of the moments is zero
$$
\Sigma M_x =0 , \Sigma M_y =0 , \Sigma M_z =0  
$$

Dot Product

$$
\vec{A} \bullet \vec{B} = ABcos\left(\theta\right)= A_xB_x + A_yB_y+A_zB_z
$$

Cross-Product

$$
\vec{C}=\vec{A} \times \vec{B} = 
\begin{vmatrix}
    \widehat{i} & \widehat{j} & \widehat{k}\\
    A_{x} & A_{y} & A_{z}\\
    B_{x} & B_{y} & B_{z}
\end{vmatrix} 
$$

Moment of Force
$$
M_0 = Fd \\
\vec{M_0}=\vec{r}\times \vec{F} = 
\begin{vmatrix}
    \widehat{i} & \widehat{j} & \widehat{k}\\
    r_{x} & r_{y} & r_{z}\\
    F_{x} & F_{y} & F_{z}
\end{vmatrix} 
$$

Moment of Force about a Specified axis
$$
\vec{M_a}=\vec{u}\bullet\vec{r}\times \vec{F} = 
\begin{vmatrix}
    u_{x} & u_{y} & u_{z} \\
    r_{x} & r_{y} & r_{z} \\
    F_{x} & F_{y} & F_{z}
\end{vmatrix} 
$$


### Statics-Example 1
#### A simple supported beam with shear-bending plots


```python
from mechpy.statics import simple_support
simple_support()
```


![png](output_6_0.png)


### Statics-Example 2
### Vector calculation method to calculate 3-D moments

$
\vec{M_{R_0}}=\Sigma \left( \vec{r} \times \vec{F} \right) = \vec{r_A}\times\vec{F_1} +\vec{r_A}\times\vec{F_2} + \vec{r_B}\times\vec{F_3}
$


```python
from mechpy.statics import moment_calc
moment_calc()
```

    Total Moment vector
    [ 30 -40  60]
    Total Force Vector about point O
    [ 20 130 -10]
    unit vector of the moment
    [ 0.38411064 -0.51214752  0.76822128]
    angles at which the moments react
    [  67.41146121  120.80698112   39.80557109]



![png](output_9_1.png)


### Distributed Loads Calculations

$
F_R = \Sigma F=\int_L w(x) \,dx  = \int_A dA \,dx  
$

example, hibbler, pg 187

$$
F_R=\int_A dA \, =  \int_{0}^{2} \, 60x^2 \, dx = 160 N
$$

$$
 \overline{x} = \frac{\int_A x \, dA}{\int_A dA} =  \frac{\int_{0}^{2} x60x^2 \, dx}{\int_{0}^{2} \, 60x^2 \, dx} = \frac{240}{160}
$$


```python
x = sp.var('x')
w = 60*x**2# N/m
sp.plot(w, (x,0,2));
```


![png](output_11_0.png)



```python
w.subs(x,2)
```




$$240$$




```python
sp.Integral(w,(x,0,2))
```




$$\int_{0}^{2} 60 x^{2}\, dx$$




```python
sp.integrate(w)
```




$$20 x^{3}$$




```python
sp.integrate(w,(x,0,2))
```




$$160$$




```python
sp.Integral(x*w,(x,0,2))
```




$$\int_{0}^{2} 60 x^{3}\, dx$$




```python
sp.integrate(w*x)
```




$$15 x^{4}$$




```python
sp.integrate(x*w,(x,0,2))
```




$$240$$




```python
FR= float(sp.integrate(w,(x,0,2)))
xbar = float(sp.integrate(x*w,(x,0,2)))/FR
print('center of load of an exponential distributed load is %f' % xbar)
```

    center of load of an exponential distributed load is 1.500000



```python
#hibbler pg 346

import numpy as np

r = np.array([    0, 3 , 5.25])
F = np.array([-13.5, 0 ,6.376])
M = -np.cross(r,F)

# moments 
print('M_x = %f \nM_y = %f \nM_z = %f' % (M[0], M[1], M[2] ))
```

    M_x = -19.128000 
    M_y = 70.875000 
    M_z = -40.500000


# Materials
[index](#Mechpy)

## Stress and Strain
Stress is a tensor that can be broken into 

$$
\overline{\sigma}=\begin{bmatrix}
 \sigma_{xx} & \sigma_{xy} & \sigma_{xz}\\
 \sigma_{yx} & \sigma_{yy} & \sigma_{yz}\\
 \sigma_{zx} & \sigma_{zy} & \sigma_{zz}
 \end{bmatrix} 
$$



## Factors of safety
In aerospace, typically 1.2 for civilian aircraft and 1.15 for military

$$FS=\frac{\sigma_{yield}}{\sigma}-1$$  

## Fastener Notes and Formulas

Finding the centroid of a bolt with coordinates, $\overline{x},\overline{y}$
$$ \overline{x}=\frac{\sum_{i=1}^{n_b}{A_i x_i} }{\sum_{i=1}^{n_b}{A_i} } \ \ \overline{y}=\frac{\sum_{i=1}^{n_b}{A_i y_i} }{\sum_{i=1}^{n_b}{A_i}}$$

Joint/Polar Moment of Inertia, $r=$ distance from centroid to fastener
$$J= \int{r^2dA}= \sum_{i=1}^{n_b}{A_k r_k^2}$$

Bearing Stress on a bolt
$$\sigma^i_{bearing}=\frac{V_{max}}{Dt}$$

Shear Stress on each bolt i due to shear force
$$\tau_f^i = \frac{P}{\sum_{i=1}^{n_b}{A_i} }$$  
Where $A_i=$ the area of ith bolt, $n_b=$number of bolts, and $P=$ shear force

Shear Stress on each bolt i due to moment
$$\tau_t^i = \frac{T r_i}{J} $$  

### Modes of failure of fastened Joints
1. Tensile Plate Failure across the net section between rivets/bolts
2. Failure of rivets through shear
3. Compression failure between rivet and plate
4. Edge shear-out at rivet hole
5. Edge tearing at rivet hole

#### 1.

$$\sigma_t =\frac{F_s}{(b-nd)t}$$

#### 2.

#### 3.

#### 4.

#### 5.



## Adhesive Joints

With members, or adherends, joined with adhesives, either the member will fail due to tensile loads or the adhesive will fail in shear.

The simple solution to finding the stress of bonded surfaces is taking the average stress
$$\tau_{avg}=\frac{P}{bL}$$, is not an accurate way to model maximum stress. A good rule of thumb based on the calculations below is 
$$\tau_{max}=2.08\tau_{avg}$$

The maximum shearing stress of an adhesive layer, $\tau_{max}$, can be computed as 
$$\tau_{max}=K_s\tau_{avg}=K_s\left(\frac{P}{bL_L}\right)$$
with $P$ as applied load, $b$ as the width ofthe adhesive layer, and $L_L$ as the length ofthe adhesive layer. The stress distribution factor, $K_s$, can be defined as $K_s=\frac{cL}{tanh(CL/2)}$ where $c=\sqrt{\frac{2G_a}{Et_mt_a}}$, where the shear modulus, $G_a=\frac{\tau}{\gamma}$, and $E$ as the modulus of elasticity.


The max shearing stress, $\tau_{max}$ in a scarf joint can be found with 
$$\tau_{max}=K_s\tau_{avg}=K_s\left[ \frac{Pcos\theta}{\left(\frac{bt}{sin\theta} \right)  } \right] = K_s\left(  \frac{P}{bt} sin\theta cos\theta \right)$$
where $t$ is the thickness of the adherend members and $\theta=tan^{-1}\frac{t}{L_s}$ is the scarf angle

*Mechanical Design of Machine Elements and Machines by Collins, Jack A., Busby, Henry R., Staab, George H. (2009)*


```python

```


```python
%matplotlib inline
```


```python
## Bolted Joint Example

# fastener Location
from mechpy.design import fastened_joint
fx = [0,1,2,3,0,1,2,3]
fy = [0,0,0,0,1,1,1,1]
# Force magnitude(x,y)
P = [-300,-500]
# Force location
l = [2,1]
df = fastened_joint(fx, fy, P, l)

df.plot(kind='scatter', x='x', y='y');
#df.plot(style='o', x='x', y='y')
plt.plot(df.xbar[0],df.ybar[0],'*')
df
#ax = plt.gca()
#ax.arrow(l[0], l[1], Pnorm[0],Pnorm[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
#x.arrow(xbar, ybar, Pnorm[0],0, head_width=0.05, head_length=0.1, fc='k', ec='k')
#ax.arrow(xbar, ybar, 0,Pnorm[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fastener</th>
      <th>x</th>
      <th>y</th>
      <th>x^2</th>
      <th>y^2</th>
      <th>xbar</th>
      <th>ybar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_24_1.png)


## Composite Mechanics
[index](#Mechpy)


```python
from mechpy.math import T3rot, T6rot
from mechpy.composites import laminate, material_plots, laminate_gen
```


```python
import importlib
importlib.reload
```




    <function importlib.reload>




```python
from mechpy.math import T2rot
T2rot(45)
```




    array([[ 0.5253, -0.8509],
           [ 0.8509,  0.5253]])




```python
from IPython.html.widgets import *

plt.figure(figsize=(12,8))
x = [-1,1, 0,-1,]
y = [-1,-1,1,-1]
xy = np.array([x,y])
plt.xlim([-11.1,11.1])
plt.ylim([-11.1,11.1])
xyR = np.dot(T2rot(30),xy)
#plt.plot(xyR[0,:],xyR[1,:])
def rot2(th, xt,yt,zt):
    xyR = np.dot(T2rot(th),xy*zt)
    xyR[0,:]+=xt
    xyR[1,:]+=yt
    plt.plot(xyR[0,:],xyR[1,:])
    plt.axis('square')
    plt.xlim([-11.1,11.1])
    plt.ylim([-11.1,11.1])    
    plt.show()

interact(rot2, th=(0,np.pi,np.pi/90), yt=(1,10,1), xt=(1,10,1), zt=(1,10,1));
```


![png](output_29_0.png)



```python
print(T6rot(45,45,45))
```

    [[ 0.25    0.25    0.5    -0.3536 -0.3536  0.25  ]
     [ 0.0214  0.7286  0.25    0.4268 -0.0732 -0.125 ]
     [ 0.7286  0.0214  0.25   -0.0732  0.4268 -0.125 ]
     [-0.25   -0.25    0.5     0.3536  0.3536  0.75  ]
     [ 0.8536 -0.1464 -0.7071  0.3536 -0.3536  0.3536]
     [-0.1464  0.8536 -0.7071 -0.3536  0.3536  0.3536]]



```python
help(laminate_gen)
```

    Help on function laminate_gen in module mechpy.composites:
    
    laminate_gen(lamthk=1.5, symang=[45, 0, 90], plyratio=2.0, matrixlayers=False, balancedsymmetric=True)
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
    



```python
material_plots()
```


![png](output_32_0.png)



![png](output_32_1.png)



![png](output_32_2.png)



![png](output_32_3.png)



![png](output_32_4.png)



```python
laminate()
```

    Using matplotlib backend: Qt4Agg
    --------------- laminate1 Stress analysis of fibers----------
    plyangles
    [30, -30, 0, 0, -30, 30]
    ply layers
    [ -4.5000e-04  -3.0000e-04  -1.5000e-04  -5.4210e-20   1.5000e-04   3.0000e-04   4.5000e-04]
    alpha
    [[ -1.8000e-08]
     [  2.4300e-05]
     [  0.0000e+00]]
    ABD=
    [[  1.0240e+08   1.8945e+07  -1.8626e-09  -5.4570e-12  -1.3642e-12   0.0000e+00]
     [  1.8945e+07   1.6250e+07  -9.3132e-10  -1.3642e-12  -1.3642e-12   0.0000e+00]
     [ -1.8626e-09  -9.3132e-10   2.0191e+07   0.0000e+00   1.1369e-13  -1.3642e-12]
     [ -5.4570e-12  -1.3642e-12   0.0000e+00   5.7792e+00   1.7657e+00   1.2611e+00]
     [ -1.3642e-12  -1.3642e-12   0.0000e+00   1.7657e+00   1.2561e+00   4.1768e-01]
     [  0.0000e+00   1.1369e-13  -1.3642e-12   1.2611e+00   4.1768e-01   1.8498e+00]]
    Ex=   8.92E+10
    Ey=   1.42E+10
    nuxy= 1.17E+00
    Gxy=  2.24E+10
    alpha_laminate
    [[ -2.1595e-06]
     [  1.6802e-05]
     [  7.5173e-22]]
    epsilon_laminate
    [[  3.2393e-04]
     [ -2.5203e-03]
     [ -1.9508e-19]
     [  4.2353e+00]
     [ -5.3989e+00]
     [ -1.6683e+00]]
    NMhatth
    [[  9.7160e+01]
     [  2.3212e+02]
     [  3.5527e-15]
     [ -6.9389e-18]
     [ -1.7347e-17]
     [  0.0000e+00]]
    sigma_laminate
    [[ -1.6193e+07]
     [ -3.8686e+07]
     [ -5.9212e-10]
     [  1.4267e+04]
     [  5.9212e-12]
     [  4.9343e-13]]
    epsilon_th
    [[ -3.8982e-04  -3.8982e-04  -3.8982e-04  -3.8982e-04   3.2123e-04   3.2123e-04   3.2123e-04   3.2123e-04  -3.8982e-04  -3.8982e-04  -3.8982e-04  -3.8982e-04]
     [  1.8358e-03   1.8358e-03   1.8358e-03   1.8358e-03   1.1247e-03   1.1247e-03   1.1247e-03   1.1247e-03   1.8358e-03   1.8358e-03   1.8358e-03   1.8358e-03]
     [ -2.4631e-03  -2.4631e-03   2.4631e-03   2.4631e-03  -1.5405e-19  -1.1276e-19  -1.1276e-19  -7.1468e-20   2.4631e-03   2.4631e-03  -2.4631e-03  -2.4631e-03]]
    epsilonbar_th
    [[  3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04   3.2393e-04]
     [ -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03  -2.5203e-03]
     [ -4.3368e-19   0.0000e+00   0.0000e+00   0.0000e+00  -1.5405e-19  -1.1276e-19  -1.1276e-19  -7.1468e-20   0.0000e+00   0.0000e+00   0.0000e+00   0.0000e+00]]
    sigma_th
    [[ -5.5178e+07  -5.5178e+07  -5.5178e+07  -5.5178e+07   5.3423e+07   5.3423e+07   5.3423e+07   5.3423e+07  -5.5178e+07  -5.5178e+07  -5.5178e+07  -5.5178e+07]
     [  2.1145e+07   2.1145e+07   2.1145e+07   2.1145e+07   1.4644e+07   1.4644e+07   1.4644e+07   1.4644e+07   2.1145e+07   2.1145e+07   2.1145e+07   2.1145e+07]
     [ -1.0838e+07  -1.0838e+07   1.0838e+07   1.0838e+07  -6.7782e-10  -4.9614e-10  -4.9614e-10  -3.1446e-10   1.0838e+07   1.0838e+07  -1.0838e+07  -1.0838e+07]]
    sigmabar_th
    [[ -2.6711e+07  -2.6711e+07  -2.6711e+07  -2.6711e+07   5.3423e+07   5.3423e+07   5.3423e+07   5.3423e+07  -2.6711e+07  -2.6711e+07  -2.6711e+07  -2.6711e+07]
     [ -7.3218e+06  -7.3218e+06  -7.3218e+06  -7.3218e+06   1.4644e+07   1.4644e+07   1.4644e+07   1.4644e+07  -7.3218e+06  -7.3218e+06  -7.3218e+06  -7.3218e+06]
     [ -3.8468e+07  -3.8468e+07   3.8468e+07   3.8468e+07  -6.7782e-10  -4.9614e-10  -4.9614e-10  -3.1446e-10   3.8468e+07   3.8468e+07  -3.8468e+07  -3.8468e+07]]
    epsilon_app
    [[ -4.9697e-04  -3.3132e-04  -7.6475e-04  -3.8237e-04  -6.3530e-04  -1.0500e-20  -1.0500e-20   6.3530e-04   3.8237e-04   7.6475e-04   3.3132e-04   4.9697e-04]
     [  1.0206e-03   6.8039e-04   1.1138e-03   5.5691e-04   8.0983e-04  -6.0443e-20  -6.0443e-20  -8.0983e-04  -5.5691e-04  -1.1138e-03  -6.8039e-04  -1.0206e-03]
     [  4.1299e-03   2.7533e-03  -2.2528e-03  -1.1264e-03   2.5024e-04   8.1161e-21   8.1161e-21  -2.5024e-04   1.1264e-03   2.2528e-03  -2.7533e-03  -4.1299e-03]]
    epsilonbar_app
    [[ -1.9059e-03  -1.2706e-03  -1.2706e-03  -6.3530e-04  -6.3530e-04  -1.0500e-20  -1.0500e-20   6.3530e-04   6.3530e-04   1.2706e-03   1.2706e-03   1.9059e-03]
     [  2.4295e-03   1.6197e-03   1.6197e-03   8.0983e-04   8.0983e-04  -6.0443e-20  -6.0443e-20  -8.0983e-04  -8.0983e-04  -1.6197e-03  -1.6197e-03  -2.4295e-03]
     [  7.5073e-04   5.0048e-04   5.0048e-04   2.5024e-04   2.5024e-04   8.1161e-21   8.1161e-21  -2.5024e-04  -2.5024e-04  -5.0048e-04  -5.0048e-04  -7.5073e-04]]
    NMbarapp
    [[  0.  ]
     [  0.  ]
     [  0.  ]
     [ 12.84]
     [  0.  ]
     [  0.  ]]
    epsilon
    [[ -8.8679e-04  -7.2113e-04  -1.1546e-03  -7.7219e-04  -3.1407e-04   3.2123e-04   3.2123e-04   9.5653e-04  -7.4431e-06   3.7493e-04  -5.8501e-05   1.0716e-04]
     [  2.8564e-03   2.5162e-03   2.9496e-03   2.3927e-03   1.9346e-03   1.1247e-03   1.1247e-03   3.1490e-04   1.2789e-03   7.2196e-04   1.1554e-03   8.1520e-04]
     [  1.6668e-03   2.9014e-04   2.1035e-04   1.3367e-03   2.5024e-04  -1.0464e-19  -1.0464e-19  -2.5024e-04   3.5896e-03   4.7160e-03  -5.2164e-03  -6.5931e-03]]
    epsilonbar
    [[ -1.5820e-03  -9.4667e-04  -9.4667e-04  -3.1137e-04  -3.1137e-04   3.2393e-04   3.2393e-04   9.5923e-04   9.5923e-04   1.5945e-03   1.5945e-03   2.2298e-03]
     [ -9.0764e-05  -9.0060e-04  -9.0060e-04  -1.7104e-03  -1.7104e-03  -2.5203e-03  -2.5203e-03  -3.3301e-03  -3.3301e-03  -4.1399e-03  -4.1399e-03  -4.9498e-03]
     [  7.5073e-04   5.0048e-04   5.0048e-04   2.5024e-04   2.5024e-04  -1.0464e-19  -1.0464e-19  -2.5024e-04  -2.5024e-04  -5.0048e-04  -5.0048e-04  -7.5073e-04]]
    sigma
    [[ -1.2950e+08  -1.0473e+08  -1.7093e+08  -1.1305e+08  -4.3082e+07   5.3423e+07   5.3423e+07   1.4993e+08   2.6969e+06   6.0572e+07  -5.6276e+06   1.9148e+07]
     [  3.2055e+07   2.8418e+07   3.2381e+07   2.6763e+07   2.2574e+07   1.4644e+07   1.4644e+07   6.7129e+06   1.5527e+07   9.9084e+06   1.3871e+07   1.0235e+07]
     [  7.3338e+06   1.2766e+06   9.2552e+05   5.8817e+06   1.1011e+06  -4.6043e-10  -4.6043e-10  -1.1011e+06   1.5794e+07   2.0750e+07  -2.2952e+07  -2.9010e+07]]
    sigmabar
    [[ -9.5465e+07  -7.2547e+07  -1.1930e+08  -7.3005e+07  -4.3082e+07   5.3423e+07   5.3423e+07   1.4993e+08   1.9582e+07   6.5876e+07   1.9124e+07   4.2042e+07]
     [ -1.9834e+06  -3.7629e+06  -1.9248e+07  -1.3285e+07   2.2574e+07   1.4644e+07   1.4644e+07   6.7129e+06  -1.3589e+06   4.6040e+06  -1.0881e+07  -1.2660e+07]
     [ -6.6290e+07  -5.7016e+07   8.8498e+07   6.3483e+07   1.1011e+06  -4.6043e-10  -4.6043e-10  -1.1011e+06   1.3452e+07  -1.1563e+07  -1.9919e+07  -1.0645e+07]]



![png](output_33_1.png)



![png](output_33_2.png)



![png](output_33_3.png)


## Composite Plates  
[index](#Mechpy)  



```python

```


```python

```


```python

```


```python

```


```python

```

# Stress Transformations
[index](#Mechpy)  

$$
\overline{\sigma}=\begin{bmatrix}
 \sigma_{xx} & \sigma_{xy} & \sigma_{xz}\\
 \sigma_{yx} & \sigma_{yy} & \sigma_{yz}\\
 \sigma_{zx} & \sigma_{zy} & \sigma_{zz}
 \end{bmatrix} 
$$

reduce to plane stress

$$
\overline{\sigma}=\begin{bmatrix}
 \sigma_{xx} & \sigma_{xy} & 0 \\
 \sigma_{yx} & \sigma_{yy} & 0\\
 0 &           0 &           \sigma_{zz}
 \end{bmatrix} 
$$

or

$$
\overline{\sigma}=\begin{bmatrix}
 \sigma_{xx} & \tau_{xy} & 0 \\
 \tau_{yx} & \sigma_{yy} & 0\\
 0 &           0 &           \sigma_{zz}
 \end{bmatrix} 
$$


$$
\overline{\sigma}=\begin{bmatrix}
 \sigma_{x} & \sigma_{xy} \\
 \sigma_{yx} & \sigma_{y} \\
 \end{bmatrix} 
$$

Transformation

$$
A=\begin{bmatrix}
 cos(\theta) & sin(\theta) \\
 -sin(\theta) & cos(\theta) \\
 \end{bmatrix} 
$$

$$
\sigma'=A \sigma A^T
$$


$$
\sigma_1 , \sigma_2 = \frac{\sigma_{x}}{2} + \frac{\sigma_{y}}{2} + \sqrt{\tau_{xy}^{2} + \left(\frac{\sigma_{x}}{2} - \frac{\sigma_{y}}{2}\right)^{2}}
$$


$$
T=\left[\begin{matrix}\sin^{2}{\left (\theta \right )} & \cos^{2}{\left (\theta \right )} & 2 \sin{\left (\theta \right )} \cos{\left (\theta \right )}\cos^{2}{\left (\theta \right )} & \\
\sin^{2}{\left (\theta \right )} & - 2 \sin{\left (\theta \right )} \cos{\left (\theta \right )}\- \sin{\left (\theta \right )} \cos{\left (\theta \right )} & \\
\sin{\left (\theta \right )} \cos{\left (\theta \right )} & \sin^{2}{\left (\theta \right )} - \cos^{2}{\left (\theta \right )}\end{matrix}\right]
$$


```python
import sympy as sp
from sympy.abc import tau, sigma
import numpy as np
sp.init_printing()
```


```python
sx,sy,txy,tp = sp.symbols('sigma_x,sigma_y,tau_xy,theta_p')
sp1 = (sx+sy)/2 + sp.sqrt( ((sx-sy)/2)**2 + txy**2 )
sp2 = (sx+sy)/2 - sp.sqrt( ((sx-sy)/2)**2 + txy**2 )
print(sp.latex(sp1))
sp1
```

    \frac{\sigma_{x}}{2} + \frac{\sigma_{y}}{2} + \sqrt{\tau_{xy}^{2} + \left(\frac{\sigma_{x}}{2} - \frac{\sigma_{y}}{2}\right)^{2}}





$$\frac{\sigma_{x}}{2} + \frac{\sigma_{y}}{2} + \sqrt{\tau_{xy}^{2} + \left(\frac{\sigma_{x}}{2} - \frac{\sigma_{y}}{2}\right)^{2}}$$




```python
tp = sp.atan(2*txy/(sx-sy) )/2
tp
```




$$\frac{1}{2} \operatorname{atan}{\left (\frac{2 \tau_{xy}}{\sigma_{x} - \sigma_{y}} \right )}$$




```python
tpp = tp.evalf(subs={sx:10,sy:15,txy:10})
tpp
```




$$-0.662908831834016$$




```python
#s,s11,s22,s33,s12 = sp.var('s,s11,s22,s33,s12')
s,s11,s22,s33,s12,s13,t,t12 = sp.symbols('sigma, sigma11,sigma22,sigma33,sigma12,sigma13,tau,tau12')
s = sp.Matrix([[s11,t12,0],[t12,s22,0],[0,0,s33]])
s
```




$$\left[\begin{matrix}\sigma_{11} & \tau_{12} & 0\\\tau_{12} & \sigma_{22} & 0\\0 & 0 & \sigma_{33}\end{matrix}\right]$$




```python
t = sp.symbols('theta')
m = sp.sin(t)
n = sp.cos(t)
T = sp.Matrix([[m**2,n**2, 2*m*n],[n**2,m**2,-2*m*n],[-m*n,m*n,m**2-n**2]])
T
```




$$\left[\begin{matrix}\sin^{2}{\left (\theta \right )} & \cos^{2}{\left (\theta \right )} & 2 \sin{\left (\theta \right )} \cos{\left (\theta \right )}\\\cos^{2}{\left (\theta \right )} & \sin^{2}{\left (\theta \right )} & - 2 \sin{\left (\theta \right )} \cos{\left (\theta \right )}\\- \sin{\left (\theta \right )} \cos{\left (\theta \right )} & \sin{\left (\theta \right )} \cos{\left (\theta \right )} & \sin^{2}{\left (\theta \right )} - \cos^{2}{\left (\theta \right )}\end{matrix}\right]$$




```python
T1 = T.subs(t, sp.pi/4)
T1
```




$$\left[\begin{matrix}\frac{1}{2} & \frac{1}{2} & 1\\\frac{1}{2} & \frac{1}{2} & -1\\- \frac{1}{2} & \frac{1}{2} & 0\end{matrix}\right]$$




```python
sprime = T1 * s * T1.inv()
sprime
```




$$\left[\begin{matrix}\frac{\sigma_{11}}{4} + \frac{\sigma_{22}}{4} + \frac{\sigma_{33}}{2} + \frac{\tau_{12}}{2} & \frac{\sigma_{11}}{4} + \frac{\sigma_{22}}{4} - \frac{\sigma_{33}}{2} + \frac{\tau_{12}}{2} & - \frac{\sigma_{11}}{2} + \frac{\sigma_{22}}{2}\\\frac{\sigma_{11}}{4} + \frac{\sigma_{22}}{4} - \frac{\sigma_{33}}{2} + \frac{\tau_{12}}{2} & \frac{\sigma_{11}}{4} + \frac{\sigma_{22}}{4} + \frac{\sigma_{33}}{2} + \frac{\tau_{12}}{2} & - \frac{\sigma_{11}}{2} + \frac{\sigma_{22}}{2}\\- \frac{\sigma_{11}}{4} + \frac{\sigma_{22}}{4} & - \frac{\sigma_{11}}{4} + \frac{\sigma_{22}}{4} & \frac{\sigma_{11}}{2} + \frac{\sigma_{22}}{2} - \tau_{12}\end{matrix}\right]$$




```python
sprime.evalf(subs={s11:10, s22:00, s33:0, t12:0})
```




$$\left[\begin{matrix}2.5 & 2.5 & -5.0\\2.5 & 2.5 & -5.0\\-2.5 & -2.5 & 5.0\end{matrix}\right]$$




```python
s.eigenvals() 
```




$$\left \{ \sigma_{33} : 1, \quad \frac{\sigma_{11}}{2} + \frac{\sigma_{22}}{2} - \frac{1}{2} \sqrt{\sigma_{11}^{2} - 2 \sigma_{11} \sigma_{22} + \sigma_{22}^{2} + 4 \tau_{12}^{2}} : 1, \quad \frac{\sigma_{11}}{2} + \frac{\sigma_{22}}{2} + \frac{1}{2} \sqrt{\sigma_{11}^{2} - 2 \sigma_{11} \sigma_{22} + \sigma_{22}^{2} + 4 \tau_{12}^{2}} : 1\right \}$$




```python
s2 = s.evalf(subs={s11:2.2, s22:3, s33:sp.pi, s12:7.3})
s2
```




$$\left[\begin{matrix}2.2 & \tau_{12} & 0\\\tau_{12} & 3.0 & 0\\0 & 0 & 3.14159265358979\end{matrix}\right]$$




```python
sigma = np.array([[90,60],
                  [60,-20]])
np.linalg.eigvals(sigma)
```




    array([ 116.39410298,  -46.39410298])




```python
# PLane Stress

tauxy = 1    # lbs/in 
sigmax = 0   # lbs/in
sigmay = 0   # lbs/in

sigma = np.array([[sigmax, tauxy,0],
                  [tauxy,   sigmay,0],
                 [0,0,0]])

sigmap = np.linalg.eig(sigma)[0]
print(sigmap)

thetap = np.linalg.eig(sigma)[1]  # degrees

print('cosine angle')
print(thetap )  # cosine angle

print('plane angle')
print(np.arccos(thetap)*180/np.pi)
```

    [ 1. -1.  0.]
    cosine angle
    [[ 0.70710678 -0.70710678  0.        ]
     [ 0.70710678  0.70710678  0.        ]
     [ 0.          0.          1.        ]]
    plane angle
    [[  45.  135.   90.]
     [  45.   45.   90.]
     [  90.   90.    0.]]



```python
# maximum in-plane shear stress
eps = 1e-16   # machine epsilon to avoid divide-by-zero error
rad_to_deg = 180/np.pi
theta1 = 0.5 * np.arctan( 2*tauxy / ((sigmax-sigmay+eps))) * rad_to_deg
print(theta1)
```

    45.0



```python
tauxy = 0    # lbs/in 
sigmax = 100   # lbs/in
sigmay = np.linspace(0,1.100)   # lbs/in

eps = 1e-16   # machine epsilon to avoid divide-by-zero error
rad_to_deg = 180/np.pi
theta1 = 0.5 * np.arctan( 2*tauxy / ((sigmax-sigmay+eps))) * rad_to_deg
print(theta1)

# sigmax = 100
# sigmay = np.linspace(0,1.100)
# tauxy = 0
# tparray = sp.atan(2*tauxy/(sigmax-sigmay) )/2
# tparray
```

    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]



```python
sigma
```




    array([[0, 1, 0],
           [1, 0, 0]])




```python
th = np.pi/4  # 45 deg
m = np.cos(th)
n = np.sin(th)
A = np.array([ [m,n],[-n,m]])

tauxy = 1    # lbs/in 
sigmax = 0   # lbs/in
sigmay = 0   # lbs/in

sigma = np.array([[sigmax, tauxy],
                  [tauxy,   sigmay]])

sigmat = A @ sigma @ A.T  # transformed stress
sigmat
```




    array([[ 1.,  0.],
           [ 0., -1.]])




```python
sigmap = np.linalg.eig(sigmat)[0]  # principal stresses
print(sigmap)

thetap = np.linalg.eig(sigmat)[1]  # principal planes
print(thetap* 180/np.pi)
```

    [ 1. -1.]
    [[ 57.29577951   0.        ]
     [  0.          57.29577951]]



```python

```


```python

```


```python

```


```python

```


```python

```


```python
from ipywidgets import IntSlider
IntSlider()
```


```python
# Principal Stresses

sx  = 63.66
sy  = 0
sz  = 0
txy = 63.66
txz = 0
tyz = 0

S = np.matrix([[sx, txy, txz],
            [txy, sy, tyz],
            [txy, txz, sz]])

print(S)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-afcfef3f49b2> in <module>()
          8 tyz = 0
          9 
    ---> 10 S = np.matrix([[sx, txy, txz],
         11             [txy, sy, tyz],
         12             [txy, txz, sz]])


    NameError: name 'np' is not defined



```python
principal_stresses = np.linalg.eigvals(S)
print(principal_stresses)
```

    [   0.          -39.34404372  103.00404372]



```python
import sympy as sp
from sympy.abc import tau, sigma
#s,s11,s22,s33,s12 = sp.var('s,s11,s22,s33,s12')
s,s11,s22,s33,s12,s13 = sp.symbols('sigma, sigma11,sigma22,sigma33,sigma12,sigma13')
s = sp.Matrix([[s11,s12,0],[s12,s22,0],[0,0,s33]])
s
```




$$\left[\begin{matrix}\sigma_{11} & \sigma_{12} & 0\\\sigma_{12} & \sigma_{22} & 0\\0 & 0 & \sigma_{33}\end{matrix}\right]$$




```python
s**2
```




$$\left[\begin{matrix}\sigma_{11}^{2} + \sigma_{12}^{2} & \sigma_{11} \sigma_{12} + \sigma_{12} \sigma_{22} & 0\\\sigma_{11} \sigma_{12} + \sigma_{12} \sigma_{22} & \sigma_{12}^{2} + \sigma_{22}^{2} & 0\\0 & 0 & \sigma_{33}^{2}\end{matrix}\right]$$




```python
s.eigenvals()  # hmm looks familiar
```




$$\left \{ \sigma_{33} : 1, \quad \frac{\sigma_{11}}{2} + \frac{\sigma_{22}}{2} - \frac{1}{2} \sqrt{\sigma_{11}^{2} - 2 \sigma_{11} \sigma_{22} + 4 \sigma_{12}^{2} + \sigma_{22}^{2}} : 1, \quad \frac{\sigma_{11}}{2} + \frac{\sigma_{22}}{2} + \frac{1}{2} \sqrt{\sigma_{11}^{2} - 2 \sigma_{11} \sigma_{22} + 4 \sigma_{12}^{2} + \sigma_{22}^{2}} : 1\right \}$$




```python
s1 = s.subs(s11,2.2).subs(s22,3).subs(s33,sp.pi).subs(s12,7.3)
s1
```




$$\left[\begin{matrix}2.2 & 7.3 & 0\\7.3 & 3 & 0\\0 & 0 & \pi\end{matrix}\right]$$




```python
# or 
s2 = s.evalf(subs={s11:2.2, s22:3, s33:sp.pi, s12:7.3})
s2
```




$$\left[\begin{matrix}2.2 & 7.3 & 0\\7.3 & 3.0 & 0\\0 & 0 & 3.14159265358979\end{matrix}\right]$$




```python
s1.eigenvals()
```




$$\left \{ \pi : 1, \quad \frac{13}{5} + \frac{\sqrt{5345}}{10} : 1, \quad - \frac{\sqrt{5345}}{10} + \frac{13}{5} : 1\right \}$$




```python
s2.eigenvals()
```




$$\left \{ \frac{314159265358979}{100000000000000} : 1, \quad \frac{13}{5} + \frac{\sqrt{5345}}{10} : 1, \quad - \frac{\sqrt{5345}}{10} + \frac{13}{5} : 1\right \}$$




```python
s2.inv()
```




$$\left[\begin{matrix}-0.0642535874919684 & 0.156350396230456 & 0\\0.156350396230456 & -0.0471192974941101 & 0\\0 & 0 & 0.318309886183791\end{matrix}\right]$$




```python
C = sp.symbols('C1:100')
C
```




$$\left ( C_{1}, \quad C_{2}, \quad C_{3}, \quad C_{4}, \quad C_{5}, \quad C_{6}, \quad C_{7}, \quad C_{8}, \quad C_{9}, \quad C_{10}, \quad C_{11}, \quad C_{12}, \quad C_{13}, \quad C_{14}, \quad C_{15}, \quad C_{16}, \quad C_{17}, \quad C_{18}, \quad C_{19}, \quad C_{20}, \quad C_{21}, \quad C_{22}, \quad C_{23}, \quad C_{24}, \quad C_{25}, \quad C_{26}, \quad C_{27}, \quad C_{28}, \quad C_{29}, \quad C_{30}, \quad C_{31}, \quad C_{32}, \quad C_{33}, \quad C_{34}, \quad C_{35}, \quad C_{36}, \quad C_{37}, \quad C_{38}, \quad C_{39}, \quad C_{40}, \quad C_{41}, \quad C_{42}, \quad C_{43}, \quad C_{44}, \quad C_{45}, \quad C_{46}, \quad C_{47}, \quad C_{48}, \quad C_{49}, \quad C_{50}, \quad C_{51}, \quad C_{52}, \quad C_{53}, \quad C_{54}, \quad C_{55}, \quad C_{56}, \quad C_{57}, \quad C_{58}, \quad C_{59}, \quad C_{60}, \quad C_{61}, \quad C_{62}, \quad C_{63}, \quad C_{64}, \quad C_{65}, \quad C_{66}, \quad C_{67}, \quad C_{68}, \quad C_{69}, \quad C_{70}, \quad C_{71}, \quad C_{72}, \quad C_{73}, \quad C_{74}, \quad C_{75}, \quad C_{76}, \quad C_{77}, \quad C_{78}, \quad C_{79}, \quad C_{80}, \quad C_{81}, \quad C_{82}, \quad C_{83}, \quad C_{84}, \quad C_{85}, \quad C_{86}, \quad C_{87}, \quad C_{88}, \quad C_{89}, \quad C_{90}, \quad C_{91}, \quad C_{92}, \quad C_{93}, \quad C_{94}, \quad C_{95}, \quad C_{96}, \quad C_{97}, \quad C_{98}, \quad C_{99}\right )$$




```python

```


```python

```


```python

```


```python
from mechpy.math import ode1
ode1()
```

    0 [ 1.] 1.0
    0.1 [ 0.1353353] 0.135335283237
    0.2 [ 0.01831564] 0.0183156388887
    0.30000000000000004 [ 0.00247875] 0.00247875217667
    0.4 [ 0.00033546] 0.000335462627903
    0.5 [  4.53999512e-05] 4.53999297625e-05
    0.6 [  6.14421567e-06] 6.14421235333e-06
    0.7 [  8.31529279e-07] 8.31528719104e-07
    0.7999999999999999 [  1.12535271e-07] 1.12535174719e-07
    0.8999999999999999 [  1.52300000e-08] 1.52299797447e-08
    0.9999999999999999 [  2.06116213e-09] 2.06115362244e-09
    1.0999999999999999 [  2.78957342e-10] 2.78946809287e-10


## Dynamics Vibrations Notes
**Jul 1, 2015**

## Introduction
<div id="sec:intro"></div>

modal analysis is similar to frequency analysis. In frequency analysis a complex signal is resolved into a set of simple sine waves with individual frequency and amplitude and phase parameters. In modal analysis, a complex deflection pattern of a vibrating structure is resolved into a set of simple mode shapes with the same individual parameters. 


## Structural Dynamics Background
<div id="sec:stdybg"></div>

Most systems are actually multiple degrees of freedom (MDOF) and have some non-linearity, but can be simplified with a superposition of SDOF linear systems 

Newtons law states that acceleration is a function of the applied force and the mass of the object, or
$$
[inertial forces] + [Dissipative forces] + [Restoring Forces] = [External Forces] \\
m\ddot{x} + c\dot{x} + kx = f(t)  \\
\zeta<1 is\ underdamped  \\
$$

some other dynamic characteristics are
$$
\omega = frequency  \\
\zeta = damping     \\
\{\phi\} = mode shape  \\
\omega^{2}_{n}=\frac{k}{m} = natural frequency \\
\zeta = \frac{c}{\sqrt{2km}}    \\
H(\omega)=Frequency\ Response \\
\phi(\omega)=Phase
$$

## Damping Model

Where there is energy dissipation, there is damping. The system can be broken into the system inputs/excitation, a system G(s), and the output response, in Laplace or space

The transfer function is a math model defining the input/output relationship of a physical system. Another definition is the Laplace transform ( $\mathcal{L}$) of the output divided by the Laplace transform of the input. 

The frequency response function (FRF) is defined in a similar manner such that FRF is the fourier transform ($ \mathcal{F} $) of the input divided by the fourier transform of the output

$$
Transfer\ Function=\frac{Output}{Input} \\
G(s) = \frac{Y(s)}{X(s)}
$$

## Modal Testing

These relationships can be further explained by the modal test process. The measurements taken during a test are frequency response function measurements. The parameter estimation routines are curve fits in the Laplace domain and result in transfer functions.

Frequency Response Matrix

$$
\begin{bmatrix}
 H_{11} & H_{12} & \cdots & H_{1n} \\
 H_{21} & H_{22} & \cdots & H_{2n} \\
 \vdots  & \vdots  & \ddots & \vdots  \\
 H_{n1} & H_{n2} & \cdots & H_{nn} 
\end{bmatrix}
$$

## Random Notes
<div id="sec:rand"></div>

The signal-analysis approach is done by measuring vibration with accelerometers and determine the frequency spectrum. The other moethod is a system -analysis where a dual-channel FFT anlayzer is used to measure the ratio of the response to the input giving the frequency response function (FRF)

a modal model allows the analysis of structural systems

a mode shape is a deflection-pattern associated with a particular modal frequency or pole location. It is not tangible or easily observed. The actual displacement of the structure will be a sum of all the mode shapes. A harmonic exitation close to the modal frequency, 95% of the displacement may be due to the particular modeshape

Modal Descriptions Assumes Linearity
 * Superposition of the component waves will result in the final wave. A swept sinosoid will give the same result as a broadband excitation

 * Homogeneity is when a measured FRF is independent of excitation level

 * Reciprocity implies that the FRF measured between any two DOFs is independent of which of them for excitation or response

 * small deflections - cannot predict buckling or catastrophic failure

 * casual - the structure will not vibrate before it is excited

 * stable - the vibrations will die out when the excitation is removd

 * time-invariant - the dynamic characteristics will not change during the measurments

## The Lumped-Parameter Model and Modal Theory

[Physical Coordinates] = [Modal Matrix][Modal Coordinates]

$$
[x] = [\phi][q]
$$

## Keywords and Notations

$$
m=mass        \\
k=stiffness   \\
c = damping coefficient  \\
c_c = critical damping coefficient  \\
$$



## Finite-Element-Method
[index](#Mechpy) 

The element connectivty is used to assemble the global stiffness matrix, the nodal force matrix, and the displacement matrix

The minimization of the potentail energy is used to solve the global equation once the boundary conditions are applied to prevent rigid body motion

$ \{F\} = [K]\{U\} $

where  

$ \{F\}=nodal\ force\ matrix $  
$ [K] = global\ stiffness\ matrix $  
$ \{U\} = nodal\ displacement\ matrix $  

Once the displacements, U are computed, the strain, $\bar{\varepsilon}$ is calcualted 

with $\{\varepsilon\}=[B]\{U\}$

where

$[B]=strain-displacement\ matrix$

and stresses, $\bar{\sigma}$ are determined via Hookes Law and 

$\{\sigma\}=[C]\{\varepsilon\}$

where  

$[C] = compliance\ matrix$


### FEM-Example-1


```python
from mechpy.fem import cst_fem
cst_fem(structure='9node')
```

    plane stress condition



![png](output_83_1.png)



![png](output_83_2.png)



    <matplotlib.figure.Figure at 0xee6cf98>



![png](output_83_4.png)


## Curve-Fitting 
[index](#Mechpy)  



```python
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
#==============================================================================
# Method 1 - polyfit
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = [1,2,3,4]
y = [3,5,7,10] # 10, not 9, so the fit isn't perfect
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y
plt.text(4,4,fit_fn)
plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.xlim(0, 5)
plt.ylim(0, 12)
```




    (0, 12)




![png](output_86_1.png)



```python
import matplotlib.pyplot as plt

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
z = np.polyfit(x, y, 3)

p = np.poly1d(z)
print(p)

p6 = np.poly1d(np.polyfit(x, y, 6))
print(' ------------------------------------------ ')
print(p6)

xp = np.linspace(-2, 6, 100)
plt.plot(x, y, '.', xp, p(xp), '-', xp, p6(xp), '--')
plt.ylim(-2,2)

plt.show()
```

             3          2
    0.08704 x - 0.8135 x + 1.693 x - 0.03968
     ------------------------------------------ 
               6           5           4          3          2
    -0.001269 x + 0.01071 x + 0.01711 x - 0.2894 x + 0.2772 x + 0.7857 x


    C:\Users\nealio\Anaconda3\lib\site-packages\numpy\lib\polynomial.py:594: RankWarning: Polyfit may be poorly conditioned
      warnings.warn(msg, RankWarning)



![png](output_87_2.png)



```python
#==============================================================================
# Hmk from numerical methods class
#==============================================================================
X = np.array([0,  1, 2, 3,  4,  5])
Y = np.array([4, -1, 6, 1, -4, -9])
order=3 # integer > 0
C = np.polyfit(X,Y,order);
C = np.flipud(C)

h = 1000
xfit = np.linspace(min(X),max(X),h);
yfit = np.zeros(len(xfit))

for i,c in enumerate(C):
    yfit += c*xfit**i

plt.plot(X,Y,'o',xfit,yfit)
plt.title('third order polynomnial least sqaures fit')
plt.show()
```


![png](output_88_0.png)



```python
#==============================================================================
# non-linear least squares
#==============================================================================

from scipy.optimize import curve_fit
X = np.array([0 ,   1 , 2 ,   3 ,   4 ,   5 ])
Y = np.array([0.1,  1 , 1.5 , 0.8 , 0.3 , 0.25 ])
fn = lambda a: (a[0]*X+a[1]) * np.exp(a[2]*X+a[3])
Sn = lambda x: np.sum( (fn(x)-Y)**2 )
A = sp.optimize.fmin(func=Sn, x0=[0,0,0,0])
plt.plot(X, Y,'o')
xplot = np.linspace(0,5,100)
fnfit = lambda x,a: (a[0]*x+a[1]) * np.exp(a[2]*x+a[3])
plt.plot(xplot, fnfit(xplot, A))
```

    Warning: Maximum number of function evaluations has been exceeded.





    [<matplotlib.lines.Line2D at 0x120f5ad8d68>]




![png](output_89_2.png)



```python
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

plt.figure(figsize=(16,7))
mpl.rcParams['font.size'] = 16

X = np.array([1, 2,    3,   4 ,    5 ])
Y = np.array([2, 3.25, 3.5, 3.75 , 3.8])

# a[0] is the x asymptote and a[1] is the y asymptote and 
fn = lambda a: -1/(X-a[0]) + a[1]
Sn = lambda x: np.sum( (fn(x)-Y)**2 )

A = scipy.optimize.fmin(func=Sn, x0=[0,0])
print(A)
plt.plot(X, Y,'o')
xplot = np.linspace(0.75 , 5,100)
fnfit = lambda x,a: -1/(x-a[0]) + a[1]

plt.subplot(1,2,1)
eps = 1e-16
xp = 0.5
yp = 4
x = np.linspace(xp,4,100)
y = -1/(x-xp+eps)+yp
plt.plot(x,y)
plt.ylim([0,yp])
plt.title('plot of how the function should look with actual parameters')


plt.plot(xplot, fnfit(xplot, A))
plt.plot(X,Y, 'o')
plt.title(r'$f(x)=\frac{-1}{x-a_0}+a_1$')
#plt.text(2,1,r'$f(x)=\frac{-1}{x-0.494}+3.969$')
```

    Optimization terminated successfully.
             Current function value: 0.015158
             Iterations: 109
             Function evaluations: 208
    [ 0.49404038  3.96935945]





    <matplotlib.text.Text at 0x120f5f92860>




![png](output_90_2.png)



```python
#==============================================================================
# 
#==============================================================================
import numpy as np
from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + np.random.normal(size=len(xdata))
popt, pcov = curve_fit(func, xdata, ydata)
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 2., 1.]))

plt.plot(xdata, y)

yfit = func(xdata, popt[0], popt[1], popt[2])
plt.plot(xdata, yfit, 'o')
```




    [<matplotlib.lines.Line2D at 0x120f7619dd8>]




![png](output_91_1.png)


## Linear Algebra with Python
[index](#Mechpy)  

Python's numpy package allows python, a generic computing language to perform powerful mathematical calculations. Although python's math syntax is not as obvious as MATLAB's, the functionality is comparable. This document is designed to be an intro to that syntax 

Some references

http://nbviewer.ipython.org/github/carljv/cython_testing/blob/master/cython_linalg.ipynb

We can either use scipy, which includes numpy, 
http://docs.scipy.org/doc/

or use numpy directly
http://docs.scipy.org/doc/numpy/

Since there are many ways to solve linear algebra problems, (eg Octave/Matlab, julia, scipy, numpy)
I tend to prefer the most matlabesc approaches due to the ubiquity of Matlab and the simplicity of the syntax, which frankly, python suffers with.

The major difference between arrays and matrices in python is that arrays are n-dimensions, where matrices are only up to 2-dimensions  
m


```python
import numpy as np
from scipy import linalg
```

Pythons list is a generic data storage object. it can be easily extended to a numpy array, which is specialized for numerical and scientific computation 


```python
np.zeros((5,3))
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])




```python
np.array([[1,2],[3,4]])
```




    array([[1, 2],
           [3, 4]])




```python
np.matrix(np.zeros((5,3)))
```




    matrix([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])




```python
np.matrix([[1,2],[3,4]])
```




    matrix([[1, 2],
            [3, 4]])




```python
# Matrix multiplication can be achieved using the dot method
i = [[1,0,0],[0,1,0],[0,0,1]]  # identiy matrix
a = [[4,3,1],[5,7,2],[2,2,2]]
np.dot(i,a)

```




    array([[4, 3, 1],
           [5, 7, 2],
           [2, 2, 2]])




```python
#Or, matrix multiplication can be done if a matrix is explicitly defined
np.matrix(i)*np.matrix(a)
```




    matrix([[4, 3, 1],
            [5, 7, 2],
            [2, 2, 2]])




```python
# Notice, when arrays are mutliplied, we get the dot product 
np.array(i)*np.array(a)
```




    array([[4, 0, 0],
           [0, 7, 0],
           [0, 0, 2]])




```python
# convert an array to a matrix
m = np.matrix(a)
m
```




    matrix([[4, 3, 1],
            [5, 7, 2],
            [2, 2, 2]])




```python
m.T  # transpose
```




    matrix([[4, 5, 2],
            [3, 7, 2],
            [1, 2, 2]])




```python
m.I  # inverse
```




    matrix([[ 0.55555556, -0.22222222, -0.05555556],
            [-0.33333333,  0.33333333, -0.16666667],
            [-0.22222222, -0.11111111,  0.72222222]])




```python
m**2
```




    matrix([[33, 35, 12],
            [59, 68, 23],
            [22, 24, 10]])




```python
np.array(a)**2
```




    array([[16,  9,  1],
           [25, 49,  4],
           [ 4,  4,  4]])




```python
m
```




    matrix([[4, 3, 1],
            [5, 7, 2],
            [2, 2, 2]])




```python
m[:,2]
```




    matrix([[1],
            [2],
            [2]])




```python
m[2,:]
```




    matrix([[2, 2, 2]])




```python
m[:2,:2]
```




    matrix([[4, 3],
            [5, 7]])




```python
m[1:,1:]
```




    matrix([[7, 2],
            [2, 2]])



## Sympy Linear Algebra


```python
# import sympy
import sympy as sp
#from sympy.mpmath import *
```


```python
x = sp.Symbol('x')   # x = var('x')
M = sp.Matrix([[2,x],[x,3]])
M
```




$$\left[\begin{matrix}2 & x\\x & 3\end{matrix}\right]$$




```python
M.eigenvals()
```




$$\left \{ - \frac{1}{2} \sqrt{4 x^{2} + 1} + \frac{5}{2} : 1, \quad \frac{1}{2} \sqrt{4 x^{2} + 1} + \frac{5}{2} : 1\right \}$$




```python
M.eigenvects()
```




$$\left [ \left ( - \frac{1}{2} \sqrt{4 x^{2} + 1} + \frac{5}{2}, \quad 1, \quad \left [ \left[\begin{matrix}- \frac{x}{\frac{1}{2} \sqrt{4 x^{2} + 1} - \frac{1}{2}}\\1\end{matrix}\right]\right ]\right ), \quad \left ( \frac{1}{2} \sqrt{4 x^{2} + 1} + \frac{5}{2}, \quad 1, \quad \left [ \left[\begin{matrix}- \frac{x}{- \frac{1}{2} \sqrt{4 x^{2} + 1} - \frac{1}{2}}\\1\end{matrix}\right]\right ]\right )\right ]$$




```python
M.eigenvects()[1][0]
```




$$\frac{1}{2} \sqrt{4 x^{2} + 1} + \frac{5}{2}$$




```python
Mval = M.eigenvects()[1][0]
Mval.evalf(subs={x:3.14})
```




$$5.67955971794838$$




```python
print(sp.latex(M))
```

    \left[\begin{matrix}2 & x\\x & 3\end{matrix}\right]


copy and paste into markdown 

$ \left[\begin{matrix}2 & x\\x & 3\end{matrix}\right] $


## Signal Processing
Page 174 Introduction for python for Science - David Pine


```python
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
width = 2.0
freq = 0.5
t = np.linspace(-10, 10, 101) # linearly space time array
g = np.exp(-np.abs(t)/width)*np.sin(2.0 * np.pi * freq * t)
dt = t[1]-t[0] # increment between times in time array
G = fftpack.fft(g) # FFT of g
f = fftpack.fftfreq(g.size, d=dt) # frequenies f[i] of g[i]
f = fftpack.fftshift(f) # shift frequencies from min to max
G = fftpack.fftshift(G) # shift G order to coorespond to f
fig = plt.figure(1, figsize=(8,6), frameon=False)
ax1 = fig.add_subplot(211)
ax1.plot(t, g)
ax1.set_xlabel('t')
ax1.set_ylabel('g(t)')
ax2 = fig.add_subplot(212)
ax2.plot(f, np.real(G), color='dodgerblue', label='real part')
ax2.plot(f, np.imag(G), color='coral', label='imaginary part')
ax2.legend()
ax2.set_xlabel('f')
ax2.set_ylabel('G(f)')
plt.show()
```

# Design

## Factors of Safety

DLL, Design Limit Load = max force or moment expected during a mission with a given statistical probability  

Al, Allowable =  allowed minimum applied load or strength of a structure at a given statistical probablity  

FS, factor of safety [1, $\infty$] = a factor applied to a DLL to decrease the chance of failure, typically around 1-3  

KD, knockdown (0,1] = a percentage reduction of Allowable load to reduce the chance of failure

A KD=0.8 would be applied to the allowable to reduce it by 20%, $Al_{new}=Al_{old}*KD$   

MS, margin of safety = a measure of reserve strength , how much applied loda can increase before the safety of the vehicle is comprimised. $ MS\geq0$ for a good design, $MS=\frac{Allowable}{DLL*FS}-1$

For example with a $FS=1.15$, $DLL=80$, $Al=100$, we have a margin of $MS=\frac{100}{80*1.15}-1=\frac{100}{92}-1=0.087$ which is passing our design checks based on the expected max load of 80

Lets Assume a knockdown of 27%, so $K=1-0.27=0.73$  

$$
FS = \frac{1}{K}
$$


We can also say we have a $FS = \frac{1}{0.73}=1.3699$

$$
\sigma_{design}=\frac{\sigma_{ult}}{FS} = \sigma_{ult}*K
$$



```python

```

# Engineering Mathematics with Python
[index](#Mechpy)


```python
from numpy import *
```


```python
r_[1:11]
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
arange(1,11)
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
linspace(1,10,10)
```




    array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])



## Functions
[index](#Index) 


```python
import sympy as sp
s,ed = sp.symbols('s,ed')
K = sp.Function('K')

Ked = 0.4*ed+.2
Ks = 0.105*s+0.63 

Ktot = Ked*Ks
sp.expand(Ktot)
```




$$0.042 ed s + 0.252 ed + 0.021 s + 0.126$$




```python
Ktot = sp.lambdify((ed,s), (0.4*ed+.2)*(0.105*s+0.63))
K(2,3.54)
```




$$K{\left (2,3.54 \right )}$$




```python
di,df,t = sp.symbols('d_i,d_f,t')
```


```python
from sympy.utilities.lambdify import lambdify, implemented_function
```


```python
sb = implemented_function(sp.Function('sb'), lambda di,t: 11445*(di/t)**3 -70429*(di/t)**2 + 145552*(di/t)  )
```


```python
Kdt = implemented_function(sp.Function('Kdt'), \
                           lambda di,t,df: 11445/sb(di, t)*(df/t)**3 - \
                           70429/sb(di, t)*(df/t)**2 + 145552/sb(di, t)*(df/t)  )
```


```python
Kdt(0.1875, 0.25, 0.246)
```




$$1.15540221170541$$




```python
sb = sp.Function('sb')
sb = 11445*(di/t)**3 -70429*(di/t)**2 + 145552*(di/t)
sb
```




$$\frac{11445 d_{i}^{3}}{t^{3}} - \frac{70429 d_{i}^{2}}{t^{2}} + \frac{145552 d_{i}}{t}$$




```python
Kdt = sp.Function('Kdt')
Kdt = 11445/sb*(df/t)**3 - 70429/sb*(df/t)**2 + 145552/sb*(df/t)
Kdt
```




$$\frac{11445 d_{f}^{3}}{t^{3} \left(\frac{11445 d_{i}^{3}}{t^{3}} - \frac{70429 d_{i}^{2}}{t^{2}} + \frac{145552 d_{i}}{t}\right)} - \frac{70429 d_{f}^{2}}{t^{2} \left(\frac{11445 d_{i}^{3}}{t^{3}} - \frac{70429 d_{i}^{2}}{t^{2}} + \frac{145552 d_{i}}{t}\right)} + \frac{145552 d_{f}}{t \left(\frac{11445 d_{i}^{3}}{t^{3}} - \frac{70429 d_{i}^{2}}{t^{2}} + \frac{145552 d_{i}}{t}\right)}$$




```python
Kdt = sp.simplify(Kdt)
```


```python
sp.latex(Kdt)
```




    '\\frac{d_{f} \\left(11445 d_{f}^{2} - 70429 d_{f} t + 145552 t^{2}\\right)}{d_{i} \\left(11445 d_{i}^{2} - 70429 d_{i} t + 145552 t^{2}\\right)}'




```python

```


```python

```


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
fig, ax = plt.subplots()

N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

Pult = (20, 35, 30, 35, 27)
Pultstd = (2, 3, 4, 1, 2)
rects1 = ax.bar(ind, Pult, width, color='r', yerr=Pultstd)

Ppl = (25, 32, 34, 20, 25)
Pplstd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, Ppl, width, color='y', yerr=Pplstd)

# add some text for labels, title and axes ticks
ax.set_ylabel('P, load, lbs')
ax.set_title('Ultimate and PL Load')
ax.set_xticks(ind + width/2)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax.legend(['Ult','Pl'])
ax.margins(0.02)


### OR

fig, ax = plt.subplots()
barwidth = 0.35       # the width of the bars
specimenNames = ['Sample1', 'Sample2', 'Sample3', 'Samlpe4']
x = np.arange(4)
y = np.random.random_integers(1, 10 ,len(x))
e = np.random.random_integers(0,1,len(x))
ax.bar(x,y, width=barwidth, yerr=e)
ax.set_xticks(x)
plt.xticks(x+barwidth/2, specimenNames, rotation='vertical')
plt.margins(0.05)
#ax.margins(0.05)
plt.show()
```

    C:\Users\ngordon\AppData\Local\Continuum\Anaconda3\lib\site-packages\ipykernel\__main__.py:33: DeprecationWarning: This function is deprecated. Please call randint(1, 10 + 1) instead
    C:\Users\ngordon\AppData\Local\Continuum\Anaconda3\lib\site-packages\ipykernel\__main__.py:34: DeprecationWarning: This function is deprecated. Please call randint(0, 1 + 1) instead



![png](output_145_1.png)



![png](output_145_2.png)


## Raw Test Data


```python
import pandas as pd
import numpy as np
import glob as gb
from matplotlib.pyplot import *
%matplotlib inline

csvdir='./examples/'
```


```python
e=[]
y=[]

for s in specimen:
    files = gb.glob(csvdir + '*.csv')  # select all csv files
    fig, ax = subplots()
    title(s)
    Pult = []
    
    for f in files:
        d1 = pd.read_csv(f, skiprows=1)
        d1 = d1[1:]  # remove first row of string
        d1.columns = ['t', 'load', 'ext']  # rename columns
        d1.head()
        # remove commas in data
        for d in d1.columns:
            #d1.dtypes
            d1[d] = d1[d].map(lambda x: float(str(x).replace(',','')))
        Pult.append(np.max(d1.load))
        plot(d1.ext, d1.load)   
        ylabel('Pult, lbs')
        xlabel('extension, in')
        
        
    e.append(np.std(Pult))
    y.append(np.average(Pult)     )
    show()


# bar chart 
barwidth = 0.35       # the width of the bars

fig, ax = subplots()
x = np.arange(len(specimen))
ax.bar(x,  y, width=barwidth, yerr=e)

#ax.set_xticks(x)
xticks(x+barwidth/2, specimen, rotation='vertical')
title('Pult with sample average and stdev of n=3')
ylabel('Pult, lbs')
margins(0.05)
show()
```


![png](output_148_0.png)



![png](output_148_1.png)


## Finding the "first" peak and delta-10 threshhold limit on force-displacement data

 http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb


```python

%matplotlib inline
from scipy import signal
from pylab import plot, xlabel, ylabel, title, rcParams, figure
import numpy as np
pltwidth = 16
pltheight = 8
rcParams['figure.figsize'] = (pltwidth, pltheight)

csv = np.genfromtxt('./examples/stress_strain1.csv', delimiter=",")
disp = csv[:,0]
force = csv[:,1]
print('number of data points = %i' % len(disp))

def moving_average(x, window):
    """Moving average of 'x' with window size 'window'."""
    y = np.empty(len(x)-window+1)
    for i in range(len(y)):
        y[i] = np.sum(x[i:i+window])/window
    return y

plt1 = plot(disp, force);
xlabel('displacement');
ylabel('force');

```

    number of data points = 42124



![png](output_150_1.png)



```python
figure()
mywindow = 1000  # the larger the filter window, the more agressive the filtering
force2 = moving_average(force, mywindow)
x2 = range(len(force2))
plot(x2,  force2);
title('Force smoothed with moving average filter');
```


![png](output_151_0.png)



```python

# Find f' using diff to find the first intersection of the 0

# mvavgforce = mvavgforce[:len(mvavgforce)/2]
force2p = np.diff(force2)
x2p = range(len(force2p))
plot(x2p, force2p);
title('Slope of the smoothed curve')
```




    <matplotlib.text.Text at 0xe81dc88>




![png](output_152_1.png)



```python
i = np.argmax(force2p<0)     
### or
# i = where(force2p<0)[0][0]
#### or
# for i, f in enumerate(force2p):
#     if f < 0:
#         break
```


```python
plot(x2p,  force2p, i,force2p[i],'o', markersize=15);
title('find the point at which the slope goes negative, indicating a switch in the slope direction');
```


![png](output_154_0.png)



```python
plot(x2,  force2, i,force2[i],'o',markersize=15);
title('using that index, plot on the force-displacement curve');
```


![png](output_155_0.png)



```python
#Now, we need to find the next point from here that is 10 less.
delta = 1

i2 = np.argmax(force2[i]-delta > force2[i:])

# If that point does not exist on the immediate downward sloping path, 
#then just choose the max point. In this case, 10 would exist very 
#far away from the point and not be desireable
if i2 > i:
    i2=0
plot(x2,  force2, i,force2[i],'o', i2+i, force2[i2+i] ,'*', markersize=15);
```


![png](output_156_0.png)



```python

```


```python

```


```python

```


```python

```


```python

```


```python
### Copy and past data into python

y = '''62606.53409
59989.34659
62848.01136
80912.28693
79218.03977
81242.1875
59387.27273
59795.68182
58303.18182
81184.09091
71876.81818
72904.77273
115563.9205
113099.7869
106939.2756
58758.11688
57349.02597
52614.77273
72899.75649
68424.51299
73514.28571
75549.83766
84867.69481
79881.41234
55882.71104
54156.54221
58260.71429
73027.5974
69470.69805
66843.99351
82758.44156
81647.72727
77519.96753'''
y = [float(x) for x in np.array(y.replace('\n',',').split(','))]
print(y, end=" ")
```

    [62606.53409, 59989.34659, 62848.01136, 80912.28693, 79218.03977, 81242.1875, 59387.27273, 59795.68182, 58303.18182, 81184.09091, 71876.81818, 72904.77273, 115563.9205, 113099.7869, 106939.2756, 58758.11688, 57349.02597, 52614.77273, 72899.75649, 68424.51299, 73514.28571, 75549.83766, 84867.69481, 79881.41234, 55882.71104, 54156.54221, 58260.71429, 73027.5974, 69470.69805, 66843.99351, 82758.44156, 81647.72727, 77519.96753] 

# Units
[index](#Mechpy)


```python
from mechunits import uc1
# uc1 uses sympy
```


```python
uc1(1.0,'psi','kPa')
```

    1.00 psi = 6.89  kPa 



```python
uc1(1.0,'newton','pound')
```

    1.00 newton = 0.22  pound 



```python
from mechunits import uc2
# uses pint
uc2(17.5,'lbf','newton')
```

    17.5 lbf = 77.84387826705874 newton





77.84387826705874 newton




```python
uc2(300,'pascal','psi')
```

    300 pascal = 0.043511321319062775 psi





0.043511321319062775 psi




```python
from mechunits import in_mm
in_mm()
```

         0 in - 0.000000 in - 0.000000 mm 
      1/16 in - 0.062500 in - 1.587500 mm 
       1/8 in - 0.125000 in - 3.175000 mm 
      3/16 in - 0.187500 in - 4.762500 mm 
       1/4 in - 0.250000 in - 6.350000 mm 
      5/16 in - 0.312500 in - 7.937500 mm 
       3/8 in - 0.375000 in - 9.525000 mm 
      7/16 in - 0.437500 in - 11.112500 mm 
       1/2 in - 0.500000 in - 12.700000 mm 
      9/16 in - 0.562500 in - 14.287500 mm 
       5/8 in - 0.625000 in - 15.875000 mm 
     11/16 in - 0.687500 in - 17.462500 mm 
       3/4 in - 0.750000 in - 19.050000 mm 
     13/16 in - 0.812500 in - 20.637500 mm 
       7/8 in - 0.875000 in - 22.225000 mm 
     15/16 in - 0.937500 in - 23.812500 mm 
         1 in - 1.000000 in - 25.400000 mm 


# Engineering-Software-APIs  
[index](#Mechpy)  

## CATIA
[index](#Mechpy)  


```python
from mechpy import catia
help(catia)
```

    Help on module mechpy.catia in mechpy:
    
    NAME
        mechpy.catia
    
    DESCRIPTION
        CATIA Python Module that connects to a Windows CATIA session through COM
        Assumes that CATIA is running.
    
    CLASSES
        builtins.object
            Catia
        
        class Catia(builtins.object)
         |  object to manipulate CATIA through windows COM
         |  
         |  Methods defined here:
         |  
         |  __init__(self)
         |      Initialize self.  See help(type(self)) for accurate signature.
         |  
         |  add_drawing(self)
         |      creates a new drawing
         |  
         |  add_geoset(self)
         |      adds a flat geoset set give as a list
         |  
         |  add_part(self)
         |      adds part to current catia object
         |  
         |  add_point(self, x=0, y=0, z=0)
         |      adds a point to the current part
         |  
         |  add_points(self)
         |      adds a bunch of random points
         |  
         |  add_product(self)
         |      adds part to current catia object
         |  
         |  change_parameter(self, strMatch='', inVal=0)
         |      enter a value for strMatch and this will search through all the 
         |      parameters and try to match them. If more than one match is found,
         |      nothing will happen, if exactly one match is found the value will 
         |      update with the
         |      Example - this is find the shaft and change the angle to 180
         |      change_parameter(strMatch = 'Shaft.1\FirstAngle', inVal = 180)
         |  
         |  connect_doc(self)
         |      connects to current active part
         |  
         |  create_bom(self)
         |      ' creates BOM for the current catia object
         |  
         |  custom_cmd(self, cmdin)
         |      runs the command when you mouse over the icon in CATIA
         |      Open, Save, Fit All In, * iso
         |  
         |  export_dxf(self, fix1=True)
         |      function to export a CATDrawing to dxf
         |      TODO - need to figure out how to export CATDrawing to dxf
         |  
         |  fit_window(self)
         |  
         |  get_plies(self, stackName='Stacking', indent=' ')
         |      returns all plies on the Stacking geoset
         |  
         |  get_ply_dir(self)
         |  
         |  get_username(self)
         |  
         |  launch_catia(self)
         |      launches a quiet catia process
         |  
         |  make_geosets(self)
         |  
         |  new_from(self, filepath='cube.CATPart')
         |      creating a new part from an existing document
         |  
         |  open_dialog(self)
         |  
         |  open_part(self, filepath='cube.CATPart')
         |      opens an existing CATIA part given an absolute filepath
         |  
         |  plybook(self, stackName='Stacking', pliesgroup='Plies Group')
         |      creates flat patterns in a catdrawing
         |      just have the CATPart composite part open
         |      given a geoset stackName, will generate a sheet for each ply
         |  
         |  print_parameters(self)
         |      shows all parameters in catia
         |  
         |  quit_catia(self)
         |      terminates the current catia object
         |  
         |  read_from(self, filepath='cube.CATPart')
         |      loading a CATIA document, faster than opening but does not show it
         |  
         |  save_current(self)
         |      saves the current file
         |  
         |  save_current_as(self, file2='new1')
         |      saves the current file as
         |  
         |  save_dialog(self)
         |  
         |  search(self, s="Name='PLY-01',all")
         |      search strings
         |      s=".Point.Name=Point.1*;All"
         |  
         |  show_body_tree(self, indent=' ')
         |      returns a list of all the bodies in the part
         |      Bodies cannot be nested like geosets
         |      part1 = self.CATIA.ActiveDocument
         |      c.CATIA.ActiveDocument.Part
         |      c.CATIA.ActiveDocument.Part
         |      c.show_body_tree(c.CATIA.ActiveDocument.Part)
         |  
         |  show_docs(self)
         |      a list of all documents or files can be generated with this command
         |  
         |  show_geoset_tree(self, indent=' ')
         |      recursuvely returns a list of all the geosets in the part
         |      c.show_geoset_tree(c.CATIA.ActiveDocument.Part)
         |      part1 = c.CATIA.ActiveDocument.Part
         |  
         |  show_product_info(self)
         |  
         |  show_selection(self)
         |      prints current selection in CATIA
         |  
         |  show_windows(self)
         |      only one catia application can be running at a time
         |      all catia windows can be found using the Windows object
         |  
         |  toggle_show(self)
         |      toggles the visiblity current CATIA session
         |  
         |  update(self)
         |      updates catia object
         |  
         |  user_selection(self)
         |  
         |  ----------------------------------------------------------------------
         |  Data descriptors defined here:
         |  
         |  __dict__
         |      dictionary for instance variables (if defined)
         |  
         |  __weakref__
         |      list of weak references to the object (if defined)
    
    FUNCTIONS
        bolt_test()
            make a bolt
        
        cube_test()
            creates a cube
            https://gist.github.com/jl2/2704426
    
    FILE
        f:\mechpy\mechpy\catia.py
    
    


## Abaqus 
[index](#Mechpy)  



# Engineering Python   
[index](#Mechpy)  

to plot inline in the ipython console or jupyter notebook, otherwise plots will be in the a seperate window

to turn on inline plotting
```python
from IPython import get_ipython
get_ipython().magic('matplotlib inline')
#or
%matplotlib inline
```

to turn off inline plotting
```python
get_ipython().magic('matplotlib')
#or
%matplotlib
#or 
%matplotlib qt
```


## Symbolic mathematics with sympy


```python
# import sympy library and initialize latex printing
import sympy as sp
#sp.init_printing()
#sp.init_printing(use_latex='matplotlib')
sp.init_printing(use_latex='mathjax')
```


```python
# add a symbolic character
x = sp.Symbol('x')
```


```python
sp.sqrt(x**2)
```




$$\sqrt{x^{2}}$$




```python
r = sp.Rational(11, 13)
r
```




$$\frac{11}{13}$$




```python
float(r)
```




$$0.8461538461538461$$




```python
f = sp.Function('f')
f
```




    f




```python
f(x)
```




$$f{\left (x \right )}$$




```python
h = sp.Lambda(x,x**2)
h
```




$$\left( x \mapsto x^{2} \right)$$




```python
w = 2*(x**2-x)-x*(x+1)
w
```




$$2 x^{2} - x \left(x + 1\right) - 2 x$$




```python
w.args
```




$$\left ( - 2 x, \quad 2 x^{2}, \quad - x \left(x + 1\right)\right )$$




```python
sp.simplify(w)
```




$$x \left(x - 3\right)$$




```python
sp.factor(x**2-1)
```




$$\left(x - 1\right) \left(x + 1\right)$$




```python
#partial fractions
y = 1/(x**2+3*x+2)
y
```




$$\frac{1}{x^{2} + 3 x + 2}$$




```python
sp.apart(y,x)
```




$$- \frac{1}{x + 2} + \frac{1}{x + 1}$$




```python
f = sp.Function('f')(x)
sp.diff(f,x)
```




$$\frac{d}{d x} f{\left (x \right )}$$




```python
y = sp.Symbol('y')
g = sp.Function('g')(x,y)
g.diff(x,y)
```




$$\frac{\partial^{2}}{\partial x\partial y}  g{\left (x,y \right )}$$




```python
a,b,c,d = sp.symbols("a b c d")
M = sp.Matrix([[a,b],[c,d]])
M
```




$$\left[\begin{matrix}a & b\\c & d\end{matrix}\right]$$




```python
M*M
```




$$\left[\begin{matrix}a^{2} + b c & a b + b d\\a c + c d & b c + d^{2}\end{matrix}\right]$$




```python
# if ipython is to be used as a calculator initialize with 
from sympy import init_session
init_session() 
```

    IPython console for SymPy 1.0 (Python 3.5.1-64-bit) (ground types: python)
    
    These commands were executed:
    >>> from __future__ import division
    >>> from sympy import *
    >>> x, y, z, t = symbols('x y z t')
    >>> k, m, n = symbols('k m n', integer=True)
    >>> f, g, h = symbols('f g h', cls=Function)
    >>> init_printing()
    
    Documentation can be found at http://docs.sympy.org/1.0/



```python
from sympy import oo, Function, dsolve, Eq, Derivative, sin,cos,symbols
from sympy.abc import x
import sympy as sp
import numpy as np
import matplotlib.pyplot as mp
get_ipython().magic('matplotlib inline')
# this will print output as unicode
```


```python
# assign a sympy variable
x = sp.var('x')
x
```




$$x$$




```python
#assign a function 
f =  sp.sin(6*x)*sp.exp(-x)
f
```




$$e^{- x} \sin{\left (6 x \right )}$$




```python
f.subs(x,3)
```




$$\frac{1}{e^{3}} \sin{\left (18 \right )}$$




```python
float(f.subs(x,3))
```




$$-0.037389453398415345$$




```python
sp.plot(f)
```


![png](output_200_0.png)





    <sympy.plotting.plot.Plot at 0x14364a8fc18>




```python
# a onetime pretty print
sp.pprint(f)
```

     -x         
    ℯ  ⋅sin(6⋅x)



```python
#or we can print the latex rendering
sp.latex(f)
```




    'e^{- x} \\sin{\\left (6 x \\right )}'




```python
# first derivative
df = f.diff()
df
```




$$- e^{- x} \sin{\left (6 x \right )} + 6 e^{- x} \cos{\left (6 x \right )}$$




```python
# differentaite f'' wrt x
sp.diff(f,x,1)
```




$$- e^{- x} \sin{\left (6 x \right )} + 6 e^{- x} \cos{\left (6 x \right )}$$




```python
# substitute x with pi
f.subs(x,np.pi)
```




$$-3.17530720082064 \cdot 10^{-17}$$




```python
#%% Numeric Computation from the documentation
from sympy.abc import x
```


```python
# lambdify using the math module, 10^2 faster than subs
expr = sp.sin(x)/x
f = sp.lambdify(x,expr)
f(3.14)
```




$$0.0005072143046136395$$




```python
# lambdify using numpy
expr = sp.sin(x)/x
f = sp.lambdify(x,expr, "numpy")
f(np.linspace(1,3.14,20))
```




    array([  8.41470985e-01,   8.06076119e-01,   7.67912588e-01,
             7.27262596e-01,   6.84424864e-01,   6.39711977e-01,
             5.93447624e-01,   5.45963742e-01,   4.97597617e-01,
             4.48688937e-01,   3.99576866e-01,   3.50597122e-01,
             3.02079129e-01,   2.54343238e-01,   2.07698064e-01,
             1.62437944e-01,   1.18840569e-01,   7.71647744e-02,
             3.76485431e-02,   5.07214305e-04])




```python
z = np.arange(0,6,.1)
z
```




    array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
            1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
            2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
            3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
            4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
            5.5,  5.6,  5.7,  5.8,  5.9])




```python
# create an array from 0 to 6 with 300 points
z = np.linspace(0,6,30)
z
```




    array([ 0.        ,  0.20689655,  0.4137931 ,  0.62068966,  0.82758621,
            1.03448276,  1.24137931,  1.44827586,  1.65517241,  1.86206897,
            2.06896552,  2.27586207,  2.48275862,  2.68965517,  2.89655172,
            3.10344828,  3.31034483,  3.51724138,  3.72413793,  3.93103448,
            4.13793103,  4.34482759,  4.55172414,  4.75862069,  4.96551724,
            5.17241379,  5.37931034,  5.5862069 ,  5.79310345,  6.        ])




```python
## ODE Sympy from documentation

# see here for example scripts
# C:\Users\Neal\Anaconda3\Lib\site-packages\sympy\mpmath\tests
from sympy import Function, dsolve, Eq, Derivative, sin,cos,symbols
from sympy.abc import x
import numpy as np
import sympy as sp
import matplotlib.pyplot as mp
```


```python
f = Function('f')
deq = dsolve(Derivative(f(x), x,x) + 9*f(x), f(x))
deq
```




$$f{\left (x \right )} = C_{1} \sin{\left (3 x \right )} + C_{2} \cos{\left (3 x \right )}$$




```python
diffeq1_sym = deq.args[1]
diffeq1_sym
```




$$C_{1} \sin{\left (3 x \right )} + C_{2} \cos{\left (3 x \right )}$$




```python
diffeq1 = diffeq1_sym.subs({'C1':1, 'C2':0.5})
diffeq1
```




$$\sin{\left (3 x \right )} + 0.5 \cos{\left (3 x \right )}$$




```python
diffeq1_f = sp.lambdify(x,diffeq1, "numpy")
diffeq1_f
```




    <function numpy.<lambda>>




```python
diffeq1arr = diffeq1_f(np.linspace(1,3.14,20))
diffeq1arr
```




    array([-0.35387624, -0.68544104, -0.93948885, -1.08728921, -1.11212728,
           -1.0111941 , -0.79590429, -0.49060511, -0.12982305,  0.24564078,
            0.59332492,  0.87390954,  1.05566313,  1.11803104,  1.05396004,
            0.87069598,  0.5889643 ,  0.24062625, -0.1349244 , -0.49521635])




```python
plt.plot(diffeq1_f(np.linspace(-10,10,2000)));
plt.title('plot of the numpy array');
```


![png](output_217_0.png)



```python
sp.plot(diffeq1, title='plot of the sympy function');
```


![png](output_218_0.png)



```python
eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)
dsolve(eq, hint='1st_exact')
```




$$\left [ f{\left (x \right )} = - \operatorname{acos}{\left (\frac{C_{1}}{\cos{\left (x \right )}} \right )} + 2 \pi, \quad f{\left (x \right )} = \operatorname{acos}{\left (\frac{C_{1}}{\cos{\left (x \right )}} \right )}\right ]$$




```python
#or
dsolve(eq, hint='almost_linear')
```




$$\left [ f{\left (x \right )} = - \operatorname{acos}{\left (\frac{C_{1}}{\sqrt{- \cos^{2}{\left (x \right )}}} \right )} + 2 \pi, \quad f{\left (x \right )} = \operatorname{acos}{\left (\frac{C_{1}}{\sqrt{- \cos^{2}{\left (x \right )}}} \right )}\right ]$$




```python
t = symbols('t')
x,y = symbols('x, y', function=True)
```


```python
eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
dsolve(eq)
```




$$\left [ x{\left (t \right )} = C_{1} x_{0} + C_{2} x_{0} \int \frac{8}{x_{0}^{2}} \left(e^{\int 7 t\, dt}\right) e^{\int 12 t\, dt}\, dt, \quad y{\left (t \right )} = C_{1} y_{0} + \operatorname{C_{2}}{\left (y_{0} \int \frac{8}{x_{0}^{2}} \left(e^{\int 7 t\, dt}\right) e^{\int 12 t\, dt}\, dt + \frac{1}{x_{0}} \left(e^{\int 7 t\, dt}\right) e^{\int 12 t\, dt} \right )}\right ]$$




```python
eq = (Eq(Derivative(x(t),t),x(t)*y(t)*sin(t)), Eq(Derivative(y(t),t),y(t)**2*sin(t)))
dsolve(eq)
```




$$\left\{x{\left (t \right )} = - \frac{e^{C_{1}}}{C_{2} e^{C_{1}} - \cos{\left (t \right )}}, y{\left (t \right )} = - \frac{1}{C_{1} - \cos{\left (t \right )}}\right\}$$




```python

```


```python
#pretty plots
from sympy import sin, symbols, dsolve, pprint, Function
from sympy.solvers.ode import odesimp
x,u2,C1 = symbols('x,u2,C1')
f = Function('f')
eq = dsolve(x*f(x).diff(x) - f(x) - x*sin(f(x)/x), f(x), hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral', simplify=False)
eq2 = odesimp(eq, f(x), 1, set([C1]), hint='1st_homogeneous_coeff_subs_indep_div_dep')
```


```python
eq
```




$$\log{\left (f{\left (x \right )} \right )} = \log{\left (C_{1} \right )} + \int^{\frac{x}{f{\left (x \right )}}} - \frac{1}{u_{2}^{2}} \left(u_{2} + \frac{1}{\sin{\left (\frac{1}{u_{2}} \right )}}\right)\, du_{2}$$




```python
eq2
```




$$f{\left (x \right )} = 2 x \operatorname{atan}{\left (C_{1} x \right )}$$




```python
f = Function('f')
eq = dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x), hint = '1st_homogeneous_coeff_best', simplify=False)
eq
```




$$\log{\left (f{\left (x \right )} \right )} = \log{\left (C_{1} \right )} - \frac{1}{3} \log{\left (\frac{3 x^{2}}{f^{2}{\left (x \right )}} + 1 \right )}$$




```python

```


```python
# Ordinary Differential Equations
plt.dps = 15
plt.pretty = True
# solves ODE y'(x) =y(x), y(0)=1
f = sympy.mpmath.odefun(lambda x,y:y,0,1)
for x in [0,1,2.5]:
    print(f(x), exp(x))
```

    1.0 1
    2.71828182845905 E
    12.1824939607035 12.1824939607035



```python

```


```python

```


```python
z = np.linspace(1,5,200)
fplot = np.array([f(x) for x in z])
fexactplot = np.array([fexact(x) for x in z])
plt.plot(z,fplot, z, fexactplot)
plt.plot(z,fexactplot-fplot)
```




    [<matplotlib.lines.Line2D at 0x14368125978>]




![png](output_233_1.png)



```python
f=lambda x: [x[0]**2 - 2*x[0] - x[1] + 0.5, x[0]**2 + 4*x[1]**2 - 4]
x,y=np.mgrid[-0.5:2.5:24j,-0.5:2.5:24j]
U,V=f([x,y])
plt.quiver(x,y,U,V,color='r', \
         linewidths=(0.2,), edgecolors=('k'), \
         headaxislength=5)
plt.show()
```


![png](output_234_0.png)



```python

```


```python
# one way to plot using symbolic math
x = sp.var('x')
f =  sp.sin(6*x)*sp.exp(-x)
df = f.diff()
w = np.array([f.subs(x,k) for k in z])
dw = np.array([df.subs(x,k) for k in z])

plt.plot(z,w,z,dw);
```


![png](output_236_0.png)



```python
#%% Sympy Plotting

# shows two plots together
from sympy import symbols
from sympy.plotting import plot
x = symbols('x')
p1 = plot(x*x)
p2 = plot(x)
p1.extend(p2)
p1.show()
```


![png](output_237_0.png)



![png](output_237_1.png)



![png](output_237_2.png)



```python
#single plot with range
plot(x**2, (x,-5,5));
```


![png](output_238_0.png)



```python
#multiple plots with range
plot(x,x**2,x**3, (x,-5,5))
```


![png](output_239_0.png)





    <sympy.plotting.plot.Plot at 0x1436833a7b8>




```python
#multiple plots with different ranges
plot( (x, (x,-5,5)), (x**2, (x,-2,2)), (x**3, (x,-3,3)), 
     title='testing multiple plots',
     xlabel='x-label',
     ylabel='ylabel')
```


![png](output_240_0.png)





    <sympy.plotting.plot.Plot at 0x14368379080>




```python
# parametric plots
from sympy import symbols, cos,sin
from sympy.plotting import plot_parametric
u = symbols('x')
plot_parametric(cos(u), sin(u), (u,-5,5))
```


![png](output_241_0.png)





    <sympy.plotting.plot.Plot at 0x143682e8ac8>




```python
#multiple parametric plots with a single range
plot_parametric((cos(u), sin(u)), (u,cos(u)))
```


![png](output_242_0.png)





    <sympy.plotting.plot.Plot at 0x143682e8160>




```python
# multiple parametric plots with multiple ranges
plot_parametric((cos(u), sin(u), (u,-5,5)), (cos(u), u, (u,-10,10)))
```


![png](output_243_0.png)





    <sympy.plotting.plot.Plot at 0x143681a5cf8>




```python
# 3d plots
from sympy import symbols
from sympy.plotting import plot3d
x,y = symbols('x,y')
plot3d(x*y, (x,-5,5), (y,-5,5))
```


![png](output_244_0.png)





    <sympy.plotting.plot.Plot at 0x1436838c5f8>




```python
# multiple plots with multiple ranges
plot3d((x**2+y**2, (x,-5,5), (y,-5,5)) , (x*y, (x,-3,3), (y,-3,3)))
```


![png](output_245_0.png)





    <sympy.plotting.plot.Plot at 0x143683ac518>




```python
# 3d parametric plots
from sympy import symbols, sin,cos
from sympy.plotting import plot3d_parametric_line
u = symbols('u')
plot3d_parametric_line(cos(u), sin(u), u, (u,-5,5))
```


![png](output_246_0.png)





    <sympy.plotting.plot.Plot at 0x14368202a58>




```python
#plotting regions
p6 = plot_implicit(y>x**2)
```


![png](output_247_0.png)



```python
#plotting using boolean
p7 = plot_implicit(And(y>x, y >=-x))
```


![png](output_248_0.png)



```python
from numpy import pi, sin
I = np.arange(0, 2*pi+0.1, 0.1)
plt.plot(I,sin(I), label='sin(I)')
plt.title('y=sin(x)')
plt.xlabel('x [rad]')
plt.ylabel(' Function y = sin(x)')
plt.text(pi/2,1, 'Max value', ha = 'center', va='bottom')
plt.text(3*pi/2,-1, 'Min value', ha = 'center', va='top')
plt.xticks(np.arange(0, 2*pi, pi/2), 
       ('0', r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{2}$'))
plt.xlim([0, 2*pi])
plt.ylim([-1.2, 1.2])
plt.grid() 
```


![png](output_249_0.png)


# Drawing and Diagrams
[index](#Mechpy)

to install pysketcher run these commands for a windows machine

check out tutorials
http://hplgit.github.io/pysketcher/doc/pub/tutorial/._pysketcher002.html  


```bash  
pip install future
choco install imagemagick -y # make sure to run as admin
git clone https://github.com/hplgit/pysketcher
cd pysketcher/
python setup.py install
```


```python
get_ipython().magic('matplotlib')  # seperate window

from pysketcher import *

L = 8.0
H = 1.0
xpos = 2.0
ypos = 3.0

drawing_tool.set_coordinate_system(xmin=0, xmax=xpos+1.2*L,
                                   ymin=0, ymax=ypos+5*H,
                                   axis=True)
drawing_tool.set_linecolor('blue')
drawing_tool.set_grid(True)
drawing_tool.set_fontsize(22)

P0 = point(xpos,ypos)
main = Rectangle(P0, L, H)
h = L/16  # size of support, clamped wall etc
support = SimplySupportedBeam(P0, h)
clamped = Rectangle(P0 + point(L, 0) - point(0,2*h), h, 6*h).set_filled_curves(pattern='/')
F_pt = point(P0[0]+L/2, P0[1]+H)
force = Force(F_pt + point(0,2*H), F_pt, '$F$').set_linewidth(3)
L_dim = Distance_wText((xpos,P0[1]-3*h), (xpos+L,P0[1]-3*h), '$L$')
beam = Composition({'main': main, 'simply supported end': support,
                    'clamped end': clamped, 'force': force,
                    'L': L_dim})
beam.draw()
beam.draw_dimensions()
drawing_tool.display()

get_ipython().magic('matplotlib inline') # inline plotting
```


```python
get_ipython().magic('matplotlib')  # seperate window

from pysketcher import *

L = 8.0
H = 1.0
xpos = 2.0
ypos = 3.0

drawing_tool.set_coordinate_system(xmin=0, xmax=xpos+1.2*L,
                                   ymin=0, ymax=ypos+5*H,
                                   axis=True)
drawing_tool.set_linecolor('blue')
drawing_tool.set_grid(True)
drawing_tool.set_fontsize(22)

P0 = point(xpos,ypos)
main = Rectangle(P0, L, H)
h = L/16  # size of support, clamped wall etc
support = SimplySupportedBeam(P0, h)
clamped = Rectangle(P0 + point(L, 0) - point(0,2*h), h, 6*h).set_filled_curves(pattern='/')
F_pt = point(P0[0]+L/2, P0[1]+H)
force = Force(F_pt + point(0,2*H), F_pt, '$F$').set_linewidth(3)
L_dim = Distance_wText((xpos,P0[1]-3*h), (xpos+L,P0[1]-3*h), '$L$')
beam = Composition({'main': main, 'simply supported end': support,
                    'clamped end': clamped, 'force': force,
                    'L': L_dim})
beam.draw()
beam.draw_dimensions()
drawing_tool.display()

get_ipython().magic('matplotlib inline') # inline plotting
```

## PLotting


```python
import matplotlib.pyplot as plt
```


```python
%matplotlib inline
```


```python
fig, ax = plt.subplots(figsize=(12, 3))

ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.05, 0.25)
ax.axhline(0)

ax.text(0, 0.1, "Text label", fontsize=14, family="serif")

ax.plot(1, 0, 'o')
ax.annotate("Annotation",
            fontsize=14, family="serif",
            xy=(1, 0), xycoords='data',
            xytext=(+20, +50), textcoords='offset points', 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=1.5"))

ax.text(2, 0.1, r"Equation: $i\hbar\partial_t \Psi = \hat{H}\Psi$", fontsize=14, family="serif")

ax.annotate('arc3', xy=(0.5, -1), xycoords='data',
            xytext=(-30, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=.2")
            )

ax.annotate('arc', xy=(1., 1), xycoords='data',
            xytext=(-40, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc,angleA=0,armA=30,rad=10"),
            )

ax.annotate('arc', xy=(1.5, -1), xycoords='data',
            xytext=(-40, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=7"),
            )
plt.show()
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /home/neal/anaconda3/lib/python3.5/site-packages/IPython/core/formatters.py in __call__(self, obj)
        337                 pass
        338             else:
    --> 339                 return printer(obj)
        340             # Finally look for special method names
        341             method = _safe_get_formatter_method(obj, self.print_method)


    /home/neal/anaconda3/lib/python3.5/site-packages/IPython/core/pylabtools.py in <lambda>(fig)
        224 
        225     if 'png' in formats:
    --> 226         png_formatter.for_type(Figure, lambda fig: print_figure(fig, 'png', **kwargs))
        227     if 'retina' in formats or 'png2x' in formats:
        228         png_formatter.for_type(Figure, lambda fig: retina_figure(fig, **kwargs))


    /home/neal/anaconda3/lib/python3.5/site-packages/IPython/core/pylabtools.py in print_figure(fig, fmt, bbox_inches, **kwargs)
        115 
        116     bytes_io = BytesIO()
    --> 117     fig.canvas.print_figure(bytes_io, **kw)
        118     data = bytes_io.getvalue()
        119     if fmt == 'svg':


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backend_bases.py in print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, **kwargs)
       2178                     orientation=orientation,
       2179                     dryrun=True,
    -> 2180                     **kwargs)
       2181                 renderer = self.figure._cachedRenderer
       2182                 bbox_inches = self.figure.get_tightbbox(renderer)


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in print_png(self, filename_or_obj, *args, **kwargs)
        525 
        526     def print_png(self, filename_or_obj, *args, **kwargs):
    --> 527         FigureCanvasAgg.draw(self)
        528         renderer = self.get_renderer()
        529         original_dpi = renderer.dpi


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in draw(self)
        472 
        473         try:
    --> 474             self.figure.draw(self.renderer)
        475         finally:
        476             RendererAgg.lock.release()


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/figure.py in draw(self, renderer)
       1157         dsu.sort(key=itemgetter(0))
       1158         for zorder, a, func, args in dsu:
    -> 1159             func(*args)
       1160 
       1161         renderer.close_group('figure')


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_base.py in draw(self, renderer, inframe)
       2322 
       2323         for zorder, a in dsu:
    -> 2324             a.draw(renderer)
       2325 
       2326         renderer.close_group('axes')


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in draw(self, renderer)
        794                     textrenderer.draw_text(gc, x, y, clean_line,
        795                                            textobj._fontproperties, angle,
    --> 796                                            ismath=ismath, mtext=mtext)
        797 
        798         gc.restore()


    /home/neal/anaconda3/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
         75                 value = type()
         76             try:
    ---> 77                 self.gen.throw(type, value, traceback)
         78                 raise RuntimeError("generator didn't stop after throw()")
         79             except StopIteration as exc:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in _wrap_text(textobj)
         58             textobj.set_text(old_text)
         59     else:
    ---> 60         yield textobj
         61 
         62 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in draw(self, renderer)
        747 
        748         with _wrap_text(self) as textobj:
    --> 749             bbox, info, descent = textobj._get_layout(renderer)
        750             trans = textobj.get_transform()
        751 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in _get_layout(self, renderer)
        350         tmp, lp_h, lp_bl = renderer.get_text_width_height_descent('lp',
        351                                                          self._fontproperties,
    --> 352                                                          ismath=False)
        353         offsety = (lp_h - lp_bl) * self._linespacing
        354 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in get_text_width_height_descent(self, s, prop, ismath)
        227             fontsize = prop.get_size_in_points()
        228             w, h, d = texmanager.get_text_width_height_descent(s, fontsize,
    --> 229                                                                renderer=self)
        230             return w, h, d
        231 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/texmanager.py in get_text_width_height_descent(self, tex, fontsize, renderer)
        673         else:
        674             # use dviread. It sometimes returns a wrong descent.
    --> 675             dvifile = self.make_dvi(tex, fontsize)
        676             dvi = dviread.Dvi(dvifile, 72 * dpi_fraction)
        677             try:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/texmanager.py in make_dvi(self, tex, fontsize)
        420                      'string:\n%s\nHere is the full report generated by '
        421                      'LaTeX: \n\n' % repr(tex.encode('unicode_escape')) +
    --> 422                      report))
        423             else:
        424                 mpl.verbose.report(report, 'debug')


    RuntimeError: LaTeX was not able to process the following string:
    b'lp'
    Here is the full report generated by LaTeX: 
    
    This is pdfTeX, Version 3.1415926-2.5-1.40.14 (TeX Live 2013/Debian)
     restricted \write18 enabled.
    entering extended mode
    (./d4969ce036b5440f12cfe4e0e57f9efe.tex
    LaTeX2e <2011/06/27>
    Babel <3.9h> and hyphenation patterns for 2 languages loaded.
    (/usr/share/texlive/texmf-dist/tex/latex/base/article.cls
    Document Class: article 2007/10/19 v1.4h Standard LaTeX document class
    (/usr/share/texlive/texmf-dist/tex/latex/base/size10.clo))
    
    ! LaTeX Error: File `type1cm.sty' not found.
    
    Type X to quit or <RETURN> to proceed,
    or enter new name. (Default extension: sty)
    
    Enter file name: 
    ! Emergency stop.
    <read *> 
             
    l.3 \renewcommand
                     {\rmdefault}{pnc}^^M
    No pages of output.
    Transcript written on d4969ce036b5440f12cfe4e0e57f9efe.log.




    <matplotlib.figure.Figure at 0x7f4759ccbf28>



```python
fig, ax = plt.subplots(figsize=(12, 3))

ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(-1, 3.5)
ax.set_ylim(-0.1, 1)
ax.axhline(0)

ax.text(0, 0.1, "Text label", fontsize=14, family="serif")

ax.text(2, 0.1, r"Equation: $i\hbar\partial_t \Psi = \hat{H}\Psi$", fontsize=14, family="serif")

plt.annotate("Important Value", (.5,.5), xycoords='data',xytext=(1,.2), arrowprops=dict(arrowstyle='->'))


ax.plot([1,0],marker=r'$\circlearrowleft$',ms=50)
ax.plot([0,.5],marker=r'$\downarrow$',ms=100)

ax.plot(1, 0, 'o')
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-101-ec3f3acd0dfe> in <module>()
         14 
         15 
    ---> 16 ax.plot([1,0],marker=r'$\circlearrowleft$',ms=50)
         17 ax.plot([0,.5],marker=r'$\downarrow$',ms=100)
         18 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/__init__.py in inner(ax, *args, **kwargs)
       1810                     warnings.warn(msg % (label_namer, func.__name__),
       1811                                   RuntimeWarning, stacklevel=2)
    -> 1812             return func(ax, *args, **kwargs)
       1813         pre_doc = inner.__doc__
       1814         if pre_doc is None:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_axes.py in plot(self, *args, **kwargs)
       1422             kwargs['color'] = c
       1423 
    -> 1424         for line in self._get_lines(*args, **kwargs):
       1425             self.add_line(line)
       1426             lines.append(line)


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_base.py in _grab_next_args(self, *args, **kwargs)
        384                 return
        385             if len(remaining) <= 3:
    --> 386                 for seg in self._plot_args(remaining, kwargs):
        387                     yield seg
        388                 return


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_base.py in _plot_args(self, tup, kwargs)
        372         ncx, ncy = x.shape[1], y.shape[1]
        373         for j in xrange(max(ncx, ncy)):
    --> 374             seg = func(x[:, j % ncx], y[:, j % ncy], kw, kwargs)
        375             ret.append(seg)
        376         return ret


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_base.py in _makeline(self, x, y, kw, kwargs)
        279         self._setdefaults(default_dict, kw, kwargs)
        280         seg = mlines.Line2D(x, y, **kw)
    --> 281         self.set_lineprops(seg, **kwargs)
        282         return seg
        283 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_base.py in set_lineprops(self, line, **kwargs)
        187     def set_lineprops(self, line, **kwargs):
        188         assert self.command == 'plot', 'set_lineprops only works with "plot"'
    --> 189         line.set(**kwargs)
        190 
        191     def set_patchprops(self, fill_poly, **kwargs):


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in set(self, **kwargs)
        935                raise TypeError('There is no %s property "%s"' %
        936                                (self.__class__.__name__, k))
    --> 937             ret.extend([func(v)])
        938         return ret
        939 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/lines.py in set_marker(self, marker)
       1069 
       1070         """
    -> 1071         self._marker.set_marker(marker)
       1072         self.stale = True
       1073 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/markers.py in set_marker(self, marker)
        253 
        254         self._marker = marker
    --> 255         self._recache()
        256 
        257     def get_path(self):


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/markers.py in _recache(self)
        191         self._capstyle = 'butt'
        192         self._filled = True
    --> 193         self._marker_function()
        194 
        195     if six.PY3:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/markers.py in _set_mathtext_path(self)
        329         props = FontProperties(size=1.0)
        330         text = TextPath(xy=(0, 0), s=self.get_marker(), fontproperties=props,
    --> 331                         usetex=rcParams['text.usetex'])
        332         if len(text.vertices) == 0:
        333             return


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/textpath.py in __init__(self, xy, s, size, prop, _interpolation_steps, usetex, *kl, **kwargs)
        443         self._vertices, self._codes = self.text_get_vertices_codes(
        444                                             prop, s,
    --> 445                                             usetex=usetex)
        446 
        447         self._should_simplify = False


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/textpath.py in text_get_vertices_codes(self, prop, s, usetex)
        523 
        524         if usetex:
    --> 525             verts, codes = text_to_path.get_text_path(prop, s, usetex=True)
        526         else:
        527             clean_line, ismath = self.is_math_text(s)


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/textpath.py in get_text_path(self, prop, s, ismath, usetex)
        148                                                     prop, s)
        149         else:
    --> 150             glyph_info, glyph_map, rects = self.get_glyphs_tex(prop, s)
        151 
        152         verts, codes = [], []


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/textpath.py in get_glyphs_tex(self, prop, s, glyph_map, return_new_glyphs_only)
        312             dvi = dviread.DviFromFileLike(dvifilelike, self.DPI)
        313         else:
    --> 314             dvifile = texmanager.make_dvi(s, self.FONT_SCALE)
        315             dvi = dviread.Dvi(dvifile, self.DPI)
        316         try:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/texmanager.py in make_dvi(self, tex, fontsize)
        420                      'string:\n%s\nHere is the full report generated by '
        421                      'LaTeX: \n\n' % repr(tex.encode('unicode_escape')) +
    --> 422                      report))
        423             else:
        424                 mpl.verbose.report(report, 'debug')


    RuntimeError: LaTeX was not able to process the following string:
    b'$\\\\circlearrowleft$'
    Here is the full report generated by LaTeX: 
    
    This is pdfTeX, Version 3.1415926-2.5-1.40.14 (TeX Live 2013/Debian)
     restricted \write18 enabled.
    entering extended mode
    (./4381b5d1fc97d373cb0ae275acd500e3.tex
    LaTeX2e <2011/06/27>
    Babel <3.9h> and hyphenation patterns for 2 languages loaded.
    (/usr/share/texlive/texmf-dist/tex/latex/base/article.cls
    Document Class: article 2007/10/19 v1.4h Standard LaTeX document class
    (/usr/share/texlive/texmf-dist/tex/latex/base/size10.clo))
    
    ! LaTeX Error: File `type1cm.sty' not found.
    
    Type X to quit or <RETURN> to proceed,
    or enter new name. (Default extension: sty)
    
    Enter file name: 
    ! Emergency stop.
    <read *> 
             
    l.3 \renewcommand
                     {\rmdefault}{pnc}^^M
    No pages of output.
    Transcript written on 4381b5d1fc97d373cb0ae275acd500e3.log.



    Error in callback <function install_repl_displayhook.<locals>.post_execute at 0x7f475bb78a60> (for post_execute):



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/pyplot.py in post_execute()
        145             def post_execute():
        146                 if matplotlib.is_interactive():
    --> 147                     draw_all()
        148 
        149             # IPython >= 2


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/_pylab_helpers.py in draw_all(cls, force)
        148         for f_mgr in cls.get_all_fig_managers():
        149             if force or f_mgr.canvas.figure.stale:
    --> 150                 f_mgr.canvas.draw_idle()
        151 
        152 atexit.register(Gcf.destroy_all)


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backend_bases.py in draw_idle(self, *args, **kwargs)
       2024         if not self._is_idle_drawing:
       2025             with self._idle_draw_cntx():
    -> 2026                 self.draw(*args, **kwargs)
       2027 
       2028     def draw_cursor(self, event):


    /home/neal/anaconda3/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
         75                 value = type()
         76             try:
    ---> 77                 self.gen.throw(type, value, traceback)
         78                 raise RuntimeError("generator didn't stop after throw()")
         79             except StopIteration as exc:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backend_bases.py in _idle_draw_cntx(self)
       1698     def _idle_draw_cntx(self):
       1699         self._is_idle_drawing = True
    -> 1700         yield
       1701         self._is_idle_drawing = False
       1702 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backend_bases.py in draw_idle(self, *args, **kwargs)
       2024         if not self._is_idle_drawing:
       2025             with self._idle_draw_cntx():
    -> 2026                 self.draw(*args, **kwargs)
       2027 
       2028     def draw_cursor(self, event):


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in draw(self)
        472 
        473         try:
    --> 474             self.figure.draw(self.renderer)
        475         finally:
        476             RendererAgg.lock.release()


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/figure.py in draw(self, renderer)
       1157         dsu.sort(key=itemgetter(0))
       1158         for zorder, a, func, args in dsu:
    -> 1159             func(*args)
       1160 
       1161         renderer.close_group('figure')


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_base.py in draw(self, renderer, inframe)
       2322 
       2323         for zorder, a in dsu:
    -> 2324             a.draw(renderer)
       2325 
       2326         renderer.close_group('axes')


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in draw(self, renderer)
        794                     textrenderer.draw_text(gc, x, y, clean_line,
        795                                            textobj._fontproperties, angle,
    --> 796                                            ismath=ismath, mtext=mtext)
        797 
        798         gc.restore()


    /home/neal/anaconda3/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
         75                 value = type()
         76             try:
    ---> 77                 self.gen.throw(type, value, traceback)
         78                 raise RuntimeError("generator didn't stop after throw()")
         79             except StopIteration as exc:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in _wrap_text(textobj)
         58             textobj.set_text(old_text)
         59     else:
    ---> 60         yield textobj
         61 
         62 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in draw(self, renderer)
        747 
        748         with _wrap_text(self) as textobj:
    --> 749             bbox, info, descent = textobj._get_layout(renderer)
        750             trans = textobj.get_transform()
        751 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in _get_layout(self, renderer)
        350         tmp, lp_h, lp_bl = renderer.get_text_width_height_descent('lp',
        351                                                          self._fontproperties,
    --> 352                                                          ismath=False)
        353         offsety = (lp_h - lp_bl) * self._linespacing
        354 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in get_text_width_height_descent(self, s, prop, ismath)
        227             fontsize = prop.get_size_in_points()
        228             w, h, d = texmanager.get_text_width_height_descent(s, fontsize,
    --> 229                                                                renderer=self)
        230             return w, h, d
        231 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/texmanager.py in get_text_width_height_descent(self, tex, fontsize, renderer)
        673         else:
        674             # use dviread. It sometimes returns a wrong descent.
    --> 675             dvifile = self.make_dvi(tex, fontsize)
        676             dvi = dviread.Dvi(dvifile, 72 * dpi_fraction)
        677             try:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/texmanager.py in make_dvi(self, tex, fontsize)
        420                      'string:\n%s\nHere is the full report generated by '
        421                      'LaTeX: \n\n' % repr(tex.encode('unicode_escape')) +
    --> 422                      report))
        423             else:
        424                 mpl.verbose.report(report, 'debug')


    RuntimeError: LaTeX was not able to process the following string:
    b'lp'
    Here is the full report generated by LaTeX: 
    
    This is pdfTeX, Version 3.1415926-2.5-1.40.14 (TeX Live 2013/Debian)
     restricted \write18 enabled.
    entering extended mode
    (./d4969ce036b5440f12cfe4e0e57f9efe.tex
    LaTeX2e <2011/06/27>
    Babel <3.9h> and hyphenation patterns for 2 languages loaded.
    (/usr/share/texlive/texmf-dist/tex/latex/base/article.cls
    Document Class: article 2007/10/19 v1.4h Standard LaTeX document class
    (/usr/share/texlive/texmf-dist/tex/latex/base/size10.clo))
    
    ! LaTeX Error: File `type1cm.sty' not found.
    
    Type X to quit or <RETURN> to proceed,
    or enter new name. (Default extension: sty)
    
    Enter file name: 
    ! Emergency stop.
    <read *> 
             
    l.3 \renewcommand
                     {\rmdefault}{pnc}^^M
    No pages of output.
    Transcript written on d4969ce036b5440f12cfe4e0e57f9efe.log.




    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /home/neal/anaconda3/lib/python3.5/site-packages/IPython/core/formatters.py in __call__(self, obj)
        337                 pass
        338             else:
    --> 339                 return printer(obj)
        340             # Finally look for special method names
        341             method = _safe_get_formatter_method(obj, self.print_method)


    /home/neal/anaconda3/lib/python3.5/site-packages/IPython/core/pylabtools.py in <lambda>(fig)
        224 
        225     if 'png' in formats:
    --> 226         png_formatter.for_type(Figure, lambda fig: print_figure(fig, 'png', **kwargs))
        227     if 'retina' in formats or 'png2x' in formats:
        228         png_formatter.for_type(Figure, lambda fig: retina_figure(fig, **kwargs))


    /home/neal/anaconda3/lib/python3.5/site-packages/IPython/core/pylabtools.py in print_figure(fig, fmt, bbox_inches, **kwargs)
        115 
        116     bytes_io = BytesIO()
    --> 117     fig.canvas.print_figure(bytes_io, **kw)
        118     data = bytes_io.getvalue()
        119     if fmt == 'svg':


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backend_bases.py in print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, **kwargs)
       2178                     orientation=orientation,
       2179                     dryrun=True,
    -> 2180                     **kwargs)
       2181                 renderer = self.figure._cachedRenderer
       2182                 bbox_inches = self.figure.get_tightbbox(renderer)


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in print_png(self, filename_or_obj, *args, **kwargs)
        525 
        526     def print_png(self, filename_or_obj, *args, **kwargs):
    --> 527         FigureCanvasAgg.draw(self)
        528         renderer = self.get_renderer()
        529         original_dpi = renderer.dpi


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in draw(self)
        472 
        473         try:
    --> 474             self.figure.draw(self.renderer)
        475         finally:
        476             RendererAgg.lock.release()


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/figure.py in draw(self, renderer)
       1157         dsu.sort(key=itemgetter(0))
       1158         for zorder, a, func, args in dsu:
    -> 1159             func(*args)
       1160 
       1161         renderer.close_group('figure')


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/axes/_base.py in draw(self, renderer, inframe)
       2322 
       2323         for zorder, a in dsu:
    -> 2324             a.draw(renderer)
       2325 
       2326         renderer.close_group('axes')


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         59     def draw_wrapper(artist, renderer, *args, **kwargs):
         60         before(artist, renderer)
    ---> 61         draw(artist, renderer, *args, **kwargs)
         62         after(artist, renderer)
         63 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in draw(self, renderer)
        794                     textrenderer.draw_text(gc, x, y, clean_line,
        795                                            textobj._fontproperties, angle,
    --> 796                                            ismath=ismath, mtext=mtext)
        797 
        798         gc.restore()


    /home/neal/anaconda3/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
         75                 value = type()
         76             try:
    ---> 77                 self.gen.throw(type, value, traceback)
         78                 raise RuntimeError("generator didn't stop after throw()")
         79             except StopIteration as exc:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in _wrap_text(textobj)
         58             textobj.set_text(old_text)
         59     else:
    ---> 60         yield textobj
         61 
         62 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in draw(self, renderer)
        747 
        748         with _wrap_text(self) as textobj:
    --> 749             bbox, info, descent = textobj._get_layout(renderer)
        750             trans = textobj.get_transform()
        751 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/text.py in _get_layout(self, renderer)
        350         tmp, lp_h, lp_bl = renderer.get_text_width_height_descent('lp',
        351                                                          self._fontproperties,
    --> 352                                                          ismath=False)
        353         offsety = (lp_h - lp_bl) * self._linespacing
        354 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/backends/backend_agg.py in get_text_width_height_descent(self, s, prop, ismath)
        227             fontsize = prop.get_size_in_points()
        228             w, h, d = texmanager.get_text_width_height_descent(s, fontsize,
    --> 229                                                                renderer=self)
        230             return w, h, d
        231 


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/texmanager.py in get_text_width_height_descent(self, tex, fontsize, renderer)
        673         else:
        674             # use dviread. It sometimes returns a wrong descent.
    --> 675             dvifile = self.make_dvi(tex, fontsize)
        676             dvi = dviread.Dvi(dvifile, 72 * dpi_fraction)
        677             try:


    /home/neal/anaconda3/lib/python3.5/site-packages/matplotlib/texmanager.py in make_dvi(self, tex, fontsize)
        420                      'string:\n%s\nHere is the full report generated by '
        421                      'LaTeX: \n\n' % repr(tex.encode('unicode_escape')) +
    --> 422                      report))
        423             else:
        424                 mpl.verbose.report(report, 'debug')


    RuntimeError: LaTeX was not able to process the following string:
    b'lp'
    Here is the full report generated by LaTeX: 
    
    This is pdfTeX, Version 3.1415926-2.5-1.40.14 (TeX Live 2013/Debian)
     restricted \write18 enabled.
    entering extended mode
    (./d4969ce036b5440f12cfe4e0e57f9efe.tex
    LaTeX2e <2011/06/27>
    Babel <3.9h> and hyphenation patterns for 2 languages loaded.
    (/usr/share/texlive/texmf-dist/tex/latex/base/article.cls
    Document Class: article 2007/10/19 v1.4h Standard LaTeX document class
    (/usr/share/texlive/texmf-dist/tex/latex/base/size10.clo))
    
    ! LaTeX Error: File `type1cm.sty' not found.
    
    Type X to quit or <RETURN> to proceed,
    or enter new name. (Default extension: sty)
    
    Enter file name: 
    ! Emergency stop.
    <read *> 
             
    l.3 \renewcommand
                     {\rmdefault}{pnc}^^M
    No pages of output.
    Transcript written on d4969ce036b5440f12cfe4e0e57f9efe.log.




    <matplotlib.figure.Figure at 0x7f4759cbf860>



```python
from pylab import *
from scipy import fft
N = 2**9
F = 25
t = arange(N)/float(N)
x = cos(2*pi*t*F) + rand(len(t))*3
subplot(2,1,1)
plot(t,x)
ylabel('x []')
xlabel('t [seconds]')
title('A cosine wave')
grid()

subplot(2,1,2)
f = t*N
xf = fft(x)
plot(f,abs(xf))
title('Fourier transform of a cosine wave')
xlabel('xf []')
ylabel('xf []')
xlim([0,N])
grid()
show()

# note the spike at 25 hz and 512-25
```
