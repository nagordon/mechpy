# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:07:50 2016

@author: ngordon
"""

#==============================================================================
# 3d animation 0 stackoverflow
#==============================================================================

import numpy as np

from matplotlib.pyplot import *
import matplotlib.animation as animation



x = [[0,0],[0,1],[1,2],[2,2],[2,1],[1,0],[1,1] ]
y = [[0,1],[0,0],[0,0],[0,1],[1,1],[1,1],[0,1] ]

for x1, y1 in zip(x,y):
    plot(x1,y1,'-o')



#x=[0,1,2]
#y=[0,0,0]
#plt.plot(x,y,'-o')


