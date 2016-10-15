# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:07:50 2016

@author: ngordon
"""

#==============================================================================
# 3d animation 0 stackoverflow
#==============================================================================

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen(num):
    """Data generation"""
    angle = num * np.pi/36    
    vx, vy, vz = np.cos(angle), np.sin(angle), 1
    ax.cla()
    ax.quiver(0, 0, 0, vx, vy, vz, pivot="tail", color="black")
    ax.quiver(0, 0, 0, vx, vy, 0, pivot="tail", color="black",
              linestyle="dashed")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=30, azim=60)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data_gen(0)
ani = animation.FuncAnimation(fig, data_gen, range(72), blit=False)
plt.show()


