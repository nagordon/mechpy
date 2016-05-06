#!/usr/bin/env python
# <MOHR: A tool for drawing Mohr's Circles.>
#
# Copyright (C) 2012 Cristian Gabriel Escudero escudero89@gmail.com
# This program is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License version 3, as published 
# by the Free Software Foundation.
# 
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranties of 
# MERCHANTABILITY, SATISFACTORY QUALITY, or FITNESS FOR A PARTICULAR 
# PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along 
# with this program.  If not, see <http://www.gnu.org/licenses/>.

# Modo de uso:
# Desde consola llamar: python Mohr "a" "b" "c" "filename" "simbolos"
# Donde a = sigma_x, b = sigma_y, c = sigma_xy = sigma_yx
# Siendo estas las componentes del tensor de tensiones.
# "filename" nombre del archivo ("" -> no guarda el archivo)
# "simbolos" son los dibujos de A B C y los grados (por si salieron feos,
# y quieres agregarlo manualmente, dejar vacio para que estren)
# 
# https://github.com/escudero89/mis_scripts/blob/master/python/mohr%20circle/Mohr.py

import math, sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Arc, Circle
import matplotlib.lines as lines

def main(sigma_x, sigma_y, sigma_xy, filename="", symbols="yes"):
    
    # Variables iniciales
    
    escala = 1
    borde = 1
    '''
    sigma_x = 0
    sigma_y = 0
    sigma_xy = 8      
    '''
    OC = float(sigma_x + sigma_y) / 2
    radio = math.sqrt(pow(float(sigma_x - sigma_y)/2, 2) + sigma_xy * sigma_xy)
    
    # Retorna alfa en radianes, asi que lo pasamos a grados, e invertimos su signo
    if (sigma_xy == 0):
        alfa = math.pi
    else:
        alfa = math.acos((sigma_x - sigma_y) / (2 * radio)) / 2
        alfa = -alfa / math.pi * 180    
    
    # Variables para plotear
    
    ancho = int( (abs(radio) + abs(OC)) * 2 * escala + borde * 2 )
    alto = int( abs(radio) * 2 * escala + borde * 2 )
    paso = 10      
    
    delta_x = ancho / 2
    delta_y = alto / 2

    # Puntos
    
    A = { "x" : sigma_x, "y" : -sigma_xy }
    B = { "x" : sigma_y, "y" : sigma_xy }
    C = { "x" : OC, "y": 0 }
      
    # Preparo para plotear
    
    p0 = [-delta_x + OC, -delta_y * 2]
    p1 = [delta_x + OC, delta_y * 2]
          
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)

    # Agrego figuras
    
    ax1.plot(A["x"], A["y"], 'o', label="A")
    ax1.plot(B["x"], B["y"], 'o', label="B")
    ax1.plot(C["x"], C["y"], 'o', label="C")   
           
    ax1.add_line(lines.Line2D((-ancho, ancho),(0, 0), linewidth=1, color="black"))
    ax1.add_line(lines.Line2D((0, 0),(-alto, alto), linewidth=1, color="black"))    

    ax1.add_line(lines.Line2D((A["x"],B["x"]),(A["y"],B["y"]),
        linewidth=1,
        color="red"))

    ax1.add_artist(Circle((C["x"], C["y"]), radio, fill=False, color="blue"))

    arc_len = radio
    ax1.add_artist(Arc((C["x"], C["y"]), 
        arc_len, arc_len, 
        theta1=alfa*2,
        theta2=0,
        color="purple",
        label="2 alfa = " + str(alfa)))
        
    escale = .1*radio

    if (symbols == "yes"):
    
        plt.annotate('A', xy=(A["x"] + escale, A["y"] + escale), size='large')
        plt.annotate('B', xy=(B["x"] + escale, B["y"] + escale), size='large')
        plt.annotate('C', xy=(C["x"] + escale, C["y"] + escale), size='large')        
                
        plt.annotate(str(round(alfa*2, 2)), 
            xy=(C["x"]+arc_len/2, C["y"]-arc_len/2), 
            color="purple", 
            size='large')
        
    # Ploteo
    
    ax1.grid(True)
    ax1.axis('equal')
    ax1.axis((p0[0], p1[0], p0[1], p1[1]))
    

        
    plt.show()        
    
if __name__ == "__main__":

    sigma_x = 1
    sigma_y = 2
    sigma_xy = 10
    
    main(sigma_x, sigma_y, sigma_xy)
