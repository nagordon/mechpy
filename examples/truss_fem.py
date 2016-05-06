####################################################
#
# Simple Truss direct stifness method implementation
# 2010/01/05
#    py27
####################################################

from scipy.linalg import solve
from numpy import matrix, array
from math import sqrt

# nodes
xy = [ 3., 0., 0., 0., 0., 4., 3., 4. ]


# elements (sorry, truss elements only)
#
# elem = [ node1, node2, E, A ]
#
elem1 = [ 1, 2, 20., 1. ]
elem2 = [ 2, 0, 300., 1. ]
elem3 = [ 0, 1, 10., 1. ]
elem4 = [ 2, 3, 10., 1. ]
elem5 = [ 3, 0, 10., 1. ]
elemlist = [ elem1, elem2, elem3, elem4, elem5 ]

# loads
FxFy = [ 13., 0., 0., 0., 0., 0., 0., 5. ]

# constraints (0 = active, 1 = inactive)
const_uv = [ 0, 0, 1, 1, 1, 0, 0, 0 ]

##############################################
#
# Defining simple list-based matrix and
# related functions, such as:
#
##############################################

def makematrix(m,n):
    listobj = []
    for i in range(m):
        make_row = []
        for j in range(n):
            make_row.append(0)
        listobj.append(make_row)
    return listobj

##############################################
#
# This is what would be called "element subroutine"
# and it returns the element's stiffness matrix
#
##############################################

def trussELMmatrix(thiselement):
    k = makematrix(4,4)
    x1 = xy[2*thiselement[0]]
    y1 = xy[2*thiselement[0]+1]
    x2 = xy[2*thiselement[1]]
    y2 = xy[2*thiselement[1]+1]
    E = thiselement[2]
    A = thiselement[3]
    L = sqrt( (x2-x1)**2 + (y2-y1)**2 )
    c = (x2-x1)/L
    s = (y2-y1)/L
    k[0][0] = k[2][2] = (E*A/L)*c**2
    k[0][1] = k[1][0] = k[2][3] = k[3][2] = (E*A/L)*c*s
    k[1][1] = k[3][3] = (E*A/L)*s**2
    k[0][2] = k[2][0] = -(E*A/L)*c**2
    k[3][0] = k[2][1] = k[1][2] = k[0][3] = -(E*A/L)*c*s
    k[3][1] = k[1][3] = -(E*A/L)*s**2
    return k

##############################################
#
# main program
#
##############################################

# Calculate ID array (which is a list)
IDarray = range(len(const_uv))
counter = 0
for i in range(len(const_uv)):
    if const_uv[i] == 0:
        counter += 1
        IDarray[i] = counter
    else:
        IDarray[i] = 0

# setup stiffness matrix
stiffness_k = makematrix(counter,counter)
matrix_b = range(counter)
for elm in elemlist:
    n1 = elm[0] # node 1
    n2 = elm[1] # node 2
    elmmatrix = trussELMmatrix(elm)
    l2g = [ IDarray[2*n1], IDarray[2*n1+1], IDarray[2*n2], IDarray[2*n2+1] ]
    for i in range(4):
        if l2g[i] != 0:
            for j in range(4):
                if l2g[j] != 0:
                    stiffness_k[l2g[i]-1][l2g[j]-1] += elmmatrix[i][j]
    
    # pra cada elemento, loopear nos elementos da matriz e ir soman
    # -do ao lugar indicado pelo IDarray correspondente

# loads matrix
for i in range(len(IDarray)):
    if IDarray[i] != 0:
        matrix_b[IDarray[i]-1] = FxFy[i]

answer = solve(matrix(stiffness_k),array(matrix_b))

reactions = []
for i in range(len(const_uv)):
    if const_uv[i] == 1:
        reac = 0
        for j in range(len(answer)):
            reac += stiffness_k[i][j]*answer[j]
        reactions.append(reac)
    else:
        reactions.append(0)


