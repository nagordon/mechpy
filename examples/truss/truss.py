from numpy import pi, array, size, zeros, outer, ones, concatenate, where, multiply, sum, column_stack, vstack
from numpy.linalg import norm, solve
from numpy.random import normal
from matplotlib import delaunay
from scipy.spatial import Delaunay
#from matplotlib.tri import Triangulation
from matplotlib.pyplot import plot, axis, arrow, xkcd, savefig, grid


# Yield strength of steel
Fy = 344*pow(10, 6)

# Elastic modulus of steel
E = 210*pow(10, 9)

# Outer diameters of some optional sizes, in meters
OUTER_DIAM = [(x+1.0)/100 for x in range(10)]

# Thickness of the wall sections of optional sizes, in meters
THICK = [d/15 for d in OUTER_DIAM]

# Cross sectional area in m^2
AREA_SEC = [pi*pow(d/2, 2) - pi*pow(d/2-d/15, 2) for d in OUTER_DIAM]

# Moment of inertia, in m^4
I_SEC = [pi*(pow(d, 4) - pow((d - 2*d/15), 4))/64 for d in OUTER_DIAM]

# Weight per length, kg/m
WEIGHT = [a*7870 for a in AREA_SEC]

# Example factor of safety
FOS_MIN = 1.25


def force_eval(D):
    """This function takes as input a dictionary D that defines the following variables:
        "Re":    encodes the permitted motion of each joint. Each joint is represented by 
                 a vector of three binary variables, indicating whether or not it is supported
                 in the, x, y and z direction.
        "Coord": This indicates the x, y, and z coordinates of each joint.
        "Load":  This indicates the x, y and z loadings placed on each joint.
        "Con":   This indicates how the joints are connected
        "E":     This indicates the elastic modullus of each connecting member described above.
        "A":     This indicates the area of each connecting member described above
       The function returns three variables.  
        F:       The forces present in each member
        U:       Node displacements
        R:       Node reactions
        """
    
    
    Tj = zeros([3, size(D["Con"], axis=1)])
    w = array([size(D["Re"], axis=0), size(D["Re"], axis=1)])
    SS = zeros([3*w[1], 3*w[1]])
    U = 1.0 - D["Re"]
    
    # This identifies joints that are unsupported, and can therefore be loaded
    ff = where(U.T.flat == 1)[0]
    
    # Step through the each member in the truss, and build the global stiffness matrix
    for i in range(size(D["Con"], axis=1)):
        H = D["Con"][:, i]
        C = D["Coord"][:, H[1]] - D["Coord"][:, H[0]]
        Le = norm(C)
        T = C/Le
        s = outer(T, T)
        G = D["E"][i]*D["A"][i]/Le
        ss = G*concatenate((concatenate((s, -s), axis=1), concatenate((-s, s), axis=1)), axis=0)
        Tj[:, i] = G*T
        e = range((3*H[0]), (3*H[0] + 3)) + range((3*H[1]), (3*H[1] + 3))
        for ii in range(6):
            for j in range(6):
                SS[e[ii], e[j]] += ss[ii, j]

    SSff = zeros([len(ff), len(ff)])
    for i in range(len(ff)):
        for j in range(len(ff)):
            SSff[i,j] = SS[ff[i], ff[j]]
                
    Loadff = D["Load"].T.flat[ff]
    Uff = solve(SSff, Loadff)
    
    ff = where(U.T==1)
    for i in range(len(ff[0])):
        U[ff[1][i], ff[0][i]] = Uff[i]
    
    F = sum(multiply(Tj, U[:, D["Con"][1,:]] - U[:, D["Con"][0,:]]), axis=0)
    R = sum(SS*U.T.flat[:], axis=1).reshape([w[1], w[0]]).T
    
    return F, U, R


def fos_eval(truss):
    D = {}
    
    M = len(truss["Con"].T)    
    N = len(truss["Coord"].T)
    
    # Add the "Re"
    D["Re"] = array([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]]).T
    for _ in range(N-5):
        D["Re"] = column_stack([D["Re"], [0,0,1]])
    
    # Add the appropriate loads
    D["Load"] = zeros([3, truss["N"]])
    D["Load"][1, 1] = -200000.0
    D["Load"][1, 3] = -200000.0
    
    # Add the area information from truss structure
    D["A"] = []
    for member_size in truss["SIZES"]:
        D["A"].append(AREA_SEC[int(member_size)])
    D["Coord"] = truss["Coord"]
    D["Con"] = truss["Con"]
    D["E"] = E*ones(M)
    
    # Do force analysis
    F, U, R = force_eval(D)
    
    # Calculate lengths
    L = zeros(M)
    for i in range(M):
        L[i] = norm(D["Coord"][:, D["Con"][0, i]] - D["Coord"][:, D["Con"][1, i]])
    
    # Calculate FOS's
    FOS = zeros(M)
    for i in range(len(F)):
        FOS[i] = D["A"][i]*Fy/F[i]
        if FOS[i] < 0:
            FOS[i] = min(pi*pi*E*I_SEC[int(truss["SIZES"][i] - 1)]/(L[i]*L[i])/-F[i], -FOS[i])
    
    return FOS, F
    

def plot_truss(truss, FOS, F):
    # Collect some information
    M = len(truss["Con"].T)    
    N = len(truss["Coord"].T)

    Hm = []
    # Plot every member
    for i in range(M):
        p1 = truss["Coord"][:, truss["Con"][0, i]]
        p2 = truss["Coord"][:, truss["Con"][1, i]]
        if FOS[i] > 1:
            color = 'g'
        else:
            color = 'r'
        if F[i] > 0:
            lst = '--'
        else:
            lst = '-'
        Hm.append(plot([p1[0], p2[0]], [p1[1], p2[1]], color, linewidth=truss["SIZES"][i]+1, linestyle = lst))
        axis('equal')
        
    # Plot supports
    Hs = []
    Hs.append(plot(truss["Coord"][0, 0], truss["Coord"][1, 0], 'ks', ms=15))
    Hs.append(plot(truss["Coord"][0, 2], truss["Coord"][1, 2], 'ko', ms=15))
    Hs.append(plot(truss["Coord"][0, 4], truss["Coord"][1, 4], 'ks', ms=15))
    
    # Plot loads
    Hl = []
    Hl.append(plot(truss["Coord"][0, 1], truss["Coord"][1, 1], 'ko', ms=10))
    Hl.append(arrow(truss["Coord"][0, 1], truss["Coord"][1, 1] + 2.0, 0.0, -1.5, 
                    fc="m", ec="m", head_width=0.3, head_length=0.6, width=0.1, zorder=3))
    Hl.append(plot(truss["Coord"][0, 3], truss["Coord"][1, 3], 'ko', ms=10))
    Hl.append(arrow(truss["Coord"][0, 3], truss["Coord"][1, 3] + 2.0, 0.0, -1.5, 
                    fc="m", ec="m", head_width=0.3, head_length=0.6, width=0.1, zorder=3))
        
    # Plot every joint
    Hj = []
    for i in range(N-5):
        Hj.append(plot(truss["Coord"][0, i + 5], truss["Coord"][1, i + 5], 'ko', ms=10))
    
    return Hm, Hj, Hl, Hs


def init_truss(N):
    # Initializes a random truss, with a few preset options
    truss = {}
    truss["N"] = N
    truss["Coord"] = array([[-5, 0, 0], [-2, 0, 0], [1, 0, 0], [3, 0, 0], [5, 0, 0]])
    for i in range(N-5):
        # Draw random location from distribution
        x = normal()*2
        y = normal()+3
        # Add to coordinates
        truss["Coord"] = vstack([truss["Coord"], array([x, y, 0])])
    truss["Coord"] = truss["Coord"].T
    # Delaunay triangulation to get connections
    cens,edg,tri,neig = Delaunay(truss["Coord"][0, :], truss["Coord"][1, :])
    truss["Con"] = edg.T
    truss["SIZES"] = ones(len(truss["Con"].T))*4
    
    return truss


if __name__ == "__main__":
    xkcd()
    tr = init_truss(10)
    FOS, F = fos_eval(tr)
    plot_truss(tr, FOS, F)
    grid("on")
    savefig("fos.png")
