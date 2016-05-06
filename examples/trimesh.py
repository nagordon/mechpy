import numpy
import matplotlib as mplt

def lineplot(vertices, indices, linewidths=1):
    """Plot 2D line segments"""
    vertices = numpy.asarray(vertices)
    indices = numpy.asarray(indices)
    
    #3d tensor [segment index][vertex index][x/y value]
    lines = vertices[numpy.ravel(indices),:].reshape((indices.shape[0],2,2))
    
    col = mplt.collections.LineCollection(lines)
    col.set_color('k')
    col.set_linewidth(linewidths)

    sub = mplt.pylab.gca()
    sub.add_collection(col,autolim=True)
    sub.autoscale_view()

def trimesh(vertices, indices, labels=False):
    """
    Plot a 2D triangle mesh
    """
    from scipy import asarray
    from matplotlib import collections
    from pylab import gca, axis, text, show, figure
    from numpy import average
    
    vertices,indices = asarray(vertices),asarray(indices)

    #3d tensor [triangle index][vertex index][x/y value]
    triangles = vertices[indices.ravel(),:].reshape((indices.shape[0],3,2))
    
    col = collections.PolyCollection(triangles)
    col.set_facecolor('grey')
    col.set_alpha(0.5)
    col.set_linewidth(1)

    figure()
    sub = gca()
    
    sub.add_collection(col,autolim=True)
    
    axis('off')
    sub.autoscale_view()
    
    if labels:
        barycenters = average(triangles,axis=1)
        for n,bc in enumerate(barycenters):
            text(bc[0], bc[1], str(n), {'color' : 'k', 'fontsize' : 8,
                                        'horizontalalignment' : 'center',
                                        'verticalalignment' : 'center'})
    show()
    
    
def regular_triangle_mesh(nx, ny):
    """Construct a regular triangular mesh in the unit square
    
    Parameters
    ----------
    nx : int
       Number of nodes in the x direction
    ny : int
       Number of nodes in the y direction

    Returns
    -------
    Vert : array
        nx*ny x 2 vertex list
    E2V : array
        Nex x 3 element list

    Examples
    --------
    >>> E2V,Vert = regular_triangle_mesh(3, 2)

    """
    nx,ny = int(nx),int(ny)

    if nx < 2 or ny < 2:
        raise ValueError('minimum mesh dimension is 2: %s' % ((nx,ny),) )

    Vert1 = numpy.tile(numpy.arange(0, nx-1), ny - 1) + numpy.repeat(numpy.arange(0, nx * (ny - 1), nx), nx - 1)
    Vert3 = numpy.tile(numpy.arange(0, nx-1), ny - 1) + numpy.repeat(numpy.arange(0, nx * (ny - 1), nx), nx - 1) + nx
    Vert2 = Vert3 + 1
    Vert4 = Vert1 + 1

    Verttmp = numpy.meshgrid(numpy.arange(0, nx, dtype='float'), numpy.arange(0, ny, dtype='float'))
    Verttmp = (Verttmp[0].ravel(),Verttmp[1].ravel())
    Vert = numpy.vstack(Verttmp).transpose()
    Vert[:,0] = (1.0 / (nx - 1)) * Vert[:,0]
    Vert[:,1] = (1.0 / (ny - 1)) * Vert[:,1]

    E2V1 = numpy.vstack((Vert1,Vert2,Vert3)).transpose()
    E2V2 = numpy.vstack((Vert1,Vert4,Vert2)).transpose()
    E2V = numpy.vstack((E2V1,E2V2))

    return Vert,E2V

if __name__=='__main__':

    meshnum = 1

    if meshnum==1:
        from pyamg.gallery import mesh
        V,E = mesh.regular_triangle_mesh(20,6)
        
    if meshnum==2:
        from scipy.io import loadmat
        mesh = loadmat('crack_mesh.mat')
        V=mesh['V']
        E=mesh['E']

    trimesh(V,E)
