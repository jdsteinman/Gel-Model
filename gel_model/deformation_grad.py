import numpy as np
from scipy.linalg import inv
import meshio 
from sim_tools import get_surface_normals, normalize

def def_grad(points, normals, u):

    npoints = points.shape[0]
    F = np.zeros((npoints, 3, 3))

    # Get two perpendicular vectors
    for i, n in enumerate(normals):
        # Choose two perpendicular vectors
        v = np.array([0, 0, 0])
        x = np.array([1, 0, 0])
        y = np.array([0, 1,  0])
        
        if np.allclose(n, x):
            v = y
        else:
            v = x

        # Find perpendicular vectors to n
        a = np.cross(n, v) 
        b = np.cross(n, a) 

        a = normalize(a) * 1.75
        b = normalize(b) * 1.75

        # create box
        p = points[i]
        box = np.zeros((8,3))

        box[0] = p + a + b
        box[1] = p + a - b
        box[2] = p - a + b
        box[3] = p - a - b
        box[4] = p + a + b + 3.5*n
        box[5] = p + a - b + 3.5*n
        box[6] = p - a + b + 3.5*n
        box[7] = p - a - b + 3.5*n
        xbar   =  np.mean(box, 0)
        vectors = box - xbar

        # Deformed box
        dbox = np.zeros(box.shape)
        for j, vert in enumerate(box):
            dbox[j] = vert + u(vert)

        dvectors = dbox - (xbar + u(xbar))

        # Solve for F:  F*vectos = dvectors
        x = vectors.T
        b = dvectors.T

        F[i,:,:] = b@x.T@inv(x@x.T)

    return F