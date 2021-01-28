import meshio
import math
import numpy as np
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

"""
Contains functions to create level sets
"""
# Normalize an array of 3-component vectors
def normalize(a):
    ss = np.sum(a**2, axis=1)**0.5
    a = a / ss[:, np.newaxis]
    return a

# Coordinate conversions
def cart2pol(x, y, z):
    r = np.sqrt(x**2 + y**2)
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(r, z)
    theta = np.arctan2(y, x)
    return(rho, phi, theta)

def pol2cart(rho, phi, theta):
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return(x, y, z)

# Convert to polar, add r + k, convert back to cartesian
def add_to_rad(inc, vertices):
    sets = {}
    for i in inc:
        polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
        polar_vert[:,0] += i
        cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
        sets[str(i)] = cart_vert
    return(sets)

# Convert to polar, multiply r*k, convert back to cartesian
def mult_rad(inc, vertices):
    sets = {}
    for i in inc:
        polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
        polar_vert[:,0] *= i
        cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
        sets[str(i)] = cart_vert
    return(sets)

# Returns dot product of displacement vectors with surface outward normal
def dots(u, vert, conn):

    # Check type
    vert = np.asarray(vert, dtype="float64")
    conn = np.asarray(conn, dtype="int64")

    # Face coordinates
    tris = vert[conn]

    # Face normals
    fn = np.cross( tris[:,1,:] - tris[:,0,:]  , tris[:,2,:] - tris[:,0,:] )

    # Normalize face normals
    fn = normalize(fn)

    # Vertex normal = sum of adjacent face normals
    n = np.zeros(vert.shape, dtype = vert.dtype)
    n[ conn[:,0] ] += fn
    n[ conn[:,1] ] += fn
    n[ conn[:,2] ] += fn

    # Normalize vertex normals
    # Mult by -1 to make them point outward
    n = normalize(n) * -1

    # vertex displacement
    disp = np.array([u(x) for x in vert])

    # normalize row-wise
    disp = normalize(disp)

    # dot products
    dp = np.sum(disp*n, axis=1)

    return dp