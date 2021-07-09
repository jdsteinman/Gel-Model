import meshio
import math
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sb
from numpy.linalg import eig
from scipy.linalg import polar, inv
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle
from dolfin import Function as fenicsFunc

"""
Functions to create level sets, calculate surface normals,
plot stuff, and more.
"""
## Normalize an array of 3-component vectors
def normalize(a):

    arr = np.asarray(a)

    assert(np.ndim(arr) <= 2)

    if arr.ndim == 1:
        ss = np.sum(arr**2)**0.5
        arr = arr/ss
    elif arr.ndim == 2:
        ss = np.sum(arr**2, axis=1)**0.5
        arr = arr / ss[:, np.newaxis]

    return arr

## Coordinate conversions
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

## Convert to polar, add r + k, convert back to cartesian
def add_to_radius(vertices, k):
    polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
    polar_vert[:,0] += k
    cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
    
    return cart_vert

## Convert to polar, multiply r*k, convert back to cartesian
def scale_radius(vertices, k):
    polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
    polar_vert[:,0] *= k
    cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
    
    return cart_vert

## Surface Normals
def get_surface_normals(vert, conn):
     # Check type
    vert = np.asarray(vert, dtype="float64")
    conn = np.asarray(conn, dtype="int64")

    # Face coordinates
    tris = vert[conn]

    # Face normals
    fn = np.cross(tris[:,1,:] - tris[:,0,:], tris[:,2,:] - tris[:,0,:] )

    # Normalize face normals
    fn = normalize(fn)

    # Vertex normal = sum of adjacent face normals
    n = np.zeros(vert.shape, dtype = vert.dtype)
    n[ conn[:,0] ] += fn
    n[ conn[:,1] ] += fn
    n[ conn[:,2] ] += fn

    # Normalize vertex normals
    # Mult by -1 to make them point outward??
    n = normalize(n) * -1

    return n

## Returns dot product of displacement vectors with surface outward normal
def dots(disp, norms):

    # normalize row-wise
    disp = normalize(disp)

    # dot products
    dp = np.sum(disp*norms, axis=1)

    return dp

"""
Converted to classes


## Isosurface Outputs ========================================================================
def generate_sets(factors, points):

    # Format inputs
    points = np.asarray(points, dtype="float64")

    # Create level sets
    set_dict = {}
    for f in factors:
        set_points = scale_radius(points, f)
        set_dict[str(f)] = set_points

    return set_dict


def save_sets(set_dict, conn, mu=None, u=None, grad_u=None, output_folder="./"):

    # Loop through isosurfaces
    set_data = {}
    for factor, points in set_dict.items():

        # Format 
        points = np.asarray(points, dtype="float64")
        conn = np.asarray(conn, dtype="int64")

        # get point data
        point_data = create_point_data(points, conn, mu, u, grad_u)

        # save set data
        set_data[factor] = point_data

        # write to vtk
        writeVTK(output_folder + "set_" + str(factor), points, conn, point_data)

    return set_data


def create_point_data(points, conn, mu=None, u=None, grad_u=None):
    
    # Surface normals
    normals = get_surface_normals(points, conn)

    # Shear modulus
    if mu is not None:
        mu = np.array([mu(p) for p in points])

    if u is not None:
        u = np.array([u(p) for p in points])

    # Deformation Data
    if grad_u is not None:
        du, F, R, U, C = deformation_tensors(points, grad_u)

        # Normal Stretches
        stretches = get_stretches(normals, C)

    # Dataframe
    df = ArraystoDF(points, normals, mu, u, du, F, C, stretches)

    # Point data dictionary
    point_data = {}
    for column in df:
        skips = ['x', 'y', 'z', 'r']
        if column in skips: continue

        dat = np.ascontiguousarray(df[column],  dtype=np.float32)
        point_data[column] = dat

    return point_data

def deformation_tensors(points, grad_u):

    # Number of points
    npoints = np.size(points, 0)

    # Displacement Gradient
    grad_u = np.array([grad_u(p) for p in points])
    grad_u.resize((npoints,3,3))

    # Deformation Gradient
    I = np.eye(3)
    F = I + grad_u

    # Polar Decomposition
    R, U = [], []
    for f in F:
        r, u = polar(f)
        R.append(r)
        U.append(u)
    R = np.array(R)
    U = np.array(U)

    # Right Cauchy-Green Tensor
    C = np.matmul(F.transpose(0,2,1), F)   

    return grad_u, F, R, U, C

def get_stretches(u, C):
    npoints = np.size(u, 0)
    u = u.reshape(npoints, 3, 1)
    stretches = np.matmul(u.transpose(0, 2, 1), np.matmul(C, u)) ** 0.5
    return stretches.ravel()

def ArraystoDF(points, normals=None, mu=None, u=None, grad_u=None, F=None, C=None, stretches=None):

    data=pd.DataFrame()
    npoints = np.size(points, 0)

    x, y, z = np.hsplit(points, 3)
    r = np.sum((points-points[0])**2, axis=1)**0.5
    data["x"] = x.flatten()
    data["y"] = y.flatten()
    data["z"] = z.flatten()
    data["r"] = r.flatten()

    # Surface Normals
    if normals is not None:
        nx, ny, nz = np.hsplit(normals, 3)
        data["nx"] = nx.flatten()
        data["ny"] = ny.flatten()
        data["nz"] = nz.flatten()

    # Shear Modulus
    if mu is not None:
        data["mu"] = mu.flatten()

    # Displacement
    if u is not None:
        ux, uy, uz = np.hsplit(u, 3)
        umag = np.sqrt(ux**2 + uy**2 + uz**2)
        data["ux"] = ux.flatten()
        data["uy"] = uy.flatten()
        data["uz"] = uz.flatten()
        data["umag"] = umag.flatten()

    if normals is not None and u is not None:
        udot = dots(u, normals)
        data["udot"] = udot

    # Displacement Gradient
    if grad_u is not None:
        columns = ['g11','g12','g13','g21','g22','g23','g31','g32','g33']
        for col, dat in zip(columns, grad_u.reshape((npoints, 9)).T):
            data[col] = dat

    # Deformation Tensor        
    if F is not None:
        columns = ['F11','F12','F13','F21','F22','F23','F31','F32','F33']
        for col, dat in zip(columns, F.reshape((npoints,9)).T):
            data[col] = dat

    # Right Cauchy-Green Tensor
    if C is not None:
        columns = ['C11','C12','C13','C21','C22','C23','C31','C32','C33']    
        for col, dat in zip(columns, C.reshape((npoints,9)).T):
            data[col] = dat
        
        # Eigenvalues/eigenvectors
        w, v = eig(C)

        # Order by decreasing eigenvalue
        sort_ind = np.argsort(w, axis=-1)
        sort_ind = np.flip(sort_ind, axis=-1)
        w = np.take_along_axis(w, sort_ind,-1)
        w = np.sqrt(w)

        for i, mat in enumerate(v):
            mat = mat[:, sort_ind[i]]

        columns = ['w1', 'w2', 'w3']
        for col, dat in zip(columns, w.transpose()):
            data[col] = dat

        columns = ["v11","v12", "v13", "v21", "v22", "v23", "v31", "v32", "v33"]
        for col, dat in zip(columns, v.reshape((npoints,9), order="F").T):
            data[col] = dat

    # Stretch
    if stretches is not None:
        data['normalstretches'] = stretches

    return data

def writeVTK(fname, points, conn, pointData=None, cellData=None):

    # x,y,z
    x = np.ascontiguousarray(points[:,0], dtype="float64")
    y = np.ascontiguousarray(points[:,1], dtype="float64")
    z = np.ascontiguousarray(points[:,2], dtype="float64")

    # cell types
    ncells = np.size(conn, 0)
    ctype = np.zeros(ncells)
    ctype[:] = VtkTriangle.tid

    # ravel conenctivity
    conn_ravel = conn.ravel().astype("int64")

    # offset begins with 1
    offset = 3 * (np.arange(ncells, dtype='int64') + 1)    

    if pointData is not None and cellData is None:
        unstructuredGridToVTK(fname, x, y, z, connectivity=conn_ravel, offsets=offset, cell_types = ctype,  pointData=pointData)
    elif pointData is None and cellData is not None:
        unstructuredGridToVTK(fname, x, y, z, connectivity=conn_ravel, offsets=offset, cell_types = ctype,  cellData=cellData)    
    elif pointData is not None and cellData is not None:
        unstructuredGridToVTK(fname, x, y, z, connectivity=conn_ravel, offsets=offset, cell_types = ctype,  pointData=pointData, cellData=cellData)
    else:
        print("Missing Data")  
         
    return

## Level Set Plots ================================================================================

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Isosurface multiplier')

def plot_sets(factors, set_data, output_folder="./"):

    disp = [data["umag"] for data in set_data.values()]

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title('Displacement on Isosurfaces')
    ax.set_ylabel(r'Displacement ($\mu$m)')
    parts = ax.violinplot(dataset=disp,
            showmeans=False, showmedians=False,
            showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('xkcd:pale teal')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(disp, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(disp, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='dimgrey', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='dimgrey', linestyle='-', lw=1)

    labels = [str(f) for f in factors]
    set_axis_style(ax, labels)

    plt.savefig(output_folder + 'isoplots.png', bbox_inches='tight')

# Data over line ============================================================

def data_over_line(point, direction, inc, bound, mu=None, u=None, grad_u=None):

    point = np.array(point)
    direction = np.array(direction)
    points = point.reshape((1,3))

    # 1000 Maximum points
    for i in range(1000):
        nextpoint = points[-1] + direction*inc
        if abs(nextpoint[0])>=bound or abs(nextpoint[1])>=bound or abs(nextpoint[2])>=bound:
            break

        points = np.vstack((points, nextpoint)) 

    npoints = np.size(points, 0)

    if u is not None:
        u = np.array([u(p) for p in points])

    if mu is not None:
        mu = np.array([mu(p)*10**-12 for p in points])

    stretches=None
    if grad_u is not None:
        du, F, R, U, C = deformation_tensors(points, grad_u)

        v = np.broadcast_to(direction, (npoints,3))
        v = np.ascontiguousarray(v)
        stretches = get_stretches(v, C)

    data = ArraystoDF(points=points, u=u, mu=mu, grad_u=du, F=F, C=C, stretches=stretches)

    return data


# Misc =========================================================================================================

def tabulate3(u):
    u_arr = u.compute_vertex_values()  # 1-d numpy array
    length = np.shape(u_arr)[0]
    u_arr = np.reshape(u_arr, (length//3, 3), order="F") # Fortran ordering
    return u_arr

"""
# Alex's Deformation Gradient Method
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

        a = normalize(a) * 2
        b = normalize(b) * 2

        # create box
        p = points[i]
        box = np.zeros((8,3))

        box[0] = p + a + b
        box[1] = p + a - b
        box[2] = p - a + b
        box[3] = p - a - b
        box[4] = p + a + b + 4*n
        box[5] = p + a - b + 4*n
        box[6] = p - a + b + 4*n
        box[7] = p - a - b + 4*n
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

        a = normalize(a) * 2
        b = normalize(b) * 2

        # create box
        p = points[i]
        box = np.zeros((8,3))

        box[0] = p + a + b
        box[1] = p + a - b
        box[2] = p - a + b
        box[3] = p - a - b
        box[4] = p + a + b + 4*n
        box[5] = p + a - b + 4*n
        box[6] = p - a + b + 4*n
        box[7] = p - a - b + 4*n
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