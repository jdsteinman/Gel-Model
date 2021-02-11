import meshio
import math
import pandas as pd
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import seaborn as sb
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle
from dolfin import Function as fenicsFunc

"""
Functions to create level sets, calculate surface normals,
plot stuff, and more.
"""
## Normalize an array of 3-component vectors
def normalize(a):
    ss = np.sum(a**2, axis=1)**0.5
    a = a / ss[:, np.newaxis]
    return a

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
def add_to_rad(inc, vertices):
    sets = {}
    for i in inc:
        polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
        polar_vert[:,0] += i
        cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
        sets[str(i)] = cart_vert
    return(sets)

## Convert to polar, multiply r*k, convert back to cartesian
def mult_rad(inc, vertices):
    sets = {}
    for i in inc:
        polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
        polar_vert[:,0] *= i
        cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
        sets[str(i)] = cart_vert
    return(sets)

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

## Level Set Plots
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

def plot_sets(disp, sets, output_folder):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title('Displacement on Isosurfaces')
    ax.set_ylabel(r'Displacement ($\mu$m)')
    parts = ax.violinplot(
            disp, showmeans=False, showmedians=False,
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

    labels = [str(s) for s in sets]
    set_axis_style(ax, labels)

    plt.savefig(output_folder + 'isoplots.png', bbox_inches='tight')

##Grand Finale
def level_sets(sets, vert, conn, u, output_folder="./"):

    # Format inputs
    vert = np.asarray(vert, dtype="float64")
    conn = np.asarray(conn, dtype="int64")

    # number of points/cells
    nvert = np.size(vert, 0)
    ncells = np.size(conn, 0)

    # Create level sets
    set_dict = mult_rad(sets, vert)
    u_sets = []

    for s in sets:
        points = np.array(set_dict[str(s)], dtype="float64")

        # x,y,z
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]

        # cell types
        ctype = np.zeros(ncells)
        ctype[:] = VtkTriangle.tid

        # ravel conenctivity
        conn_ravel = conn.ravel().astype("int64")

        # offset begins with 1
        offset = 3 * (np.arange(ncells, dtype='int64') + 1)

        # Data
        disp = np.array([u(p) for p in points])
        ux, uy, uz = disp[:,0], disp[:,1], disp[:,2]
        ux = np.ascontiguousarray(ux, dtype=np.float32)
        uy = np.ascontiguousarray(uy, dtype=np.float32)
        uz = np.ascontiguousarray(uz, dtype=np.float32)

        # magnitude
        u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
        u_sets.append(u_mag)

        # normals
        normals = get_surface_normals(points, conn)

        # dot product
        u_dot = dots(disp, normals)

        # signed magnitude
        s_mag = u_mag * np.abs(u_dot) / u_dot

        unstructuredGridToVTK(output_folder + "set_" + str(s), x, y, z, connectivity=conn_ravel, offsets=offset, cell_types = ctype, 
        pointData={"u_x" : ux, "u_y" : uy, "u_z" : uz, "u_mag" : u_mag, "u_dot" : u_dot, "u_mag_signed":s_mag})
    
    plot_sets(u_sets, sets, output_folder)

## Data Output
def toDataFrame(points, u=None, mu=None, grad_u=None, C=None):

    data=pd.DataFrame()
    npoints = np.size(points, 0)

    x, y, z = np.hsplit(points, 3)
    r = np.sum((points-points[0])**2, axis=1)**0.5
    data["x"] = x.flatten()
    data["y"] = y.flatten()
    data["z"] = z.flatten()
    data["r"] = r.flatten()

    if u is not None:
        u = np.array([u(p) for p in points])
        ux, uy, uz = np.hsplit(u, 3)
        mag = np.sqrt(ux**2 + uy**2 + uz**2)
        data["ux"] = ux.flatten()
        data["uy"] = uy.flatten()
        data["uz"] = uz.flatten()
        data["U_mag"] = mag.flatten()

    if mu is not None:
        mu = np.array([mu(p)*10**-12 for p in points])
        data["mu"] = mu.flatten()

    if grad_u is not None:
        grad_u = np.array([grad_u(p) for p in points])
        columns = ['g11','g12','g13','g21','g22','g23','g31','g32','g33']
        for col, dat in zip(columns, grad_u.T):
            data[col] = dat
            
        grad_u.resize((npoints,3,3))
        I = np.eye(3)
        F = I + grad_u
        columns = ['F11','F12','F13','F21','F22','F23','F31','F32','F33']
        for col, dat in zip(columns, F.reshape((npoints,9)).T):
            data[col] = dat

        columns = ['C11','C12','C13','C21','C22','C23','C31','C32','C33']
        C = np.matmul(F.transpose(0,2,1), F)      
        for col, dat in zip(columns, C.reshape((npoints,9)).T):
            data[col] = dat
        
        w, v = LA.eig(C)

        columns = ['xstretch', 'ystretch', 'zstretch']
        for col, dat in zip(columns, w.transpose()):
            data[col] = dat

    return data
        
def tabulate3(u):
    u_arr = u.compute_vertex_values()  # 1-d numpy array
    length = np.shape(u_arr)[0]
    u_arr = np.reshape(u_arr, (length//3, 3), order="F") # Fortran ordering
    return u_arr