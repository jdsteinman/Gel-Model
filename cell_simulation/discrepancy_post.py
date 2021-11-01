import h5py
import os
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from post_tools import get_surface_normals
import matplotlib.pyplot as plt
def main(idx, L):
    """
    idk: surface node for normal
    """

    folder = "./output/bird/single_field/fixed/"

    # Surface vertices
    surface_points = np.loadtxt("../cell_meshes/bird/cell_surface_vertices.txt")
    surface_faces  = np.loadtxt("../cell_meshes/bird/cell_surface_faces.txt")
    surface_normals = get_surface_normals(surface_points, surface_faces)

    direction = surface_normals[idx]
    p = surface_points[idx]
    step = 0.3
    max_iter = 1000
    centroid = np.array([72.20045715,72.90093189,47.46392168])
    normal_line = []
    for i in range(max_iter):
        p = p + step*direction

        p_centered = p-centroid
        if abs(p_centered[0])>=L/2 or abs(p_centered[1])>=L/2 or abs(p_centered[2])>L/2:
            break
        normal_line.append(p)
    normal_line = np.array(normal_line)
    X,Y,Z = normal_line.T

    # Mesh points, connectivity
    f = h5py.File(os.path.join(folder, "u.h5"), 'r')
    mesh_points = f['Mesh']['mesh']['geometry'][()]
    mesh_points = np.array(mesh_points)
    x,y,z = mesh_points.T

    # u sim
    u = f['VisualisationVector']['0'][()]
    u = np.array(u)

    sim_x = griddata((x,y,z), u[:,0], (X,Y,Z))
    sim_y = griddata((x,y,z), u[:,1], (X,Y,Z))
    sim_z = griddata((x,y,z), u[:,2], (X,Y,Z))

    u_sim_line = np.array((sim_x, sim_y, sim_z)).T

    # u data
    u_data = pd.read_csv("./output/bird/single_field/interpolated_data_fine.csv", index_col=False)
    u_data = -1*u_data.loc[:,'u':'w'].to_numpy()

    data_x = griddata((x,y,z), u_data[:,0], (X,Y,Z))
    data_y = griddata((x,y,z), u_data[:,1], (X,Y,Z))
    data_z = griddata((x,y,z), u_data[:,2], (X,Y,Z))

    u_data_line = np.array((data_x, data_y, data_z)).T

    res = u_data_line - u_sim_line
    res_mag = np.sum(res**2, axis=1)**0.5
    
    fig, ax = plt.subplots(1,1)
    r = np.sum((normal_line-surface_points[idx])**2, axis=1)**0.5
    ax.plot(r, res_mag)
    plt.show()


if __name__=="__main__":
    main(251, 200)