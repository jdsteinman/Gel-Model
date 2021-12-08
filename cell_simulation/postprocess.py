import h5py
import os
import argparse
import numpy as np
import pandas as pd
from post_tools import UnstructuredData


cell = "claw"
directory = "./output/" + cell + "/homogeneous"

# Surface vertices
surface_points = np.loadtxt("../cell_data/"+cell+"/NI/meshes/cell_surface_coarse_vertices.txt")

# Mesh points, connectivity
f = h5py.File(os.path.join(directory, "u.h5"), 'r')
points = f['Mesh']['mesh']['geometry'][()]
points = np.array(points)

conn = f['Mesh']['mesh']['topology'][()]

# u sim
u = f['VisualisationVector']['0'][()]
u = np.array(u)

# u data
u_data = pd.read_csv("../cell_data/"+cell+"/NI/displacements/interpolated_data_coarse.csv", index_col=False)
u_data = u_data.loc[:,'u':'w'].to_numpy()

f = h5py.File(os.path.join(directory, "F.h5"), 'r')
F = f['VisualisationVector']['0'][()]
F = np.array(F).reshape((-1,3,3))

if os.path.exists(os.path.join(directory, "mu.h5")):
    f = h5py.File(os.path.join(directory, "mu.h5"), 'r')
    mu = f['VisualisationVector']['0'][()]
    mu = np.array(mu)*10**6 # Pa)
else:
    mu = np.zeros((points.shape[0]))+100

data = UnstructuredData(surface_points, points, conn, u, u_data, F, mu)
data.save_to_vtk(os.path.join(directory, "displacements.vtk"))
