import h5py
import os
import argparse
import numpy as np
import pandas as pd
from post_tools import UnstructuredData

def main(folder):
    # Surface vertices
    surface_points = np.loadtxt("../cell_meshes/bird/cell_surface_vertices.txt")

    # Mesh points, connectivity
    f = h5py.File(os.path.join(folder, "u.h5"), 'r')
    points = f['Mesh']['mesh']['geometry'][()]
    points = np.array(points)

    conn = f['Mesh']['mesh']['topology'][()]

    # u sim
    u = f['VisualisationVector']['0'][()]
    u = np.array(u)

    # u data
    u_data = pd.read_csv("./output/single_field/coarse/interpolated_data.csv", index_col=False)
    u_data = -1*u_data.loc[:,'u':'w'].to_numpy()

    f = h5py.File(os.path.join(folder, "F.h5"), 'r')
    F = f['VisualisationVector']['0'][()]
    F = np.array(F).reshape((-1,3,3))

    if os.path.exists(os.path.join(folder, "mu.h5")):
        f = h5py.File(os.path.join(folder, "mu.h5"), 'r')
        mu = f['VisualisationVector']['0'][()]
        mu = np.array(mu)*10**6 # Pa)
    else:
        mu = np.zeros((points.shape[0]))+100

    data = UnstructuredData(surface_points, points, conn, u, u_data, F, mu)
    data.save_to_vtk(os.path.join(folder, "displacements.vtk"))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Postprocessing")
    parser.add_argument('directory', metavar='d', type=str,
                         help="directory containing outputfiles")

    args = parser.parse_args()
    main(args.directory)
