import h5py 
import os
import argparse
import numpy as np
from post_tools import PointData

def main(folder):
    # Load data
    f = h5py.File(os.path.join(folder, "U.h5"), 'r')
    x = f['Mesh']['mesh']['geometry'][()]
    x = np.array(x)

    conn = f['Mesh']['mesh']['topology'][()]

    u = f['VisualisationVector']['0'][()]
    u = np.array(u)

    f = h5py.File(os.path.join(folder, "F.h5"), 'r')
    F = f['VisualisationVector']['0'][()]
    F = np.array(F).reshape((-1,3,3))

    f = h5py.File(os.path.join(folder, "mu.h5"), 'r')
    mu = f['VisualisationVector']['0'][()]
    mu = np.array(mu)*10**6 # Pa)

    point_data = PointData(x, conn, u, F, mu)
    point_data.save_to_vtk(os.path.join(folder, "sim_output.vtk"))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Postprocessing")
    parser.add_argument('directory', metavar='d', type=str,
                        help="directory containing outputfiles")

    args = parser.parse_args()
    main(args.directory)