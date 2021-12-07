import h5py
import os
import argparse
import numpy as np
import pandas as pd
import meshio

def main(folder):

    # Mesh
    mesh = meshio.read("../cell_meshes/bird/hole_coarse.xdmf")

    # Simulated Data
    u_file = h5py.File(os.path.join(folder, "u.h5"), 'r')
    u_sim = u_file['VisualisationVector']['0'][()]
    u_sim = np.array(u_sim)
    u_sim_mag = np.sum(u_sim**2, axis=1)**0.5

    # Experimental data
    u_data = pd.read_csv("./output/single_field/coarse/interpolated_data.csv", index_col=False)
    u_data = -1*u_data.loc[:,'u':'w'].to_numpy()
    u_data_mag = np.sum(u_data**2, axis=1)**0.5

    # Degradation
    degradation = np.zeros(u_sim.shape[0])
    discrepancy = u_data_mag-u_sim_mag
    max_d = np.max(discrepancy)
    # print(max_d)
    # degradation = discrepancy * 0.999/4

    Dmax = 0.9
    degradation = np.zeros(u_sim.shape[0])
    discrepancy = u_data_mag-u_sim_mag
    pos_ind = np.where(discrepancy>=0)[0]   # underestimation
    dpos = discrepancy[pos_ind]
    neg_ind = np.where(discrepancy<0)[0]    # overestimation
    dneg = discrepancy[neg_ind]

    degradation[pos_ind] = dpos * -0.5/5
    print(degradation)
    # degradation[neg_ind] = dneg_scaled
    degradation[neg_ind] = 0

    mesh_out = meshio.Mesh(
        mesh.points,
        mesh.cells,
        point_data = {"degradation":degradation}
    )

    mesh_out.write(
        os.path.join(folder, "degradation.xdmf")
    )

    np.savetxt(os.path.join(folder, "degradation.txt"), degradation)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Postprocessing")
    parser.add_argument('directory', metavar='d', type=str,
                         help="directory containing outputfiles")

    args = parser.parse_args()
    main(args.directory)
