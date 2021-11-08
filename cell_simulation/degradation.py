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

    # Experimental data
    u_data = pd.read_csv("./output/single_field/coarse/interpolated_data.csv", index_col=False)
    u_data = -1*u_data.loc[:,'u':'w'].to_numpy()

    # Degradation
    Dmax = 0.5
    degradation = np.zeros(u_sim.shape[0])
    discepancy = u_data-u_sim
    dpos = discepancy[discepancy>0]
    dneg = discepancy[discepancy<0]
    pos_ind = np.where(discepancy>0)
    neg_ind = np.where(discepancy<0)
    dpos_scaled = (dpos-np.min(dpos))/(np.max(dpos)-np.min(dpos)) * Dmax
    dneg_scaled = (dneg-np.min(dneg))/(np.max(dneg)-np.min(dneg)) * -Dmax
    degradation[pos_ind] = dpos_scaled
    degradation[neg_ind] = dneg_scaled

    meshio.write(os.path.join(folder, "degradation.xdmf"),
        points=mesh.points,
        cells=mesh.cells,
        point_data=degradation
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Postprocessing")
    parser.add_argument('directory', metavar='d', type=str,
                         help="directory containing outputfiles")

    args = parser.parse_args()
    main(args.directory)
