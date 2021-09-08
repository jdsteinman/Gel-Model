import h5py 
import os
import numpy as np
from deformation import PointData

def main(folder):
    # Verties and displacement
    f = h5py.File(os.path.join(folder, "U.h5"), 'r')
    x = f['Mesh']['mesh']['geometry'][()]
    x = np.array(x)

    conn = f['Mesh']['mesh']['topology'][()]
    print(conn)

    u = f['VisualisationVector']['0'][()]
    u = np.array(u)

    # Deformation Gradient
    f = h5py.File(os.path.join(folder, "F.h5"), 'r')
    F = f['VisualisationVector']['0'][()]
    F = np.array(F).reshape((-1,3,3))

    point_data = PointData(x, conn, u, F)
    point_data.save_to_vtk(os.path.join(folder, "deformation"))



if __name__=="__main__":
    main("output/corners/150")