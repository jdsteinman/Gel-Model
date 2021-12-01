import numpy as np
import pandas as pd
"""
Extract surface vertices and displacements
"""
path = "new_cell/"
# file = "isochoric_surface_data_2_um_gridsize.csv"
file = "interpolated_data.csv"
dat = pd.read_csv(path + file)

vert = dat.loc[:,'X':'Z']
vert.to_csv(path + "surface_1000_vertices.txt", sep=" ", index=False, header=False)

disp=dat.loc[:, 'u':'w']
disp.to_csv(path + "surface_1000_displacements.txt", sep=" ", index=False, header=False)

# faces = np.loadtxt(path+"CytoD_faces.txt")
# faces = faces-1
# np.savetxt(path+"CytoD_faces.txt", faces, fmt="%d")