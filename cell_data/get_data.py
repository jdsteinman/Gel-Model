import numpy as np
import pandas as pd
"""
Extract surface vertices and displacements
"""
path = "star_destroyer/"
# file = "isochoric_surface_data_2_um_gridsize.csv"
file = "interpolated_data.csv"
dat = pd.read_csv(path + file, index_col=False)

disp=-1*dat.loc[:, 'u':'w']
disp.to_csv(path + "cell_surface_1000_displacements.txt", sep=" ", index=False, header=False)

# faces = np.loadtxt(path+"CytoD_faces.txt")
# faces = faces-1
# np.savetxt(path+"CytoD_faces.txt", faces, fmt="%d")