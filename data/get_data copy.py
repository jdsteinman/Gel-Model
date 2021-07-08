import numpy as np
import pandas as pd

dat = pd.read_csv("isochoric_surface_data_2_um_gridsize.csv")

vert = dat.loc[:,'X':'Z']
vert.to_csv("CytoD_vertices.txt", sep=" ", index=False, header=False)

disp=dat.loc[:, 'u':'w']
disp.to_csv("displacements.txt", sep=" ", index=False, header=False)