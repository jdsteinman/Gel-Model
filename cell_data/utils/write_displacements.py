import numpy as np
import pandas as pd
import os

"""
Get displacements from interpolated data
"""

idir  = "../star_destroyer/"
ifile = "interpolated_NI_surface_data_coarse.csv"
odir  = "../star_destroyer/IN/displacements"
ofile = "surface_displacements_IN_coarse.txt"
input_filename = os.path.join(idir, ifile)
output_filename = os.path.join(odir, ofile)

dat = pd.read_csv(input_filename, index_col=False)
disp = -1*dat.loc[:, 'u':'w']
disp = disp.to_numpy()
np.savetxt(output_filename, disp)
