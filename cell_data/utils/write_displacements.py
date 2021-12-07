import numpy as np
import pandas as pd
import os

"""
Get displacements from interpolated data
"""

input_filename = "../triangle/interpolated_NI_surface_coarse_data.csv"
output_filename = "../triangle/NI/displacements/surface_displacements_coarse.txt"

dat = pd.read_csv(input_filename, index_col=False)
disp = dat.loc[:, 'u':'w']
disp = disp.to_numpy()
np.savetxt(output_filename, disp)