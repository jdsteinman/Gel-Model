import numpy as np
import pandas as pd
import os

"""
Get displacements from interpolated data
"""

input_filename = "../finger/interpolated_NI_surface_data.csv"
output_filename = "../finger/NI/displacements/surface_displacements.txt"

dat = pd.read_csv(input_filename, index_col=False)
disp = dat.loc[:, 'u':'w']
disp = disp.to_numpy()
np.savetxt(output_filename, disp)
