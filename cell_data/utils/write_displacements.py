import numpy as np
import pandas as pd
import os

"""
Get displacements from interpolated data
"""

cell = "triangle"
input_filename = "../"+cell+"/interpolated_IN_surface_data_coarse.csv"
output_filename = "../"+cell+"/NI/displacements/surface_displacements_coarse.txt"

dat = pd.read_csv(input_filename, index_col=False)
disp = -1*dat.loc[:, 'u':'w']
disp = disp.to_numpy()
np.savetxt(output_filename, disp)
