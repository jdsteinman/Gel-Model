import numpy as np
import pandas as pd
import os

"""
Get displacements from interpolated data
"""

input_filename = "../bird/interpolated_NI_surface_coarse_data.csv"
output_filename = "../bird/IN/displacements/surface_displacements_IN_coarse.txt"

dat = pd.read_csv(input_filename, index_col=False)
disp = -1*dat.loc[:, 'u':'w']
disp = disp.to_numpy()
np.savetxt(output_filename, disp)
