import numpy as np
import pandas as pd
import os

directory = "./star_destroyer"

displacements_file = os.path.join(directory, "interpolated_NI_surface_displacements.csv")
displacements_NI = pd.read_csv(displacements_file, index_col=False).loc[:, 'u':'w']
displacements_NI = displacements_NI.to_numpy()
displacements_IN = -1* displacements_NI

inactive_vertices_file = os.path.join(directory, "cell_surface_inactive_vertices.txt")
inactive_vertices = np.loadtxt(inactive_vertices_file)

normal_vertices = inactive_vertices + displacements_IN
np.savetxt(os.path.join(directory, "cell_surface_normal_vertices.txt"), normal_vertices)



