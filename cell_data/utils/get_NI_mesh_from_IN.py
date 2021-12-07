import numpy as np
import pandas as pd
import meshio
import os

# Files
directory = "../triangle"
dir_in = os.path.join(directory, "IN/meshes")
dir_out = os.path.join(directory, "NI/meshes")
filename = "cell_surface_coarse.stl"

# Displacements
displacements_file = os.path.join(directory, "interpolated_NI_surface_coarse_data.csv")
displacements_NI = pd.read_csv(displacements_file, index_col=False).loc[:, 'u':'w'].to_numpy()
displacements_IN = -1*displacements_NI

# Mesh
mesh_filename = os.path.join(dir_in, filename)
mesh = meshio.read(mesh_filename)
surface_points = mesh.points + displacements_IN

for cell in mesh.cells:
    if cell.type == "triangle":
        surface_faces = cell.data

# Write
out_filename = os.path.join(dir_out, filename)
meshio.write(out_filename, meshio.Mesh(
                points=surface_points,
                cells=[("triangle", surface_faces)]))


