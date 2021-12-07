import meshio
import os
import numpy as np

"""
Write stl mesh of cell surface

Input:
    points: array
    faces:  array
Output:
    surface stl mesh
"""

# Directories
dir_in = "../triangle"
dir_out = "../triangle/IN/meshes"

# Load Data
surface_points = np.loadtxt(os.path.join(dir_in, "CytoD_vertices.txt"))
surface_faces = np.loadtxt(os.path.join(dir_in, "CytoD_faces.txt"), dtype=int)

# Mesh
out_filename = os.path.join(dir_out, "cell_surface.stl")
meshio.write(out_filename, meshio.Mesh(
                points=surface_points,
                cells=[("triangle", surface_faces)]))