import meshio
import os
import numpy as np

"""
Input:
    points: array
    faces:  array
Output:
    surface stl mesh
"""

# Directories
# dir_in = "../star_destroyer/IN/meshes"
dir_in = "../star_destroyer"
dir_out = "../star_destroyer/IN/meshes"

# Load Data
surface_points = np.loadtxt(os.path.join(dir_in, "CytoD_vertices.txt"))
surface_faces = np.loadtxt(os.path.join(dir_in, "CytoD_faces.txt"), dtype=int)

# Mesh
out_filename = os.path.join(dir_out, "cell_surface.stl")
meshio.write(out_filename, meshio.Mesh(
                points=surface_points,
                cells=[("triangle", surface_faces)]))