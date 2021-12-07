import meshio
import os
import numpy as np

directory = "../triangle/NI/meshes/"
file = "hole_coarse.xdmf"

input_filename = os.path.join(directory, file)
print("Converting %s" % input_filename)
output_filename = os.path.splitext(input_filename)[0]

# Read Mesh
mesh = meshio.read(input_filename)

# Vertices
points = mesh.points 
np.savetxt(output_filename + "_vertices.txt", points)





