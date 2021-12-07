import meshio
import os
import numpy as np

"""
Extract points and faces from cell surface mesh
"""

directories = ['../bird/IN/meshes']

for directory in directories:
    for file in os.listdir(directory):
        if file.endswith(".stl"):
                input_filename = os.path.join(directory, file)    
                print("Converting %s" % input_filename)
                output_filename = os.path.splitext(input_filename)[0]

                # Read stl
                mesh = meshio.read(input_filename)
                
                # Save Vertices
                points = mesh.points 
                np.savetxt(output_filename + "_vertices.txt", points)

                # Save Faces
                for cell in mesh.cells:
                    if cell.type == "triangle":
                        triangle_cells = cell.data
                        
                np.savetxt(output_filename + "_faces.txt", triangle_cells, fmt="%d")
