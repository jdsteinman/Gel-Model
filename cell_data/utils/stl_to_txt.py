import meshio
import os
import numpy as np

"""
Extract points and faces from cell surface mesh
"""

directories = ['../triangle', '../bird', '../star_destroyer']
subdirectories = ['IN/meshes', 'NI/meshes']

for directory in directories:
    for subdirectory in subdirectories:
        path = os.path.join(directory, subdirectory)
        for file in os.listdir(path):
            if file.endswith(".stl"):
                    input_filename = os.path.join(path, file)    
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
