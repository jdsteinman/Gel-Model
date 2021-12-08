import meshio
import os
import numpy as np

# directories = ['../triangle', '../bird', '../star_destroyer']
# subdirectories = ['IN/meshes', 'NI/meshes']

directories = ['../triangle']
subdirectories = ['NI/meshes']

for directory in directories:
    for subdirectory in subdirectories:
        path = os.path.join(directory, subdirectory)
        for file in os.listdir(path):
            if file.endswith(".msh"):
                input_filename = os.path.join(path, file)
                print("Converting %s" % input_filename)
                output_filename = os.path.splitext(input_filename)[0]

                mesh = meshio.read(input_filename)
                points = mesh.points

                for cell in mesh.cells:
                    if cell.type == "triangle":
                        triangle_cells = cell.data
                    elif  cell.type == "tetra":
                        tetra_cells = cell.data

                # Get physical labels
                for key in mesh.cell_data_dict["gmsh:physical"].keys():
                    if key == "triangle":
                        triangle_data = mesh.cell_data_dict["gmsh:physical"][key]
                    elif key == "tetra":
                        tetra_data = mesh.cell_data_dict["gmsh:physical"][key]

                # Separate meshes
                try:
                    print("\tWriting mesh")
                    meshio.write(output_filename + ".xdmf", meshio.Mesh(points=points, cells={"tetra": tetra_cells}))
                    print("\tWriting domains")
                    meshio.write(output_filename + "_domains.xdmf", meshio.Mesh(points=points, 
                                cells={"tetra": tetra_cells},
                                cell_data={"domains":[tetra_data]}))
                except Exception as e: 
                    print("\t", e)
                    quit()

                try:
                    print("\tWriting boundaries")
                    triangle_mesh = meshio.Mesh(points=points,
                                        cells=[("triangle", triangle_cells)],
                                        cell_data={"boundaries":[triangle_data]})
                    meshio.write(output_filename +  "_boundaries.xdmf", triangle_mesh)
                except Exception as e: 
                    print("\t",e)
                    quit()
