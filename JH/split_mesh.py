import meshio
import numpy as np

"""
Separates triangle and tetrahadral elements of a .msh file.
Saves each set of elements in separate mesh.xdmf file.
"""

path = "./plate_with_circular_hole/"
filename = "very_fine"
msh = meshio.read(path + filename + ".msh")

# Get triangle and tet connectivity
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "line":
        line_cells = cell.data

# Get physical labels
for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "line":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]

# Separate meshes
triangle_mesh = meshio.Mesh(points=msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"triangle":[triangle_data]})

line_mesh = meshio.Mesh(points=msh.points, cells={"line": line_cells})

# write to xdmf
meshio.write(path + filename + "_triangle.xdmf", triangle_mesh)
meshio.write(path + filename +  "_line.xdmf", line_mesh)



