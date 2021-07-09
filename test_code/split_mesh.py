import meshio
import numpy as np

"""
Separates triangle and tetrahadral elements of a .msh file.
Saves each set of elements in separate mesh.xdmf file.
"""

path = "./mesh/"
filename = "mesh_gradient"
msh = meshio.read(path + filename + ".msh")

# Get triangle and tet connectivity
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

# Get physical labels
for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "tetra":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]

# Separate meshes
try:
    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
    meshio.write(path + filename + "_tetra.xdmf", tetra_mesh)
except:
    pass

try:
    triangle_mesh =meshio.Mesh(points=msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"triangle":[triangle_data]})
    meshio.write(path + filename +  "_triangle.xdmf", triangle_mesh)
except:
    pass


