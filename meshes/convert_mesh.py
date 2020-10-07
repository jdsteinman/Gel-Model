import meshio
import numpy as np

filename = "/home/john/research/meshes/sphere_in_cube"
#filename = "/home/john/research/meshes/unit_cube"

msh = meshio.read(filename+".msh")
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

tetra = False
triangle = False

for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
        triangle = True
    elif key == "tetra":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]
        tetra = True

if tetra:
    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})

if triangle:
    triangle_mesh =meshio.Mesh(points=msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"name_to_read":[triangle_data]})

meshio.write(filename+".xdmf", tetra_mesh)
