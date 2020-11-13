import meshio
import numpy as np

filename = "ellipsoidw2pods"

msh = meshio.read(filename+".msh")
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "tetra":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]

tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})

triangle_mesh =meshio.Mesh(points=msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"triangle":[triangle_data]})

meshio.write("TwoPseudopodsTetra.xdmf", tetra_mesh)
meshio.write("TwoPseudopodsTriangle.xdmf", triangle_mesh)

