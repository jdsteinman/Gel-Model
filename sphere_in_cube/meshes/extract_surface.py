import meshio
import numpy as np

"""
Extracts surface from a cell-in-bounding-box mesh.
Outputs:
    - surface mesh as .xdmf file
    - vertices/connectivity as .txt files
"""

path = "./"
filename = "hole_2"
physical_num = 202  # physical number marking surface

# Read mesh
msh = meshio.read(path + filename + ".msh")

# Get triangle and tet connectivity
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data  # 2D connectivity

    elif  cell.type == "tetra":
        tetra_cells = cell.data     # 3D connectivity

# Get physical labels
for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "tetra":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]

# Get surface cells
surf_cells = np.column_stack((triangle_cells, triangle_data)) 
surf_cells = surf_cells[surf_cells[:,-1] == physical_num]    # Extract cells on surface
surf_cells = surf_cells[:,0:-1]

# Get Nodes
vertices = msh.points # all nodes
surf_vert = []
vert_map = {}      # maps old nodes to new nodes
num_vert = 0

# Rewrite connectivity
for i, face in enumerate(surf_cells):
    for j, vert in enumerate(face):

        if vert in vert_map:
            surf_cells[i][j] = vert_map[vert]
        else:
            vert_map[vert] = num_vert  # add key to map
            surf_cells[i][j] = num_vert
            surf_vert.append(vertices[int(vert)]) # add vert
            num_vert += 1

surf_vert = np.array(surf_vert)

# Save mesh
surface_mesh = meshio.Mesh(points=surf_vert, cells=[("triangle", surf_cells)])
meshio.write(path + filename +  "_surface.xdmf", surface_mesh)

# txt output
np.savetxt(path + "nodes.txt", vertices, delimiter=" ")
np.savetxt(path + "surface_nodes.txt", surf_vert, delimiter=" ")
np.savetxt(path + "surface_faces.txt", surf_cells, delimiter=" ", fmt='%d')