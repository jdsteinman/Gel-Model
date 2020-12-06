import meshio
import numpy as np

path = "./"
out_path = "./"
filename = "cytod_uncentered_unpca"
cell_surface_num = 200
box_surface_num = 201
volume_num = 300

# Read mesh
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
tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})

triangle_mesh =meshio.Mesh(points=msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"triangle":[triangle_data]})

# write to xdmf
meshio.write(path + filename + "_tetra.xdmf", tetra_mesh)
meshio.write(path + filename +  "_triangle.xdmf", triangle_mesh)

# Get cell surface cells
surf_cells = np.column_stack((triangle_cells, triangle_data)) 
surf_cells = surf_cells[surf_cells[:,-1] == cell_surface_num]
surf_cells = surf_cells[:,0:-1]

cell_nodes = np.unique(surf_cells)
np.savetxt("surface_nodes_check.txt", cell_nodes, fmt = "%d", delimiter=" ")

# Get Nodes
nodes = msh.points 
np.savetxt(out_path + "gel_vertices.txt", nodes, delimiter=" ")
# nodes_new = []
# surf_cells_new = surf_cells
# vert_map = {}
# num_vert = 0

# for i, face in enumerate(surf_cells_new):
#     for j, vert in enumerate(face):

#         if vert in vert_map:
#             surf_cells_new[i][j] = vert_map[vert]
#         else:
#             vert_map[vert] = num_vert          # add key to map
#             surf_cells_new[i][j] = num_vert    
#             nodes_new.append(nodes[int(vert)]) # add node
#             num_vert += 1

# nodes_new = np.array(nodes_new)

# np.savetxt(out_path + "surf_cells.txt", surf_cells, delimiter=" ", fmt='%d')
# np.savetxt(out_path + "nodes.txt", nodes, delimiter=" ")

# np.savetxt(out_path + "surf_cells_new.txt", surf_cells_new, delimiter=" ", fmt='%d')
# np.savetxt(out_path + "nodes_new.txt", nodes_new, delimiter=" ")



