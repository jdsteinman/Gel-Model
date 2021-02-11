import os
import post_tools as pt
import meshio 
import numpy as np
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

output_folder = "./test/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

path = "../meshes/ellipsoid/ellipsoid_surface.xdmf"
mesh = meshio.read(path)
vert = np.array(mesh.points)
conn = np.array(mesh.cells[0].data)

# x,y,z
x, y, z = np.hsplit(vert, 3)
x = np.ascontiguousarray(x, dtype=np.float32)
y = np.ascontiguousarray(y, dtype=np.float32)
z = np.ascontiguousarray(z, dtype=np.float32)

# cell types
ncells = np.size(conn, 0)
ctype = np.zeros(ncells)
ctype[:] = VtkTriangle.tid

# ravel conenctivity
conn_ravel = conn.ravel().astype("int64")

# offset begins with 1
offset = 3 * (np.arange(ncells, dtype='int64') + 1)

# normals
normals = pt.get_surface_normals(vert, conn)
nx, ny, nz = np.hsplit(normals, 3)

nx = np.ascontiguousarray(nx, dtype=np.float32)
ny = np.ascontiguousarray(ny, dtype=np.float32)
nz = np.ascontiguousarray(nz, dtype=np.float32)

rand = np.random.rand(np.size(vert, 0))

# VTK write
unstructuredGridToVTK(output_folder + "normals", x, y, z, connectivity=conn_ravel, offsets=offset, cell_types = ctype,
                        # pointData={"rand":rand})
                        pointData={"nx":nx.flatten(), "ny":ny.flatten(), "nz":nz.flatten()})