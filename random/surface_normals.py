import meshio
import numpy as np
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

def normalize(a):
    ss = np.sum(a**2, axis=1)**0.5
    a = a / ss[:, np.newaxis]
    return a

mesh_path = "../meshes/ellipsoid/"
surf_mesh = meshio.read(mesh_path + "ellipsoid_surface.xdmf")
surf_vert = np.array(surf_mesh.points)
surf_conn = np.array(surf_mesh.cells[0].data, dtype="int64")

# Face coordinates
tris = surf_vert[surf_conn]

# Face normals
fnorm = np.cross( tris[:,1,:] - tris[:,0,:] , tris[:,2,:] - tris[:,0,:] )

# Normalize face normals
fnorm = normalize(fnorm)

# Vertex normal = sum of adjacent face normals
vnorm = np.zeros(surf_vert.shape)
vnorm[ surf_conn[:,0] ] += fnorm
vnorm[ surf_conn[:,1] ] += fnorm
vnorm[ surf_conn[:,2] ] += fnorm

# Normalize vertex normals
vnorm = normalize(vnorm)

# VTK
ncells = np.size(surf_conn,0)
conn = np.ravel(surf_conn)

x, y, z = np.split(surf_vert, 3, axis=1)
x=np.ascontiguousarray(x)
y=np.ascontiguousarray(y)
z=np.ascontiguousarray(z)

nx, ny, nz = np.split(fnorm, 3, axis=1)
nx = np.ravel(nx)
ny = np.ravel(ny)
nz = np.ravel(nz)

vx, vy, vz = np.split(vnorm, 3, axis=1)
vx = np.ravel(vx)
vy = np.ravel(vy)
vz = np.ravel(vz)

offset = 3 * (np.arange(ncells) + 1)

ctype = np.zeros(ncells)
ctype[:] = VtkTriangle.tid

unstructuredGridToVTK("face_normals", x, y, z, connectivity=conn.astype('float64'), offsets=offset.astype('float64'), cell_types = ctype,
                        cellData={"nx" : nx, "ny" : ny, "nz" : nz}, pointData={"nx" : vx, "ny" : vy, "nz" : vz})




