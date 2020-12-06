import meshio
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd
import sys
import time
import json
from dolfin import *
import os

"""
Extremely messy first-attempt at BC-simulation with Alex's mapping

TODO:
- try-except block to save solution from last valid solver_call
- resolutions shouldn't be hardcoded (merge w/ pre_volume.py)
- better i/o
- better documentation
- switch from face bc's to vertex bc's

How to run:
- cd ./code
- Type in the terminal: nohup python3 bc_sim_xf.py 100 &
- After the code finishes, one shall find a result folder ./output/casename
- Load solution.pvd in ParaView to visualize the field. 

"""

## Define objects and functions
class bc_nw(UserExpression):
    def __init__(self, mesh, cell2trans_dict, **kwargs):
        self.mesh = mesh  # input mesh
        self._cell2trans_dict = cell2trans_dict  # dict does what?
        self.cell_record = []
        self.x_record = []
        self.error_log = []
        super().__init__(**kwargs)

    def value_shape(self):
        return (3,)

    def eval_cell(self, value, x, cell):
        try:
            value[0], value[1], value[2] = self._cell2trans_dict[cell.index]
        except KeyError:
            value[0], value[1], value[2] = (0, 0, 0)
            self.error_log.append(cell)
        self.cell_record.append(cell)
        self.x_record.append(x)


class Surface(SubDomain):
    # Creates a class representing Surface
    def init_record(self):
        self.x_record = []

    def inside(self, x, on_boundary):
        self.x_record.append(x)
        return on_boundary


def create_surf_midpoints(surf_mesh):
    cell_dict = dict(surf_mesh.cells)
    midpoints = np.zeros(cell_dict['triangle'].shape)
    for idx, triangle in enumerate(cell_dict['triangle']):
        midpoints[idx] = surf_mesh.points[cell_dict['triangle'][idx]].mean(0)
    return midpoints


def solver_call(u, du, bcs):
    d = len(u)
    I = Identity(d)
    F = I + grad(u)
    B = Constant((0.0, 0.0, 0.0))
    T = Constant((0.0, 0.0, 0.0))
    C = F.T * F
    Ic = tr(C)
    J = det(F)

    # Material properties
    # shr0, nu0 = 1.0, 0.45
    # mu = 3
    # lmbda = 1

    mu = 325
    nu = 0.49

    E = 2*mu*(1+nu)
    lmbda = E*nu/((1 + nu)*(1 - 2*nu))

    # Math
    psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2
    Pi = psi * dx - dot(B, u) * dx - dot(T, u) * ds
    F = derivative(Pi, u, v)
    J = derivative(F, u, du)

    ## Solving
    solve(F == 0, u, bcs, J=J, solver_parameters={"newton_solver": {"linear_solver": "mumps"}})

    return u, du


## Import data
date_str = "06092019"
gel_str = "1"
path = "../data/" + date_str + "_G" + gel_str + "/"
chunks = int(100) # How many steps to you want to break the prescribed displacement into? # what does this actually do?
output_folder = "output/" + date_str + "_G" + gel_str + "_uncentered_unpca"
cytod_surf = meshio.read(path + "cytod_uncentered_unpca" + ".msh")
cytod_faces = cytod_surf.cells[0].data
cytod_vol = meshio.read(path + "cytod_uncentered_unpca_vol_r5" + ".msh")

# with open("Data/" + input_folder + "/metadata.json", "r") as j_file:
#     meta_dict = json.load(j_file)

mesh = Mesh()
with XDMFFile(path + "cytod_uncentered_unpca_vol_r5" + ".xdmf") as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, 2)
# with XDMFFile(path + "cytod_uncentered_unpca_vol_r5" + "_function.xdmf") as infile:
#      infile.read(mvc)

# Create derivative data
subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc)
subdomains.set_all(0)

# x_bound, y_bound, z_bound = meta_dict["Image bounds (um)"]
# x_pixels, y_pixels, z_pixels = meta_dict["Image size (pixels)"]
# x_res = x_bound/x_pixels
# y_res = y_bound/y_pixels
# z_res = z_bound/z_pixels
x_bound = 149.9457
y_bound = 149.9457
z_bound = 120
x_res = 0.29286
y_res = 0.29286
z_res = 0.8

surf_mesh1_midpoints = create_surf_midpoints(cytod_surf)
#TODO: Why is this labeled Vertical Displacement?
vert_disp = pd.read_csv(path + "displacements_cytod_to_normal_uncentered_unpca.csv",
                        header=None).values
#TODO: Currently dividing each element by 100?
vert_disp = vert_disp/chunks
midpoint_disp = np.zeros((cytod_faces.shape[0], 3)) # create an array #faces by 3

for idx, face in enumerate(cytod_faces):
    # Assign average displacement to tetra midpoint
    midpoint_disp[idx, :] = np.mean((vert_disp[face[0]],
                                     vert_disp[face[1]],
                                     vert_disp[face[2]]), axis=0)

#TODO: What boundary is this Marking?                                     
mesh_boundaries = np.vstack((cytod_vol.points.min(0), cytod_vol.points.max(0))).T

## Mark surface facets of mesh (2D)
domains = MeshFunction("size_t", mesh, 2)
surf = Surface()
surf.init_record()
surf.mark(domains, 1) # mark all surfaces 1

## Revert domain marking of OUTER boundary of gel && create cell_idx -> transformation dict
change_log = []
cell_idx_list = np.zeros(midpoint_disp.shape[0])
for index, face in enumerate(faces(mesh)):

    x, y, z = face.midpoint().array() # get location of midpoint
    if domains.array()[index] == 1:
        # If face is on outer boundary, change MeshFunction to 0
        if np.isclose(x, mesh_boundaries[0, 0], atol=1) or np.isclose(x, mesh_boundaries[0, 1], atol=1):
            domains.array()[index] = 0
            change_log.append(index)
        elif np.isclose(y, mesh_boundaries[1, 0], atol=1) or np.isclose(y, mesh_boundaries[1, 1], atol=1):
            domains.array()[index] = 0
            change_log.append(index)
        elif np.isclose(z, mesh_boundaries[2, 0], atol=1) or np.isclose(z, mesh_boundaries[2, 1], atol=1):
            domains.array()[index] = 0
            change_log.append(index)
        # If face is on cell surface, add
        else:
            dist_mat = distance_matrix(np.array([[x, y, z]]), surf_mesh1_midpoints)
            cell_idx_list[np.argmin(dist_mat)] = face.entities(3)[0]

## Setting up simulation
dx = Measure('dx', domain=mesh, subdomain_data=subdomains, metadata={'quadrature_degree': 2})

V = VectorFunctionSpace(mesh, "Lagrange", 1)
sF = FunctionSpace(mesh, "Lagrange", 1)
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V)

# Gel boundary conditions
zero = Constant((0.0, 0.0, 0.0))
bcs = []
sbd = [] # surface boundary conditions

sbd.append(CompiledSubDomain("near(x[0], side)", side = 0))
sbd.append(CompiledSubDomain("near(x[1], side)", side = 0))
sbd.append(CompiledSubDomain("near(x[2], side)", side = 0))
sbd.append(CompiledSubDomain("near(x[0], side)", side = x_bound))
sbd.append(CompiledSubDomain("near(x[1], side)", side = y_bound))
sbd.append(CompiledSubDomain("near(x[2], side)", side = z_bound))
[bcs.append((DirichletBC(V, zero, sub))) for sub in sbd]
bcs.append(None) # why?

total_start = time.time()
for idx in range(chunks):
    iter_start = time.time()
    print()
    print("solver Call: ", idx)
    print("----------------")

    ## Create boundary condition function
    cell2trans_dict = dict(zip(cell_idx_list,
                               midpoint_disp*(idx+1)))
    boundary_func = bc_nw(mesh, cell2trans_dict)
    bcs[-1] = DirichletBC(V, boundary_func, domains, 1)
    print("bc created")
    u, du = solver_call(u, du, bcs)
    print("Time: ", time.time() - iter_start)

    if idx==1:
        """
        Really bad code. Meant to prematurely break out of loop so that the solution doesn't not
        converge.
        """
        # break

print()
print("Total Time: ", time.time() - total_start)


## Exporting Data
hdf5_file = HDF5File(mesh.mpi_comm(),
                     "output/" + output_folder + "/function_dump.h5",
                     "w")
hdf5_file.write(u, "/function")
hdf5_file.close()

file = File("output/" + output_folder + "/solution.pvd")
file << u

# Tabulate dof coordinates
u_arr = u.compute_vertex_values()  # 1-d numpy array
length = np.shape(u_arr)[0]
u_arr = np.reshape(u_arr, (length//3, 3), order="F") # Fortran ordering

# Save txt solutions
np.savetxt(output_folder + "vertex_disp_sim.txt", u_arr, delimiter=" ")

