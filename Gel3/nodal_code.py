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
J&J's adaptation of bc_sim_xf.py
Prescribes displacements at cell surface nodes

TODO:
- Does Gel3 data domain match our mesh?

Prerequisites:
- Gel Volume Mesh
- Surface mesh
- Displacements correspoding to surface mesh nodes
"""

## Define objects and functions ===========================================================================
class bc_nw(UserExpression):
    def __init__(self, mesh, cell2trans_dict, **kwargs):
        self.mesh = mesh  # input mesh
        self._cell2trans_dict = cell2trans_dict 
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

    mu = 325 * 10**12  # 2nd lame parameter (shear modulus)
    nu = 0.49   # Poisson's ratio
    E = 2*mu*(1+nu)
    lmbda = E*nu/((1 + nu)*(1 - 2*nu))   # 1st  lame parameter

    # Math
    psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2
    Pi = psi * dx - dot(B, u) * dx - dot(T, u) * ds
    F = derivative(Pi, u, v)
    J = derivative(F, u, du)

    ## Solving
    solve(F == 0, u, bcs, J=J, solver_parameters={"newton_solver": {"linear_solver": "mumps"}})

    return u, du

## Simulation Setup =======================================================================================
chunks = int(100) 
data_path = "../data/Gel3/"
mesh_path = "../meshes/Gel3/"
output_folder =  "./output/Gel3/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

outer_number = 200
inner_number = 201
volume_number = 300

cytod_surf = meshio.read(mesh_path + "cytod_uncentered_unpca_surface" + ".xdmf")   # for disp mapping
cytod_faces = cytod_surf.cells[0].data

# Read volume mesh
mesh = Mesh()
with XDMFFile(mesh_path + "cytod_uncentered_unpca_tetra.xdmf") as infile:
    infile.read(mesh)

# Get surface numbers
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(mesh_path + "cytod_uncentered_unpca_triangle.xdmf") as infile:
    infile.read(mvc, "triangle")    # store physical numbers

# Mesh Functions
domains = cpp.mesh.MeshFunctionSizet(mesh, mvc)   
subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc) # Derivative data
subdomains.set_all(0)

# Read displacements
surf_mesh1_midpoints = create_surf_midpoints(cytod_surf)
vert_disp = pd.read_csv(data_path + "displacements_cytod_to_normal_uncentered_unpca.csv",
                        header=None).values
vert_disp = vert_disp/chunks
midpoint_disp = np.zeros((cytod_faces.shape[0], 3)) # create an array #faces by 3

for idx, face in enumerate(cytod_faces):
    # Assign average displacement to tetra midpoint
    midpoint_disp[idx, :] = np.mean((vert_disp[face[0]],
                                     vert_disp[face[1]],
                                     vert_disp[face[2]]), axis=0)

# Get face mapping
cell_idx_list = np.zeros(midpoint_disp.shape[0])
i = 0
for index, face in enumerate(faces(mesh)):
    x, y, z = face.midpoint().array() # get location of midpoint
    if domains.array()[index] == inner_number:
        i += 1
        dist_mat = distance_matrix(np.array([[x, y, z]]), surf_mesh1_midpoints)
        cell_idx_list[np.argmin(dist_mat)] = face.entities(3)[0]
    elif domains.array()[index] == outer_number:
        continue
    else:
        domains.array()[index] = 199  # Easir visualization in paraview

# Setting up simulation
dx = Measure('dx', domain=mesh, subdomain_data=subdomains, metadata={'quadrature_degree': 2})
V = VectorFunctionSpace(mesh, "Lagrange", 1)
sF = FunctionSpace(mesh, "Lagrange", 1)
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V, name="disp")

# Gel boundary conditions
zero = Constant((0.0, 0.0, 0.0))
bcs = []
bcs.append(DirichletBC(V, zero, domains, outer_number))
bcs.append(None) 

## Simulation ===================================================================================================
total_start = time.time()
for idx in range(chunks):
    iter_start = time.time()
    print()
    print("solver Call: ", idx)
    print("----------------")

    # Create boundary condition function
    cell2trans_dict = dict(zip(cell_idx_list,
                               midpoint_disp*(idx+1)))

    boundary_func = bc_nw(mesh, cell2trans_dict)
    bcs[-1] = DirichletBC(V, boundary_func, domains, inner_number)
 
    print("bc created")
    u, du = solver_call(u, du, bcs)
    print("Time: ", time.time() - iter_start)

    # Logs
    with open(output_folder + 'error_log.txt', 'a+') as f:
        f.write("Iteration: %d\n" % idx)
        for item in boundary_func.error_log:
            f.write("%s\n" % item)

    ## Uncomment following line to prematurely break out of solver
    # if idx==1: break

print()
print("Total Time: ", time.time() - total_start)

## Output ============================================================================================
u.set_allow_extrapolation(True)
beads_init = np.genfromtxt("../data/Gel3/beads_init.txt", delimiter=" ")
beads_disp = np.array([u(p) for p in beads_init])

# Tabulate dof coordinates
u_arr = u.compute_vertex_values()  # 1-d numpy array
length = np.shape(u_arr)[0]
u_arr = np.reshape(u_arr, (length//3, 3), order="F") # Fortran ordering

# Save txt solutions
np.savetxt(output_folder + "sim_beads_disp.txt", beads_disp, delimiter=" ")
np.savetxt(output_folder + "sim_vertex_disp.txt", u_arr, delimiter=" ")

# Paraview output
disp_file = File(output_folder + "displacement.pvd")
disp_file << u



