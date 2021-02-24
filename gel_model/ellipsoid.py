import os
import time
import meshio
import numpy as np
import pandas as pd
from dolfin import *
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle
from sim_tools import generate_sets, plot_sets, toDataFrame

"""
Written by: John Steinman
Fenics simulation of ellipsoidal model with functionally graded gel
- mu(r) = mu_bulk * (r/r_max) ** k
    - k detrmines shape of profile:
        - k = 0:     uniform 
        - 0 < k < 1: concave
        - k = 1:     linear
        - k > 1:     convex
- outputs:
    - displacement, gradient, and Jacobian fields (XDMF)
    - displacement on isosurfaces (VTK)
    - displacement at each vertex (txt)
    - summary o f simulation parameters (txt)
""" 

## Functions and Class Definitions =========================================================
class shear_modulus(UserExpression):
    def __init__ (self, vert, conn, **kwargs):
        super()._ _init__(**kwargs)
        self._vert = np.asarray(vert, dtype="float64")  # surface vertices

    def set_params(self, mu_bulk, k, rmax):    
        self._mu =  mu_bulk
        self._k = k
        self._rmax = rmax

    def eval(self, value, x):
        px = np.array([x[0], x[1], x[2]], dtype="float64")

        # Distance to surface
        r = px - self._vert
        r = np.sum(np.abs(r)**2, axis=-1)**(1./2)
        r = np.amin(r)

        if r < self._rmax:
            value[0] = self._mu*(r/self._rmax)**self._k + self._mu*.01  # Power Model
            # value[0] = self._mu*0.5 + self._k*.01   # Step function
        else:
            value[0] = self._mu * 1.01

    def value_shape(self):
        return ()

class inner_bc(UserExpression):
    def __init__(self, mesh, face2disp_dict, **kwargs):
        self.mesh = mesh 
        self._face_dict = face2disp_dict
        super().__init__(**kwargs)

    def value_shape(self):
        return (3,)

    def eval_cell(self, value, x, cell):
        try:
            value[0], value[1], value[2] = self._face_dict[cell.index]
        except KeyError:
            value[0], value[1], value[2] = (0, 0, 0)

def get_vert_disp(vert, a, b, c):
    vert_disp = np.zeros(vert.shape)
    vert_disp[:,0] = a/10 * vert[:,0]
    vert_disp[:,1] = b/10 * vert[:,1]
    vert_disp[:,2] = c/20 * vert[:,2]

    return vert_disp

def get_midpoints(surf_mesh):
    cell_dict = dict(surf_mesh.cells)
    midpoints = np.zeros((cell_dict['triangle'].shape[0], 3))
    for idx, triangle in enumerate(cell_dict['triangle']):
        midpoints[idx] = surf_mesh.points[cell_dict['triangle'][idx]].mean(0)
    return midpoints

def get_midpoint_disp(vert_disp, faces):
    midpoint_disp = np.zeros((faces.shape[0], 3))
    for idx, face in enumerate(faces):
        midpoint_disp[idx, :] = np.mean((vert_disp[face[0]],
                                     vert_disp[face[1]],
                                     vert_disp[face[2]]), axis=0)
    return midpoint_disp

def get_face_mapping(midpoints, mesh, mf, inner_number):
    face_map = np.zeros(midpoints.shape[0])

    for index, face in enumerate(faces(mesh)):
        # if mesh face is on inner boundary
        if mf.array()[index] == inner_number:
            mesh_midpoint = face.midpoint().array().reshape((1,3)) 
            dist_mat = distance_matrix(mesh_midpoint, midpoints)
            face_map[np.argmin(dist_mat)] = face.entities(3)[0]

    return face_map

def solver_call(u, du, bcs, mu, lmbda):
    ## Kinematics
    B = Constant((0, 0, 0))  # Body force per unit volume
    T = Constant((0, 0, 0))  # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    ## Invariants of deformation tensors
    Ic = tr(C)
    Jac  = det(F)

    ## Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(Jac) + (lmbda/2)*(ln(Jac))**2

    ## Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

    ## Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, w)
    J = derivative(F, u, du)

    # Create nonlinear variational problem and solve
    problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()

    return u, du, Jac

### Simulation Setup ================================================================================

## Files
tag = "test"
mesh_path = "../meshes/ellipsoid/"
output_folder = "./output/" + tag + "/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

## Meshes
mesh = Mesh()
with XDMFFile(mesh_path + "ellipsoid_tetra.xdmf") as infile:
    infile.read(mesh)

surf_mesh = meshio.read(mesh_path + "ellipsoid_surface.xdmf")
surf_vert = np.array(surf_mesh.points)
surf_conn = np.array(surf_mesh.cells[0].data)

## Function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

## Subdomain markers
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(mesh_path + "ellipsoid_triangle.xdmf") as infile:
    infile.read(mvc, "triangle")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
outer_number = 200
inner_number = 201
volume_number = 300

# Material parameters
nu = 0.49                        # Poisson's ratio
mu_bulk = 325 * 10**12           # Bulk Modulus
lmbda = 2*nu*mu_bulk / (1-2*nu)  # 1st Lame Parameter
k = 0.                           # profile shape
l = np.amax(mesh.coordinates())  # side length of gel
rmax = 10                        # graded model region

mu = shear_modulus(surf_vert, surf_conn)
mu.set_params(mu_bulk, k, rmax)

##  Boundary Conditions
vert_disp = get_vert_disp(surf_vert, 1, 1, -2)
midpoints = get_midpoints(surf_mesh)
midpoint_disp = get_midpoint_disp(vert_disp, surf_conn)
face_map = get_face_mapping(midpoints, mesh, mf, inner_number)

zero = Constant((0.0, 0.0, 0.0))
bcs = []
bcs.append(DirichletBC(V, zero, mf, outer_number))
bcs.append(None) 

## Functions
du, w = TrialFunction(V), TestFunction(V)    # Incremental displacement
u = Function(V)

## Run Sim ==================================================================================
chunks = 1
midpoint_disp /= chunks

total_start = time.time()
for i in range(chunks):
    iter_start = time.time()
    print("Solver Call: ", i)
    print("----------------")

    ## Inner BC
    face2disp_dict = dict(zip(face_map, midpoint_disp*(i+1)))
    boundary_func = inner_bc(mesh, face2disp_dict)
    bcs[-1] = DirichletBC(V, boundary_func, mf, inner_number)

    ## Solver
    u, du, Jac = solver_call(u, du, bcs, mu, lmbda)

    print("Time: ", time.time() - iter_start)
    print()

print("Total Time: ", time.time() - total_start)
u.set_allow_extrapolation(True) # Temp fix for evaluating on surface

# Deformation
d = u.geometric_dimension()
I = Identity(d)      # Identity tensor
F = I + grad(u)      # Deformation gradient
C = F.T*F            # Right Cauchy-Green tensor

# Projections
grad_u = project(grad(u), TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3)))
grad_u.set_allow_extrapolation(True)
mu = project(mu, FunctionSpace(mesh, "DG", 1))
mu.set_allow_extrapolation(True)

### Outputs ==================================================================================

## Isosurfaces
sets = [1, 1.2, 1.4, 1.6, 1.8, 2]
set_disp = generate_sets(sets, surf_vert, surf_conn, u, grad_u, output_folder)
plot_sets(sets, set_disp, output_folder)

## Table Outputs
npoints = 100
zaxis = np.column_stack((np.zeros(npoints), np.zeros(npoints), np.linspace(20, l, npoints) ))
yaxis = np.column_stack((np.zeros(npoints), np.linspace(10, l, npoints), np.zeros(npoints) ))

zdata = toDataFrame(zaxis, u, mu, grad_u)
ydata = toDataFrame(yaxis, u, mu, grad_u)

zdata.to_csv(output_folder+"data_z.csv", sep=",")
ydata.to_csv(output_folder+"data_y.csv", sep=",")

# XDMF Outputs
disp_file = XDMFFile(output_folder + "displacement_" + tag + ".xdmf")
u.rename("u","displacement")
disp_file.write(u)

mu_file = XDMFFile(output_folder + "mu_" + tag + ".xdmf")
mu.rename("mu", "shear_modulus")
mu_file.write(mu)

## Save Simulation Parameters
f = open(output_folder + "profile.txt", "w+")
f.write("mu(r) = mu_bulk * (r/r_max) ** k")
f.write("\nk = " + str(k))
f.write("\nmu_bulk = " + str(mu_bulk))
f.write("\nlambda = " + str(lmbda))
f.write("\rmax = " + str(rmax))
f.close()
