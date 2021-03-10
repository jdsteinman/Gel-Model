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
from sim_tools import *
from deformation_grad import def_grad

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
    - summary of simulation parameters (txt)
""" 

parameters['linear_algebra_backend'] = 'PETSc'
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 2

## Functions and Class Definitions =========================================================
class shear_modulus(UserExpression):
    def __init__ (self, vert, conn, **kwargs):
        super().__init__(**kwargs)
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
            if k >= 0:
                value[0] = self._mu*(r/self._rmax)**self._k + self._mu*.01  # Power function
            else:
                value[0] = self._mu*0.5 + self._k*.01   # Step function
        else:
            value[0] = self._mu * 1.01

    def value_shape(self):
        return ()

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
    #'''
    solver = NonlinearVariationalSolver(problem)
    print(solver.parameters['newton_solver'].keys())
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    #solver.parameters['newton_solver']['linear_solver'] = 'cg'
    #solver.parameters['newton_solver']['preconditioner'] = 'amg'
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    #'''
    solver.solve()

    return u, du, Jac


### Simulation Setup ================================================================================

## Files
tag = "step012"
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
V = VectorFunctionSpace(mesh, "CG", 2)
du, w = TrialFunction(V), TestFunction(V)    # Incremental displacement
u = Function(V)
u.vector()[:] = 0

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
k = -1.                           # profile shape
l = np.amax(mesh.coordinates())  # side length of gel
rmax = 10                        # graded model region

# mu = Expression(mu_compiled_code)
# mu.muBulk = mu_bulk
# mu.rmax = rmax
# mu.k = k

mu = shear_modulus(surf_vert, surf_conn)
mu.set_params(mu_bulk, k, rmax)

##  Boundary Conditions
zero = Constant((0.0, 0.0, 0.0))
bcs = []
bcs.append(DirichletBC(V, zero, mf, outer_number))
bcs.append(None) 

u_D = Expression(["t*x[0]*a/10", "t*x[1]*b/10", "-t*x[2]*c/20"], a=0, b=1, c=2, t=0, degree=1)
testbc = DirichletBC(V, u_D, mf, inner_number) 

## Run Sim ==================================================================================
chunks = 2

total_start = time.time()
for i in range(chunks):
    iter_start = time.time()
    print("Solver Call: ", i)
    print("----------------")

    # Increment Boundary Conditions
    u_D.t = (i+1)/chunks
    bcs[-1] = testbc

    ## Solver
    u, du, Jac = solver_call(u, du, bcs, mu, lmbda)

    print("Time: ", time.time() - iter_start)
    print()

print("Total Time: ", time.time() - total_start)
u.set_allow_extrapolation(True) # Temp fix for evaluating on surface

# Projections
mu = project(mu, FunctionSpace(mesh, "CG", 1))
mu.set_allow_extrapolation(True)

grad_space = TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3))
grad_u = project(grad(u), V=grad_space, solver_type = 'cg', preconditioner_type = 'amg')
grad_u.set_allow_extrapolation(True)

## Isosurfaces
factors = [1.01, 1.2, 1.4, 1.6, 1.8, 2]
set_dict = generate_sets(factors, surf_vert)
set_data = save_sets(set_dict, surf_conn, mu, u, grad_u, output_folder)

## Deformation Test
points = scale_radius(surf_vert, 1.2)
normals = get_surface_normals(points, surf_conn)
F = def_grad(surf_vert, normals, u)
df = ArraystoDF(points, F=F)
point_data = {}
for column in df:
    skips = ['x', 'y', 'z', 'r']
    if column in skips: continue

    dat = np.ascontiguousarray(df[column],  dtype=np.float32)
    point_data[column] = dat

writeVTK(output_folder + "F" + "1.2", points, surf_conn, point_data)

## XDMF Outputs
disp_file = XDMFFile(output_folder + "displacement_" + tag + ".xdmf")
u.rename("u","displacement")
disp_file.write(u)

grad_file = XDMFFile(output_folder + "gradient_" + tag + ".xdmf")
grad_u.rename("grad_u","displacement gradient")
grad_file.write(grad_u)

## Table Outputs
xpoint = [10,0,0]
xdir = [1,0,0]
xdata = data_over_line(xpoint, xdir, 0.1, l, mu, u, grad_u)
xdata.to_csv(output_folder+"data_x.csv", sep=",")

ypoint = [0,10,0]
ydir = [0,1,0]
ydata = data_over_line(ypoint, ydir, 0.1, l, mu, u, grad_u)
ydata.to_csv(output_folder+"data_y.csv", sep=",")

zpoint = [0,0,20]
zdir = [0,0,1]
zdata = data_over_line(zpoint, zdir, 0.1, l, mu, u, grad_u)
zdata.to_csv(output_folder+"data_z.csv", sep=",")