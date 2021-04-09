import os
import time
import meshio
import numpy as np
import pandas as pd
from dolfin import *
from pyevtk.hl import unstructuredGridToVTK
from classes import Isosurfaces, LineData
from sim_tools import *
from deformation_grad import def_grad

"""
Written by: John Steinman
Fenics simulation with standard properties
- outputs:
    - displacement and displacement gradient fields (XDMF)
    - isosurfaces (VTK)
    - data along axes (txt)
    - summary of simulation parameters (txt)
""" 

parameters['linear_algebra_backend'] = 'PETSc'
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 2

def solver_call(u, du, w, bcs, mu, lmbda):
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
tag = ""
mesh_path = "../meshes/ellipsoid/"
output_folder = "./output/bctest/ijk/" + tag + "/"
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
mu = Constant(325 * 10**12)      # Bulk Modulus
lmbda = 2*nu*mu/ (1-2*nu)  # 1st Lame Parameter
length = np.amax(mesh.coordinates())  # side length of gel

##  Boundary Conditions
zero = Constant((0.0, 0.0, 0.0))
bcs = []

# Boundary Function
# def ellipsoid_surface(x, on_boundary):
    # return on_boundary and abs(x[0]) < 30 and abs(x[1]) < 30 and abs(x[2]) < 30

# u_D = Expression(["t*x[0]*a/10", "t*x[1]*b/10", "-t*x[2]*c/20"], a=0.1, b=0.1, c=0.2, t=0, degree=3)

def front(x, on_boundary):
    return on_boundary and x[0] == length

def back(x, on_boundary):
    return on_boundary and x[0] == -length

def right(x, on_boundary):
    return on_boundary and x[1] == length

def left(x, on_boundary):
    return on_boundary and x[1] == -length

def top(x, on_boundary):
    return on_boundary and x[2] == length

def bottom(x, on_boundary):
    return on_boundary and x[2] == -length

bc_front = DirichletBC(V, Constant((0.0, 0.0, 0.0)), front) 
bc_back = DirichletBC(V, Constant((-0.0, 0.0, 0.0)), back) 
bc_right = DirichletBC(V, Constant((0.0, 0.0, 0.0)), right) 
bc_left = DirichletBC(V, Constant((0.0, -0.0, 0.0)), left) 
bc_top = DirichletBC(V, Constant((0.0, 0.0, 1.0)), top) 
bc_bottom = DirichletBC(V, Constant((0.0, 0.0, -1.0)), bottom) 

for bc in [bc_front, bc_back, bc_right, bc_left, bc_top, bc_bottom]:
    bcs.append(bc)

# bc_outer = DirichletBC(V, u_D, mf, outer_number) 
bc_inner = DirichletBC(V, zero, mf, inner_number) 
bcs.append(bc_inner)
# bcs.append(None) 

## Run Sim ==================================================================================
chunks = 1

total_start = time.time()
for i in range(chunks):
    iter_start = time.time()
    print("Solver Call: ", i)
    print("----------------")

    # Increment Boundary Conditions
    # u_D.t = (i+1)/chunks
    # bcs[-1] = bc_outer

    ## Solver
    u, du, Jac = solver_call(u, du, w, bcs, mu, lmbda)

    print("Time: ", time.time() - iter_start)
    print()

print("Total Time: ", time.time() - total_start)

# Projections
mu = project(mu, FunctionSpace(mesh, "CG", 1))
F = Identity(3) + grad(u)
F = project(F, V=TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

## XDMF Outputs
disp_file = XDMFFile(output_folder + "U_" + tag + ".xdmf")
u.rename("U","displacement")
disp_file.write(u)

F_file = XDMFFile(output_folder + "F_" + tag + ".xdmf")
F.rename("F","deformation gradient")
F_file.write(F)

## Isosurfaces
sets = [1.2]
iso = Isosurfaces(sets, surf_vert, surf_conn, u, F, mu)
iso.save_sets(output_folder)

## Data over line
xdata = LineData(u, F, mu, [10.1,0,0], [1,0,0], length)
xdata.save_to_csv(fname = output_folder + "Xdata.csv")

ydata = LineData(u, F, mu, [0,10.1,0], [0,1,0], length)
ydata.save_to_csv(fname = output_folder + "Ydata.csv")

zdata = LineData(u, F, mu, [0,0,20.1], [0,0,1], length)
zdata.save_to_csv(fname = output_folder + "Zdata.csv")

