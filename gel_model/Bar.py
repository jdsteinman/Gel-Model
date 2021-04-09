import os
import time
import numpy as np
import numpy.linalg as LA
import pandas as pd
from dolfin import *
from matplotlib import pyplot as plt
from classes import LineData

parameters['linear_algebra_backend'] = 'PETSc'
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 2

def solver_call(u, du, bcs, mu, lmbda):

    # Kinematics
    B = Constant((0, 0, 0))  # Body force per unit volume
    T = Constant((0, 0, 0))  # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    Jac  = det(F)

    ## Elasticity parameters
    nu = 0.49  # Poisson's ratio
    mu_bulk = 325 * 10**12  # Bulk Modulus
    lmbda = 2*nu*mu_bulk / (1-2*nu)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(Jac) + (lmbda/2)*(ln(Jac))**2

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, w)
    J = derivative(F, u, du)

    # Create nonlinear variational problem and solve
    problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
    solver = NonlinearVariationalSolver(problem)
    # print(solver.parameters['newton_solver'].keys())
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    #solver.parameters['newton_solver']['linear_solver'] = 'cg'
    #solver.parameters['newton_solver']['preconditioner'] = 'amg'
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    solver.solve()

    return u

# Simulation setup
output_folder = "./output/bar/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
tag = ["step","Uniform"]

mesh = BoxMesh(Point(0.0, -0.5, -0.5), Point(10.0, 0.5, 0.5), 100, 10, 10)
V = VectorFunctionSpace(mesh, "CG", 2)
V0 = FunctionSpace(mesh, "DG", 1)

# Define Boundary Conditions
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 10.0)
sides = CompiledSubDomain("on_boundary && (near(abs(x[1]), side) || near(abs(x[2]), side))", side=0.5)

zero = Constant((0.0, 0.0, 0.0))
u_b = Constant((-1, 0.0, 0.0))

bc1 = DirichletBC(V, u_b, left) 
bc2 = DirichletBC(V, zero, right)
bc3 = DirichletBC(V.sub(1), Constant(0), sides)
bc4 = DirichletBC(V.sub(2), Constant(0), sides)

bcs = [bc1, bc2, bc3, bc4]

# Sim
lmbda = 1.5925 * 10**16
mu_bulk = 325 * 10**12  # Bulk Modulus
mu_expr = []
mu_expr.append(Expression("(x[0] < 5) ? mu_bulk/2 : mu_bulk", degree=1, mu_bulk = mu_bulk))

total_start = time.time()
for i, mu in enumerate(mu_expr):
    # Solver
    du, w = TrialFunction(V), TestFunction(V)   # Incremental displacement
    u = Function(V, name="disp" + tag[i])           
    u = solver_call(u, du, bcs, mu, lmbda)

    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Projections
    mu = project(mu, FunctionSpace(mesh, "DG", 1))
    grad_u = project(grad(u), TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)))

    # Plot
    point = [0.01, 0, 0]
    direction = [1, 0, 0]
    bound = 10

    data = LineData(u, grad_u, mu, point, direction, bound)
    data.save_to_csv(output_folder + "bar.txt")

    ## XDMF Outputs
    disp_file = XDMFFile(output_folder + "displacement_" + tag[i] + ".xdmf")
    u.rename("u","displacement")
    disp_file.write(u)

    grad_space = TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3))
    grad_u = project(grad(u), V=grad_space, solver_type = 'cg', preconditioner_type = 'amg')
    grad_u.set_allow_extrapolation(True)

    grad_file = XDMFFile(output_folder + "gradient_" + tag[i] + ".xdmf")
    grad_u.rename("grad_u","displacement gradient")
    grad_file.write(grad_u)

print("Total Time: ", time.time() - total_start)