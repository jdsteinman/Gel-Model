import os
import time
import numpy as np
import numpy.linalg as LA
import pandas as pd
import sim_tools as st
from dolfin import *
from matplotlib import pyplot as plt

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
    solver.solve()

    return u

# Simulation setup
output_folder = "./output/bar/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
tag = ["step","Uniform"]

mesh = BoxMesh(Point(0.0, -0.5, -0.5), Point(10.0, 0.5, 0.5), 100, 10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 1)

# Define Boundary Conditions
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 10.0)

zero = Constant((0.0, 0.0, 0.0))
u_b = Constant((-1.0, 0.0, 0.0))

bc1 = DirichletBC(V, u_b, left) 
bc2 = DirichletBC(V, zero, right)
bcs = [bc1, bc2]

# Sim
lmbda = 1.5925 * 10**16
mu_bulk = 325 * 10**12  # Bulk Modulus
mu_expr = []
# mu_expr.append(Expression("mu_bulk", degree=1, mu_bulk = mu_bulk))
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
    grad_u = project(grad(u), TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3)))
    C = project(C, TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3)))

    # Plot
    npoints = 100
    points = np.column_stack([np.linspace(0.01, 9.99, npoints), np.zeros(npoints), np.zeros(npoints) ])
    
    data = st.toDataFrame(points, u, mu, grad_u)
    data.to_csv(output_folder+"data.csv", sep=",")

print("Total Time: ", time.time() - total_start)