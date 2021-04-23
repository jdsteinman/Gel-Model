import os
import time
import meshio
import numpy as np
from dolfin import *
from classes import LineData

"""
Written by: John Steinman
Test with xml input
- outputs:
    - displacement and deformation gradient fields (XDMF)
""" 

parameters['linear_algebra_backend'] = 'PETSc'
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 2

def solver_call(u, du, w, bcs, mu, lmbda):
    ## Kinematics
    B = Constant((0, 0, 0))     # Body force per unit volume
    T = Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    ## Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    ## Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    ## Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

    ## Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, w)
    Jac = derivative(F, u, du)

    # Create nonlinear variational problem and solve
    problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=Jac)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    solver.solve()

    return u, du

class outer_boundary(SubDomain):
    def inside(self, x, on_boundary):
        cond = abs(x[0]) < 30 and abs(x[1]) < 30 and abs(x[2]) < 30
        return on_boundary and not cond

class inner_boundary(SubDomain):
    def inside(self, x, on_boundary):
        cond = abs(x[0]) < 30 and abs(x[1]) < 30 and abs(x[2]) < 30
        return on_boundary and cond

### Simulation Setup ================================================================================

## Files
mesh_path = "./meshes/"
output_folder = "./output/xml/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

## Meshes
mesh = Mesh("../meshes/ellipsoid/ellipsoid.xml")

## Function space
U = VectorElement('Lagrange', mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, U)
du, w = TrialFunction(V), TestFunction(V)    # Incremental displacement
u = Function(V)
u.vector()[:] = 0

# Material parameters
nu = 0.49                        # Poisson's ratio
mu = Constant(325 * 10**12)      # Bulk Modulus
lmbda = 2*nu*mu/ (1-2*nu)        # 1st Lame Parameter

## BCs
mf = cpp.mesh.MeshFunctionSizet(mesh, mesh.topology().dim()-1)
mf.set_all(0)
inner = inner_boundary()
inner.mark(mf, 1)
outer = outer_boundary()
outer.mark(mf, 2)

##  Outer BC
zero = Constant((0.0, 0.0, 0.0))
bcs = []
bcs.append(DirichletBC(V, zero, mf, 2))

# Inner BC
u_D = Expression(["t*x[0]*a/10", "t*x[1]*b/10", "-t*x[2]*c/20"], a=1, b=1, c=2, t=1, degree=2)
innerbc = DirichletBC(V, u_D, mf, 1) 

innerbc.apply(u.vector())
bc_file = XDMFFile(output_folder + "bc.xdmf")
u.rename("U","displacement")
bc_file.write(u)
u.vector()[:] = 0
bcs.append(None)

## Run Sim ==================================================================================
chunks = 1

total_start = time.time()
for i in range(chunks):
    iter_start = time.time()
    print("Solver Call: ", i)
    print("----------------")

    # Increment Boundary Conditions
    u_D.t = (i+1)/chunks
    bcs[-1] = innerbc

    ## Solver
    u, du = solver_call(u, du, w, bcs, mu, lmbda)

    print("Time: ", time.time() - iter_start)
    print()

print("Total Time: ", time.time() - total_start)

# Projections
F = Identity(3) + grad(u)
F = project(F, V=TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
mu = project(mu, FunctionSpace(mesh, "CG", 1))

## XDMF Outputs
disp_file = XDMFFile(output_folder + "U.xdmf")
u.rename("U","displacement")
disp_file.write(u)

F_file = XDMFFile(output_folder + "F.xdmf")
F.rename("F","deformation gradient")
F_file.write(F)

# plot over line
zdata = LineData(u, F, mu, [0,0,20], [0,0,1], 25)
zdata.save_to_csv(fname = output_folder + "Zdata.csv")