import os
from dolfin import *
from matplotlib import pyplot as plt

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True

# Read mesh and define function space
lpath = "../meshes/"
mesh = Mesh()
with XDMFFile(lpath + "basicPseudopodsTetra.xdmf") as infile:
    infile.read(mesh)

# What do MeshValueCollection and MeshFunction do?
# How to use MeshFunction to define bc?
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(lpath + "basicPseudopodsTriangle.xdmf") as infile:
    infile.read(mvc, "basicPseudopodsTriangle")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define Boundary Conditions
zero = Constant((0.0, 0.0, 0.0))
u_0 = Expression(("a*x[0]","b*x[1]","x[2]*c"), degree=1, a = .25, b = .25, c = -.50)

bc1 = DirichletBC(V, u_0, mf, 1)  # inner bc
bc2 = DirichletBC(V, zero, mf, 2) # outer bc
bcs = [bc1, bc2]

# Define functions
du, w = TrialFunction(V), TestFunction(V)      # Incremental displacement
u  = Function(V, name="disp")                  # Displacement from previous iteration
b  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
h  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E  = 1000.0  # Young's Modulus
nu = 0.3   # Poisson Ratio
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(b, u)*dx - dot(h, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, w)

# Compute Jacobian of F
J = derivative(F, u, du)

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem)
solver.solve()

# Save solution in VTK format
file = File("output/hyper_solution.pvd")
file << u

