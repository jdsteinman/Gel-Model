import os
import math
from dolfin import *
from matplotlib import pyplot as plt

## Define classes
class cell_bc(UserExpression):
    def __init__(self, contract, expand, **kwargs):
        self.c1 = contract
        self.c2 = expand

        super().__init__(**kwargs)

    def value_shape(self):
        return (3,)

    def eval(self, values, x):
        r = math.sqrt(x[0]*x[0] + x[2]*x[2] )
        theta = math.atan(x[2]/x[0])

        r_new = r*self.c1
        x_new = r_new * math.cos(theta)
        z_new = r_new * math.sin(theta)
        
        if x[0] > 0:
            values[0] = x_new
            values[1] = x[1]*self.c2
            values[2] = z_new
        else:
            values[0] = x_new*-1
            values[1] = x[1]*self.c2
            values[2] = z_new*-1


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True

# Read mesh and define function space
lpath = "../meshes/"
filename = "ellipsoidw2pods"
mesh = Mesh()
with XDMFFile(lpath + filename + "Tetra.xdmf") as infile:
    infile.read(mesh)

# What do MeshValueCollection and MeshFunction do?
# How to use MeshFunction to define bc?
# Outer surfaces
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(lpath + filename + "Triangle.xdmf") as infile:
    infile.read(mvc, "triangle")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define Boundary Conditions
zero = Constant((0.0, 0.0, 0.0))
u_0 = Expression(("a*x[0]*1.5/11.5","x[1]*1.5/7.6","-x[2]*3.5/18.75"), degree=1, a = 0.05, b = .05, c = -.1)
#u_0 = cell_bc(-0.2, 0.1)

bc1 = DirichletBC(V, u_0, mf, 1)  # inner bc
bc2 = DirichletBC(V, zero, mf, 2) # outer bc
bcs = [bc1, bc2]

# Define functions
du, w = TrialFunction(V), TestFunction(V)      # Incremental displacement
u  = Function(V, name="disp")                  # Displacement from previous iteration
b  = Constant((0, 0, 0))  # Body force per unit volume
h  = Constant((0, 0, 0))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E  = 5000.0  # Young's Modulus
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
file = File("output/Solution.pvd")
file << u

#try saving in xdmf format instead