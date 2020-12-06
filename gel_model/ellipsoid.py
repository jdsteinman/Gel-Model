import os
import numpy as np
from dolfin import *
from matplotlib import pyplot as plt

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True

# Read mesh and define function space
ipath = "../meshes/"
opath = "./output/ellipsoid/"
mesh = Mesh()
with XDMFFile(ipath + "ellipsoid_tetra.xdmf") as infile:
    infile.read(mesh)

# Outer surfaces
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(ipath + "ellipsoid_triangle.xdmf") as infile:
    infile.read(mvc, "triangle")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define Boundary Conditions
zero = Constant((0.0, 0.0, 0.0))
# u_0 = Expression(("a*x[0]","b*x[1]","c*x[2]"), degree=1, a = 0.05, b = .05, c = -.1)
u_0 = Expression(("a*x[0]*1.5/11.5","x[1]*1.5/7.6","x[2]*3.5/18.75"), degree=1, a = 0.05, b = .05, c = -.1)

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

## Elasticity parameters
mu = 325*10^12
nu = 0.49
E = 2*mu*(1+nu)
lmbda = E*nu/((1 + nu)*(1 - 2*nu))

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

# Compute gradient
grad_u = project(grad(u), TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)))
# grad_u_x, grad_u_y = grad_u.split(deepcopy=True)

# Evaluate at bead locations
u.set_allow_extrapolation(True)
beads_init = np.genfromtxt("../data/Gel3/beads_init.txt", delimiter=" ")
beads_final = np.genfromtxt("../data/Gel3/beads_final.txt", delimiter=" ")
real_disp = beads_final - beads_init

x_c = np.array([72.2393420333335, 72.9653489566665, 46.8100199433333])
x_c.shape = (1,3)
x_c = np.repeat(x_c, np.shape(beads_init)[0], axis=(0))

beads_init = beads_init - x_c
beads_disp = np.array([u(p) for p in beads_init])
beads_final = beads_init + beads_disp
np.savetxt(opath + "ellipsoid_beads_init.txt", beads_disp, delimiter=" ")
np.savetxt(opath + "ellipsoid_beads_disp.txt", beads_disp, delimiter=" ")

# Calculate Error
rss = np.sqrt(np.sum(np.square(real_disp - beads_disp), 0) )
r = np.corrcoef(real_disp.T, beads_disp.T)

# Save solution in VTK format
disp_file = File(opath + "displacement.pvd")
disp_file << u

grad_file = File(opath + "gradient.pvd")
grad_file << grad_u

# Tabulate dof coordinates
u_arr = u.compute_vertex_values()  # 1-d numpy array
length = np.shape(u_arr)[0]
u_arr = np.reshape(u_arr, (length//3, 3), order="F") # Fortran ordering

# Save solution
np.savetxt(opath + "vertex_disp.txt", u_arr, delimiter=" ")

#try saving in xdmf format instead