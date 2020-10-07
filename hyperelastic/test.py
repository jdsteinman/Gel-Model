from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True

# Read mesh and define function space
fname = '/home/john/research/meshes/sphere_in_cube.xdmf'
file_in = XDMFFile(fname)
mesh = Mesh()
tdim = mesh.topology().dim()
# Create facet function over mesh, unitialized values
# boundaries = MeshFunction('size_t', mesh, tdim - 1)
# plot(boundaries, title='unitialized values', interactive=True)


file_in.read(mesh)
width = 5
radius = 1

#mesh = UnitCubeMesh(16, 16, 16)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

def outer(x, on_boundary):
    cond = near(abs(x[0]), width) or near(abs(x[1]), width) or near(abs(x[2]), width)
    return cond and on_boundary

def inner(x, on_boundary):
    r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
    cond = near(r, radius)
    if cond:
        print(r)
    return cond 

# Define Dirichlet boundary (x = 0 or x = 1)
zero = Constant((0.0, 0.0, 0.0))
u_D = Expression(('0','0','1 + x[0]'), degree=1) # expression defined solution values on boundary

bc1 = DirichletBC(V, u_D, inner)
bc2 = DirichletBC(V, zero, outer)

bcs = [bc1, bc2]

# Define functions
du, w = TrialFunction(V), TestFunction(V)            # Incremental displacement

u  = Function(V)                 # Displacement from previous iteration
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
E, nu = 10.0, 0.3
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
problem = NonlinearVariationalProblem(F, u, bcs=bc1, J=J)
solver = NonlinearVariationalSolver(problem)
solver.solve()

# Save solution in VTK format
file = File("hyperelastic/displacement2.pvd")
file << u


