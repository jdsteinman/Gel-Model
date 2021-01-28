import os
import meshio
import numpy as np
import post_tools as pt
from dolfin import *
from matplotlib import pyplot as plt
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

"""
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

## Functions and Class Definitions =========================================================
class shear_modulus(UserExpression):
    def __init__(self, vert, conn, **kwargs):
        super().__init__(**kwargs)
        self._vert = np.asarray(vert, dtype="float64")  # surface vertices

    def set_params(self, mu_bulk, k, rmax):    
        self._mu = mu_bulk
        self._k = k
        self._rmax = rmax

    def eval(self, value, x):
        px = np.array([x[0], x[1], x[2]], dtype="float64")

        # Distance to surface
        r = px - self._vert
        r = np.sum(np.abs(r)**2, axis=-1)**(1./2)
        r = np.amin(r)

        if r < self._rmax:
            value[0] = self._mu*(r/self._rmax)**self._k
        else:
            value[0] = self._mu

    def value_shape(self):
        return ()

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
    lmbda = 1.5925 * 10**16

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

    return u, du, Jac

## Simulation Setup ================================================================================

# Files
mesh_path = "../meshes/ellipsoid/"
output_folder = "./output/convex/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
tag = "_convex"

# Meshes
mesh = Mesh()
with XDMFFile(mesh_path + "ellipsoid_tetra.xdmf") as infile:
    infile.read(mesh)

surf_mesh = meshio.read(mesh_path + "ellipsoid_surface.xdmf")
surf_vert = np.array(surf_mesh.points)
surf_conn = np.array(surf_mesh.cells[0].data)

# Function space    
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Subdomain markers
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(mesh_path + "ellipsoid_triangle.xdmf") as infile:
    infile.read(mvc, "triangle")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

outer_number = 200
inner_number = 201
volume_number = 300

# Define Boundary Conditions
zero = Constant((0.0, 0.0, 0.0))
u_0 = Expression(("a*x[0]/11.5", "b*x[1]/7.6","c*x[2]/18.75"), degree=1, a = 1, b = 1, c = -2)

bc1 = DirichletBC(V, u_0, mf, inner_number)  # inner bc
bc2 = DirichletBC(V, zero, mf, outer_number) # outer bc
bcs = [bc1, bc2]

# Define functions
du, w = TrialFunction(V), TestFunction(V)     # Incremental displacement
u = Function(V, name="disp" + tag)            # Displacement from previous iteration

lmbda = 1.5925 * 10**16
mu_bulk = 325 * 10**12  # Bulk Modulus
k = 2.
rmax = np.amax(mesh.coordinates()) # side length of gel

mu = shear_modulus(surf_vert, surf_conn)
mu.set_params(mu_bulk, k, rmax)

u, du, Jac = solver_call(u, du, bcs, mu, lmbda)

## Isosurfaces ====================================================================================

npoints = np.shape(surf_vert)[0]
ncells = np.shape(surf_conn)[0]

sets = [1.2, 1.4, 1.6, 1.8, 2]
iso_points = pt.mult_rad(sets, surf_vert)  # from other file

for s in sets:
    points = np.array(iso_points[str(s)], dtype="float64")

    # VTK setup
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    conn = surf_conn.ravel().astype("int64")

    ctype = np.zeros(ncells)
    ctype[:] = VtkTriangle.tid

    offset = 3 * (np.arange(ncells, dtype='int64') + 1)

    # Data
    disp = np.array([u(p) for p in points])
    ux, uy, uz = disp[:,0], disp[:,1], disp[:,2]
    ux = np.ascontiguousarray(ux, dtype=np.float32)
    uy = np.ascontiguousarray(uy, dtype=np.float32)
    uz = np.ascontiguousarray(uz, dtype=np.float32)

    # magnitude
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)

    # dot product
    u_dot = pt.dots(u, points, surf_conn)

    # signed magnitude
    s_mag = u_mag * np.abs(u_dot) / u_dot

    unstructuredGridToVTK(output_folder + "level_set_" + str(s), x, y, z, connectivity=conn, offsets=offset, cell_types = ctype, 
    pointData={"u_x" : ux, "u_y" : uy, "u_z" : uz, "u_mag" : u_mag, "u_dot" : u_dot, "u_mag_signed":s_mag})


## Other outputs ========================================================================================================

# Compute gradient and Jacobian
grad_u = project(grad(u), TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3)))
grad_u.rename("grad" + tag, "displacement gradient")

Jac_proj = project(Jac, FunctionSpace(mesh, "DG", 0))
Jac_proj.rename("jac" + tag + tag, "Jacobian")

# Save solution in XDMF format
disp_file = XDMFFile(output_folder + "displacement" + tag + ".xdmf")
disp_file.write(u)

grad_file = XDMFFile(output_folder + "gradient" + tag + ".xdmf")
grad_file.write(grad_u)

jac_file = XDMFFile(output_folder + "jacobian" + tag + ".xdmf")
jac_file.write(Jac_proj)

# Tabulate dof coordinates
u_arr = u.compute_vertex_values()  # 1-d numpy array
length = np.shape(u_arr)[0]
u_arr = np.reshape(u_arr, (length//3, 3), order="F") # Fortran ordering

# Save txt solution
np.savetxt(output_folder + "displacement" + tag + ".txt", u_arr, delimiter=" ")

# Profile expression
f = open(output_folder + "profile.txt", "w+")
f.write("mu(r) = mu_bulk * (r/r_max) ** k")
f.write("\nk = " + str(k))
f.write("\nmu_bulk = " + str(mu_bulk))
f.write("\nlambda = " + str(lmbda))
f.write("\rmax = " + str(rmax))
f.close()