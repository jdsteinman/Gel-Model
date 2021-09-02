from sys import set_asyncgen_hooks
import dolfin as df
import os
import numpy as np
from scipy.optimize import root_scalar

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3
# df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
# df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
# df.parameters['krylov_solver']['maximum_iterations'] = 10000

def main():
    params = {}

    params["mu_ff"] = 100
    params["c"] = 0.1
    params["stretch"] = 1.1

    params["output_folder"] = "output/linear/" + "Cx" + str(params["c"]) + "/"

    solver(params)

def solver(params):
    # Mesh
    length = 10
    width = 1
    mesh = df.BoxMesh(df.Point(0.0, -width/2, -width/2), df.Point(length, width/2, width/2), 50, 3, 3)

    dx = df.Measure("dx")
    ds = df.Measure("ds")

    # Function Space
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    # Material parameters
    mu_ff = params["mu_ff"]
    c = params["c"]
    mu = df.Expression("x[0]<5 ? mu_ff : c*mu_ff", mu_ff=mu_ff, c=c, degree=1)
    nu = 0.3 
    kappa = 2*mu_ff*(1+nu)/3/(1-2*nu)
    lmbda = 2*mu_ff*nu/(1-2*nu)

    # Define strain and stress
    def epsilon(u):
        return df.sym(df.grad(u))

    def sigma(u):
        d = u.geometric_dimension()
        return lmbda*df.tr(epsilon(u))*df.Identity(d) + 2*mu*epsilon(u)
        
    # Define variational problem
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    B = df.Constant((0, 0, 0))
    T = df.Constant((0, 0, 0))
    a = df.inner(sigma(u), epsilon(v))*dx
    L = df.dot(B, v)*dx + df.dot(T, v)*ds

    # Solve for analytical Jacobian
    stretch = params["stretch"]
    def func(J):
        return kappa/2*J**(8/3)-kappa/2*J**(5/3)+mu_ff/2/3/stretch*J-mu_ff/2/3*stretch**2
    
    J_analytical = root_scalar(func, bracket=[0,5], method="brentq")
    J_analytical = J_analytical.root

    u_right = length*stretch - length
    u_sides = width*np.sqrt(J_analytical/stretch) - width
    u_sides = 0

    # Boundary Conditions
    left   = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right  = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = length)
    top    = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = width/2)
    bottom = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = -width/2)
    front  = df.CompiledSubDomain("near(x[2], side) && on_boundary", side = width/2)
    back   = df.CompiledSubDomain("near(x[2], side) && on_boundary", side = -width/2)    

    bc_left   = df.DirichletBC(V, df.Constant((0.0, 0.0, 0.0)), left)
    bc_right  = df.DirichletBC(V.sub(0), df.Constant(u_right), right)
    bc_top    = df.DirichletBC(V.sub(1), df.Constant(u_sides), top)
    bc_bottom = df.DirichletBC(V.sub(1), df.Constant(-u_sides), bottom)
    bc_front  = df.DirichletBC(V.sub(2), df.Constant(u_sides), front)
    bc_back   = df.DirichletBC(V.sub(2), df.Constant(-u_sides), back)
    bcs = [bc_left, bc_right, bc_top, bc_bottom, bc_front, bc_back]

    # Compute solution
    u = df.Function(V)
    df.solve(a == L, u, bcs)

    # Projections
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    mu = df.project(mu, df.FunctionSpace(mesh, "DG", 1))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mu_file = df.XDMFFile(output_folder + "mu.xdmf")
    mu.rename("mu", "Shear Modulus")
    mu_file.write(mu)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

if __name__ == "__main__":
    main()

