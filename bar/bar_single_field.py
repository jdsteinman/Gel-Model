import dolfin as df
import os
import numpy as np
from scipy.optimize import root_scalar
import tools 

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3

def main():
    params = {}

    params["nu"] = 0.49
    params["mu"] = 100e3
    params["degradation"] = 0.5
    params["stretch"] = 1.05

    params["output_folder"] = "output/single_field/mu"

    solver(params)

def solver(params):
    # Mesh
    L = 10
    W = 1
    mesh = df.BoxMesh(df.Point(0.0, -W/2, -W/2), df.Point(L, W/2, W/2), 100, 3, 3)

    dx = df.Measure("dx")
    ds = df.Measure("ds")

    # Function Space
    element_u = df.VectorElement("CG",mesh.ufl_cell(),2)
    V = df.FunctionSpace(mesh, element_u)
    u = df.Function(V)
    u.rename('u','displacement')

    # Set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0, 0.0)), V)
    df.assign(u, u_0)
    
    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    J = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/J**(2/d)            # Isochoric decomposition

    # Invariants of deformation tensors
    IC_bar = df.tr(C_bar)

    # Material parameters
    nu_ff = params["nu"]
    mu_ff = params["mu"]
    D = params["degradation"]
    mu = df.Expression("x[0]<5 ? mu : D*mu", mu=mu_ff, D=D, degree=1)
    nu = df.Expression("x[0]<5 ? nu : D*nu", nu=nu_ff, D=D, degree=1)
    kappa = 2*mu_ff*(1+nu)/3/(1-2*nu)
    kappa = 2*mu_ff*(1+nu_ff)/3/(1-2*nu_ff)

    c1 = mu/2
    c2 = kappa

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 

    # Total potential energy
    Pi = psi*dx - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, u)
    Dres = df.derivative(res, u)

    # Boundary Conditions
    left   = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right  = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = L)
    top_bottom  = df.CompiledSubDomain("near(abs(x[1]), side) && abs(x[2])<0.1", side = W/2)
    front_back  = df.CompiledSubDomain("near(abs(x[2]), side) && abs(x[1])<0.1", side = W/2)
  
    # Solve for analytical Jacobian
    stretch = params["stretch"]
    u_right = L*(stretch-1)

    bc_left   = df.DirichletBC(V.sub(0), df.Constant(0.0), left)
    bc_right  = df.DirichletBC(V.sub(0), df.Constant(u_right), right)
    bc_top_bottom  = df.DirichletBC(V.sub(2), df.Constant(0), top_bottom, method="pointwise")
    bc_front_back  = df.DirichletBC(V.sub(1), df.Constant(0), front_back, method="pointwise")
    bcs = [bc_left, bc_right, bc_top_bottom, bc_front_back]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, u, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'lu'
    solver.solve()

    # Projections
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    J = df.project(J, df.FunctionSpace(mesh, "DG", 0))
    mu = df.project(mu, df.FunctionSpace(mesh, "DG", 0))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # linedata = tools.LineData(u, F, mu, point=[0,0,0], direction=[1,0,0], bound=10, step=0.1)
    # linedata.save_to_csv(output_folder+"x_axis.csv")
    # linedata.save_to_vtk(output_folder+"x_axis")

    mu_file = df.XDMFFile(os.path.join(output_folder, "mu.xdmf"))
    mu.rename("mu", "Shear Modulus")
    mu_file.write(mu)

    disp_file = df.XDMFFile(os.path.join(output_folder, "U.xdmf"))
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(os.path.join(output_folder, "F.xdmf"))
    F.rename("F","deformation gradient")
    F_file.write(F)

    J_file = df.XDMFFile(os.path.join(output_folder, "J.xdmf"))
    J.rename("J","Jacobian")
    J_file.write(J)

    # with open(output_folder+"log_params", "w") as f:
    #     f.write("mu_ff = {:e}\n".format(mu_ff))
    #     f.write("kappa = {:e}\n".format(kappa))
    #     f.write("c =     {:f}\n".format(c))

if __name__ == "__main__":
    main()

