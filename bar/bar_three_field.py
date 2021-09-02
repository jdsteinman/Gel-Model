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

    params["nu"] = 0.499
    params["mu_ff"] = 100e12
    params["c"] = 1
    params["stretch"] = 1.1

    params["output_folder"] = "output/three_field/nu=" + str(params["nu"]) + "/Cx" + str(params["c"]) + "/"

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
    element_p = df.FiniteElement("DG",mesh.ufl_cell(),0)
    element_J = df.FiniteElement("DG",mesh.ufl_cell(),0)
  
    V = df.FunctionSpace(mesh,df.MixedElement([element_u,element_p,element_J]))
    xi = df.Function(V)
    xi.rename('xi','mixed solution')
    xi_ = df.TestFunction(V)
    dxi = df.TrialFunction(V)

    # Set initial values
    V_u = V.sub(0).collapse()
    V_p = V.sub(1).collapse()
    V_J = V.sub(2).collapse()
    u_0 = df.interpolate(df.Constant((0.0, 0.0, 0.0)), V_u)
    p_0 = df.interpolate(df.Constant(0.0), V_p)
    J_0 = df.interpolate(df.Constant(1.), V_J)

    df.assign(xi,[u_0,p_0,J_0])
    u,p,J = df.split(xi)
    
    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/d)            # Isochoric decomposition

    # Invariants of deformation tensors
    IC_bar = df.tr(C_bar)

    # Material parameters
    nu = params["nu"]
    mu_ff = params["mu_ff"]
    c = params["c"]
    mu = df.Expression("x[0]<5 ? mu_ff : c*mu_ff", mu_ff=mu_ff, c=c, degree=1)
    kappa = 2*mu_ff*(1+nu)/3/(1-2*nu)

    c1 = mu/2
    c2 = df.Constant(kappa)

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    left   = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right  = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = L)
    top    = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = W/2)
    bottom = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = -W/2)
    front  = df.CompiledSubDomain("near(x[2], side) && on_boundary", side = W/2)
    back   = df.CompiledSubDomain("near(x[2], side) && on_boundary", side = -W/2)    

    # Solve for analytical Jacobian
    stretch = params["stretch"]
    u_right = L*(stretch-1)
    u_sides = W*(1/np.sqrt(stretch)-1)

    bc_left   = df.DirichletBC(V_u.sub(0), df.Constant(0.0), left)
    bc_right  = df.DirichletBC(V_u.sub(0), df.Constant(u_right), right)
    bc_top    = df.DirichletBC(V_u.sub(1), df.Constant(u_sides), top)
    bc_bottom = df.DirichletBC(V_u.sub(1), df.Constant(-u_sides), bottom)
    bc_front  = df.DirichletBC(V_u.sub(2), df.Constant(u_sides), front)
    bc_back   = df.DirichletBC(V_u.sub(2), df.Constant(-u_sides), back)
    # bcs = [bc_left, bc_right, bc_top, bc_bottom, bc_front, bc_back]
    bcs = []

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'lu'
    solver.solve()
   
    u, p, J = xi.split(True)

    # Projections
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    mu = df.project(mu, df.FunctionSpace(mesh, "DG", 1))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    linedata = tools.LineData(u, F, mu, point=[0,0,0], direction=[1,0,0], bound=10, step=0.1)
    linedata.save_to_csv(output_folder+"x_axis.csv")
    linedata.save_to_vtk(output_folder+"x_axis")

    mu_file = df.XDMFFile(output_folder + "mu.xdmf")
    mu.rename("mu", "Shear Modulus")
    mu_file.write(mu)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

    with open(output_folder+"log_params", "w") as f:
        f.write("mu_ff = {:e}\n".format(mu_ff))
        f.write("kappa = {:e}\n".format(kappa))
        f.write("c =     {:f}\n".format(c))

if __name__ == "__main__":
    main()

