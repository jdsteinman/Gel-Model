import dolfin as df
import numpy as np
import sys
import os
import time

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 10000

"""
Written by: John Steinman
"""

def main():

    params = {}

    params['mesh'] = "./meshes/hole.xdmf"
    params['domains'] = "./meshes/hole_domains.xdmf"
    params['boundaries'] = "./meshes/hole_boundaries.xdmf"

    params['mu_ff'] = 100e12
    params['c'] = 0.1
    params['near_field'] = 50

    params['output_folder'] = './output/hole/C=' + str(params['c']) + '/'

    solver_call(params)

class ShearModulus(df.UserExpression):
    def __init__(self, mu_ff, c, L, *args, **kwargs):
        self.mu_ff = mu_ff
        self.c = c
        self.L = L
        super().__init__(*args, **kwargs)

    def value_shape(self):
        return()

    def eval(self, value, x):
        if abs(x[0])<=self.L/2 and abs(x[1])<=self.L/2 and abs(x[2])<=self.L/2:
            value[0]=self.mu_ff*self.c
        else:   
            value[0]=self.mu_ff

def solver_call(params):
    # Mesh
    mesh = df.Mesh()
    with df.XDMFFile(params["mesh"]) as infile:
        infile.read(mesh)

    # df.plot(mesh)
    # import matplotlib.pyplot as plt
    # plt.show()
    # quit()
    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(params["domains"]) as infile:
        infile.read(mvc, "domains") 
    domains = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(params["boundaries"]) as infile:
        infile.read(mvc, "boundaries") 
    boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    element_u = df.VectorElement("CG", mesh.ufl_cell(), 2)
    element_p = df.FiniteElement("DG", mesh.ufl_cell(), 0)
    element_J = df.FiniteElement("DG", mesh.ufl_cell(), 0)
  
    V = df.FunctionSpace(mesh,df.MixedElement([element_u,element_p,element_J]))
    xi = df.Function(V)
    xi.rename('xi','mixed solution')
    xi_ = df.TestFunction(V)
    dxi = df.TrialFunction(V)

    # Set initial values
    V_u = V.sub(0)
    V_p = V.sub(1)
    V_J = V.sub(2)
    u_0 = df.interpolate(df.Constant((0.0, 0.0, 0.0)), V_u.collapse())
    p_0 = df.interpolate(df.Constant(0.0), V_p.collapse())
    J_0 = df.interpolate(df.Constant(1.), V_J.collapse())

    df.assign(xi,[u_0,p_0,J_0])
    u, p, J = df.split(xi)

    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/d)            # Isochoric decomposition
    IC_bar = df.tr(C_bar)          # Invariant of isochoric C

    # Material parameters
    mu_ff = params["mu_ff"]
    c = params["c"]
    near_field = params["near_field"]

    mu = ShearModulus(mu_ff, c, near_field)
    nu = 0.499 
    kappa = 2*mu_ff*(1+nu)/3/(1-2*nu)

    c1 = mu/2
    c2 = df.Constant(kappa)

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Gateaux Derivative
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_inner = df.Expression(["x[0]/15*c*t","x[1]/15*c*t","x[2]/15*c*t"], c=, t=0, degree=1)

    xsides = df.CompiledSubDomain("near(abs(x[0]), 75) && abs(x[1])<20 && abs(x[2])<20 && on_boundary")

    outer_bc_y = df.DirichletBC(V_u.sub(1), df.Constant(0.), xsides)
    outer_bc_z = df.DirichletBC(V_u.sub(2), df.Constant(0.), xsides)
    inner_bc = df.DirichletBC(V_u, u_inner, boundaries, 202)
    bcs = [inner_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'lu'

    # Solve
    chunks = 5
    total_start = time.time()
    for i in range(chunks):
        print("\nIteration: ", i)
        start = time.time()
        u_inner.t = (i+1)/chunks
        solver.solve()
        print("Time: ", time.time()-start) 
    print("Total Time: ", time.time() - total_start, "s")

    u, p, J = xi.split(True)

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
    F.rename("F", "Deformation Gradient")
    F_file.write(F)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

    with open(output_folder+"log_params", "w+") as f:
        f.write("Mesh: {:s}\n".format(params["mesh"]))
        f.write("No. Elements: {:d}\n".format(mesh.num_cells()))
        f.write("mu_ff = {:e}\n".format(mu_ff))
        f.write("kappa = {:e}\n".format(kappa))
        f.write("c =     {:f}\n".format(c))

if __name__ == "__main__":
    main()
