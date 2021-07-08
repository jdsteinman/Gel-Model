import dolfin as df
from dolfin.function.expression import UserExpression
import numpy as np
import os
import time
import meshio
import ufl


"""
Written by: John Steinman
Fenics simulation of hyperelastic ellipsoidal model with thermal contraction.
- outputs:
    - displacement and deformation gradient fields (XDMF)
""" 

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

class TempExpression(UserExpression):
    def __init__(self, domains, dT_x, dT_y, **kwargs):
        super().__init__(**kwargs)
        self.domains = domains
        self.dT_x = dT_x
        self.dT_y = dT_y
    
    def eval_cell(self, value, x, cell):
        if self.domains[cell.index] == 203:
            value[0] = self.dT_x
        elif self.domains[cell.index] == 204:
            value[0] = self.dT_y
        else:
            value[0] = 0.0
    
    def value_shape(self):
        return()

def main():

    params = {}

    params['output_folder'] = './output/plate/'

    params['mesh'] = df.Mesh("./meshes/plate_with_cross.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/plate_with_cross_physical_region.xml")

    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/plate_with_cross_facet_region.xml")

    solver_call(params)


def solver_call(params):
    from dolfin import ln, dot, inner, sym, det, tr, grad, Identity

    # Mesh
    mesh = params["mesh"]

    domains = params["physical_region"]
    boundaries = params["facet_region"]

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    V = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    W = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    R = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)   
    M = df.FunctionSpace(mesh, df.MixedElement([V,W,R]))
    V, W, R = M.split()

    # Trial and Test functions
    dxi = df.TrialFunction(M)
    xi_ = df.TestFunction(M)

    # Functions from most recent iteration
    xi = df.Function(M) 

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V.collapse())
    p_0 = df.interpolate(df.Constant((0.0)), W.collapse())
    J_0 = df.interpolate(df.Constant((1.0)), R.collapse())
    df.assign(xi, [u_0, p_0, J_0])

    # Variational forms
    u, p, Jt = df.split(xi)

    # Kinematics
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Constant((0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    b = F*F.T                   # Left Cauchy-Greem tensor

    # Invariants of deformation tensors
    J = det(F)
    I = tr(b) 
    Ibar = I * J**-1

    # Material parameters
    nu = df.Constant(0.49)         # Poisson's ratio
    mu = df.Constant(1.)           # Bulk Modulus
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter
    kappa = lmbda + 2*mu/3

    c1 = kappa/4
    c2 = mu/2

    # Linear Thermoelastic model
    VT = df.FunctionSpace(mesh, "CG", 1)       
    T_expr = TempExpression(domains=domains, dT_x=0.0, dT_y=1.0e-5)
    dT = df.interpolate(T_expr, VT)
    alpha = df.Constant(1e-5)

    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))-alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

    # Stored strain energy density
    psi_1 = c1*(Jt**2-1-2*ln(Jt))
    psi_2 = c2*(Ibar-2)
    psi_3 = p*(J-Jt)

    psi_H = psi_1 + psi_2 + psi_3         # Hyperelastic 
    psi_T = inner(sigma(u, dT), eps(u))   # Thermoelastic

    # Total potential energy
    Pi = psi_H*dx(201) + psi_T*dx(202) + psi_T*dx(203) - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative)
    F = df.derivative(Pi, xi, xi_)
    Jac = df.derivative(F, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0))
    bcs = [df.DirichletBC(V, zero, boundaries, 101)]
    # bcs.append(df.DirichletBC(V.sub(0),  df.Constant(0.0), boundaries, 102))

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    
    start = time.time()
    solver.solve()
    u, p, Jt = xi.split() 
    print("Total Time: ", time.time() - start, "s")

    # Projections
    F = Identity(2) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

    # Extract values in outer region
    submesh = df.SubMesh(mesh, domains, 201)
    ele = df.VectorElement('Lagrange', submesh.ufl_cell(), 2)
    u_outer = df.interpolate(u, df.FunctionSpace(submesh, ele))
    F_outer = df.interpolate(F, df.TensorFunctionSpace(submesh, "CG", 1, shape=(2, 2)))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u_outer.rename("U","displacement")
    disp_file.write(u_outer)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F_outer.rename("F","deformation gradient")
    F_file.write(F_outer)

    T_file = df.XDMFFile(output_folder+"dT.xdmf")
    dT.rename("dT","Temperature change")
    T_file.write(dT)

if __name__ == "__main__":
    main()