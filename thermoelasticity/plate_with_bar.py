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
    def __init__(self, domains, dT, **kwargs):
        super().__init__(**kwargs)
        self.domains = domains
        self.dT = dT
    
    def eval_cell(self, value, x, cell):
        if self.domains[cell.index] == 203:
            value[0] = self.dT
        else:
            value[0] = 0.0
    
    def value_shape(self):
        return()

def main():

    params = {}

    params['output_folder'] = './output/plate_with_bar/'

    params['mesh'] = df.Mesh("./meshes/plate_with_bar.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/plate_with_bar_physical_region.xml")

    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/plate_with_bar_facet_region.xml")

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
    U = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    V = df.FunctionSpace(mesh, U)

    du, w = df.TrialFunction(V), df.TestFunction(V) 
    u = df.Function(V)
    u.vector()[:] = 0

    # Kinematics
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Constant((0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Material parameters
    nu = 0.49                      # Poisson's ratio
    mu = df.Constant(1.)       # Bulk Modulus
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter

    # Linear Thermoelastic model
    VT = df.FunctionSpace(mesh, "CG", 1)       
    T_expr = TempExpression(domains=domains, dT=-50.)
    dT = df.interpolate(T_expr, VT)
    alpha = df.Constant(1e-2)

    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))-alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

    # Stored strain energy density (compressible neo-Hookean model)
    psi_1 = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2
    psi_2 = inner(sigma(u, dT), eps(u))

    # Total potential energy
    Pi = psi_1*dx(201) + psi_2*dx(202) + psi_2*dx(203) - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, u, w)
    Jac = df.derivative(F, u, du)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0))
    bcs = [df.DirichletBC(V, zero, boundaries, 101)]
    # bcs.append(df.DirichletBC(V.sub(0),  df.Constant(0.0), boundaries, 102))

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, u, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    
    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    # Projections
    F = Identity(2) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

    # Extract values in outer region
    submesh = df.SubMesh(mesh, domains, 201)
    u_outer = df.interpolate(u, df.FunctionSpace(submesh, U))
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