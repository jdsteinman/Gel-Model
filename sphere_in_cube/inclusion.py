import dolfin as df
import numpy as np
import sys
import os
import time

"""
Written by: John Steinman
"""

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

class shearModulus(df.UserExpression):
    def __init__(self, cell_function, params, **kwargs):
        assert(cell_function.dim()==3)
        self.cell_function = cell_function
        self.outer = params["outer"]["mu"]
        self.inner = params["inner"]["mu"]

        super().__init__(**kwargs)

    def value_shape(self):
        return ()

    def eval_cell(self, value, x, ufl_cell):
        if self.cell_function[ufl_cell.index] == 301:
            value[0] = self.outer
        elif self.cell_function[ufl_cell.index] == 302:
            value[0] =self.inner
        else:
            raise ValueError("CellFunction with bad value")

def main():

    params = {}

    params['inner'] = {}
    params['outer'] = {}

    params['inner']['mu'] = 325e12
    params['outer']['mu'] = 325e24

    params['output_folder'] = './output/inclusion/'

    params['mesh'] = df.Mesh("./meshes/inclusion.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/inclusion_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/inclusion_facet_region.xml")

    solver_call(params)

def solver_call(params):
    from dolfin import ln, dot, det, tr, grad, Identity

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
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Material parameters
    nu = 0.49                           # Poisson's ratio
    mu = shearModulus(domains, params)  # Bulk Modulus
    lmbda = 2*nu*mu/ (1-2*nu)           # 1st Lame Parameter

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - d) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx(301) + psi*dx(302) - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, u, w)
    Jac = df.derivative(F, u, du)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_a = df.Expression(["-x[0]/10","-x[1]/10","-x[2]/10"], degree=2)

    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V, u_a, boundaries, 202)

    bcs = [inner_bc, outer_bc]

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
    F = Identity(3) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

    with df.XDMFFile(output_folder+"out.xdmf") as outfile:
        outfile.write(mesh)
        outfile.write_checkpoint(u, "u", 0, append=True)
        outfile.write_checkpoint(F, "F", 0, append=True)

if __name__ == "__main__":
    main()