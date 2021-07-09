import dolfin as df
import numpy as np
import mshr
import math
import time
import os
import sys
import shutil
from scipy.sparse import csr_matrix, coo_matrix
from mpi4py import MPI

'''
    A 3D test
'''

#res_dir = 'results_complex_c2_10'
res_dir = 'output_gradient_3'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
df.set_log_level(40)

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False 
df.parameters['form_compiler']['quadrature_degree'] = 2
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 100000

if rank == 0:
    print('result folder = {}'.format(res_dir))
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
        shutil.copyfile('main.py',os.path.join(res_dir,'main.py'))

mesh = df.Mesh()
with df.XDMFFile("./mesh/mesh_gradient_tetra.xdmf") as infile:
    infile.read(mesh)
for n in range(0):
    mesh = df.refine(mesh)
val = np.array(len(mesh.cells()),'d')
val_sum = np.array(0.,'d')
comm.Reduce(val, val_sum, op=MPI.SUM, root=0)
if rank == 0:
    print('Total number of elements = {:d}'.format(int(val_sum)))
'''
# only work in serial
for p in mesh.coordinates():
    print(p)
    print(math.sqrt(p[0]**2+p[1]**2))
    print('-'*3)
print(mesh.topology().dim())
print(mesh.geometry().dim())
'''
#I noticed that order 2 for u is necessary for nearly incompressible material
#when the BC is complex. Double check!
element_u = df.VectorElement("CG",mesh.ufl_cell(),3)
element_p = df.FiniteElement("DG",mesh.ufl_cell(),0)
element_J = df.FiniteElement("DG",mesh.ufl_cell(),0)
W = df.FunctionSpace(mesh,df.MixedElement([element_u,element_p,element_J]))
xi = df.Function(W)
xi.rename('xi','mixed solution')

V_u = W.sub(0).collapse()
V_p = W.sub(1).collapse()
V_J = W.sub(2).collapse()
u_0 = df.interpolate(df.Constant((0.0, 0.0, 0.0)), V_u)
p_0 = df.interpolate(df.Constant(0.0), V_p)
J_0 = df.interpolate(df.Constant(1.), V_J)
#print(u_0.vector()[:])
#print(p_0.vector()[:])
#print(J_0.vector()[:])
'''
fa = df.FunctionAssigner(W,[V_u,V_p,V_J])
fa.assign(xi,[u_0, p_0, J_0])
'''
df.assign(xi,[u_0,p_0,J_0])
u,p,J = xi.split(deepcopy=True)
#print(u.vector()[:])
#print(p.vector()[:])
#print(J.vector()[:])
u.rename('u','displacement')
p.rename('p','pressure')
J.rename('J','Jacobian')
disp_File = df.File(os.path.join(res_dir,'u.pvd'))
disp_File << (u,0)
p_File = df.File(os.path.join(res_dir,'p.pvd'))
p_File << (p,0)
J_File = df.File(os.path.join(res_dir,'J.pvd'))
J_File << (J,0)
#u_copy,p_copy,J_copy = xi.split(deepcopy=True)
u,p,J = df.split(xi)

xi_ = df.TestFunction(W)
dxi = df.TrialFunction(W)

d = mesh.geometry().dim()
I = df.Identity(d)
F = I+df.grad(u)
C = F.T*F
b = F*F.T
E = (C-I)/2
IC = df.tr(C)
C_bar = C/J**(2/d)
b_bar = b/J**(2/d)
IC_bar = df.tr(C_bar)
Ju = df.det(F)

c1 = df.Constant(1.)
c2 = df.Constant(100.)
psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)
#psi = c1*(IC-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)
#psi = c1*(IC_bar-d)
'''
PK2 stress
'''
C = df.variable(C)
S = 2*df.diff(psi,C)

B = df.Constant((0.,0.,0.)) 

dx = df.Measure('dx',domain=mesh)
Pi = psi*dx - df.dot(B,u)*dx
res = df.derivative(Pi,xi,xi_)
Dres = df.derivative(res,xi,dxi)

class top(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[2],1)
class bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[2],0)
class inner(df.SubDomain):
    def inside(self, x, on_boundary):
        r = math.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2+(x[2]-0.5)**2)
        return on_boundary and r < 0.2
class outer(df.SubDomain):
    def inside(self, x, on_boundary):
        r = math.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2+(x[2]-0.5)**2)
        return on_boundary and r > 0.2

u_inner = df.Expression(("0.2*t*(x[0]-0.5)","0.2*t*(x[1]-0.5)","-0.5*t*(x[2]-0.5)"),t=0,degree=2)
#u_inner = df.Expression(("0.","0.","0.5*t*(x[2]-0.5)"),t=0,degree=2)

boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
top().mark(boundaries,1)
bottom().mark(boundaries,2)
inner().mark(boundaries,3)
outer().mark(boundaries,4)
'''
with np.printoptions(threshold=np.inf):
    print(boundaries.array())
'''
bc1 = df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0, 0.0)), boundaries, 1)
bc2 = df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0, 0.0)), boundaries, 2)
bc3 = df.DirichletBC(W.sub(0), u_inner, boundaries, 3)
bc4 = df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0, 0.0)), boundaries, 4)
bcs = [bc3]

problem = df.NonlinearVariationalProblem(res,xi,bcs,J=Dres)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
'''
# TODO: switch to PETSc SNES solver, which may work better!
solver = df.PETScSNESSolver()
solver.parameters['linear_solver'] = 'mumps'
solver.parameters['line_search'] = 'bt'
solver.parameters['relative_tolerance'] = 1e-3
solver.parameters['preconditioner'] = 'ilu'
solver.parameters['method'] = 'newtonls'
solver.parameters['solution_tolerance'] = 1e-5
'''

chunks = 1
output_freq = 1
start = time.time()
sys.stdout.flush() 
for n in range(1,chunks+1):
    if rank == 0:
        print('n = {}'.format(n))
    u_inner.t = 1.0*n/chunks
    #print(df.assemble(res)[:])
    solver.solve()
    #print(df.assemble(res)[:])
    '''
    M = df.assemble(Dres)
    Mmat = df.as_backend_type(M).mat()
    mat = coo_matrix(csr_matrix(Mmat.getValuesCSR()[::-1]))
    print(mat.data)
    '''
    if n%output_freq == 0:
        if rank == 0:
            print('>> Checking out')
        u,p,J = xi.split(deepcopy=True)
        u.rename('u','displacement')
        p.rename('p','pressure')
        J.rename('J','Jacobian')
        disp_File << (u,n)
        p_File << (p,n)
        J_File << (J,n)
    
    end = time.time()
    time_elapsed = end - start
    if rank == 0:
        print('Time elapsed = {:2.1f}s'.format(time_elapsed))
    sys.stdout.flush() 

if rank == 0:
    print('Done')

