from dolfin import *
import matplotlib.pyplot as plt
import os
import dolfin as dlf
import numpy as np
import math 

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"


# traction 
class Traction(UserExpression):
    def __init__(self):
        super().__init__(self)
        self.t = 0.0
    def eval(self, values, x):
        values[0] = self.t
        values[1] = 0.0
        values[2] = 0.0
    def value_shape(self):
        return (3,)

# Kinematics
def pk1Stress(u,pressure,E,nu):
    G = E/(2*(1+nu))
    c1 = G/2.0
    
    I = Identity(V.mesh().geometry().dim())  # Identity tensor
    F = I + grad(u)          # Deformation gradient
    C = F.T*F                # Right Cauchy-Green tensor
    Ic = tr(C)               # Invariants of deformation tensors
    J = det(F)
    pk2 = 2*c1*I-pressure*inv(C) # second PK stress
    return pk2, (J-1)


def geometry_3d():
    mesh = UnitCubeMesh(6, 6, 6)

    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    x0 = AutoSubDomain(lambda x: near(x[0], 0))
    x1 = AutoSubDomain(lambda x: near(x[0], 1))
    y0 = AutoSubDomain(lambda x: near(x[1], 0))
    z0 = AutoSubDomain(lambda x: near(x[2], 0))
    x0.mark(boundary_parts, 1)
    y0.mark(boundary_parts, 2)
    z0.mark(boundary_parts, 3)
    x1.mark(boundary_parts, 4)
    return boundary_parts



# Create mesh and define function space ============================================
facet_function = geometry_3d()
mesh = facet_function.mesh()
gdim = mesh.geometry().dim()
dx = Measure("dx")
ds = Measure("ds", subdomain_data=facet_function, subdomain_id=4)
print('Number of nodes: ',mesh.num_vertices())
print('Number of cells: ',mesh.num_cells())


# Limit quadrature degree
dx = dx(degree=4)
ds = ds(degree=4)


# Create function space
element_v = VectorElement("P", mesh.ufl_cell(), 1) #  
element_s = FiniteElement("P", mesh.ufl_cell(), 1) #
mixed_element = MixedElement([element_v, element_s]) #  
V = FunctionSpace(mesh, mixed_element)


# Define test and trial functions
dup = TrialFunction(V)
_u, _p = TestFunctions(V)

_u_p = Function(V)
u, p = split(_u_p)


# Create tensor function spaces for stress and stretch output
W_DFnStress = TensorFunctionSpace(mesh, "DG", degree=0)
defGrad = Function(W_DFnStress, name='F')
PK1_stress = Function(W_DFnStress, name='PK1')


# Displacement from previous iteration
b = Constant((0.0,0.0, 0.0)) # Body force per unit mass
h = Traction() # Traction force on the boundary


# Define Dirichlet boundary
bc0 = DirichletBC(V.sub(0).sub(0), Constant(0.), facet_function, 1)
bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), facet_function, 2)
bc2 = DirichletBC(V.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('1.0*time_'),time_=0.0 , degree=0)
bc3 = DirichletBC(V.sub(0).sub(0), tDirBC, facet_function, 4)
bcs = [bc0,bc1,bc2,bc3]

# material parameters
E, nu = 1e3, 0.5

# # Total potential energy
pkstrs, hydpress =  pk1Stress(u,p,E,nu)
I = Identity(V.mesh().geometry().dim())
dgF = I + grad(u)
F1 = inner(dot(dgF,pkstrs), grad(_u))*dx - dot(b, _u)*dx - dot(h, _u)*ds
F2 = hydpress*_p*dx                                               
F = F1+F2
J = derivative(F, _u_p,dup)

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, _u_p, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'


# Time stepping parameters
dt = 0.1
t, T = 0.0, 10*dt

# Save solution in VTK format
file_results = XDMFFile("./Results/TestUniaxialLoading/Uniaxial.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

stretch_vec = []
stress_vec=[]
stress_ana=[]

while t <= T:
    print('time: ', t)

    # Increase traction
    # h.t = t

    # tDirBC.time_ = t

    # solve and save disp
    solver.solve()
    
    # Extract solution components
    u_plot, p_plot = _u_p.split()
    u_plot.rename("u", "displacement")
    p_plot.rename("p", "pressure")
    
    # get stretch at a point for plotting
    point = (0.5,0.5,0)
    DF = I + grad(u_plot)
    defGrad.assign(project(DF, W_DFnStress))
    stretch_vec.append(defGrad(point)[0])

    # get stress at a point for plotting
    PK1_s,thydpress = pk1Stress(u_plot,p_plot,E,nu)
    PK1_stress.assign(project(PK1_s, W_DFnStress))
    stress_vec.append(PK1_stress(point)[0])
    
    # save xdmf file
    file_results.write(u_plot, t)
    file_results.write(defGrad, t)
    file_results.write(PK1_stress, t)

    # time increment
    t += float(dt)

    
# get analytical solution
stretch_vec = np.array(stretch_vec)
stress_vec = np.array(stress_vec)
G = E/(2*(1+nu))
c1 = G/2.0
for i in range(len(stretch_vec)):
    pk1_ana = 2*c1*(stretch_vec[i] - 1/(stretch_vec[i]*stretch_vec[i])) #PK1
    pk2_ana = (1/stretch_vec[i])*pk1_ana # PK2
    stress_ana.append(pk2_ana)  
stress_ana =  np.array(stress_ana)  

# plot results
f = plt.figure(figsize=(12,6))
plt.plot(stretch_vec, stress_vec,'r-')
plt.plot(stretch_vec, stress_ana,'k.')

plt.show()