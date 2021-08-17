from dolfin import *
mesh = UnitIntervalMesh(5)
V    = FunctionSpace(mesh, 'CG', 1)

u = Function(V)
u_vec = u.vector()
u_vec[:] = MPI.comm_world.rank + 1

v_vec = Vector(MPI.comm_self, u_vec.local_size())
u_vec.gather(v_vec, V.dofmap().dofs())

print("Original vs copied: ", u_vec.get_local(), v_vec.get_local())