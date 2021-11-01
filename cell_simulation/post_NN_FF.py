import h5py
import os
import numpy as np
import dolfin as df
from post_tools import PointData

def main():
    # Bead data
    init  = np.loadtxt('../cell_data/bird/beads_init_filtered.txt')
    final = np.loadtxt('../cell_data/bird/beads_final_filtered.txt')
    u_data = final-init

    # Simulated displacement
    mesh = df.Mesh()
    with df.XDMFFile("output/bird/1000/hole.xdmf") as infile:
        infile.read(mesh)

    V = df.VectorFunctionSpace(mesh, "CG", 2)
    u = df.Function(V)
    u_file = df.XDMFFile("output/bird/1000/out.xdmf")
    u_file.read_checkpoint(u, "u", 0)
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

    points = []
    u_sim = []
    u_data = []
    F_arr = []
    for i, (x,X) in enumerate(zip(init,final)):
        try:
            u_sim.append(u(x))
            points.append(x) 
            F_arr.append(F(x).reshape((3,3)))
            u_data.append(X-x)
        except: 
            print(x)

    point_data = PointData(points, u_sim, u_data, F_arr)
    point_data.save_to_vtk("output/bird/1000/bead_data.vtk")
if __name__=="__main__":
    main()
