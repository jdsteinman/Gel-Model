import numpy as np
import meshio

cell = 'star_destroyer'
exp = 'NI'

surface_points = np.loadtxt("../"+cell+"/"+exp+"/meshes/cell_surface_coarse_vertices.txt")
surface_faces = np.loadtxt("../"+cell+"/"+exp+"/meshes/cell_surface_coarse_faces.txt", dtype=int)
surface_disp = np.loadtxt("../"+cell+"/"+exp+"/displacements/surface_displacements_coarse.txt")

mesh = meshio.Mesh(surface_points, 
        cells=[("triangle", surface_faces)],
        point_data={"u":surface_disp})

mesh.write("../"+cell+"/"+exp+"/displacements/surface_displacements_coarse.vtk")