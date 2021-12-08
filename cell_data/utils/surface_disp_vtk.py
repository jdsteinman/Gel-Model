import numpy as np
import meshio

cell = 'finger'
exp = 'NI'

surface_points = np.loadtxt("../"+cell+"/"+exp+"/meshes/cell_surface_vertices.txt")
surface_faces = np.loadtxt("../"+cell+"/"+exp+"/meshes/cell_surface_faces.txt", dtype=int)
surface_disp = np.loadtxt("../"+cell+"/"+exp+"/displacements/surface_displacements.txt")

mesh = meshio.Mesh(surface_points, 
        cells=[("triangle", surface_faces)],
        point_data={"u":surface_disp})

mesh.write("../"+cell+"/"+exp+"/displacements/surface_displacements.vtk")