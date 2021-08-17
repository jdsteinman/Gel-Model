import pyvtk
import numpy as np

"""
Write surface displacements to vtk file
"""

path = 'bird/'
points = np.loadtxt(path+'surface_vertices_500.txt')
faces = np.loadtxt(path+'surface_faces_500.txt', dtype=int)
disp = np.loadtxt('../cell_data/bird/surface_displacements_500.txt')
print(disp)

vtk = pyvtk.VtkData(\
    pyvtk.UnstructuredGrid(points,
        triangle=faces),
    pyvtk.PointData(pyvtk.Vectors(disp, name="displacement")),
    "Surface Displacement")
vtk.tofile(path+'cell_500_test')