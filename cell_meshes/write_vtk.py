import pyvtk
import numpy as np

"""
Write surface displacements to vtk file
"""

path = 'new_cell/'
points = np.loadtxt(path+'cell_surface_1000_vertices.txt')
faces = np.loadtxt(path+'cell_surface_1000_faces.txt', dtype=int)
disp = np.loadtxt('../cell_data/new_cell/surface_1000_displacements.txt')

vtk = pyvtk.VtkData(\
    pyvtk.UnstructuredGrid(points,
        triangle=faces),
    pyvtk.PointData(pyvtk.Vectors(disp, name="displacement")),
    "Surface Displacement")
vtk.tofile(path+'cell_surface')