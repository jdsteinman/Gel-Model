import pyvtk
import numpy as np

path = 'bird/'
surface_points = np.loadtxt(path+'CytoD_vertices.txt')
surface_faces = np.loadtxt(path+'CytoD_faces.txt', dtype=int)
surface_disp = np.loadtxt(path+'displacements.txt')

vtk = pyvtk.VtkData(\
    pyvtk.UnstructuredGrid(surface_points,
        triangle=surface_faces),
    pyvtk.PointData(pyvtk.Vectors(surface_disp, name="displacement")),
    "Surface Displacements")
vtk.tofile(path+'surface_displacements')

beads_init = np.loadtxt(path+'beads_init_filtered.txt')
beads_final = np.loadtxt(path+'beads_final_filtered.txt')
u_beads = beads_final - beads_init

vtk = pyvtk.VtkData(\
    pyvtk.PolyData(beads_init),
    pyvtk.PointData(pyvtk.Vectors(u_beads, name="displacement")),
    "Bead Displacements"
    )
vtk.tofile(path+'bead_dispalcements_filtered')
