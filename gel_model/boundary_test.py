import os
import time
import meshio
import numpy as np
import pandas as pd
from dolfin import *
from pyevtk.hl import unstructuredGridToVTK
from classes import Isosurfaces, LineData
from sim_tools import *
from deformation_grad import def_grad

import matplotlib.pyplot as plt

## Files
tag = ""
mesh_path = "../meshes/ellipsoid/"
output_folder = "./output/bctest/bcs/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

## Meshes
mesh = Mesh()
with XDMFFile(mesh_path + "ellipsoid_tetra.xdmf") as infile:
    infile.read(mesh)

## Function space
V = VectorFunctionSpace(mesh, "CG", 2)
du, w = TrialFunction(V), TestFunction(V)    # Incremental displacement
u = Function(V)
u.vector()[:] = 0

## Subdomain markers
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(mesh_path + "ellipsoid_triangle.xdmf") as infile:
    infile.read(mvc, "triangle")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
outer_number = 200
inner_number = 201
volume_number = 300

##  Boundary Conditions
zero = Constant((0.0, 0.0, 0.0))
bcs = []

# Boundary Function
def ellipsoid_surface(x, on_boundary):
    return on_boundary and abs(x[0]) < 30 and abs(x[1]) < 30 and abs(x[2]) < 30

u_D = Expression(["t*x[0]*a/10", "t*x[1]*b/10", "-t*x[2]*c/20"], a=.1, b=.1, c=.2, t=1, degree=1)

testbc = DirichletBC(V, u_D, mf, inner_number) 
# testbc = DirichletBC(V, u_D, ellipsoid_surface) 


testbc.apply(u.vector())
# Save to Paraview

F = Identity(3) + grad(u)
F = project(F, V=TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

## XDMF Outputs
disp_file = XDMFFile(output_folder + "displacement_" + tag + ".xdmf")
u.rename("u","displacement")
disp_file.write(u)

F_file = XDMFFile(output_folder + "F_" + tag + ".xdmf")
F.rename("F","deformation gradient")
F_file.write(F)

mf_file = XDMFFile(output_folder + "mf_" + tag + ".xdmf")
mf_file.write(mf)

