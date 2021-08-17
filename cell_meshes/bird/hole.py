import gmsh
import sys
import os

# Initialize and create model
gmsh.initialize()
gmsh.model.add("bird_with_hole")

# Merge cell surface
gmsh.merge("cell_surface_500.stl")

gmsh.model.geo.synchronize()

# Lengths
l = 150.
h = 150.
d = 150.
lc = 1.

# Move cell to center of box
# gmsh.model.geo.translate([2, 1], l/2, h/2, d/2)

# Points
gmsh.model.geo.addPoint(l, h, d, lc, 1)
gmsh.model.geo.addPoint(l, h, 0, lc, 2)
gmsh.model.geo.addPoint(0, h, d, lc, 3)
gmsh.model.geo.addPoint(0, 0, d, lc, 4)
gmsh.model.geo.addPoint(l, 0, d, lc, 5)
gmsh.model.geo.addPoint(l, 0, 0, lc, 6)
gmsh.model.geo.addPoint(0, h, 0, lc, 7)
gmsh.model.geo.addPoint(0, 0, 0, lc, 8)

# Lines
gmsh.model.geo.addLine(3, 1, 1)
gmsh.model.geo.addLine(3, 7, 2)
gmsh.model.geo.addLine(7, 2, 3)
gmsh.model.geo.addLine(2 ,1, 4)
gmsh.model.geo.addLine(1, 5, 5)
gmsh.model.geo.addLine(5, 4, 6)
gmsh.model.geo.addLine(4, 8, 7)
gmsh.model.geo.addLine(8, 6, 8)
gmsh.model.geo.addLine(6, 5, 9)
gmsh.model.geo.addLine(6, 2, 10)
gmsh.model.geo.addLine(3, 4, 11)
gmsh.model.geo.addLine(8, 7, 12)

# Curve Loops
gmsh.model.geo.addCurveLoop([-6,-5,-1,11], 1)
gmsh.model.geo.addCurveLoop([4,5,-9,10], 2)
gmsh.model.geo.addCurveLoop([-3,-12,8,10], 3)
gmsh.model.geo.addCurveLoop([7,12,-2,11], 4)
gmsh.model.geo.addCurveLoop([-4,-3,-2,1], 5)
gmsh.model.geo.addCurveLoop([8,9,6,7], 6)

# Planes
gmsh.model.geo.addPlaneSurface([1], 2)
gmsh.model.geo.addPlaneSurface([1], 3)
gmsh.model.geo.addPlaneSurface([1], 4)
gmsh.model.geo.addPlaneSurface([1], 5)
gmsh.model.geo.addPlaneSurface([1], 6)
gmsh.model.geo.addPlaneSurface([1], 7)

# Surface Loop
gmsh.model.geo.addSurfaceLoop([1, 2, 7, -4, 6, 3, -5], 1)

# Gel Volume
gmsh.model.geo.addVolume([1], 2)

# Physical Groups
gmsh.model.geo.addPhysicalGroup(2, [2, 7, -4, 6, 3, -5], 201)
gmsh.model.geo.addPhysicalGroup(2, [1], 202)
gmsh.model.geo.addPhysicalGroup(3, [2], 301)

# Mesh density
gmsh.option

# Synchronize
gmsh.model.geo.synchronize()

# We finally generate and save the mesh:
gmsh.model.mesh.generate(2)
# gmsh.write("t2.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
