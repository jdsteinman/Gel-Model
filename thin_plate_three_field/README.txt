Three-field hyperelastic formulation of thin plate simulation. Compiled meshes and output are already included.

Mesh descriptions:
    plate.geo                     -   thin, square plate
    quarter_plate_with_hole.msh   -   top right quadrant of spare plate with circular hole
    
Necessary mesh files are already included, but to regenerate them (must have gmsh installed):
    1. Run geo file and save 2D mesh as version 2.2
    2. Run "python convert_mesh.py" to create XML files

To run simulation:
    1. Run "python plate.py" or "python quarter_plate_with_hole.py"
    2. Outputs should appear "./output/*" 

To view paraview state file:
    1. Open paraview
    2. Load State
    3. Select "quarter_plate.pvsm" or "plate.pvsm"