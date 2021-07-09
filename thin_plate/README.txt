Simulation of 2D thin plate. Includes linear elastic, hyperelastic single and three-field formulations.

Mesh descriptions:
    plate_with_hole.geo           -   square plate with circular hole 
    quarter_plate_with_hole.msh   -   top right quadrant of spare plate with circular hole
    solid_plate.geo               -   thin, square plate
    
Necessary mesh files are already included, but to regenerate them (must have gmsh installed):
    1. Run geo file and save 2D mesh as version 2.2
    2. Run "python convert_mesh.py" to create XML files
