Cube with spherical hole example.
    hole.geo converges
    hole_with_inner_cube.geo does not

File descriptions:
    hole.geo           -   geo file that generates mesh
    hole.msh           -   mesh file created by gmsh
    convert_to_xdmf.py           -   Takes .msh file and converts to xdmf
    hole.py            -   main script that executes simulation
    
Necessary mesh files are already included, but to regenerate them (must have gmsh installed):
    1. Run "hole.geo" and save 3D mesh as sphere_in_cube.msh (version 2.2)
    2. Run "python convert__to_xdmf.py" to create XML files

To run simulation:
    1. Run "python spherical_hole.py"
    2. Outputs should appear "./output/*" 

To view paraview state file:
    1. Open paraview
    2. Load State
    3. Select "output/plots.pvsm"