Cube with ellipsoidal hole example. Compiled meshes and output are already included.

File descriptions:
    sphere_in_cube.geo           -   geo file that generates mesh
    sphere_in_cube.msh           -   mesh file created by gmsh
    convert_meshes.py            -   Takes .msh file and converts to xml
    sphere_in_cube.py            -   main script that executes simulation
    
Necessary mesh files are already included, but to regenerate them (must have gmsh installed):
    1. Run "sphere_in_cube.geo" and save 3D mesh as sphere_in_cube.msh (version 2.2)
    2. Run "python convert_mesh.py" to create XML files

To run simulation:
    1. Run "python sphere_in_cube.py"
    2. Outputs should appear "./output/*" 

To view paraview state file:
    1. Open paraview
    2. Load State
    3. Select "output/plots.pvsm"