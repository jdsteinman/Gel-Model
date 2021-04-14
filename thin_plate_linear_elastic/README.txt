WORKING example. Compiled meshes and output are already included.

File descriptions:
    plate_with_hole.geo           -   geo file that generates mesh
    plate_with_hole.msh           -   mesh file created by gmsh
    convert_meshes.py             -   Takes .msh file and converts to xml
    plate_with_hole.py            -   main script that executes simulation
    
Necessary mesh files are already included, but to regenerate them (must have gmsh installed):
    1. Run "plate_with_hole.geo" and save 2D mesh as plate_with_hole.msh (version 2.2)
    2. Run "python convert_mesh.py" to create XML files

To run simulation:
    1. Run "python plate_with_hole_elastic.py"
    2. Outputs should appear "./output/*" 

To view paraview state file:
    1. Open paraview
    2. Load State
    3. Select "output/plots.pvsm"
    4. Change file locations if necessary