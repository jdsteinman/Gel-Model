# Cell Data

## Directory skeleton
<pre>
cell_data
|-- cell_1/
|   |-- CytoD_vertices.txt
|   |-- CytoD_faces.txt
|   |-- interpolated_NI_surface_data.csv
|   |-- NI/
|   |   |-- meshes/
|   |   |   |-- hole.geo
|   |   |-- displacements/
|   |-- IN/
|   |   |-- meshes/
|   |   |   |-- hole.geo
|   |   |-- displacements/
|-- utils
|   |-- get_NI_mesh_from_IN.py
|   |-- msh_to_xdmf.py
|   |-- stl_to_txt.py
|   |-- write_displacements.py
|   |-- write_surface_stl.py


</pre>

## How to configure IN data
1. Use `write_surface_stl.py` to create surface stl mesh in `IN/meshes`
2. Use `stl_to_txt.py` to save surface vertices/faces in `IN/meshes`
3. Use `gmsh` to create volume meshes in .msh format
4. Use `msh_to_xdmf.py` to convert to xdmf
5. Use `write_displacements.py` to save surface displacements in `IN/displacements`. Multiply by -1 for contraction.

## How to configure NI data
6. Use `get_NI_mesh_from_IN.py` to create normal surface mesh in `NI/meshes`
7. Follow steps 2-5 as above. 

## How to coarsen/refine surface data
8. Use meshlab to create a coarsened/refined stl surface mesh
9. Use `stl_to_txt.py` to save surface vertices/faces in `IN/meshes`
10. Interpolate surface data and save in `cell_data/`
11. Continue with steps 3-7 above