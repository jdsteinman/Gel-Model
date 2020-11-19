Merge "cytod_uncentered_unpca.stl";
CreateTopology;

Physical Surface(1) = {1};  // VIC surface


// Generate Mesh
Mesh.CharacteristicLengthFactor = 10;
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "cytod_uncentered_unpca_surface.msh";
