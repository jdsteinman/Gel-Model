SetFactory("OpenCASCADE");

// Cube
l = 50; // Side length of box
Box(1) = {-l/2, -l/2, -l/2, l, l, l};

// Sphere
Sphere(2) = {0, 0, 0, 10};
BooleanDifference{Volume{1}; Delete;}{Volume{2};};

// Beam
Box(3) = {-1, -1, -5, 2, 2, 10};
BooleanDifference{Volume{2}; Delete;}{Volume{3};};

// Physical Volumes
Physical Volume(301) = {1}; // Box
Physical Volume(302) = {2}; // Sphere
Physical Volume(303) = {3}; // Rod

// // Physical Surfaces
Physical Surface(201) = {8, 9, 10, 11, 12, 13};
Physical Surface(202) = {7};
Physical Surface(203) = {14, 15, 16, 17, 18, 19};

Mesh.CharacteristicLengthFactor = 0.25;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;

