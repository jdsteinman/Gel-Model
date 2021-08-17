SetFactory("OpenCASCADE");

// Parameters
l = 150; // Side length of box
r = 12.5; // radius of sphere

// Geometry
Box(1) = {-l/2, -l/2, -l/2, l, l, l};
Sphere(2) = {0, 0, 0, r};
BooleanDifference{Volume{1}; Delete;}{Volume{2}; Delete; }

// Physical groups
Physical Volume(301) = {1};
Physical Surface(201) = {8, 9, 10, 11, 12, 13};
Physical Surface(202) = {7};

Mesh.CharacteristicLengthFactor = 4;
Characteristic Length{PointsOf{Physical Surface{202};}} = 1;
Characteristic Length{PointsOf{Physical Surface{201};}} = 2;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "hole.msh";
