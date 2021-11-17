SetFactory("OpenCASCADE");

// Cube
l = 50; // Side length of box
Box(1) = {0, -l/2, -l/2, l, l, l};

// Sphere
Sphere(2) = {0, 0, 0, 1};
psax = 10.;
psay = 2.5;
psaz = 2.5;
Dilate{{0, 0, 0}, {psax, psay, psaz}} {Volume{2};}

// Difference
BooleanFragments{Volume{1,2}; Delete;}{}
Delete{Volume{3}; Surface{10};}

// Physical Volumes
Physical Volume(301) = {1};  // Gel
Physical Volume(302) = {2};  // Cell

// Physical Surfaces
Physical Surface(201) = {1,2,3,4,5,8,9};
Physical Surface(202) = {6,7};

// Mesh.CharacteristicLengthFactor = 0.5;
Mesh.Algorithm = 5;
Characteristic Length{PointsOf{Physical Surface{201};}} = 5;
Characteristic Length{PointsOf{Physical Surface{202};}} = 1;
