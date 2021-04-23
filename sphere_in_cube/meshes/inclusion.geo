SetFactory("OpenCASCADE");

// Cube
l = 50; // Side length of box
Box(1) = {-l/2, -l/2, -l/2, l, l, l};

// Sphere
Sphere(2) = {0, 0, 0, 10};

// Difference
BooleanDifference(3) = {Volume{1}; Delete; }{Volume{2}; };

// Physical Volumes
Physical Volume(301) = {3};
Physical Volume(302) = {2};

// Physical Surfaces
Physical Surface(201) = {8, 9, 10, 11, 12, 13};
Physical Surface(202) = {7};

Mesh.CharacteristicLengthFactor = 0.6;
Characteristic Length{PointsOf{Physical Surface{202};}} = 1;
Characteristic Length{PointsOf{Physical Surface{201};}} = 3;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
