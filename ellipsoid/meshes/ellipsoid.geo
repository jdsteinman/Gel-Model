SetFactory("OpenCASCADE");

// Cube
l = 75; // Side length of box
Box(1) = {-l/2, -l/2, -l/2, l, l, l};

// Sphere
Sphere(2) = {0, 0, 0, 1};
psax = 10.;
psay = 10.;
psaz = 20.;
Dilate{{0, 0, 0}, {psax, psay, psaz}} {Volume{2};}

// Difference
BooleanDifference{Volume{1}; Delete;}{Volume{2}; Delete; }

// Physical Volumes
Physical Volume(301) = {1};

// Physical Surfaces
Physical Surface(201) = {8, 9, 10, 11, 12, 13};
Physical Surface(202) = {7};

Mesh.CharacteristicLengthFactor = 0.75;
Characteristic Length{PointsOf{Physical Surface{202};}} = 1;
Characteristic Length{PointsOf{Physical Surface{201};}} = 5;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;

