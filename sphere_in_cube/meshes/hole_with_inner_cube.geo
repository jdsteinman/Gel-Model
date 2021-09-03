SetFactory("OpenCASCADE");

// Parameters
LL = 150;  // Side length of box
ll = 50;   // Side length of near field
D  = 25;  // diameter of sphere

// Box
Box(1) = {-LL/2, -LL/2, -LL/2, LL, LL, LL};
Box(2) = {-ll/2, -ll/2, -ll/2, ll, ll, ll};

// Sphere
Sphere(3) = {0, 0, 0, D/2};

// Difference
BooleanDifference{Volume{1}; Delete;}{Volume{2}; }
BooleanDifference{Volume{2}; Delete;}{Volume{3}; Delete; }

// Physical Volumes
Physical Volume(301) = {1};
Physical Volume(302) = {2};

// Physical Surfaces
Physical Surface(203) = {7, 8, 9, 10, 11, 12};
Physical Surface(201) = {14, 15, 16, 17, 18, 19};
Physical Surface(202) = {13};

Mesh.CharacteristicLengthFactor = 4;
Characteristic Length{PointsOf{Physical Surface{202};}} = 2;
Characteristic Length{PointsOf{Physical Surface{203};}} = 3;
Characteristic Length{PointsOf{Physical Surface{201};}} = 5;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "hole_with_inner_cube_150.msh";
