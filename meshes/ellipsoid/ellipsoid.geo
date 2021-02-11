SetFactory("OpenCASCADE");

l = 150; // Side length of box
Sphere(1) = {0, 0, 0, 1};

// principal semi-major axes
// psax = 11.5;
// psay = 7.6;
// psaz = 18.75;
psax = 10.;
psay = 10.;
psaz = 20.;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{1};}

Box(2) = {-l/2, -l/2, -l/2, l, l, l};

BooleanDifference(3) = {Volume{2}; Delete; }{Volume{1}; Delete; };
Physical Volume(300) = {3};

Physical Surface(201) = {7};
Physical Surface(200) = {4, 5, 3, 2, 6, 1};

Mesh.Algorithm = "Delaunay";
Characteristic Length{:} = 5;
// Characteristic Length{PointsOf{Physical Surface{201};}} = 5;
// Characteristic Length{PointsOf{Physical Volume{300};}}  = 10;
// Characteristic Length{PointsOf{Physical Surface{200};}} = 5;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "ellipsoid.msh";
