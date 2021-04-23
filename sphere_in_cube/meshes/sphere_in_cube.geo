SetFactory("OpenCASCADE");

l = 100; // Side length of box
Sphere(1) = {0, 0, 0, 1};

// principal semi-major axes
psax = 10.;
psay = 10.;
psaz = 10.;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{1};}

Box(2) = {-l/2, -l/2, -l/2, l, l, l};

BooleanDifference(3) = {Volume{2}; Delete; }{Volume{1}; Delete; };
Physical Volume(300) = {3};

Physical Surface(201) = {7};
Physical Surface(200) = {4, 5, 3, 2, 6, 1};

Characteristic Length{PointsOf{Physical Surface{201};}} = 1;
Characteristic Length{PointsOf{Physical Surface{200};}} = 3;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "sphere_in_cube.msh";
