SetFactory("OpenCASCADE");

l = 300; // Side length of box
Sphere(1) = {0, 0, 0, 1};

// principal semi-major axes
psax = 11.5;
psay = 7.6;
psaz = 18.75;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{1};}

Box(2) = {-l/2, -l/2, -l/2, l, l, l};

BooleanDifference(3) = {Volume{2}; Delete; }{Volume{1}; Delete; };
Physical Volume(300) = {3};

Physical Surface(201) = {7};
Physical Surface(200) = {4, 5, 3, 2, 6, 1};

Mesh.Algorithm = 6;
Characteristic Length{:} = 25;
Characteristic Length{PointsOf{Physical Volume{4};}}  = 25;
Characteristic Length{PointsOf{Surface{7};}} = 1;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "ellipsoid.msh";
