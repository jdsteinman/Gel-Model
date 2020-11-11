SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 1};

// principal semi-major axes
psax = 11.5;
psay = 7.6;
psaz = 18.75;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{1};}

Physical Surface(1) = {7};
Physical Surface(2) = {4, 5, 3, 2, 6, 1};

Sphere(2) = {0, 0, 0, 1};

// principal semi-major axes
psax = 5;
psay = 3;
psaz = 30;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{2};}

BooleanUnion(3) = {Volume{1}; Delete; }{Volume{2}; Delete; };
Physical Volume(4) = {3};

l = 150; // Side length of box
Box(5) = {-l/2, -l/2, -l/2,  l, l, l};

BooleanDifference(6) = {Volume{5}; Delete; }{Volume{3}; Delete; };
Physical Volume(7) = {6};

Mesh.Algorithm = 6;
Characteristic Length{:} = 15;
Characteristic Length{PointsOf{Physical Volume{4};}}  = 15;
Characteristic Length{PointsOf{Surface{7};}} = 1;
