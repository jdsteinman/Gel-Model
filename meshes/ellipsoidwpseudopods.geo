SetFactory("OpenCASCADE");

// Ellipsoid
Sphere(1) = {0, 0, 0, 1};

// principal semi-major axes
psax = 11.5;
psay = 7.6;
psaz = 18.75;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{1};}
Sphere(2) = {0, 0, 0, 1};

// Protrusions

// principal semi-major axes
psax = 5;
psay = 3;
psaz = 30;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{2};}

// Merge Volumes
BooleanUnion(3) = {Volume{1}; Delete; }{Volume{2}; Delete; };
Physical Volume(4) = {3};

// Gel
l = 150; // Side length of box
Box(5) = {-l/2, -l/2, -l/2,  l, l, l};
BooleanDifference(6) = {Volume{5}; Delete; }{Volume{3}; Delete; };

// Physical Entities
Physical Volume(7) = {6};
Physical Surface(1) = {7, 8, 9};            // Cell Surface
Physical Surface(2) = {4, 5, 3, 2, 6, 1};   // Box Surface


Mesh.Algorithm = 6;
Characteristic Length{:} = 15;
Characteristic Length{PointsOf{Physical Volume{4};}}  = 15;
Characteristic Length{PointsOf{Physical Surface{1};}} = 1;
