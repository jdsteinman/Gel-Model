SetFactory("OpenCASCADE");

// Ellipsoid
Sphere(1) = {0, 0, 0, 1};

// principal semi-major axes
psax = 11.5;
psay = 7.6;
psaz = 13;
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{1};}
Sphere(2) = {0, 0, 0, 1};
Sphere(3) = {0, 0, 0, 1};

// Protrusions

// principal semi-major axes
psax = 5;
psay = 3;
psaz = 18.75;

Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{2};}
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{3};}

// Rotate Pseudopods
Rotate {{0, 1, 0}, {0, 0, 0}, 0} { Volume{2} ; }
Rotate {{0, 1, 0}, {0, 0, 0}, Pi/2} { Volume{3} ; }

// Merge Volumes
BooleanUnion(4) = {Volume{1}; Delete; }{Volume{2}; Delete; };
//Physical Volume(5) = {4};
BooleanUnion(5) = {Volume{4}; Delete; }{Volume{3}; Delete; };
Physical Volume(6) = {5};

// Gel
l = 150; // Side length of box
Box(7) = {-l/2, -l/2, -l/2,  l, l, l};
BooleanDifference(8) = {Volume{7}; Delete; }{Volume{5}; Delete; };

// Physical Entities
Physical Volume(9) = {8};
Physical Surface(1) = {7, 8, 9, 10, 11};            // Cell Surface
Physical Surface(2) = {4, 5, 3, 2, 6, 1};   // Box Surface

Mesh.Algorithm = 6;
Characteristic Length{:} = 15;
Characteristic Length{PointsOf{Physical Volume{4};}}  = 10;
Characteristic Length{PointsOf{Physical Surface{1};}} = 1;

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "ellipsoidw4pods.msh";