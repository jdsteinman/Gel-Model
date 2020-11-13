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
Sphere(4) = {0, 0, 0, 1};

// Protrusions

// principal semi-major axes
psax = 5;
psay = 3;
psaz = 18.75;

Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{2};}
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{3};}
Dilate {{0, 0, 0}, {psax, psay, psaz}} {Volume{4};}

// Rotate Pseudopods
Rotate {{0, 1, 0}, {0, 0, 0}, Pi/3} { Volume{3} ; }
Rotate {{0, 1, 0}, {0, 0, 0}, -Pi/3} { Volume{4} ; }

// Delete Ends
BooleanDifference {Volume{2}; Delete; }{Volume{1}; }
Recursive Delete {Volume {5}; }
BooleanDifference {Volume{3}; Delete; }{Volume{1}; }
Recursive Delete {Volume {8}; }
BooleanDifference {Volume{4}; Delete; }{Volume{1}; }
Recursive Delete {Volume {9}; }

// Merge Volumes
BooleanUnion(10) = {Volume{1}; Delete; }{Volume{6}; Delete; };
BooleanUnion(11) = {Volume{10}; Delete; }{Volume{7}; Delete; };
BooleanUnion(12) = {Volume{11}; Delete; }{Volume{8}; Delete; };

//Physical Volume(0) = {12}; // Cell volume

// Gel
l = 150; // Side length of box
Box(13) = {-l/2, -l/2, -l/2,  l, l, l};
BooleanDifference(14) = {Volume{13}; Delete; }{Volume{12}; Delete;};

// Physical Entities
Physical Volume(0) = {14};
Physical Surface(1) = {7, 8, 9, 10};          // Cell Surface
Physical Surface(2) = {1, 2, 3, 4, 5, 6};   // Box Surface

Mesh.Algorithm = 6;
Characteristic Length{:} = 15;
Characteristic Length{PointsOf{Physical Volume{0};}}  = 15;
Characteristic Length{PointsOf{Physical Surface{1};}} = 1;
