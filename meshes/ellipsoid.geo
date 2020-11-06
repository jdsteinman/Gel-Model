SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 0.5};
Dilate {{0, 0, 0}, {1.5, 1, 0.5}} {Volume{1};}

Box(2) = {-1, -1, -1, 2, 2, 2};

BooleanDifference(3) = {Volume{2}; Delete; }{Volume{1}; Delete; };
Physical Volume(4) = {3};

Mesh.Algorithm = 6;
Characteristic Length{:} = .15;

//+
Physical Surface(1) = {7};
//+
Physical Surface(2) = {4, 5, 3, 2, 6, 1};

