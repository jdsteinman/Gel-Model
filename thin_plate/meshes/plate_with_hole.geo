SetFactory("OpenCASCADE");

fine = 0.1;
coarse = 2.0;

s = 50;
r = 5;

Rectangle(1) = {-s, -s, 0, 2*s, 2*s};
Disk(2) = {0,0,0,r};

BooleanDifference{Surface{1}; Delete;}{Surface{2}; Delete;}

// Interface
Physical Line(11) = {5};

// Outer
Physical Line(10) = {6, 7, 8, 9};

// Surface
Physical Surface(20) = {1};

// Mesh
Characteristic Length{:} = 0.5;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "plate_with_hole.msh";
