SetFactory("OpenCASCADE");

fine = 0.2;
reg = 1.0;
coarse = 5.0;

s = 1;
r = 0.05;

Rectangle(1) = {-s/2, -s/2, 0, s, s};
Disk(2) = {0,0,0,r};

BooleanDifference{Surface{1}; Delete;}{Surface{2}; Delete;}

// Outer edge
Physical Line(101) = {6, 7, 8, 9};

// Inner edge
Physical Line(102) = {5};

// Surface
Physical Surface(201) = {1};

// Mesh resolution
Mesh.CharacteristicLengthFactor = 0.05;
Characteristic Length{:} = coarse;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "plate_with_hole_coarse.msh";
