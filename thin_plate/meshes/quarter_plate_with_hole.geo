SetFactory("OpenCASCADE");

fine = 0.5;
coarse = 2.0;

s = 50;
r = 10;

Rectangle(1) = {0, 0, 0, s, s, 0};
Disk(2) = {0,0,0,r};

BooleanDifference{Surface{1}; Delete;}{Surface{2}; Delete;}

// Inner edge
Physical Line(101) = {1};

// Outer edges
Physical Line(102) = {5}; // bottom
Physical Line(103) = {2}; // left
Physical Line(104) = {3}; // top
Physical Line(105) = {4}; // right

// Surface
Physical Surface(201) = {1};

// Mesh resolution
Mesh.CharacteristicLengthFactor = 1;
Characteristic Length{:} = coarse;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "quarter_plate_with_hole.msh";
