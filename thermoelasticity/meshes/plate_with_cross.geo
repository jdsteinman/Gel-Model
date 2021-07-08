SetFactory("OpenCASCADE");

fine = 0.75;
coarse = 2.0;

s = 100;
r = 10;

Rectangle(1) = {-s/2, -s/2, 0, s, s};
Disk(2) = {0,0,0,r};
Rectangle(3) = {-r/20, -3*r/4, 0, r/10, 3*r/2};
Rectangle(4) = {-3*r/4, -r/20, 0, 3*r/2, r/10};

BooleanFragments{Surface{3}; Delete;}{Surface{4}; Delete;}
BooleanDifference{Surface{1}; Delete;}{Surface{2}; }
BooleanDifference{Surface{2}; Delete;}{Surface{3,4,5,6,7}; }

// Physical lines
Physical Line(101) = {10, 11, 12, 13};  // outer
Physical Line(102) = {5};               // circle

// Physical Surfaces
Physical Surface(201) = {1};
Physical Surface(202) = {2};
Physical Surface(203) = {6,7};
Physical Surface(204) = {3,5};

// Mesh resolution
Mesh.CharacteristicLengthFactor = 1;
Characteristic Length{:} = fine;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "circle.msh";