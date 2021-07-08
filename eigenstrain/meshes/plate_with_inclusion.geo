SetFactory("OpenCASCADE");

fine = 0.5;
coarse = 2.0;

s = 100;
r = 10;

Rectangle(1) = {-s/2, -s/2, 0, s, s};
Disk(2) = {0,0,0,r};

BooleanDifference{Surface{1}; Delete;}{Surface{2};}

// Outer edge
Physical Line(101) = {6, 7, 8, 9};

// Inner edge
Physical Line(102) = {5};

// Surface
Physical Surface(201) = {1};
Physical Surface(202) = {2};

// Mesh resolution
Mesh.CharacteristicLengthFactor = 1;
Characteristic Length{:} = coarse;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "plate_with_hole.msh";
