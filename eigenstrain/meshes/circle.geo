SetFactory("OpenCASCADE");

fine = 0.5;
coarse = 2.0;

r2 = 100;
r1 = 10;

Disk(1) = {0,0,0,r2};
Disk(2) = {0,0,0,r1};

BooleanDifference{Surface{1}; Delete;}{Surface{2};}

// Edges
Physical Line(101) = {3};  // Outer
Physical Line(102) = {2};  // Inner

// Surface
Physical Surface(201) = {1};
Physical Surface(202) = {2};

// Mesh resolution
Mesh.CharacteristicLengthFactor = 1;
Characteristic Length{:} = coarse;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "circle.msh";
