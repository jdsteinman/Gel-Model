SetFactory("OpenCASCADE");

fine = 0.5;
coarse = 2.0;

s = 100;
r = 10;

Rectangle(1) = {-s/2, -s/2, 0, s, s};

// Outer edge
Physical Line(101) = {1};
Physical Line(102) = {2};
Physical Line(103) = {3};
Physical Line(104) = {4};

// Surface
Physical Surface(201) = {1};

// Mesh resolution
Mesh.CharacteristicLengthFactor = 1;
Characteristic Length{:} = coarse;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "plate.msh";
