SetFactory("OpenCASCADE");

fine = 0.01;
coarse = 0.1;

s = 1;

Rectangle(1) = {-s/2, -s/2, 0, s, s};

// Outer edge
Physical Line(100) = {1, 2, 3, 4};
Physical Line(101) = {1};
Physical Line(102) = {2};
Physical Line(103) = {3};
Physical Line(104) = {4};

// Surface
Physical Surface(201) = {1};

// Mesh resolution
Mesh.CharacteristicLengthFactor = 1;
Characteristic Length{:} = fine;

// Generate Mesh
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "plate.msh";
