/*
  Generate gel volume mesh with cell in center.
  Needs cytod_uncentered_unpca.stl

  Note: Cannot currently use OpenCascade with stl files
*/

Merge "cell_surface_coarse.stl";

// Centroid
cx = 76.9589;
cy = 75.9575;
cz = 69.6501;

// Lengths
l = 250.;
length = l;
height = l;
depth = l;
lcar1 = 0.5;

// Outer Boundary Points
Point(1) = {length/2,height/2,depth/2,lcar1}; 
Point(2) = {length/2,height/2,-depth/2, lcar1}; 
Point(3) = {-length/2,height/2,depth/2, lcar1}; 
Point(4) = {-length/2,-height/2,depth/2,lcar1}; 
Point(5) = {length/2,-height/2,depth/2,lcar1}; 
Point(6) = {length/2,-height/2,-depth/2,lcar1}; 
Point(7) = {-length/2,height/2,-depth/2,lcar1}; 
Point(8) = {-length/2,-height/2,-depth/2,lcar1}; 

// Translate Box
Translate{cx, cy, cz} {Point{1,2,3,4,5,6,7,8};}

// Outer Box
Line(1) = {3,1};
Line(2) = {3,7};
Line(3) = {7,2};
Line(4) = {2,1};
Line(5) = {1,5};
Line(6) = {5,4};
Line(7) = {4,8};
Line(8) = {8,6};
Line(9) = {6,5};
Line(10) = {6,2};
Line(11) = {3,4};
Line(12) = {8,7};
Line Loop(13) = {-6,-5,-1,11};
Plane Surface(14) = {13};
Line Loop(15) = {4,5,-9,10};
Plane Surface(16) = {15};
Line Loop(17) = {-3,-12,8,10};
Plane Surface(18) = {17};
Line Loop(19) = {7,12,-2,11};
Plane Surface(20) = {19};
Line Loop(21) = {-4,-3,-2,1};
Plane Surface(22) = {21};
Line Loop(23) = {8,9,6,7};
Plane Surface(24) = {23};

// Define Volumes
Surface Loop(25) = {1,14,24,-18,22,16,-20};
Volume(26) = {25};

// Physical Entities
Physical Surface(201) = {14,16,18,20,22,24};  // outer box
Physical Surface(202) = {1};                  // cell surface
Physical Volume(301) = {26};                  // gel

// Characteristic Length
Mesh.CharacteristicLengthFactor = 1;
Characteristic Length{PointsOf{Physical Surface{201};}} = 20;

