/*
  Generate gel volume mesh with cell in center.
  Needs cytod_uncentered_unpca.stl

  Note: Cannot currently use OpenCascade with stl files
*/

Merge "cell_surface_1000.stl";

// Lengths
l=250.;
length = l;
height = l;
depth = l;
lcar1 = 0.5;

// Translate {length, height, depth} {Surface{1};}

// Outer Boundary Points
Point(1) = {length/2,height/2,depth/2,lcar1}; 
Point(2) = {length/2,height/2,-depth/2, lcar1}; 
Point(3) = {-length/2,height/2,depth/2, lcar1}; 
Point(4) = {-length/2,-height/2,depth/2,lcar1}; 
Point(5) = {length/2,-height/2,depth/2,lcar1}; 
Point(6) = {length/2,-height/2,-depth/2,lcar1}; 
Point(7) = {-length/2,height/2,-depth/2,lcar1}; 
Point(8) = {-length/2,-height/2,-depth/2,lcar1}; 

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
Line Loop(2) = {-6,-5,-1,11};
Plane Surface(2) = {2};
Line Loop(3) = {4,5,-9,10};
Plane Surface(3) = {3};
Line Loop(4) = {-3,-12,8,10};
Plane Surface(4) = {4};
Line Loop(5) = {7,12,-2,11};
Plane Surface(5) = {5};
Line Loop(6) = {-4,-3,-2,1};
Plane Surface(6) = {6};
Line Loop(7) = {8,9,6,7};
Plane Surface(7) = {7};

// Inner box Points
Point(9) = {length/4,height/4,depth/4,lcar1}; 
Point(10) = {length/4,height/4,-depth/4, lcar1}; 
Point(11) = {-length/4,height/4,depth/4, lcar1}; 
Point(12) = {-length/4,-height/4,depth/4,lcar1}; 
Point(13) = {length/4,-height/4,depth/4,lcar1}; 
Point(14) = {length/4,-height/4,-depth/4,lcar1}; 
Point(15) = {-length/4,height/4,-depth/4,lcar1}; 
Point(16) = {-length/4,-height/4,-depth/4,lcar1}; 

// Inner Box
Line(13) = {11,9}; 
Line(14) = {11,15};
Line(15) = {15,10}; 
Line(16) = {10,9};  
Line(17) = {9,13};
Line(18) = {13,12};
Line(19) = {12,16};
Line(20) = {16,14};
Line(21) = {14,13};
Line(22) = {14,10};
Line(23) = {11,12};
Line(24) = {16,15};

Line Loop(8) = {-18,-17,-13,23};
Plane Surface(8) = {8};
Line Loop(9) = {16,17,-21,22};
Plane Surface(9) = {9};
Line Loop(10) = {-15,-24,20,22};
Plane Surface(10) = {10};
Line Loop(11) = {19,24,-14,23};
Plane Surface(11) = {11};
Line Loop(12) = {-16,-15,-14,13};
Plane Surface(12) = {12};
Line Loop(13) = {20,21,18,19};
Plane Surface(13) = {13};

// Define Volumes
Surface Loop(1) = {1,8,12,-10,12,9,-11};
Volume(1) = {1};
Surface Loop(2) = {8,12,-10,12,9,-11,2,7,-4,6,3,-5};
Volume(2) = {2};

// Physical Entities
Physical Surface(201) = {2,3,4,5,6,7};  // outer box
Physical Surface(202) = {1};            // cell surface
Physical Surface(203) = {8,9,10,11,12,13};  // interface

Physical Volume(301) = {1};             // near field
Physical Volume(302) = {2};             // near field

// Characteristic Length
Mesh.CharacteristicLengthFactor = 3;
Characteristic Length{PointsOf{Physical Surface{201};}} = 15;
Characteristic Length{PointsOf{Physical Surface{203};}} = 5;

// Generate Mesh
//Mesh 3;
//Mesh.MshFileVersion = 2.2;
// Save "hole.msh";

