Mesh.MshFileVersion = 2.2;

lc = .2;
r = 1; // radius of sphere

// Sphere
Point(1) = {0.0,0.0,0.0,lc};
Point(2) = {r,0.0,0.0,lc};
Point(3) = {0,r,0.0,lc};
Circle(1) = {2,1,3};
Point(4) = {-r,0,0.0,lc};
Point(5) = {0,-r,0.0,lc};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};
Point(6) = {0,0,-r,lc};
Point(7) = {0,0,r,lc};
Circle(5) = {3,1,6};
Circle(6) = {6,1,5};
Circle(7) = {5,1,7};
Circle(8) = {7,1,3};
Circle(9) = {2,1,7};
Circle(10) = {7,1,4};
Circle(11) = {4,1,6};
Circle(12) = {6,1,2};

// surfaces of 8 octants
Line Loop(13) = {2,8,-10};
Surface(14) = {13};
Line Loop(15) = {10,3,7};
Surface(16) = {15};
Line Loop(17) = {-8,-9,1};
Surface(18) = {17};
Line Loop(19) = {-11,-2,5};
Surface(20) = {19};
Line Loop(21) = {-5,-12,-1};
Surface(22) = {21};
Line Loop(23) = {-3,11,6};
Surface(24) = {23};
Line Loop(25) = {-7,4,9};
Surface(26) = {25};
Line Loop(27) = {-4,12,-6};
Surface(28) = {27};

// Surface of sphere
Surface Loop(29) = {28,26,16,14,20,24,22,18};
//Volume(30) = {29};

// Inner Surface
Physical Surface("inner_surface", 1) = {14, 16, 18, 20, 22, 24, 26, 28};

// What does this do?
// try also netgen:
// Mesh.Algorithm3D = 4;

lc2 = 2;
w = 5; // 0.5 width of cube
Point(10) = {-w,-w,-w,lc2};
Point(11) = {w,-w,-w,lc2};
Point(12) = {w,w,-w,lc2};
Point(13) = {-w,w,-w,lc2};
Point(14) = {-w,-w,w,lc2};
Point(15) = {w,-w,w,lc2};
Point(16) = {w,w,w,lc2};
Point(17) = {-w,w,w,lc2};
Line(31) = {14,15};
Line(32) = {15,11};
Line(33) = {11,10};
Line(34) = {10,14};
Line(35) = {14,17};
Line(36) = {17,13};
Line(37) = {13,10};
Line(38) = {13,12};
Line(39) = {12,16};
Line(40) = {16,17};
Line(41) = {16,15};
Line(42) = {11,12};
Line Loop(43) = {38,39,40,36};
Plane Surface(44) = {43};
Line Loop(45) = {36,37,34,35};
Plane Surface(46) = {45};
Line Loop(47) = {35,-40,41,-31};
Plane Surface(48) = {47};
Line Loop(49) = {32,42,39,41};
Plane Surface(50) = {49};
Line Loop(51) = {42,-38,37,-33};
Plane Surface(52) = {51};
Line Loop(53) = {33,34,31,32};
Plane Surface(54) = {53};
// Surface of cube
Surface Loop(55) = {44,52,-50,54,-46,48};

// Outer Physical Surfaces
Physical Surface("outer_surface", 2) = {44, 46, 48, 50, 52, 54};


// Volume between cube and sphere
Volume(56) = {55,29};
Physical Volume ("Volume", 3) = {56};
