Mesh.MshFileVersion = 2.2;

// Cube
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

Volume(56) = {55};
Physical Volume (57) = {56};
