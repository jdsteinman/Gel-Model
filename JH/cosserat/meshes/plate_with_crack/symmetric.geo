// Symmetrical crack mesh, just a square, but we need to do refinement near the tip
// This may not be the best refinement strategy, might be best to add more points or region
// for refinement

width = 4;
height = 4;
crack_length = 1.0;

fine = crack_length/80.0;
coarse = width/10.0;

Point(1) = {0.0, 0.0, 0.0, coarse};
Point(5) = {crack_length, 0.0, 0.0, fine};
Point(6) = {crack_length + 0.3*crack_length, 0.0, 0.0, fine};
Point(2) = {width, 0.0, 0.0};
Point(3) = {width, height, 0.0};
Point(4) = {0.0, height, 0.0, coarse};

Line(1) = {1,5};
Line(2) = {5,6};
Line(6) = {6,2};
Line(3) = {2,3};
Line(4) = {3,4};
Line(5) = {4,1};

Line Loop(1) = {1,2,6,3,4,5};
Plane Surface(1) = 1;
