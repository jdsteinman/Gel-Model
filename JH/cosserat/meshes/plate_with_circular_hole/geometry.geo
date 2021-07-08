fine = 0.1;
coarse = 2.0;

Point(1) = {0.216, 0.0, 0.0, fine};
Point(2) = {16.2, 0.0, 0.0, coarse};
Point(3) = {16.2, 16.2, 0.0, coarse};
Point(4) = {0.0, 16.2, 0.0, coarse};
Point(5) = {0.0, 0.216, 0.0, fine};
Point(6) = {0.0, 0.0, 0.0, fine};

Line(12) = {1,2};
Line(23) = {2,3};
Line(34) = {3,4};
Line(45) = {4,5};
Circle(561) = {5,6,1};

Line Loop(1) = {12,23,34,45,561};
Plane Surface(1) = 1;

// Interface
Physical Line(1) = {561};

// Outer
Physical Surface(0) = {1};
