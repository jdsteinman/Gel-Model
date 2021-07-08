fine = 0.007;
coarse = 0.3;

a = 1.0;
b = 20.0;

Point(1) = {a, 0.0, 0.0, fine};
Point(2) = {b/3.0, 0.0, 0.0, fine};
Point(3) = {b, 0.0, 0.0, coarse};
Point(4) = {b, b, 0.0, coarse};
Point(5) = {0.0, b, 0.0, coarse};
Point(6) = {0.0, b/3.0, 0.0, fine};
Point(7) = {0.0, a, 0.0, fine};
Point(8) = {0.0, 0.0, 0.0, fine};

Point(100) = {b/3.0, b/3.0, 0.0, fine};

Line(12) = {1,2};
Line(23) = {2,3};
Line(34) = {3,4};
Line(45) = {4,5};
Line(56) = {5,6};
Line(67) = {6,7};

Line(18) = {1,8};
Line(87) = {8,7};

Circle(781) = {7,8,1};

Line Loop(1) = {12,23,34,45,56,67,781};
Line Loop(2) = {781,18,87};

Plane Surface(1) = 1;
Plane Surface(2) = 2;

Physical Line(1) = {781};
Physical Surface(0) = {2};
Physical Surface(1) = {1};
