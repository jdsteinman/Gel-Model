fine = 0.01;
coarse = fine;

a = 1.0;
b = 3.0; 

Point(1) = {a, 0.0, 0.0, fine};
Point(2) = {b, 0.0, 0.0, fine};
Point(3) = {b, b, 0.0, fine};
Point(4) = {0.0, b, 0.0, fine};
Point(5) = {0.0, a, 0.0, fine};
Point(6) = {0.0, 0.0, 0.0, fine};

Line(12) = {1,2};
Line(23) = {2,3};
Line(34) = {3,4};
Line(45) = {4,5};

Line(16) = {1,6};
Line(65) = {6,5};

Circle(561) = {5,6,1};

Line Loop(1) = {12,23,34,45,561};
Line Loop(2) = {561,16,65};

Plane Surface(1) = 1;
Plane Surface(2) = 2;

Physical Line(1) = {561};
Physical Surface(0) = {2};
Physical Surface(1) = {1};
