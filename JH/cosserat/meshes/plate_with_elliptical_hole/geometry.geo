g = 2.5; // the ratio c/r which parameterises results in Nakamura and Lakes
r = 0.5; // radius of curvature, fixed
width = 24.5;
height = 9;

c = g*r; 
b = r*g*Sqrt(1.0/g);

fine = c/80.0;
coarse = width/10.0;

Point(1) = {c, 0.0, 0.0, fine};
Point(2) = {width, 0.0, 0.0, coarse};
Point(3) = {width, height, 0.0, coarse};
Point(4) = {0.0, height, 0.0, coarse};
Point(5) = {0.0, b, 0.0, fine};
Point(6) = {0.0, 0.0, 0.0, fine};
Point(7) = {c/2.0, 0.0, 0.0, fine};
Point(8) = {2.0*c, 0.0, 0.0, fine};


Line(18) = {1, 8};
Line(82) = {8, 2};
Line(23) = {2, 3};
Line(34) = {3, 4};
Line(45) = {4, 5};
Ellipse(561) = {5, 6, 6, 1};

Line Loop(1) = {18,82,23,34,45,561};
Plane Surface(1) = 1;
