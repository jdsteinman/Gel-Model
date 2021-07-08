// WARNING: This is not a valid gmsh file, see plate_with_elliptical_hole.geo for runnable example

major_axis_length = ${major_axis_length};
minor_axis_length = ${minor_axis_length};
x_dim = ${x_dim};
y_dim = ${y_dim};

fine = major_axis_length/200.0;
coarse = x_dim/10.0;

Point(1) = {minor_axis_length, 0.0, 0.0, fine};
Point(2) = {x_dim, 0.0, 0.0, coarse};
Point(3) = {x_dim, y_dim, 0.0, coarse};
Point(4) = {0.0, y_dim, 0.0, coarse};
Point(5) = {0.0, major_axis_length, 0.0, fine};
Point(6) = {0.0, 0.0, 0.0, fine};
Point(8) = {0.0, 2.0*major_axis_length, 0.0, fine};

Line(12) = {1, 2};
Line(23) = {2, 3};
Line(34) = {3, 4};
Line(48) = {4, 8};
Line(85) = {8, 5};
Ellipse(561) = {5, 6, 6, 1};

Line Loop(1) = {12,23,34,48,85,561};
Plane Surface(1) = 1;
