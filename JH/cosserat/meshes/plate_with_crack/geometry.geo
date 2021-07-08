width = 4.0;
height = 8.0;
crack_length = 1.0;

fine = crack_length/2.0;
coarse = width/2.0;

Point(1) = {0.0, 0.0, 0.0, coarse};
Point(2) = {width, 0.0, 0.0, coarse};
Point(3) = {width, height, 0.0, coarse};
Point(4) = {0.0, height, 0.0, coarse};
Point(5) = {0.0, height/2.0, 0.0, coarse};
Point(6) = {crack_length, height/2.0, 0.0, fine};

Line(12) = {1,2};
Line(23) = {2,3};
Line(34) = {3,4};
Line(45) = {4,5};
Line(51) = {5,1};

Line(56) = {5,6};

Line Loop(1) = {12,23,34,45,51};

Plane Surface(1) = {1};
Line {56} In Surface{1};

Physical Surface('plate') = {1};
Physical Line('crack') = {56};
Physical Point('openend') = {5};
