SetFactory("OpenCASCADE");

i = 3;
R = 2;
Sphere(1) = {0, 0, 0, R};
Affine{ 1,0,0,0, 0,10,0,0, 0,0,1,0 }
	{ Volume{i}; }
Rotate {{Sqrt(2), Sqrt(2) , 0},
	{0, 0, 0}, Pi/3} Volume{i};}
