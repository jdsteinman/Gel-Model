clc
clear all
close all

inpath = "../star_destroyer/";
outpath = "../star_destroyer/IN/meshes";
vertices = textread(inpath + "cell_surface_vertices.txt");
faces = textread(inpath + "cell_surface_faces.txt");

for i = 1:size(vertices, 1)
    vertices(i, 1) = vertices(i, 1);
    vertices(i, 2) = vertices(i, 2);
    vertices(i, 3) = vertices(i, 3);
end

TR = triangulation(faces, vertices);
stlwrite(TR, outpath + "cell_surface.stl")
