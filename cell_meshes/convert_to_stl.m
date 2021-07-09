clc
clear all
close all

inpath = "../cell_data/bird/";
outpath = "bird/";
vertices = textread(inpath + "CytoD_vertices.txt");
faces = textread(inpath + "CytoD_faces.txt");

centroid = mean(vertices);
for i = 1:size(vertices, 1)
    vertices(i, 1) = vertices(i, 1) - centroid(1);
    vertices(i, 2) = vertices(i, 2) - centroid(2);
    vertices(i, 3) = vertices(i, 3) - centroid(3);
end

TR = triangulation(faces, vertices);
stlwrite(TR, outpath + "cell_surface.stl")
