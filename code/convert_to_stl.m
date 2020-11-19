clc
clear all
close all

vertices = textread('CytoD_vertices.txt');
faces = textread('CytoD_faces.txt');
centroid = mean(vertices);

TR = triangulation(faces, vertices);
stlwrite(TR, 'cytod_uncentered_unpca_new.stl')
