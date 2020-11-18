clc
clear all
close all

vertices = textread('CytoD_vertices.txt');
faces = textread('CytoD_faces.txt');

TR = triangulation(faces, vertices);
stlwrite(TR, 'cytod_uncentered_unpca_new.stl')
