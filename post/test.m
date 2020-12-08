%% post processing

clc 
clear all
close all

% this code plots the displacement vs distance from cell surface data for
% 3D TFM experiments in solid angles around the cell

% Written by: John Steinman


%% Import data

sim_path = '../gel_model/output/ellipsoid/';
data_path = './ellipsoid/';
output_path = "./Gel3_small_mu_post/";
if ~exist(output_path, 'dir')
   mkdir(output_path)
end

% Initial surf or final surf vertices
vertices = textread(strcat(data_path, 'nodes.txt'));
surf_vertices = textread(strcat(data_path, 'nodes_new.txt'));
faces = textread(strcat(data_path, 'surf_cells_new.txt'));
faces = faces+1;

% import bead locations in the initial and final configuration
displacements = textread(strcat(sim_path, 'vertex_disp.txt')); 
initial = vertices;
final = initial + displacements;

%% Compute unit normal vectors at the cell vertices

% pre-allocation of surface normals
normals = zeros(size(surf_vertices));

% use a Matlab routine to find surface normals
TR = triangulation(faces, surf_vertices);
P = incenter(TR);
F = -faceNormal(TR); % inverted so the normals face outwards

% adds each facet normal to the nodes they are adjacent too
for i = 1:size(faces,1)
    for q = 1:size(faces,2)
        normals(faces(i,q),:) = normals(faces(i,q),:) + F(i,:);
    end
end

% normalizes the node normals after all the summation is complete
for i = 1:length(normals)
    normals(i,:) = normals(i,:)/norm(normals(i,:));
end

% Change nan to 0
normals(isnan(normals)) = 0;

triangle_mesh_vtkwrite('normals.vtk','polydata','triangle',surf_vertices(:,1),surf_vertices(:,2),surf_vertices(:,3),faces,'vectors','normals',normals(:,1),normals(:,2),normals(:,3))




