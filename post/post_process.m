%% post processing

clc 
clear all
close all

% this code plots the displacement vs distance from cell surface data for
% 3D TFM experiments in solid angles around the cell

% Written by: John Steinman

%% Import data

% import cell vertices and faces
path = '../gel_model/output/';
vertices = textread('nodes.txt');
surf_vertices = textread('nodes_new.txt');
faces = textread('surf_cells_new.txt');
faces = faces+1;

% import bead locations in the initial and final configuration
displacements = textread('../gel_model/output/ellipsoid_sol.txt'); 
initial = vertices;

%% Import data

data_path = '../gel_model/output/';
mesh_path = './ellipsoid/';
vertices = textread(strcat(mesh_path, 'nodes.txt'));
surf_vertices = textread(strcat(mesh_path, 'nodes_new.txt'));
faces = textread(strcat(mesh_path, 'surf_cells_new.txt'));
faces = faces+1;

% import bead locations in the initial and final configuration
displacements = textread(strcat(data_path, 'ellipsoid_sol.txt')); 
initial = vertices;
final = initial + displacements;

beads_init = textread(strcat(data_path, 'ellipsoid_beads_init.txt')); 
beads_disp = textread(strcat(data_path, 'ellipsoid_beads_disp.txt')); 
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

%% Get directions of displacements wrt cell surface

% Calculates nearest distance to cell surface and displacement direction
% (e.g. going towards the cell or away from the cell)
for i = 1:length(initial)
    dist = sqrt(sum((surf_vertices - initial(i,:)).^2,2)); % computes distance between bead and all cell vertices
    closest(i) = min(dist); % finds closest cell vertex to bead
    indx = find(dist == closest(i)); % finds which cell vertex is closest
    direction(i) = dot(normals(indx(1),:),displacements(i,:)/norm(displacements(i,:))); % dot product of cell vertex normal and bead displacement vector
end

% Calculate magnitude
disp_mag = vecnorm(displacements, 2, 2);

% Change nan to 0
direction(isnan(direction)) = 0;

%%



%% Output file
ofile = fopen('for_paraview.csv', 'w');
fprintf(ofile, 'p_x,p_y,p_z,u_x,u_y,u_z,dot\n');
for i =1:length(initial)
   fprintf(ofile, '%10.9f,%10.9f,%10.9f,', vertices(i,:));
   fprintf(ofile, '%10.9f,%10.9f,%10.9f,', displacements(i,:));
   fprintf(ofile, '%10.9f\n', direction(i));
end

% fprintf(ofile, '%10.9f\n', direction);
% fclose(ofile);


