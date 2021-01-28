%% post processing

clc 
clear all
close all

% Written by: John Steinman


%% Import data

sim_path = '../gel_model/output/func_grad/';
mesh_path = '../meshes/ellipsoid/';
output_file  = 'for_paraview.csv';
output_path = "./ellipsoid/";
if ~exist(output_path, 'dir')
   mkdir(output_path)
end

% Initial surf or final surf vertices
gel_vertices = textread(strcat(mesh_path, 'vertices.txt'));
surf_vertices = textread(strcat(mesh_path, 'surf_vertices.txt'));
faces = textread(strcat(mesh_path, 'surf_faces.txt'));
faces = faces+1;

% Simulation
displacements = textread(strcat(sim_path, 'displacementlinear.txt')); 
initial = gel_vertices;
final = initial + displacements;

% Experimental
% initial = textread(strcat(data_path, 'beads_init.txt')); 
% final = textread(strcat(data_path, 'beads_final.txt'));
% displacements = final-initial;


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

%triangle_mesh_vtkwrite('normals.vtk','polydata','triangle',surf_vertices(:,1),surf_vertices(:,2),surf_vertices(:,3),faces,'vectors','normals',normals(:,1),normals(:,2),normals(:,3))

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


%% Output file
ofile = fopen(strcat(output_path, output_file), 'w+');
fprintf(ofile, 'p_x,p_y,p_z,u_x,u_y,u_z,mag,dot\n');
for i =1:length(initial)
   fprintf(ofile, '%10.9f,%10.9f,%10.9f,', initial(i,:));
   fprintf(ofile, '%10.9f,%10.9f,%10.9f,', displacements(i,:));
   fprintf(ofile, '%10.9f,', disp_mag(i));
   fprintf(ofile, '%10.9f\n', direction(i));
end




