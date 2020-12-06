clc
clear all
close all

%% Import data

soln_path = '../gel_model/output/';
mesh_path = './ellipsoid/';
data_path = '../data/Gel3/';
vertices = textread(strcat(mesh_path, 'nodes.txt'));
surf_vertices = textread(strcat(mesh_path, 'nodes_new.txt'));
faces = textread(strcat(mesh_path, 'surf_cells_new.txt'));
faces = faces+1;

% import bead locations in the initial and final configuration
displacements = textread(strcat(soln_path, 'ellipsoid_sol.txt')); 
initial = vertices;
final = initial + displacements;

sim_init = textread(strcat(soln_path, 'ellipsoid_beads_init.txt')); 
sim_disp = textread(strcat(soln_path, 'ellipsoid_beads_disp.txt')); 
sim_mag = sqrt(sum(sim_disp.^2, 2));

real_init = textread(strcat(data_path, 'beads_init.txt')); 
real_final = textread(strcat(data_path, 'beads_final.txt')); 
real_disp = real_final - real_init;

real_mag = sqrt(sum(real_disp.^2,2));


%%

scatter(real_mag, sim_mag)
xlabel('real displacement (\mum)')
ylabel('simulated displacement (\mum)')
title('displacement magnitudes')
grid on
                
r = corrcoeff(real_mag', sim_mag');