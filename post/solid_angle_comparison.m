%% solid angle analysis

clc 
clear all
close all

% this code plots the displacement vs distance from cell surface data for
% 3D TFM experiments in solid angles around the cell

% Written by: Alex Khang, John Steinman 

% Last Updated: 11/12/2020

%% Import data
data_path = '../data/Gel3/';
sim_path = '../gel_model/output/Gel3/';
output_path = './Gel3_real/';
if ~exist(output_path, 'dir')
   mkdir(output_path)
end

vertices = textread(strcat(data_path, 'CytoD_vertices.txt'));
surf_vertices = textread(strcat(data_path, 'CytoD_vertices.txt'));
faces = textread(strcat(data_path, 'CytoD_faces.txt'));
% faces = faces+1;

% Bead Locations
initial = textread(strcat(data_path, 'beads_init.txt'));

% Experimental data
final = textread(strcat(data_path, 'beads_final.txt'));
disp_real = final - initial; 

% Simulation Data
disp_sim = textread(strcat(sim_path, 'sim_beads_disp.txt')); 

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

%% Computes quantity of interests 

% Calculates nearest distance to cell surface and displacement direction
% (e.g. going towards the cell or away from the cell)
for i = 1:length(initial)
    dist = sqrt(sum((surf_vertices - initial(i,:)).^2,2)); % computes distance between bead and all cell vertices
    closest(i) = min(dist); % finds closest cell vertex to bead
    indx = find(dist == closest(i)); % finds which cell vertex is closest
    direction_real(i) = dot(normals(indx(1),:),disp_real(i,:)/norm(disp_real(i,:)));
    direction_sim(i) = dot(normals(indx(1),:),disp_sim(i,:)/norm(disp_sim(i,:)));% dot product of cell vertex normal and bead displacement vector
end

clear q idx
% finds 10 closest beads to every bead
for i = 1:length(initial)
    dist = sqrt(sum((initial - initial(i,:)).^2,2));
    dist_sort = sort(dist);
    [q(i,1:10), idx(i,1:10)] = ismember(dist_sort(2:11), dist, 'rows');
end

% computes neighbor score: mean dot product between displacement unit
% vectors of 10 nearest neighbors 
for i = 1:size(idx,1)
    for j = 1:size(idx,2)
        scores_real(j) = dot(disp_real(i,:)/norm(disp_real(i,:)),disp_real(idx(i,j),:)/norm(disp_real(idx(i,j),:)));
        scores_sim(j) = dot(disp_sim(i,:)/norm(disp_sim(i,:)),disp_sim(idx(i,j),:)/norm(disp_sim(idx(i,j),:)));
    end
    neighbor_score_real(i) = mean(scores_real);
    neighbor_score_sim(i) = mean(scores_sim);
end
    
    
% finds beads that are moving away and towards cell surface
towards_real = find(direction_real < 0);
away_real = find(direction_real > 0);
towards_sim = find(direction_sim < 0);
away_sim = find(direction_sim > 0);

% displacement magnitude
d_real = sqrt(sum(disp_real.^2,2));
d_sim = sqrt(sum(disp_sim.^2,2));

% turns disp_real towards the cell negative
% d(towards) = -d(towards);


%% Solid angle analysis

% center beads wrt to cell center
centered_beads = initial - mean(surf_vertices);

% centerd cell
centered_nodes = surf_vertices-mean(surf_vertices);
max_radius = max(sqrt(sum(centered_nodes.^2,2)));

% finds unit direction vectors to each bead
for i = 1:length(centered_beads)
    unit_vector(i,1:3) = centered_beads(i,:)/norm(centered_beads(i,:));
end

% Convert Cartesian parameterized coordinates to spherical coordinates
[phi, theta] = cart2sph(unit_vector(:,1), unit_vector(:,2), unit_vector(:,3));

% Convert angles into 0:pi and 0:2*pi range
ind = find(phi<0);
phi(ind) = phi(ind) + 2*pi;
theta = pi/2 - theta;

% creates intervals for phi and theta for solid angle analysis. increase
% last input of linspace for smaller and more solid angles
phi_intervals = linspace(0,2*pi,3);
theta_intervals = linspace(0,pi,3);

iter = 0;
plotting_preamble
% iterates through all solid angles/wedges
for i = 1:length(theta_intervals)
    if i < length(theta_intervals)
        for j = 1:length(phi_intervals)
            if j < length(phi_intervals)
                iter = iter + 1;               
                
                % computes steradians 
                syms p t
                sr = sin(t);
                sr = int(sr,p,[phi_intervals(j) phi_intervals(j+1)]);
                steradians(i,j) = double(int(sr,t,[theta_intervals(i) theta_intervals(i+1)]));
                
                % string name for solid wedge
                trial = strcat('theta',num2str(theta_intervals(i)),'to',num2str(theta_intervals(i+1)),'phi',num2str(phi_intervals(j)),'to',num2str(phi_intervals(j+1)),'.csv');
                
                % finds the theta and phi interval for the solid wedge
                ind = find(theta > theta_intervals(i) & theta < theta_intervals(i+1));
                ind2 = find(phi > phi_intervals(j) & phi < phi_intervals(j+1));
                real_ind = intersect(ind,ind2);
                
                % compiles data to be outputted 
                %data = [initial(real_ind,:),final(real_ind,:),closest(real_ind)',disp_real(real_ind,1),disp_real(real_ind,2),disp_real(real_ind,3),d(real_ind),direction(real_ind)',neighbor_score(real_ind)'];
                data = [1];
                
                % only outputs data and plots if data exists 
                if ~isempty(data)
                
                % pre-allocation
                d_neg_real = zeros(1,length(direction_real(real_ind)));
                closest_neg_real = zeros(1,length(direction_real(real_ind)));
                d_pos_real = zeros(1,length(direction_real(real_ind)));
                closest_pos_real = zeros(1,length(direction_real(real_ind)));
                
                d_neg_sim = zeros(1,length(direction_sim(real_ind)));
                closest_neg_sim = zeros(1,length(direction_sim(real_ind)));
                d_pos_sim = zeros(1,length(direction_sim(real_ind)));
                closest_pos_sim = zeros(1,length(direction_sim(real_ind)));
                
                
                % makes disp towards the cell negative 
                for k = 1:length(direction_real(real_ind))
                    if direction_real(real_ind(k)) < 0
                        d_neg_real(k) = -d_real(real_ind(k));
%                         d_neg(k) = d(real_ind(k));
                        closest_neg_real(k) = closest(real_ind(k));
                    else
                        d_pos_real(k) = d_real(real_ind(k));
                        closest_pos_real(k) = closest(real_ind(k));
                    end
                end
                
                for k = 1:length(direction_sim(real_ind))
                    if direction_sim(real_ind(k)) < 0
                        d_neg_sim(k) = -d_sim(real_ind(k));
%                         d_neg(k) = d(real_ind(k));
                        closest_neg_sim(k) = closest(real_ind(k));
                    else
                        d_pos_sim(k) = d_sim(real_ind(k));
                        closest_pos_sim(k) = closest(real_ind(k));
                    end
                end
                
                
                % gets rid of zero values
                %d_real_nz = nonzeros(d_real);
%                 closest_neg_nz = nonzeros(closest_neg);
                %d_sim_nz = nonzeros(d_sim);
%                 closest_pos = nonzeros(closest_pos);        
              
                %creates unit sphere for plotting
                [x,y,z] = sphere(50);
                lightGrey = 0.8*[1 1 1]; % It looks better if the lines are lighter
                colors = zeros(size(x,1),size(x,2));
                [phi_temp, theta_temp] = cart2sph(x, y, z);

                %Convert angles into 0:pi and 0:2*pi range 
                ind = find(phi_temp<0);
                phi_temp(ind) = phi_temp(ind) + 2*pi;
                theta_temp = pi/2 - theta_temp;

                ind_temp = find(theta_temp > theta_intervals(i) & theta_temp < theta_intervals(i+1));
                ind2_temp = find(phi_temp > phi_intervals(j) & phi_temp < phi_intervals(j+1));
                real_ind_temp = intersect(ind_temp,ind2_temp);
                colors(real_ind_temp) = 100;
                
                % Correlation
                corr_mat = corrcoef(d_real(real_ind), d_sim(real_ind));
                r = corr_mat(1,2);          

                % plotting
                fig = figure('Position', get(0, 'Screensize'));
                % plots cell mesh
                subplot(2,3,[1,4])
                trimesh(faces,centered_nodes(:,1),centered_nodes(:,2),centered_nodes(:,3),'FaceColor','r','EdgeColor','k'); hold on;
                view([ 1 1 1])
                % plots unit sphere
                subplot(2,3,[1,4])
                surface(max_radius*x,max_radius*y,max_radius*z,colors,'FaceAlpha',0.6);
                s.EdgeColor = 'none';
                scatter3(unit_vector(real_ind,1),unit_vector(real_ind,2),unit_vector(real_ind,3)); hold on;
                xlabel('x-axis')
                ylabel('y-axis')
                zlabel('z-axis')
                view([1 1 1])
                
                
                % plots toward and away disp_real vs distance from cell
                subplot(2,3,2)
                scatter(closest_neg_real,d_neg_real,40,'r'); hold on;
                scatter(closest_pos_real,d_pos_real,40,'b'); hold on;
                rotation_axis = 0;
                line([0,150],[rotation_axis,rotation_axis]); hold on;
                ylim([-4 4])
                xticks(0:30:150);
                title('real displacement')
                xlabel('distance from cell surface (\mum)')
                ylabel('marker displacement (\mum)')
                legend('Towards Cell', 'Away from Cell')
                txt = ['Correlation: ' num2str(r)];
                text(80,1.75,txt)
                
                grid on
                % plots a histogram of the displacement magnitudes in the solid
                % angle
                subplot(2,3,3)
                histogram(abs(d_real(real_ind)),'Normalization','probability','FaceAlpha',0.8); hold on;
                xlim([0 2])
                ylim([0 1])
                xlabel('displacement (\mum)')
                ylabel('probability')
                title('real displacement magnitudes')
                grid on
                
                % plots toward and away disp_sim vs distance from cell 
                subplot(2,3,5)
                scatter(closest_neg_sim,d_neg_sim,40,'r'); hold on;
                scatter(closest_pos_sim,d_pos_sim,40,'b'); hold on;
                rotation_axis = 0;
                line([0,150],[rotation_axis,rotation_axis]); hold on;
                ylim([-4 4])
                xticks(0:30:150);
                title('sim displacement')
                xlabel('distance from cell surface (\mum)')
                ylabel('sim marker displacement (\mum)')
                legend('Towards Cell', 'Away from Cell')
                txt = ['Correlation: ' num2str(r)];
                text(80,1.75,txt)
                grid on
                
                
                % plots a histogram of the displacement magnitudes in the solid
                % angle
                subplot(2,3,6)
                histogram(abs(d_sim(real_ind)),'Normalization','probability','FaceAlpha',0.8); hold on;
                xlim([0 2])
                ylim([0 1])
                xlabel('displacement (\mum)')
                ylabel('probability')
                title('displacement magnitudes')
                grid on
                
                filename = strcat(output_path, "solid_angle_plot", num2str(iter));
                saveas(fig, filename, 'png');
                close(fig)
                
                clear d_neg_real closest_neg_real d_pos_real closest_pos_real
                clear d_neg_sim closest_neg_sim d_pos_sim closest_pos_sim
                
               
                end
            end
        end
    end
end

