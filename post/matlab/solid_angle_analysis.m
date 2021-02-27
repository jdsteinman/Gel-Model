%% solid angle analysis

clc 
clear all
close all

% this code plots the displacement vs distance from cell surface data for
% 3D TFM experiments in solid angles around the cell

% Written by: Alex Khang

% Last Updated: 11/12/2020

%% Import data

% import cell vertices and faces
path = '../data/Gel3/';
vertices = textread(strcat(path, 'CytoD_vertices.txt'));
faces = textread(strcat(path, 'CytoD_faces.txt'));

% import bead locations in the initial and final configuration
initial = textread(strcat(path, 'beads_init.txt'));
final = textread(strcat(path, 'beads_final.txt'));
displacements = final - initial;

%% Compute unit normal vectors at the cell vertices

% pre-allocation of surface normals
normals = zeros(size(vertices));

% use a Matlab routine to find surface normals
TR = triangulation(faces,vertices);
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
    dist = sqrt(sum((vertices - initial(i,:)).^2,2)); % computes distance between bead and all cell vertices
    closest(i) = min(dist); % finds closest cell vertex to bead
    indx = find(dist == closest(i)); % finds which cell vertex is closest
    direction(i) = dot(normals(indx(1),:),displacements(i,:)/norm(displacements(i,:))); % dot product of cell vertex normal and bead displacement vector
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
        scores(j) = dot(displacements(i,:)/norm(displacements(i,:)),displacements(idx(i,j),:)/norm(displacements(idx(i,j),:)));
    end
    neighbor_score(i) = mean(scores);
end
    
    
% finds beads that are moving away and towards cell surface
towards = find(direction < 0);
away = find(direction > 0);

% displacement magnitude
d = sqrt(sum(displacements.^2,2));

% turns displacements towards the cell negative
% d(towards) = -d(towards);


%% Solid angle analysis

% center beads wrt to cell center
centered_beads = initial - mean(vertices);

% centerd cell
centered_nodes = vertices-mean(vertices);
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

% iterates through all solid angles/wedges
for i = 1:length(theta_intervals)
    if i < length(theta_intervals)
        for j = 1:length(phi_intervals)
            if j < length(phi_intervals)
                
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
                data = [initial(real_ind,:),final(real_ind,:),closest(real_ind)',displacements(real_ind,1),displacements(real_ind,2),displacements(real_ind,3),d(real_ind),direction(real_ind)',neighbor_score(real_ind)'];
                
                % only outputs data and plots if data exists 
                if ~isempty(data)
                
                % pre-allocation
                d_neg = zeros(1,length(direction(real_ind)));
                closest_neg = zeros(1,length(direction(real_ind)));
                d_pos = zeros(1,length(direction(real_ind)));
                closest_pos = zeros(1,length(direction(real_ind)));
                
                
                % makes displacements towards the cell negative 
                for k = 1:length(direction(real_ind))
                    if direction(real_ind(k)) < 0
                        d_neg(k) = -d(real_ind(k));
%                         d_neg(k) = d(real_ind(k));
                        closest_neg(k) = closest(real_ind(k));
                    else
                        d_pos(k) = d(real_ind(k));
                        closest_pos(k) = closest(real_ind(k));
                    end
                end
                
                % gets rid of zero values
                d_neg = nonzeros(d_neg);
                closest_neg = nonzeros(closest_neg);
                d_pos = nonzeros(d_pos);
                closest_pos = nonzeros(closest_pos);        
              
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

                % plotting
                figure
                % plots cell mesh
                subplot(1,3,1)
                trimesh(faces,centered_nodes(:,1),centered_nodes(:,2),centered_nodes(:,3),'FaceColor','r','EdgeColor','k'); hold on;
                view([ 1 1 1])
                % plots unit sphere
                subplot(1,3,1)
                surface(max_radius*x,max_radius*y,max_radius*z,colors,'FaceAlpha',0.6);
                s.EdgeColor = 'none';
                scatter3(unit_vector(real_ind,1),unit_vector(real_ind,2),unit_vector(real_ind,3)); hold on;
                xlabel('x-axis')
                ylabel('y-axis')
                zlabel('z-axis')
                view([1 1 1])
                % plots toward and away displacements vs distance from cell 
                subplot(1,3,2)
                scatter(closest_neg,d_neg,40,'r'); hold on;
                scatter(closest_pos,d_pos,40,'b'); hold on;
                rotation_axis = 0;
                line([0,150],[rotation_axis,rotation_axis]); hold on;
                ylim([-10 10])
                xticks(0:30:150);
                xlabel('distance from cell surface (\mum)')
                ylabel('marker displacement (\mum)')
                legend('Towards Cell', 'Away from Cell')
                grid on
                % plots a histogram of the displacement magnitudes in the solid
                % angle
                subplot(1,3,3)
                histogram(abs(d(real_ind)),'Normalization','probability','FaceAlpha',0.8); hold on;
                xlabel('displacement (\mum)')
                ylabel('probability')
                title('displacement magnitudes')
                grid on
                
                clear d_neg closest_neg d_pos closest_pos
               
                   
                end
            end
        end
    end
end

