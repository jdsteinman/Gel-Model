%% Calculates Deformation Tensors from DIsplacement Gradient over line
%  Written By: John Steinman

clc
clear all
close all
 
%% Load Paraview Data

gradx = readtable("../../gel_model/output/faceBC/grad_x.csv");
gradx = gradx(~any(ismissing(gradx),2),:);
gradx = gradx(gradx.Points_0 > 0, :);

grady = readtable("../../gel_model/output/faceBC/grad_y.csv");
grady = grady(~any(ismissing(grady),2),:);
grady = grady(grady.Points_1 > 0, :);

gradz = readtable("../../gel_model/output/faceBC/grad_z.csv");
gradz = gradz(~any(ismissing(gradz),2),:);
gradz = gradz(gradz.Points_2 > 0, :);

%% Load txt Data
datax = readtable("../../gel_model/output/faceBC/data_x.csv");
datay = readtable("../../gel_model/output/faceBC/data_y.csv");
dataz = readtable("../../gel_model/output/faceBC/data_z.csv");

%% Displacement Plots
plotting_preamble()
figure(1)
hold on
% h(1) = plot(datax.x - 10, datax.ux, "r-o", "LineWidth", 2);
% h(2) = plot(datay.y - 10, datay.uy, "g-o", "LineWidth", 2);
% h(3) = plot(dataz.z - 20, dataz.uz, "b-o", "LineWidth", 2);
h(1) = scatter(datax.x - 10, datax.ux, "r");
h(2) = scatter(datay.y - 10, datay.uy, "g");
h(3) = scatter(dataz.z - 20, dataz.uz, "b");
hold off
grid on
title("Comparison of Displacement over Axes")
xlabel("Distance from Surface along Axes")
ylabel("Displacement Component")
legend("ux on x", "uy on y", "uz on z")


%% Gradient Plots
plotting_preamble()
figure(1)
subplot(1,3,1)
hold on
h(1) = plot(gradx.Points_0 - 10, gradx.grad_u_0, "r", "LineWidth", 3);
h(2) = plot(datax.x - 10, datax.g11, "b", "LineWidth", 3);
hold off
grid on
title("Displacement Gradient on X Axis")
xlabel("Distance from Surface along X Axis")
ylabel("11 Component of Grad(u)")
legend("Paraview Output", "Text Output", "Location", "best")

subplot(1,3,2)
hold on
h(1) = plot(grady.Points_1 - 10, grady.grad_u_4, "r", "LineWidth", 3);
h(2) = plot(datay.y - 10, datay.g22, "b", "LineWidth", 3);
hold off
grid on
title("Displacement Gradient on Y Axis")
xlabel("Distance from Surface along Y Axis")
ylabel("22 Component of Grad(u)")
legend("Paraview Output", "Text Output", "Location", "best")

subplot(1,3,3)
hold on
h(1) = plot(gradz.Points_2 - 20, gradz.grad_u_8, "r", "LineWidth", 3);
h(2) = plot(dataz.z - 20, dataz.g33, "b", "LineWidth", 3);
hold off
grid on
title("Displacement Gradient on Z Axis")
xlabel("Distance from Surface along Z Axis")
ylabel("33 Component of Grad(u)")
legend("Paraview Output", "Text Output", "Location", "best")

%% Calculate Deformations on X axis

G = table2array(datax(:,11:19));

% for i=1:size(G,1)
Farr = zeros(length(G), 9);
Carr = zeros(length(G), 9);
w  = zeros(length(G), 3);
v1 = zeros(length(G), 3);
v2 = zeros(length(G), 3);
v3 = zeros(length(G), 3);

for i=1:length(G)
    gradu = reshape(G(i,:), [3,3]);
    gradu = gradu';
    
    F = eye(3) + gradu;
    
    [U,S,V] = svd(F);
    R = U*V;
    U = V*S*V';
    
    C = F'*F;
    [v, w] = eig(C);
    
    Farr(i,:) = reshape(F', [1,9]);
    Carr(i,:) = reshape(C', [1,9]);
    w(i,:)    = diag(w);
    v1(i,:)   = v(:,1);
    v2(i,:)   = v(:,2);
    v3(i,:)   = v(:,3);
end
