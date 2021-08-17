%%
clc
clear all
close all
%% Plotting Preamble
set(gca,'fontname','helvetica')
set(groot,'defaultAxesTitleFontSizeMultiplier',1)
set(groot,'defaultErrorbarLineWidth', 0.5)
set(groot,'defaultAxesFontName','Helvetica')
set(groot,'defaultAxesPlotBoxAspectRatioMode','manual')
set(groot,'defaultAxesPlotBoxAspectRatio',[1 1 1])
set(groot,'defaultAxesDataAspectRatioMode','auto')
set(groot,'defaultAxesDataAspectRatio',[1 1 1])
% set(groot,'defaultAxesDataAspectRatio','default')

set(groot,'defaultAxesFontWeight','Normal')
set(0,'DefaultAxesTitleFontWeight','normal');

set(groot,'defaultAxesFontSizeMode','manual')
set(groot,'defaultAxesFontSize',15)
set(groot,'defaultAxesLabelFontSizeMultiplier',1)
set(groot,'defaultAxesLineWidth',2)
set(groot,'defaultScatterLineWidth',2)
% set(groot,'defaultScatterMarkerFaceColor','k')
% set(groot,'defaultScatterMarkerEdgeColor','k')
set(groot,'defaultScatterMarkerFaceColor','default')
set(groot,'defaultScatterMarkerEdgeColor','default')
set(groot,'defaultLineColor','k')
set(groot,'defaultLineLineWidth',2)
set(groot,'defaultLineMarkerSize',1)
co = [0,0,0;0 0 1;0 0.5 0;1 0 0;0 0.75 0.75;0.75 0 0.75;0.75 0.75 0;0.25 0.25 0.25];
set(groot,'defaultAxesColorOrder',co)
close

%% Inputs
U_data = readtable("output/mu108_kappa7500/bead_displacements_sim.txt");
U_data = table2array(U_data);
U_data = U_data(:,5:end);

U_sim = readtable("output/graded/bead_displacements_sim.txt");
U_sim = table2array(U_sim);
U_sim = U_sim(:,5:end);

%% Plots
figure
scatter(U_data(:,1), U_sim(:,1));
hold on
plot(U_data(:,1), U_data(:,1))
hold off
title("X-displacement (\mum)")
xlabel("Homogeneous Model (\mu=108 Pa)")
ylabel("Graded Model (p=0.5)")
% xlabel("Observed X-Displacement (\mum)")
% ylabel("Simulated X-Displacement (\mum)")
xlim([-2 4])
ylim([-2 4])
pbaspect([1 1 1])
corr_x = corrcoef(U_data(:,1), U_sim(:,1));
text(2,-1,sprintf("r=%f",corr_x(1,2)))

figure
scatter(U_data(:,2), U_sim(:,2));
hold on
plot(U_data(:,2), U_data(:,2))
hold off
title("Y-displacement (\mum)")
xlabel("Homogeneous Model (\mu=108 Pa)")
ylabel("Graded Model (p=0.5)")
% xlabel("Observed Y-Displacement (\mum)")
% ylabel("Simulated Y-Displacement (\mum)")
xlim([-2 4])
ylim([-2 4])
pbaspect([1 1 1])
corr_y = corrcoef(U_data(:,2), U_sim(:,2));
text(2,-1,sprintf("r=%f",corr_y(1,2)))

figure
scatter(U_data(:,3), U_sim(:,3));
hold on
plot(U_data(:,1), U_data(:,1))
hold off
title("Z-displacement (\mum)")
xlabel("Homogeneous Model (\mu=108 Pa)")
ylabel("Graded Model (p=0.5)")
% xlabel("Observed Z-Displacement (\mum)")
% ylabel("Simulated Z-Displacement (\mum)")
xlim([-2 4])
ylim([-2 4])
pbaspect([1 1 1])
corr_z = corrcoef(U_data(:,3), U_sim(:,3));
text(2,-1,sprintf("r=%f",corr_z(1,2)))

figure
U_mag_data = sqrt(U_data(:,1).^2 + U_data(:,2).^2 + U_data(:,3).^2);
U_mag_sim = sqrt(U_sim(:,1).^2 + U_sim(:,2).^2 + U_sim(:,3).^2);
scatter(U_mag_data, U_mag_sim)
hold on
plot(U_mag_data, U_mag_data)
hold off
title("Displacement Magnitude (\mum)")
xlabel("Homogeneous Model (\mu=108 Pa)")
ylabel("Graded Model (p=0.5)")
% xlabel("Observed Displacement (\mum)")
% ylabel("Simulated Displacement (\mum)")
xlim([0,4])
ylim([0,4])
pbaspect([1 1 1])
corr_mag = corrcoef(U_mag_data, U_mag_sim);
text(1,2,sprintf("r=%f",corr_mag(1,2)))