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
set(groot,'defaultLineMarkerSize',2)
co = [0,0,0;0 0 1;0 0.5 0;1 0 0;0 0.75 0.75;0.75 0 0.75;0.75 0.75 0;0.25 0.25 0.25];
set(groot,'defaultAxesColorOrder',co)