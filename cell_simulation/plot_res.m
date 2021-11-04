clc
clear all
close all

data = readtable("res.csv");
data = table2array(data);
res = data(:,1:3);
res_mag = sqrt(sum(res.^2,2));
D = (res_mag - min(res_mag)) / (max(res_mag)-min(res_mag)) * 0.5;
r = data(:,4);

figure(1)
plot(r,res_mag,'r',"LineWidth",2)
ylabel("Discrepancy, (\mum)")
xlabel("Normal Distance from Cell Surface (\mum)")

figure(2)
plot(r,D,"LineWidth",2)
ylabel("Degratation Parameter, D")
xlabel("Normal Distance from Cell Surface (\mum)")
