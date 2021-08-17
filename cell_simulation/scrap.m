clc
clear all
close all

mu = 108;
rmax = 100;

r = 0:80;
figure
set(0, 'DefaultLineLineWidth', 1.5);

p1=plot(r, func(r, 0), "DisplayName", "p=0");
hold on
p2=plot(r, func(r, 0.25), "DisplayName", "p=0.25");
p3=plot(r, func(r, 1), "DisplayName", "p=1");
p4=plot(r, func(r, 4), "DisplayName", "p=4");
hold off
title("Shear Modulus Profiles")
xlabel("Distance from cell surface (\mum)")
ylabel("Shear Modulus (Pa)")
legend()


function[mu] = func(r,p)
    mu = zeros(length(r),1);
    mu_bulk = 108;
    rmax = 40;
    
    for i=1:length(mu)
        if r(i) < rmax
            mu(i) = mu_bulk*(r(i)/rmax).^p; 
        else
            mu(i) = mu_bulk;
        end
    end
end