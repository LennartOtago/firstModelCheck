clear all
close all

data = importdata('LCurve.txt', '\t' ,1);

res_norm = data.data(:,1);
sol_norm = data.data(:,2);
reg_param = data.data(:,3);

A = importdata('A_lin.txt', '\t' ,1).data;
L = importdata('GraphLaplacian.txt', '\t' ,1).data;
y = importdata('dataY.txt', '\t' ,1).data;

%%

[U,s,V] = csvd(A);


[reg_corner,rho,eta,reg_param] = l_curve(U,s,y,'Tikh',L,V);

%[reg_c,rho_c,eta_c] = l_corner(res_norm,sol_norm,reg_param,U,s,y,'Tikh',1e10);