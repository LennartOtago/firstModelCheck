clear all
close all

data = importdata('LCurve.txt', '\t' ,1);

res_norm = data.data(:,1);
sol_norm = data.data(:,2);
reg_param = data.data(:,3);

A = importdata('ForWardMatrix.txt', '\t' ,1).data;
L = importdata('GraphLaplacian.txt', '\t' ,1).data;
y = importdata('dataY.txt', '\t' ,1).data;

%%

[U,s,V] = csvd(A);


[reg_corner,rho,eta,reg_param] = l_curve(U,s,y,'Tikh',L,V);

%[reg_corner,rho_c,eta_c] = l_corner(res_norm, sol_norm, reg_param);%,U,s,y,'Tikh')



%%

% rho = res_norm;
% eta = sol_norm;
% 
% % Set default parameters for treatment of discrete L-curve.
% deg   = 2;  % Degree of local smooting polynomial.
% q     = 2;  % Half-width of local smoothing interval.
% order = 4;  % Order of fitting 2-D spline curve.
% 
% 
% % Convert to logarithms.
% lr = length(rho);
% lrho = log(rho); leta = log(eta); slrho = lrho; sleta = leta;
% 
% % For all interior points k = q+1:length(rho)-q-1 on the discrete
% % L-curve, perform local smoothing with a polynomial of degree deg
% % to the points k-q:k+q.
% v = (-q:q)'; A = zeros(2*q+1,deg+1); A(:,1) = ones(length(v),1);
% for j = 2:deg+1, A(:,j) = A(:,j-1).*v; end
% for k = q+1:lr-q-1
% cr = A\lrho(k+v); slrho(k) = cr(1);
% ce = A\leta(k+v); sleta(k) = ce(1);
% end
% 
% % Fit a 2-D spline curve to the smoothed discrete L-curve.
% sp = spmak((1:lr+order),[slrho';sleta']);
% pp = ppbrk(sp2pp(sp),[4,lr+1]);

%%

headers = ['l_curve output data of Hansens regtoolbox\nThe optimal parameter is: \n' + string(reg_corner) + '\n'];
fileID = fopen('l_curve_output.txt','w');
fprintf(fileID,headers);
fprintf(fileID,'%1s %2s %3s\n', 'rho', 'eta', 'reg_param');
fprintf(fileID,'%1.5f %2.5f %3.5f\n', [res_norm'; sol_norm'; reg_param']);
fclose(fileID);

close all

%[reg_c,rho_c,eta_c] = l_corner(res_norm,sol_norm,reg_param,U,s,y,'Tikh',1e10);