clear all
close all

data = importdata('LCurve.txt', '\t' ,1);

resid_norm = data.data(:,1);
reg_norm = data.data(:,2);
reg_param = data.data(:,3);

A = importdata('ForWardMatrix.txt', '\t' ,1).data;

L = importdata('GraphLaplacian.txt', '\t' ,1).data;
b = importdata('dataY.txt', '\t' ,1).data;

%%

[U,s,V] = csvd(A);
[UU,ss,XX,VV,WW] = cgsvd(A,L);


%[reg_corner,rho,eta,reg_param] = l_curve(U,s,b,'Tikh',sqrt(L),V);
%[reg_corner,rho,eta,reg_param_out] = l_curve(UU,ss,b,'Tikh',sqrt(L),VV);
% 
% [x_lambda,rho,eta] = tikhonov(UU,ss,VV,b,reg_param);
% reg_param_out = reg_param;
%[reg_corner,rho_c,eta_c] = l_corner(rho, eta, reg_param);%,U,s,y,'Tikh')
[reg_corner,rho,eta] = l_corner(resid_norm, reg_norm,  reg_param);%,U,s,b,'Tikh');
reg_param_out = reg_corner;

%[reg_min,G,reg_param] = gcv(UU,ss,b,'Tikh');

%%  % The L-curve is differentiable; computation of curvature in
  

% 
% nargin = 6;
% % Set default parameters for treatment of discrete L-curve.
% deg   = 2;  % Degree of local smooting polynomial.
% q     = 2;  % Half-width of local smoothing interval.
% order = 4;  % Order of fitting 2-D spline curve.
% 
% 
% % Ensure that rho and eta are column vectors.
% rho = sol_norm(:); eta = reg_norm(:);
% 
% % Set default regularization method.
% if (nargin <= 3)
%   method = 'none';
%   if (nargin==2), reg_param = (1:length(rho))'; end
% else
%   if (nargin==6), method = 'Tikh'; end
% end
% 
% % Initialization.
% if (length(rho) < order)
%   error('Too few data points for L-curve analysis')
% end
% if (nargin > 3)
%   [p,ps] = size(s); [m,n] = size(U);
%   beta = U'*b; b0 = b - U*beta;
%   if (ps==2)
%     s = s(p:-1:1,1)./s(p:-1:1,2);
%     beta = beta(p:-1:1);
%   end
%   xi = beta./s;
%   if (m>n)  % Take of the least-squares residual.
%       beta = [beta;norm(b0)];
%   end
% end
% 
% % log-log scale is easy.
% 
%   % Compute g = - curvature of L-curve.
%   g = lcfun(reg_param,s,beta,xi);
% 
%   % Locate the corner.  If the curvature is negative everywhere,
%   % then define the leftmost point of the L-curve as the corner.
%   [~,gi] = min(g);
%   reg_c = fminbnd('lcfun',...
%     reg_param(min(gi+1,length(g))),reg_param(max(gi-1,1)),...
%     optimset('Display','off'),s,beta,xi); % Minimizer.
%   kappa_max = - lcfun(reg_c,s,beta,xi); % Maximum curvature.
% 
%   if (kappa_max < 0)
%     lr = length(rho);
%     reg_c = reg_param(lr); rho_c = rho(lr); eta_c = eta(lr);
%   else
%     f = (s.^2)./(s.^2 + reg_c^2);
%     eta_c = norm(f.*xi);
%     rho_c = norm((1-f).*beta(1:length(f)));
%     if (m>n), rho_c = sqrt(rho_c^2 + norm(b0)^2); end
%   end
% 

%%

headers = ['l_curve output data of Hansens regtoolbox\nThe optimal parameter is: \n' + string(reg_corner) + '\n'];
fileID = fopen('l_curve_output.txt','w');
fprintf(fileID,headers);
fprintf(fileID,'%1s %2s %3s\n', 'rho/ resid_norm', 'eta/ reg_norm', 'reg_param');
fprintf(fileID,'%1.5f %2.5f %3.5f\n', [rho'; eta'; reg_param_out']);
fclose(fileID);

close all

%[reg_c,rho_c,eta_c] = l_corner(res_norm,sol_norm,reg_param,U,s,y,'Tikh',1e10);