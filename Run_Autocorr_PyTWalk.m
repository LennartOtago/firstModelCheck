munlock('UWerr')
clear all
close all


data = importdata('/Users/lennart/PycharmProjects/firstModelCheck/MargPostDat.txt');
data = readmatrix('/Users/lennart/PycharmProjects/firstModelCheck/MargPostDat.txt');
deltas = data(:,1); %str2num(cell2mat(data.textdata(2:end,1)));
gammas = data(:,2); %str2num(cell2mat(data.textdata(2:end,2)));
lambdas = deltas./gammas;

burn = 50;
[value_lam,dvalue_lam,ddvalue_lam,tauint_lam,dtauint_lam,Qval_lam,rho_lam,drho_lam,tmax_lam] = UWerr(lambdas(burn:end));
[value_gam,dvalue_gam,ddvalue_gam,tauint_gam,dtauint_gam,Qval_gam,rho_gam,drho_gam,tmax_gam] = UWerr(gammas(burn:end));
[value_del,dvalue_del,ddvalue_del,tauint_del,dtauint_del,Qval_del,rho_del,drho_del,tmax_del] = UWerr(deltas(burn:end));

%% Save rho_... for Autocorrelation plot
l = length(rho_lam);
if length(rho_lam) < length(rho_gam) && length(rho_del) < length(rho_gam)
    l = length(rho_gam);
elseif length(rho_lam) < length(rho_del) && length(rho_gam) < length(rho_del)
    l = length(rho_del) ;   
end

x = zeros(1,l);
y = zeros(1,l);
z = zeros(1,l);

x(1,1:length(rho_del)) = rho_del;
y(1,1:length(rho_gam)) = rho_gam;
z(1,1:length(rho_lam)) = rho_lam;

headers ='Integrated Autocorrelation time\n';
fileID = fopen('/Users/lennart/PycharmProjects/firstModelCheck/autoCorrPyTWalk.txt','w');
fprintf(fileID,headers);
fprintf(fileID,'%1.5f %2.5f %3.5f\n', (2*tauint_del), (2*tauint_gam), (2*tauint_lam));
fprintf(fileID,'%1s %2s %3s\n', 'rho_delta', 'rho_gamma', 'rho_lambda');
fprintf(fileID,'%1.5f %2.5f %3.5f\n', [ x; y; z]);
fclose(fileID);

close all