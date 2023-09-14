from importetFunctions import *
import time
import pickle as pl
#import matlab.engine
from functions import *
from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import testReal
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from numpy import inf
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as colors

from matplotlib.ticker import FuncFormatter
def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)

""" analayse forward map without any real data values"""

MinH = 5
MaxH = 95
R = 6371 # earth radiusin km
ObsHeight = 300 # in km
scalingConstkm = 1e-3
#FakeObsHeight = MaxH + 5




"""
    make a plot with conditionnumber on y axis and layers on x axis, 
    with stable measurment numbers
"""
# #cond is max(s_i)/min
# coeff = 0.05
# min_m = 15
# max_m = 180
# max_l = 90
# min_l = 15
# num_meas = np.linspace(min_m,max_m,max_m-min_m+1)
# num_lay = np.linspace(min_l, max_l, (max_l - min_l) + 1)
# cond_A_exp = np.zeros((len(num_meas),len(num_lay)))
# cond_ATA_exp = np.zeros((len(num_meas),len(num_lay)))
# for j in range(len(num_meas)):
#     meas_ang = min_ang + (
#                 (max_ang - min_ang) * np.exp(coeff * (np.linspace(0, int(num_meas[j]) - 1, int(num_meas[j])+1 ) - (int(num_meas[j]) - 1))))
#
#     for i in range(len(num_lay)):
#         layers = np.linspace(min_h, max_h, int(num_lay[i])+1)
#         A, tang_heights = gen_forward_map(meas_ang[0:-1], layers, obs_height, R)
#         ATA = np.matmul(A.T, A)
#         ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
#         Au, As, Avh = np.linalg.svd(A)
#         cond_A_exp[j,i]= np.max(As)/np.min(As)
#         cond_ATA_exp[j,i] = np.max(ATAs)/np.min(ATAs)
#
#
# #linear case
# num_meas = np.linspace(min_m,max_m,max_m-min_m+1)
# num_lay = np.linspace(min_l, max_l, (max_l - min_l) + 1)
# cond_A_lin = np.zeros((len(num_meas),len(num_lay)))
# cond_ATA_lin = np.zeros((len(num_meas),len(num_lay)))
# for j in range(len(num_meas)):
#     meas_ang = np.linspace(min_ang,max_ang,int(num_meas[j])+1)
#     for i in range(len(num_lay)):
#         layers = np.linspace(min_h, max_h, int(num_lay[i])+1)
#         A, tang_heights = gen_forward_map(meas_ang[0:-1], layers, obs_height, R)
#         ATA = np.matmul(A.T, A)
#         ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
#         Au, As, Avh = np.linalg.svd(A)
#         cond_A_lin[j,i]=  np.max(As)/np.min(As)
#         cond_ATA_lin[j,i] =  np.max(ATAs)/np.min(ATAs)
#
#
#
#
#
#
# #cond_A[cond_A == inf] = np.nan
# vmin_lin = np.min((np.min(cond_A_lin[cond_A_lin != -inf]),np.min(cond_ATA_lin[cond_A_lin != -inf])) )
# vmax_lin = np.max((np.max(cond_A_lin[cond_A_lin != inf]), np.max(cond_A_lin[cond_A_lin != inf])))
# vmin_exp = np.min((np.min(cond_A_exp[cond_A_exp != -inf]),np.min(cond_ATA_exp[cond_A_exp != -inf]) ))
# vmax_exp = np.max((np.max(cond_A_exp[cond_A_exp != inf]), np.max(cond_A_exp[cond_A_exp != inf])))
#
#
# vmax = np.log10(np.max((vmax_lin, vmax_exp)))# np.max(cond_A_exp[cond_A_exp != inf])) # np.max((vmax_lin, vmax_exp))
# vmin = np.log10( np.min((vmin_lin, vmin_exp)))#np.min(cond_A_exp[cond_A_exp != -inf])) #np.min((vmin_lin, vmin_exp))
# cmap = mpl.pyplot.viridis()#cmr.sunburst
# ticks = np.linspace(np.ceil(vmin),np.floor(vmax),11 ,dtype = 'int')
# ticklabels = ['1e' + str(tick) for tick in ticks]
#
# # Creating figure for exp case
# fig, axs = plt.subplots(1,2,figsize=(12,6))
# pl1 = axs[0].imshow(np.log10(cond_A_exp), cmap=cmap , vmin=vmin, vmax=vmax,extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]], aspect = 'auto')
# axs[0].set_ylabel('Number of Measurement')
# axs[0].set_xlabel('Number of Layers in Model')
# pl2 = axs[1].imshow(np.log10(cond_ATA_exp), cmap=cmap,vmin=vmin, vmax=vmax, extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]], aspect = 'auto')
# axs[1].set_ylabel('Number of Measurement')
# axs[1].set_xlabel('Number of Layers in Model')
# #ax2.get_ylim([num_meas[0],num_meas[-1]])
# cbar = fig.colorbar(pl1,ax=axs,location='bottom')#, orientation='horizontal')
# cbar.set_ticks(ticks)
# cbar.set_ticklabels(ticklabels)
# axs[1].set_title('Condition Number of $A^T$ A')
# axs[0].set_title('Condition Number of A')
# fig.suptitle('Conditionnumber for exponentially spaced measurement', fontsize=16)
# plt.savefig('cond_A_exp.png', dpi=300)
# plt.show()
#
#
# # Creating figure for linear case
# fig, axs = plt.subplots(1,2,figsize=(12,6))
# pl1 = axs[0].imshow(np.log10(cond_A_lin), cmap=cmap, vmin=vmin, vmax=vmax, extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]], aspect='auto')
# axs[0].set_ylabel('Number of Measurement')
# axs[0].set_xlabel('Number of Layers in Model')
# #ax[1].imshow(cond_ATA, cmap='hot', interpolation='nearest',norm=LogNorm())
#
# pl2 = axs[1].imshow(np.log10(cond_ATA_lin), cmap=cmap, vmin=vmin, vmax=vmax, extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]] ,aspect='auto')
# axs[1].set_ylabel('Number of Measurement')
# axs[1].set_xlabel('Number of Layers in Model')
# #ax2.get_ylim([num_meas[0],num_meas[-1]])
# cbar = fig.colorbar(pl1,ax=axs,location='bottom')#, orientation='horizontal')
# cbar.set_ticks(ticks)#, orientation='horizontal')
# cbar.set_ticklabels(ticklabels)
# axs[1].set_title('Condition Number of $A^T$ A')
# axs[0].set_title('Condition Number of A')
# fig.suptitle('Conditionnumber for equally spaced measurements', fontsize=16)
# plt.savefig('cond_A_lin.png', dpi=300)
# plt.show()






''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas = 105
SpecNumLayers = 46
LayersCore = np.linspace(MinH, MaxH, SpecNumLayers)
layers = np.zeros(SpecNumLayers + 2)
layers[1:-1] =  LayersCore
layers[0]= MinH-3
layers[-1] = MaxH+5
#fing minimum and max angle in radians
MaxAng = np.arcsin((layers[-1]+ R) / (R + ObsHeight))
MinAng = np.arcsin((layers[0] + R) / (R + ObsHeight))
#add zero layers
SpecNumLayers = SpecNumLayers + 2
#meas_ang = MinAng + ((MaxAng - MinAng) * np.exp(coeff * (np.linspace(0, int(spec_num_meas) - 1, int(spec_num_meas) + 1) - (int(spec_num_meas) - 1))))
#meas_ang = np.linspace(min_ang, max_ang, num_meas+ 1)

# A_exp, tang_heights_exp = gen_forward_map(meas_ang[0:-1],layers,obs_height,R)
# A_expu, A_exps, A_expvh = np.linalg.svd(A_exp)
# ATA_exp = np.matmul(A_exp.T,A_exp)
# #condition number for A
# cond_A_exp =  np.max(A_exps)/np.min(A_exps)
# print("normal: " + str(orderOfMagnitude(cond_A_exp)))
#
# #to test that we have the same dr distances
# tot_r = np.zeros(spec_num_meas)
# #calculate total length
# for j in range(0,spec_num_meas):
#     tot_r[j] = 2*np.sqrt( (layers[-1] + R)**2 - (tang_heights_exp[j] + R )**2 )
# print('Distance trhough layers check: ' + str(np.allclose( sum(A_exp.T), tot_r)))
# #dont consider last h_val as we have layers from lowest h_val up to second highest
# ATA_expu, ATA_exps, ATA_expvh = np.linalg.svd(ATA_exp)#plot_svd(ATA, layers[0:-1])
# print("ATA: " + str(orderOfMagnitude(np.max(np.sqrt(ATA_exps))/np.min(np.sqrt(ATA_exps)))))
# # #plot sing vec and sing vals
# fig, axs = plt.subplots(1,2,figsize=(12,6))
# axs[0].scatter(range(spec_num_meas),tang_heights_exp)
# #axs.title('Measurement Setup with Conditonnumber for Forward map '+  str(cond_A))
# axs[0].set_xlabel('Number of Measurements')
# axs[0].set_ylabel('Height in km')
# axs[1].scatter(range(0,spec_num_layers),A_exps)
# fig.suptitle('Singular values of A with Conditionnumber ' + str(np.around(cond_A_exp)))
# axs[1].set_yscale('log')
# axs[1].set_ylabel('Value')
# axs[1].set_xlabel('Index')
# axs[1].set_ylim([1e1, 1e4])
# plt.savefig('ExpScalExp.png')
# plt.show()


#find best configuration of layers and num_meas
#so that cond(A) is not inf
#meas_ang = min_ang + ((max_ang - min_ang) * np.exp(coeff * (np.linspace(0, int(num_meas) - 1, int(num_meas)+1) - (int(num_meas) - 1))))
meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas + 1)
A_lin, tang_heights_lin = gen_forward_map(meas_ang[0:-1],layers,ObsHeight,R)
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
A_lin = A_lin
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros(SpecNumMeas)
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = 2*np.sqrt( (layers[-1] + R)**2 - (tang_heights_lin[j] + R )**2 )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T), tot_r)))


# ATA_linu, ATA_lins, ATA_linvh = np.linalg.svd(ATA_lin)
# print("ATA: " + str(orderOfMagnitude(np.max(np.sqrt(ATA_lins))/np.min(np.sqrt(ATA_lins)))))
# # #plot sing vec and sing vals
# fig, axs = plt.subplots(1,2,figsize=(12,6))
# axs[0].scatter(range(SpecNumMeas), tang_heights_lin)
# #axs.title('Measurement Setup with Conditonnumber for Forward map '+  str(cond_A))
# axs[0].set_xlabel('Number of Measurements')
# axs[0].set_ylabel('Height in km')
# axs[1].scatter(range(0, SpecNumLayers - 1), A_lins)
# fig.suptitle('Singular values of $F^T F$ with Conditionnumber ' + str(np.around(cond_A_lin)))
# axs[1].set_yscale('log')
# axs[1].set_ylabel('Value')
# axs[1].set_xlabel('Index')
# axs[1].set_ylim([1e1, 1e4])
# plt.savefig('SingularScalLin.png')
# plt.show()


#
#
# # analyse singlar vectors for A.T A for specific num of layers
# # specifiy layers in km
#
# gradient = np.vstack(
#     (uniform(0, 1, SpecNumLayers - 1), uniform(0, 1, SpecNumLayers - 1), uniform(0, 1, SpecNumLayers - 1))).T
#
#
# # Indices to step through colormap.
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# for i in range(SpecNumLayers - 1):
#     ax1.plot(ATA_linu[:, i], color=gradient[i], linewidth=2,
#              path_effects=[path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()])
#     text = ax1.text(i, ATA_linu[i, i], f'{i}', color=gradient[i], fontsize=20)
#     text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
#                            path_effects.Normal()])
#     ax2.scatter(i, ATA_lins[i], color=gradient[i], s=50)  # gradient[:,i]
#     ax2.text(i, ATA_lins[i], f'{i}')
#
# ax2.set_yscale('log')
# ax1.set_xlabel('index singular vector')
# ax2.set_xlabel('index singular value')
# ax1.set_ylabel('value')
# ax2.set_ylabel('value')
# plt.savefig('svd_lin.png')
# plt.show()





# # Indices to step through colormap.
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# for i in range(spec_num_layers - 1):
#     ax1.plot(ATA_expu[:, i], color=gradient[i], linewidth=2,
#              path_effects=[path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()])
#     text = ax1.text(i, ATA_expu[i, i], f'{i}', color=gradient[i], fontsize=20)
#     text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
#                            path_effects.Normal()])
#     ax2.scatter(i, ATA_exps[i], color=gradient[i], s=50)  # gradient[:,i]
#     ax2.text(i, ATA_exps[i], f'{i}')
#     ax2.set_yscale('log')
#
#
# ax2.set_yscale('log')
# ax1.set_xlabel('index singular vector')
# ax2.set_xlabel('index singular value')
# ax1.set_ylabel('value')
# ax2.set_ylabel('value')
# plt.savefig('svd_exp.png')
# plt.show()





#graph Laplacian
neigbours = np.zeros((len(layers)-1,2))
neigbours[0] = np.nan, 1
neigbours[-1] = len(layers)-3, np.nan
for i in range(1,len(layers)-2):
    neigbours[i] = i-1, i+1
L = generate_L(neigbours)
np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')



#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
#/Users/lennart/PycharmProjects/firstModelCheck
#obs_height= 300 #in km
#filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml' #/home/lennartgolks/Python /Users/lennart/PycharmProjects/firstModelCheck/tropical.O3.xml
filename = 'tropical.O3.xml'

VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
d_height = layers[1::] - layers[0:-1]

pressure_values = pressure_values * 1e2 # in cgs
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

# plt.plot(pressure_values, height_values)
# plt.plot(VMR_O3, layers)
# plt.show()

temp_values = get_temp_values(layers[0:-1] + d_height/2 )
#x = VMR_O3 * N_A * pressure_values /(R_gas * temp_values)#* 1e-13
#https://hitran.org/docs/definitions-and-units/
#files = '/home/lennartgolks/Python/firstModelCheck/634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects
files = '634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
F = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))


for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])




#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1


mol_M = 48 #g/mol for Ozone
ind = 293

#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
f_0 = c_cgs*v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")

C1 =2 * scy.constants.h * scy.constants.c**2 * v_0**3 * 1e8
C2 = scy.constants.h * scy.constants.c * 1e2 * v_0  / (scy.constants.Boltzmann * temp_values )
#plancks function
Source = np.array(C1 /(np.exp(C2) - 1) ).reshape((47,1))

#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K usually
p_ref = pressure_values[0]



''' calculate voigt funciton... does not work yet.. do later'''
#calc pressure HWHM
#calculate voigt function
# gamma = (T_ref/ temp_values)**n_air[ind]  * g_self[ind] * pressure_values
#
# #doppler half width
#
# sigm_gaus = 7.17*1e-7 * f_0 * np.sqrt(250/48) /2  / (2/np.sqrt(np.log(2)))
# delta_gamma2 = np.sqrt( 2 * k_b_cgs * np.log(2) * N_A* temp_values / (48)) * f_0/ c_cgs

#calc Dopple HWHM for different temperatures
# C3 = np.sqrt( N_A  * k_b_cgs * 2* np.log(2))/c_cgs #[7.17e-7
# alpha =  np.array([ C3*v_0 * np.sqrt(temp/mol_M) for temp in temp_values])
# sigma = alpha / np.sqrt(2 * np.log(2))
#
# sigma_gauss = v_0/c_cgs * np.sqrt(N_A * k_b_cgs * 260/ mol_M)
# x_wvnmbrs = np.linspace(-1*v_0,1*v_0,200)
# norm_Voigt = V(x_wvnmbrs,sigma[20],gamma[20]) /sum(V(x_wvnmbrs,sigma[20],gamma[20]))
# norm_Gauss = G(x_wvnmbrs,sigm_gaus)# /sum(G(x_wvnmbrs,sigma_gauss))
# norm_Lorenz = Lorenz(x_wvnmbrs,gamma[20])/ sum(Lorenz(x_wvnmbrs,gamma[20]))
# plt.plot(x_wvnmbrs, norm_Voigt, 'red')
# plt.plot(x_wvnmbrs, norm_Lorenz, 'green')
# plt.plot(x_wvnmbrs, norm_Gauss, 'blue')
# plt.show()


# p = np.logspace(3,-3,200)
# ax = plt.subplot()
# plt.plot(f_0 + p, p)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.show()

# vmr might not have correct units
#C4 =  [V(x_wvnmbrs[500],sigma[temp],gamma[temp])/sum(V(x_wvnmbrs,sigma[temp],gamma[temp])) for temp in range(0, len(temp_values)) ] #scaling trhough dopller/voigt profile

'''weighted absorption cross section according to Hitran and MIPAS instrument description
S is: The spectral line intensity (cm^−1/(molecule cm^−2))
f_broad in (1/cm^-1) is the broadening due to pressure and doppler effect,
 usually one can describe this as the convolution of Lorentz profile and Gaussian profile
 VMR_O3 is the ozone profile in units of molecule (unitless)
 has to be extended if multiple gases are to be monitored
 I multiply with 1e-4 to go from cm^2 to m^2
 '''
f_broad = 1
w_cross = S[ind,0] * VMR_O3 * f_broad * 1e-4
w_cross[0], w_cross[-1] = 0, 0



''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''
#take linear
num_mole = pressure_values / (scy.constants.Boltzmann * temp_values)
scalingConst = 1e16
theta =(num_mole * w_cross * Source * scalingConst )
Ax = np.matmul(A_lin, theta)

#convolve measurements and add noise
y = add_noise(Ax, 0.01)
#y[y < 0] = 0
#ATy = np.matmul(A_lin.T, y)
ATy = np.matmul(A_lin.T, y)

np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')

#plt.plot( y, tang_heights_lin)

# fig2, ax = plt.subplots()
# #plt.plot( VMR_O3 * 1e6 ,layers)
# plt.plot( theta ,layers[0:-1] + d_height/2)
# #ax.set_ylim([tang_heights_lin])
# plt.xlabel('Volume Mixing Ratio Ozone in ppm')
# plt.ylabel('Height in km')
# plt.savefig('measurement.png')
# plt.show()




"""start the mtc algo with first guesses of noise and lumping const delta"""

tol = 1e-8
vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MargPost(params):#, coeff):

    gamma = params[0]
    delta = params[1]
    # ATA_lin = coeff[0]
    # L = coeff[1]
    if delta < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers-1

    Bp= ATA_lin + delta/gamma * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A_lin, L,  delta/gamma)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(delta) + 0.5 * G + 0.5 * gamma * F + 1e-4 * ( delta + gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MargPost, [1/np.var(y),1/(2*np.mean(vari))])

#minimum = optimize.minimize(MargPost, [1/np.var(y), 1/(2*np.mean(vari))], args = [ATA_lin, L])
print(minimum)
print(minimum[1]/minimum[0])


""" finaly calc f and g with a linear solver adn certain lambdas
 using the gmres"""

lam= np.logspace(-4,14,100)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))



for j in range(len(lam)):

    B = (ATA_lin + lam[j] * L)

    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    #print(exitCode)

    CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
    if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol:
        f_func[j] = f(ATy, y, B_inv_A_trans_y)
    else:
        f_func[j] = np.nan

    g_func[j] = g(A_lin, L, lam[j])

# fig,axs = plt.subplots(1,2, figsize=(14, 5))
# axs[1].plot(lam,g_func)
# axs[0].plot(lam,f_func)
#
# #axs.set_yscale('log')
# axs[0].set_xscale('log')
# axs[1].set_xscale('log')
# axs[0].set_yscale('log')
#
# axs[0].set_xlabel('$\lambda$')
# axs[1].set_xlabel('$\lambda$')
#
# axs[0].set_ylabel('f($\lambda$)')
# axs[1].set_ylabel('g($\lambda$)')
# #plt.savefig('f_and_g.png')
# plt.show()

# lam = 1e3
# #number is approx 130 so 10^2
# #B is symmetric and positive semidefinite and normal
# B = (ATA_lin + lam* L) #* 1e-14eta/gamma
# Bu, Bs, Bvh = np.linalg.svd(B)
# fig, axs = plt.subplots(1,1,figsize=(12,6))
# axs.set_yscale('log')
# plt.scatter(range(0, SpecNumLayers-1), Bs)
# plt.savefig('SingVal_B_exp.png')
# plt.show()

# #condition number for B
# cond_B =  np.max(Bs)/np.min(Bs)
# print("Cond B: " + str(orderOfMagnitude(cond_B)))
#
# B_inv_L = np.zeros(np.shape(B))
# for i in range(len(B)):
#     B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=1e-7, restart=25)
#     #print(exitCode)
#
# CheckB_inv_L = np.matmul(B, B_inv_L)
# print(np.allclose(CheckB_inv_L, L, atol=1e-6))
#



'''check error in g(lambda)'''


B = (ATA_lin + minimum[1]/minimum[0] * L)

B_inv = np.zeros(np.shape(B))
for i in range(len(B)):
    e = np.zeros(len(B))
    e[i] = 1
    B_inv[:, i], exitCode = gmres(B, e, tol=tol, restart=25)
    if exitCode!= 0 :
        print(exitCode)

B_inv_L = np.matmul(B_inv, L)
num_sam = 10
trace_B_inv_L_1 = g_MC_log_det(B_inv_L, num_sam)
trace_B_inv_L_2 = g_MC_log_det(np.matmul(B_inv_L, B_inv_L), num_sam)
stdL1 = np.sqrt(np.var(trace_B_inv_L_1))
stdL2 = np.sqrt(np.var(trace_B_inv_L_2))

MCErrL1 = stdL1/ np.sqrt(num_sam)
MCErrL2 = stdL2/ np.sqrt(num_sam)


''' check taylor series in f(lambda)
around lam0 delta_lam = '''

lam0 =minimum[1] / minimum[0]
lam_try = np.linspace(lam0-1e4,lam0+1e4,101)
f_try_func = np.zeros(len(lam_try))
g_try_func = np.zeros(len(lam_try))

g_func_tay = np.ones(len(lam_try)) * g(A_lin, L, lam0)

B = (ATA_lin + lam0* L)
B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
f_func_tay = np.ones(len(lam_try)) *  f(ATy, y, B_inv_A_trans_y)

for j in range(len(lam_try)):

    B = (ATA_lin + lam_try[j] * L)

    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    #print(exitCode)

    CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
    # CheckB_inv_L = np.matmul(B, B_inv_L)
    # print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)
    if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol :
        f_try_func[j] = f(ATy, y, B_inv_A_trans_y)
    else:
        f_try_func[j] = np.nan
    delta_lam = lam_try[j] - lam0

    g_try_func[j] = g(A_lin, L, lam_try[j])

    B_inv_L = np.zeros(np.shape(B))
    for i in range(len(B)):
        B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
        if exitCode != 0:
            print(exitCode)

    B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
    B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
    B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
    B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)

    f_func_tay[j] = f_func_tay[j] + f_tayl(delta_lam, B_inv_A_trans_y, ATy[0::, 0], B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4, B_inv_L_5)
    g_func_tay[j] = g_func_tay[j] + g_tayl(delta_lam, B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4, B_inv_L_5)

# fig,axs = plt.subplots()
# axs.plot(lam_try,f_func_tay, color = 'red',linewidth = 5)
# axs.plot(lam_try,f_try_func)
# axs.set_xscale('log')
# axs.set_yscale('log')
# plt.show()
#
# ''' check taylor series in g(lambda)
# around lam0 delta_lam = '''
#
# fig,axs = plt.subplots()
# axs.plot(lam_try,g_func_tay, color = 'red',linewidth = 5)
# axs.plot(lam_try,g_try_func)
# axs.set_xscale('log')
# axs.set_yscale('log')
# plt.show()




'''do the sampling'''
 #10**(orderOfMagnitude(abs_tol * np.linalg.norm(L[:,1]))-2)
#hyperarameters
number_samples = 10000
gammas = np.zeros(number_samples)
deltas = np.zeros(number_samples)
lambdas = np.zeros(number_samples)

#inintialize sample
gammas[0] = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
deltas[0] =  minimum[1] #0.275#1/(2*np.mean(vari))0.1#
lambdas[0] = deltas[0]/gammas[0]

ATy = np.matmul(A_lin.T, y)

B = (ATA_lin + lambdas[0] * L)

B_inv_L = np.zeros(np.shape(B))
for i in range(len(B)):
    B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)



k = 0
wLam = 2e4
wgam = 1e-5
wdelt = 1e-1
betaG = 1e-4
betaD = 1e-4
alphaG = 1
alphaD = 1
rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambdas[0]
# draw gamma with a gibs step
shape = (SpecNumLayers - 1) / 2 + alphaD + alphaG

startTime = time.time()
for t in range(number_samples-1):


    # # draw new lambda
    lam_p = normal(lambdas[t], wLam)

    while lam_p < 0:
            lam_p = normal(lambdas[t], wLam)

    delta_lam = lam_p - lambdas[t]
    delta_f = f_tayl(delta_lam, B_inv_A_trans_y, ATy[0::, 0], B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4,B_inv_L_5)
    delta_g = g_tayl(delta_lam, B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4, B_inv_L_5)

    log_MH_ratio = ((SpecNumLayers - 1)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

    #accept or rejeict new lam_p
    u = uniform()
    if np.log(u) <= log_MH_ratio:
    #accept
        k = k + 1
        lambdas[t + 1] = lam_p
        #only calc when lambda is updated
        #B = (ATA_lin + lambdas[t+1] * L)
        B = (ATA_lin + lam_p * L)
        B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        # if exitCode != 0:
        #     print(exitCode)
        # CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
        # print(np.allclose(CheckB_inv_ATy, ATy[0::, 0], rtol=relative_tol_ATy))

        B_inv_L = np.zeros(np.shape(B))
        for i in range(len(B)):
            B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
            # if exitCode != 0:
            #    print(exitCode)
        #CheckB_inv_L = np.matmul(B, B_inv_L)
        #print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

        B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
        B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
        B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
        B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)

        rate = f(ATy, y, B_inv_A_trans_y)/2 + betaG + betaD * lam_p#lambdas[t+1]

    else:
        #rejcet
        lambdas[t + 1] = np.copy(lambdas[t])




    gammas[t+1] = np.random.gamma(shape, scale = 1/rate)

    deltas[t+1] = lambdas[t+1] * gammas[t+1]




elapsed = time.time() - startTime
print('acceptance ratio: ' + str(k/number_samples))
np.savetxt('samples.txt', np.vstack((gammas, deltas, lambdas)).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')

#delt_aav, delt_diff, delt_ddiff, delt_itau, delt_itau_diff, delt_itau_aav, delt_acorrn = uWerr(deltas, acorr=None, s_tau=1.5, fast_threshold=5000)

import matlab.engine
eng = matlab.engine.start_matlab()
eng.Run_Autocorr_Ana_MTC(nargout=0)
eng.quit()


AutoCorrData = np.loadtxt("auto_corr_dat.txt", skiprows=3, dtype='float')
#IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'

with open("auto_corr_dat.txt") as fID:
    for n, line in enumerate(fID):
       if n == 1:
            IntAutoDelt, IntAutoGam, IntAutoLam = [float(IAuto) for IAuto in line.split()]
            break



#refine according to autocorrelation time
burnIn = 50

new_lamb = lambdas[burnIn::math.ceil(IntAutoLam)]
#SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
new_gam = gammas[burnIn::math.ceil(IntAutoGam)]
#SetGamma = new_gam[np.random.randint(low = 0,high =len(new_gam),size =1)]
new_delt = deltas[burnIn::math.ceil(IntAutoDelt)]
#SetDelta = new_delt[np.random.randint(low = 0,high =len(new_delt),size =1)]


fig, axs = plt.subplots(3, 1,tight_layout=True)
n_bins = 20

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(new_gam,bins=n_bins)#int(n_bins/math.ceil(IntAutoGam)))
axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
axs[1].hist(new_delt,bins=n_bins)#int(n_bins/math.ceil(IntAutoDelt)))
axs[1].set_title(str(len(new_delt)) + ' effective $\delta$ samples')
axs[2].hist(new_lamb,bins=n_bins)#10)
axs[2].xaxis.set_major_formatter(scientific_formatter)
axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
plt.savefig('HistoResults.png')
#plt.show()



#draw paramter samples
paraSamp = 10
Results = np.zeros((paraSamp,len(theta)))
NormRes = np.zeros(paraSamp)
xTLxRes = np.zeros(paraSamp)

for p in range(paraSamp):
    # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
    SetGamma = new_gam[np.random.randint(low=0, high=len(new_gam), size=1)] #minimum[0]
    SetDelta  = new_delt[np.random.randint(low=0, high=len(new_delt), size=1)] #minimum[1]
    v_1 = np.sqrt(SetGamma) * np.random.multivariate_normal(np.zeros(len(ATA_lin)), ATA_lin)
    v_2 = np.sqrt(SetDelta) * np.random.multivariate_normal(np.zeros(len(L)), L)

    SetB = SetGamma * ATA_lin + SetDelta * L

    SetB_inv = np.zeros(np.shape(SetB))
    for i in range(len(SetB)):
        e = np.zeros(len(SetB))
        e[i] = 1
        SetB_inv[:, i], exitCode = gmres(SetB, e, tol=tol, restart=25)
        if exitCode != 0:
            print(exitCode)

    CheckB_inv = np.matmul(SetB, SetB_inv)
    print(np.linalg.norm(np.eye(len(SetB)) - CheckB_inv) / np.linalg.norm(np.eye(len(SetB))) < tol)

    Results[p, :] = np.matmul(SetB_inv, (SetGamma * ATy[0::, 0] + v_1 + v_2))

    NormRes[p] = np.linalg.norm( np.matmul(A_lin,Results[p, :]) - y[0::,0])
    xTLxRes[p] = np.sqrt(np.matmul(np.matmul(Results[p, :].T, L), Results[p, :]))


scalConst = scalingConst * scalingConstkm
fig3, ax1 = plt.subplots(tight_layout=True)
#plt.plot(theta,layers[0:-1] + d_height/2, color = 'red')
line1 = plt.plot(theta/ (scalConst),layers[0:-1] + d_height/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value', zorder=0)
#line1, = plt.plot(theta* max(np.mean(Results,0))/max(theta),layers[0:-1] + d_height/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
#line2, = plt.plot(np.mean(Results,0),layers[0:-1] + d_height/2,color = 'green', label = 'MC estimate')
# for i in range(paraSamp):
#     line2, = plt.plot(Results[i,:],layers[0:-1] + d_height/2,color = 'green', label = 'MC estimate')
line2 = plt.errorbar(np.mean(Results / (scalConst),0),layers[0:-1] + d_height/2,capsize=4,yerr = np.zeros(len(layers[0:-1])),color = 'red', label = 'MC estimate')#, label = 'MC estimate')
line4 = plt.errorbar(np.mean(Results / (scalConst),0),layers[0:-1] + d_height/2,capsize=4, xerr = np.sqrt(np.var(Results /(scalConst),0))/2 ,color = 'red', label = 'MC estimate')#, label = 'MC estimate')
ax2 = ax1.twiny() # ax1 and ax2 share y-axis
line3 = ax2.plot(y,tang_heights_lin, color = 'gold', label = 'data')
ax2.spines['top'].set_color('gold')
ax2.set_xlabel('Data')
ax2.tick_params(labelcolor="gold")
ax1.set_xlabel(r'Spectral Ozone radiance $\frac{W}{m^2 sr}\frac{1}{\frac{1}{cm}}$')
multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', 'gold'),axis='y')
ax1.legend(['true parameter value', 'MC estimate'])
plt.ylabel('Height in km')
fig3.savefig('FirstRecRes.png')
plt.show()

# fig4, ax1 = plt.subplots()
# #plt.plot(theta,layers[0:-1] + d_height/2, color = 'red')
# #line1 = plt.plot(theta,layers[0:-1] + d_height/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
# #line2 = plt.plot(num_mole,layers[0:-1] + d_height/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
# line3 = plt.plot(VMR_O3,layers[0:-1] + d_height/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
# #line4 = plt.plot(Source,layers[0:-1] + d_height/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
# line5 = plt.plot(num_mole * w_cross * Source,layers[0:-1] + d_height/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
# ax1.set_xlabel('Ozone Source Value')
# ax1.set_ylabel('Height in km')
# plt.show()
# fig3.savefig('TrueProfile.png')

fig5, ax1 = plt.subplots()
line2 = plt.plot(num_mole[1:-1,0],layers[1:-2] + d_height[1:-1]/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')

#plt.plot(theta,layers[0:-1] + d_height/2, color = 'red')np.mean(Results,0)[1:-1]/( num_mole[1:-1,0] * Source[1:-1,0] *)
#line1 = plt.plot(np.mean(Results,0)[1:-1]/(S[ind,0] * f_broad * 1e-4 * scalingConst*Source[1:-1,0]  ),layers[1:-2] + d_height[1:-1]/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
ax1.set_xlabel('Ozone Source Value')
ax1.set_ylabel('Height in km')
plt.show()
#fig3.savefig('TrueProfile.png')


#doesnt work cause too sensitive to noise when close to zero
# XRES = np.copy(RecX)
# XRES[XRES<0] = 0
# RecO3 = XRES[1:-1]*max(theta)/(num_mole[1:-1,0] *max(XRES) * w_cross[1:-1,0] *  Source[1:-1,0])
# RecO3[-3] = 0
# fig4, ax = plt.subplots()
# plt.plot(VMR_O3[1:-1],layers[1:-2] + d_height[1:-1]/2, color = 'red')
# #plt.plot(theta* max(XRES)/max(theta),layers[0:-1] + d_height/2, color = 'red')
# plt.plot(RecO3,layers[1:-2] + d_height[1:-1]/2, color = 'green')
#
# plt.show()






print('MTC Done')


import pytwalk
def MargPostInit(minimum):
    Params = np.zeros(2)
	# Params[0] = np.random.gamma( shape=1, scale=1e4) #gamma
	# Params[1] = np.random.gamma( shape=1, scale=1e4) #delta
    Params[0] = minimum[0] #gamma 8e-5
    Params[1] = minimum[1] #delta

    return Params

def MargPostU(Params):
    n = SpecNumLayers - 1

    Bp= ATA_lin + Params[1]/Params[0] * L

    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A_lin, L,  Params[1]/Params[0])
    F = f(np.matmul(A_lin.T, y), y, B_inv_A_trans_y)

    return -n/2 * np.log(Params[1]) + 0.5 * G + 0.5 * Params[0] * F + 1e-4 * (Params[0] + Params[1])


def MargPostSupp(Params):
	return all(0 < Params)

MargPost = pytwalk.pytwalk( n=2, U=MargPostU, Supp=MargPostSupp)
startTime = time.time()
tWalkSampNum= 100000
MargPost.Run( T=tWalkSampNum, x0=MargPostInit(minimum), xp0=np.array([normal(minimum[0], minimum[0]/4), normal(minimum[1],minimum[1]/4)]) )
elapsedtWalkTime = time.time() - startTime
print('Elapsed Time for t-walk: ' + str(elapsedtWalkTime))
MargPost.Ana()
#MargPost.TS()

#MargPost.Hist( par=0 )
#MargPost.Hist( par=1 )

MargPost.SavetwalkOutput("MargPostDat.txt")

#load data and make histogram
SampParas = np.loadtxt("MargPostDat.txt")


eng = matlab.engine.start_matlab()
eng.Run_Autocorr_PyTWalk(nargout=0)
eng.quit()

AutoCorrDataPyTWalk= np.loadtxt("autoCorrPyTWalk.txt", skiprows=3, dtype='float')
#IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'

with open("autoCorrPyTWalk.txt") as fID:
    for n, line in enumerate(fID):
       if n == 1:
            IntAutoDeltaPyT, IntAutoGamPyT, IntAutoLamPyT = [float(IAuto) for IAuto in line.split()]
            break

lambasPyT = SampParas[:,1]/SampParas[:,0]


fig, axs = plt.subplots(3, 1, tight_layout=True)
n_bins = 20
#burnIn = 50
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(SampParas[burnIn::math.ceil(IntAutoGamPyT),0],bins=n_bins)
axs[0].set_title( str(len(SampParas[burnIn::math.ceil(IntAutoGamPyT),0]))+ ' effective $\gamma$ sample' )
axs[1].hist(SampParas[burnIn::math.ceil(IntAutoDeltaPyT),1],bins=n_bins)
axs[1].set_title(str(len(SampParas[burnIn::math.ceil(IntAutoDeltaPyT),1])) + ' effective $\delta$ samples')
axs[2].hist(lambasPyT[burnIn::math.ceil(IntAutoLamPyT)],bins=n_bins)
axs[2].xaxis.set_major_formatter(scientific_formatter)
axs[2].set_title(str(len(lambasPyT[burnIn::math.ceil(IntAutoLamPyT)])) + ' effective $\lambda =\delta / \gamma samples $')
plt.savefig('PyTWalkHistoResults.png')
#plt.show()


#plot trace
fig, axs = plt.subplots( 2,1, tight_layout=True)
axs[0].plot(range(len(gammas)), neg_log_likehood(gammas,y, Ax).T)
axs[0].set_xlabel('mtc samples')
axs[0].set_ylabel('neg-log-likelihood')
axs[1].plot(range(len(SampParas[:,0])), neg_log_likehood(SampParas[:,0],y, Ax).T)
axs[1].set_xlabel('t-walk samples')
axs[1].set_ylabel('-log $\pi(y |  x ,\gamma)$')
with open('TraceMC.pickle', 'wb') as filID: # should be 'wb' rather than 'w'
    pl.dump(fig, filID)
plt.savefig('TraceMC.png')
#plt.show()

#plot para traces for MTC
fig, axs = plt.subplots( 3,1,  tight_layout=True, figsize=(7, 8))
fig.suptitle(str(number_samples)+' mtc samples in ' + str(math.ceil(elapsed)) + 's')
axs[0].plot(range(len(gammas)), gammas)
axs[0].set_xlabel(r'samples with $\tau_{int}=$ ' + str(math.ceil(IntAutoGam)))
axs[0].set_ylabel('$\gamma$')
axs[1].plot(range(len(deltas)), deltas)
axs[1].set_xlabel(r'samples with $\tau_{int}$= ' + str(math.ceil(IntAutoDelt)))
axs[1].set_ylabel('$\delta$')
axs[2].plot(range(len(lambdas)), lambdas)
axs[2].set_xlabel(r'samples with $\tau_{int}$= ' + str(math.ceil(IntAutoLam)))
axs[2].set_ylabel('$\lambda$')
with open('TraceMTCPara.pickle', 'wb') as filID: # should be 'wb' rather than 'w'
    pl.dump(fig, filID)
plt.savefig('TraceMTCPara.png')
#plt.show()

# #to open figure
# fig_handle = pl.load(open('TraceMTCPara.pickle','rb'))
# fig_handle.show()

#plot para traces for t-walk
fig, axs = plt.subplots( 2,1, tight_layout=True)
fig.suptitle(str(tWalkSampNum)+' t-walk samples in ' + str(math.ceil(elapsedtWalkTime)) + 's')
axs[0].plot(range(len(SampParas[:,0])), SampParas[:,0])
axs[0].set_xlabel(r'samples with $\tau_{int}=$ ' + str(math.ceil(IntAutoGamPyT)))
axs[0].set_ylabel('$\gamma$')
axs[1].plot(range(len(SampParas[:,1])), SampParas[:,1])
axs[1].set_xlabel(r'samples with $\tau_{int}$= ' + str(math.ceil(IntAutoDeltaPyT)))
axs[1].set_ylabel('$\delta$')
with open('TracetWalkPara.pickle', 'wb') as filID: # should be 'wb' rather than 'w'
    pl.dump(fig, filID)
plt.savefig('TracetWalkPara.png')
#plt.show()



print('t-walk Done')

'''make figure for f and g including the best lambdas and taylor series'''

B_MTC = ATA_lin + np.mean(new_lamb) * L
B_MTC_inv_A_trans_y, exitCode = gmres(B_MTC, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

f_MTC = f(ATy, y, B_MTC_inv_A_trans_y)

lamPyT = np.mean(lambasPyT[burnIn::math.ceil(IntAutoLamPyT)])
B_tW = ATA_lin + lamPyT * L
B_tW_inv_A_trans_y, exitCode = gmres(B_tW, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)


f_tW = f(ATy, y, B_tW_inv_A_trans_y)

fig,axs = plt.subplots(1,2, figsize=(14, 5))
axs[0].plot(lam,f_func)
axs[0].scatter(lam0,f_try_func[50], color = 'green', s= 70, zorder=4)
axs[0].annotate('mode $\lambda_0$ of marginal posterior',(lam0+2e4,f_try_func[50]), color = 'green', fontsize = 14.7)
axs[0].scatter(np.mean(lambdas),f_MTC, color = 'red', zorder=5)
axs[0].annotate('MTC $\lambda$ sample mean',(np.mean(lambdas)+1e4,f_MTC), color = 'red')
axs[0].scatter(lamPyT,f_tW, color = 'k', s = 35, zorder=5)
axs[0].annotate('T-Walk $\lambda$ sample mean',(lamPyT+1e5,f_tW+2e6), color = 'k')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylabel('f($\lambda$)')
axs[0].set_xlabel('$\lambda$')
inset_ax = axs[0].inset_axes([0.05,0.41,0.55,0.55])
inset_ax.scatter(lam0,f_try_func[50], color = 'green', s=60, zorder=3)
inset_ax.annotate('$\lambda_0$',(lam0+1e3,f_try_func[50]-3e5), color = 'green', fontsize = 20 )
inset_ax.plot(lam_try,f_func_tay, color = 'red',linewidth = 5, label = '$5^{th}$ Taylor series')
inset_ax.plot(lam_try,f_try_func, label = 'f($\lambda$)')
inset_ax.set_xscale('log')
inset_ax.set_yscale('log')
inset_ax.legend(loc = 'upper left', facecolor = 'none')
inset_ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False)

#axs.set_yscale('log')
axs[1].plot(lam,g_func)
axs[1].scatter(lam0,g_try_func[50], color = 'green', s=70, zorder=4)
axs[1].annotate('mode $\lambda_0$ of marginal posterior',(lam0+3e5,g_try_func[50]), color = 'green')
#axs[1].scatter(np.mean(lambdas),g_func[239], color = 'red', zorder=5)
axs[1].errorbar(np.mean(lambdas),g(A_lin, L, np.mean(lambdas) ), color = 'red', zorder=5, xerr=np.sqrt(np.var(lambdas)), fmt='o')
axs[1].annotate('MTC $\lambda$ sample mean',(np.mean(lambdas)+1e4,g(A_lin, L, np.mean(lambdas) )-45), color = 'red')
axs[1].scatter(lamPyT,g(A_lin, L, lamPyT) , color = 'k', s=35, zorder=5)
axs[1].annotate('T-Walk $\lambda$ sample mean',(lamPyT+1e6,g(A_lin, L, lamPyT) +50), color = 'k')
axs[1].set_xscale('log')
axs[1].set_xlabel('$\lambda$')
axs[1].set_ylabel('g($\lambda$)')
inset_ax = axs[1].inset_axes([0.05,0.41,0.55,0.55])
inset_ax.plot(lam_try,g_func_tay, color = 'red',linewidth = 5,label = '$5^{th}$ Taylor series')
inset_ax.plot(lam_try,g_try_func, label = 'g($\lambda$)')
inset_ax.scatter(lam0,g_try_func[50], color = 'green', s=60, zorder=3)
inset_ax.annotate('$\lambda_0$',(lam0+1e3,g_try_func[50]-2), color = 'green', fontsize = 20 )
inset_ax.set_xscale('log')
inset_ax.set_yscale('log')
inset_ax.legend(loc = 'upper left', facecolor = 'none')
inset_ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False)
with open('f_and_g.pickle', 'wb') as filID: # should be 'wb' rather than 'w'
    pl.dump(fig, filID)
plt.savefig('f_and_g.png')
#plt.show()



print('bla')


'''L-curve refularoization
'''
tol = 1e-4
lamLCurve = np.logspace(-20,20,500)
#lamLCurve = np.linspace(1e-1,1e4,300)

NormLCurve = np.zeros(len(lamLCurve))
xTLxCurve = np.zeros(len(lamLCurve))
for i in range(len(lamLCurve)):
    B = (ATA_lin + lamLCurve[i] * L)

    x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    NormLCurve[i] = np.linalg.norm( np.matmul(A_lin,x) - y[0::,0])
    #NormLCurve[i] =np.linalg.norm( np.matmul(A_lin,x))
    #NormLCurve[i] = np.sqrt(np.sum((np.matmul(A_lin, x) - y)**2))
    xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))
    #xTLxCurve[i] = np.linalg.norm(np.matmul(L,x))

A_linu, A_lins, A_linvh = csvd(A_lin)
#reg_c = l_cuve(A_linu, A_lins, y[0::,0], plotit=True)
#reg_c = l_corner(NormLCurve,xTLxCurve,lamLCurve,A_linu,A_lins,y[0::,0])
#B = (ATA_lin + reg_c * L)
B = (ATA_lin + minimum[1]/minimum[0] * L)

x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

lamLCurveZoom = np.logspace(4,10,500)
NormLCurveZoom = np.zeros(len(lamLCurve))
xTLxCurveZoom = np.zeros(len(lamLCurve))
for i in range(len(lamLCurveZoom)):
    B = (ATA_lin + lamLCurveZoom[i] * L)

    x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    NormLCurveZoom[i] = np.linalg.norm( np.matmul(A_lin,x) - y[0::,0])
    xTLxCurveZoom[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))

A_linu, A_lins, A_linvh = csvd(A_lin)
B = (ATA_lin + minimum[1]/minimum[0] * L)
x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

fig, axs = plt.subplots( 1,1, tight_layout=True)
axs.scatter(NormLCurve,xTLxCurve, zorder = 0, color = 'black')
axs.scatter(np.linalg.norm(np.matmul(A_lin, x) - y[0::, 0]),np.sqrt(np.matmul(np.matmul(x.T, L), x)), color = 'black')
#axs.annotate('$\lambda_0$ = ' + str(math.ceil(minimum[1]/minimum[0])), (np.linalg.norm(np.matmul(A_lin, x) - y[0::, 0]),np.sqrt(np.matmul(np.matmul(x.T, L), x))))
#axs.annotate('$\lambda$ = 1e' + str(orderOfMagnitude(lamLCurve[0])), (NormLCurve[0],xTLxCurve[0]))
#axs.annotate('$\lambda$ = 1e' + str(orderOfMagnitude(lamLCurve[-1])), (NormLCurve[-1],xTLxCurve[-1]))
axs.scatter(NormRes, xTLxRes, color = 'red')#, marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')

x1, x2, y1, y2 = NormLCurveZoom[0], NormLCurveZoom[-1], xTLxCurveZoom[0], xTLxCurveZoom[-1] # specify the limits
#axins = mplT.axes_grid1.inset_locator.inset_axes( parent_axes = axs,  bbox_transform=axs.transAxes, bbox_to_anchor =(0.05,0.05,0.75,0.75) , width = '100%' , height = '100%')#,  loc= 'lower left')
axins = axs.inset_axes([0.01,0.05,0.75,0.75])
axins.scatter(NormRes, xTLxRes, color = 'red')
axins.scatter(NormLCurveZoom,xTLxCurveZoom, color = 'black')
#axins.scatter(NormRes, xTLxRes)
#,'o', color='black')
axins.set_xscale('log')
axins.set_yscale('log')
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y2, 1.1*max(xTLxRes) ) # apply the y-limits (negative gradient)
axins.set_xticklabels([])
axins.set_yticklabels([])
axs.indicate_inset_zoom(axins, edgecolor="black")

axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel(r'$\sqrt{x^T L x}$')
axs.set_xlabel(r'$|| Ax - y ||$')
axs.set_title('L-curve for m=' + str(SpecNumMeas))
plt.savefig('LCurve.png')
plt.show()

print('bla')

