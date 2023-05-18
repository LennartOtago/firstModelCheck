import numpy as np
import scipy as sc
from functions import *
from scipy import constants
from scipy.sparse.linalg import gmres
import testReal
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import uniform
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from numpy import inf
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as colors

""" analayse forward map without any real data values"""

min_h = 5
max_h = 95
R = 6371 # earth radiusin km
obs_height = 300 # in km
#fing minimum and max angle in radians
max_ang = np.arcsin( (max_h + R) / (R + obs_height) )
min_ang = np.arcsin( (min_h + R) / (R + obs_height) )
coeff = 0.05

"""
    make a plot with conditionnumber on y axis and layers on x axis, 
    with stable measurment numbers
"""
# #cond is max(s_i)/min
# min_m = 15
# max_m = 180
# max_l = 90
# min_l = 15
# num_meas = np.linspace(min_m,max_m,max_m-min_m+1)
# num_lay = np.linspace(min_l, max_l, (max_l - min_l) + 1)
# cond_A = np.zeros((len(num_meas),len(num_lay)))
# cond_ATA = np.zeros((len(num_meas),len(num_lay)))
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
#         cond_A[j,i]= np.max(As)/np.min(As)
#         cond_ATA[j,i] = np.max(ATAs)/np.min(ATAs)
#
#
# #cond_A[cond_A == inf] = np.nan
# vmin = np.min(cond_A[cond_A != inf])
# vmax = np.max(cond_A[cond_A != inf])
# # Creating figure
# fig, axs = plt.subplots(1,2,figsize=(12,6))
# #ax1 = plt.subplot(1, 2, 1)
# pl1 = axs[0].imshow(cond_A, cmap='jet',norm=colors.LogNorm( vmin=vmin, vmax=vmax), extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]], aspect='auto')
# axs[0].set_ylabel('Number of Measurement')
# axs[0].set_xlabel('Number of Layers in Model')
# #ax[1].imshow(cond_ATA, cmap='hot', interpolation='nearest',norm=LogNorm())
# # ax2 = plt.subplot(1, 2, 2)
# pl2 = axs[1].imshow(cond_ATA, cmap='jet',norm=colors.LogNorm( vmin=vmin, vmax=vmax), extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]] ,aspect='auto')
# axs[1].set_ylabel('Number of Measurement')
# axs[1].set_xlabel('Number of Layers in Model')
# #ax2.get_ylim([num_meas[0],num_meas[-1]])
# fig.colorbar(pl1,ax=axs,location='bottom')#, orientation='horizontal')
# axs[1].set_title('Condition Number of $A^T$ A')
# axs[0].set_title('Condition Number of A')
# fig.suptitle('Conditionnumber for different measurement and model setups', fontsize=16)
# plt.savefig('cond_A_exp.png')
# plt.show()
#
#
#
# #make a plot with conditionnumber on y axis and layers on x axis, with stable measurment numbers
# #cond is max(s_i)/min
# min_m = 15
# max_m = 180
# max_l = 90
# min_l = 15
# num_meas = np.linspace(min_m,max_m,max_m-min_m+1)
# num_lay = np.linspace(min_l, max_l, (max_l - min_l) + 1)
# cond_A = np.zeros((len(num_meas),len(num_lay)))
# cond_ATA = np.zeros((len(num_meas),len(num_lay)))
# for j in range(len(num_meas)):
#     meas_ang = np.linspace(min_ang,max_ang,int(num_meas[j])+1)
#     for i in range(len(num_lay)):
#         layers = np.linspace(min_h, max_h, int(num_lay[i])+1)
#         A, tang_heights = gen_forward_map(meas_ang[0:-1], layers, obs_height, R)
#         ATA = np.matmul(A.T, A)
#         ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
#         Au, As, Avh = np.linalg.svd(A)
#         cond_A[j,i]=  np.max(As)/np.min(As)
#         cond_ATA[j,i] =  np.max(ATAs)/np.min(ATAs)
#
#
#
# #cond_A[cond_A == inf] = np.nan
# vmin = np.min(cond_A[cond_A != inf])
# vmax = np.max(cond_A[cond_A != inf])
# # Creating figure
# fig, axs = plt.subplots(1,2,figsize=(12,6))
# #ax1 = plt.subplot(1, 2, 1)
# pl1 = axs[0].imshow(cond_A, cmap='jet',norm=colors.LogNorm( vmin=vmin, vmax=vmax), extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]], aspect='auto')
# axs[0].set_ylabel('Number of Measurement')
# axs[0].set_xlabel('Number of Layers in Model')
# #ax[1].imshow(cond_ATA, cmap='hot', interpolation='nearest',norm=LogNorm())
# # ax2 = plt.subplot(1, 2, 2)
# pl2 = axs[1].imshow(cond_ATA, cmap='jet',norm=colors.LogNorm( vmin=vmin, vmax=vmax), extent=[num_lay[0],num_lay[-1],num_meas[0],num_meas[-1]] ,aspect='auto')
# axs[1].set_ylabel('Number of Measurement')
# axs[1].set_xlabel('Number of Layers in Model')
# #ax2.get_ylim([num_meas[0],num_meas[-1]])
# fig.colorbar(pl1,ax=axs,location='bottom')#, orientation='horizontal')
# axs[1].set_title('Condition Number of $A^T$ A')
# axs[0].set_title('Condition Number of A')
# fig.suptitle('Conditionnumber for different measurement and model setups', fontsize=16)
# plt.savefig('cond_A_lin.png')
# plt.show()





#analyse singlar vectors for A.T A for specific num of layers


#specifiy layers in km

# num_layers = 31 #46
# layers = np.linspace(min_h, max_h,num_layers+1)
# gradient = np.vstack(
#     (uniform(0, 1, num_layers - 1), uniform(0, 1, num_layers - 1), uniform(0, 1, num_layers - 1))).T

# #specify num of measurements
# min_m = 20
# max_m = 120
# num_meas_spec = np.linspace(min_m,max_m,int((max_m-min_m)/5)+1)
# for j in range(len(num_meas_spec)):
#     meas_ang = min_ang + (
#                 (max_ang - min_ang) * np.exp(coeff * (np.linspace(0, int(num_meas_spec[j]) - 1, int(num_meas_spec[j])) - (int(num_meas_spec[j]) - 1))))
#
#     A, tang_heights = gen_forward_map(meas_ang[0:-1],layers,obs_height,R)
#     ATA = np.matmul(A.T, A)
#
#     # Indices to step through colormap.
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
#     for i in range(num_layers - 1):
#         ax1.plot(ATAu[:, i], color=gradient[i], linewidth=2,
#                  path_effects=[path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()])
#         text = ax1.text(i, ATAu[i, i], f'{i}', color=gradient[i], fontsize=20)
#         text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
#                                path_effects.Normal()])
#         ax2.scatter(i, ATAs[i], color=gradient[i], s=50)  # gradient[:,i]
#         ax2.text(i, ATAs[i], f'{i}')
#         ax2.set_yscale('log')
#
#     plt.show()
#

#find best configuration of layers and num_meas
#so that cond(A) is not inf
num_meas = 100
num_layers = 45
layers = np.linspace(min_h, max_h,num_layers+1)
meas_ang = min_ang + ((max_ang - min_ang) * np.exp(coeff * (np.linspace(0, int(num_meas) - 1, int(num_meas)+1) - (int(num_meas) - 1))))
#meas_ang = np.linspace(min_ang, max_ang, num_meas+ 1)

A, tang_heights = gen_forward_map(meas_ang[0:-1],layers,obs_height,R)
Au, As, Avh = np.linalg.svd(A)
ATA = np.matmul(A.T,A)
#condition number for A
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))



#to test that we have the same dr distances
tot_r = np.zeros(num_meas)
#calculate total length
for j in range(0,num_meas):
    tot_r[j] = 2*np.sqrt( (layers[-1] + R)**2 - (tang_heights[j] + R )**2 )
print('Distance trhough layers check: ' + str(np.allclose( sum(A.T), tot_r)))

#dont consider last h_val as we have layers from lowest h_val up to second highest
ATAu, ATAs, ATAvh = np.linalg.svd(ATA)#plot_svd(ATA, layers[0:-1])
print("ATA: " + str(orderOfMagnitude(np.max(np.sqrt(ATAs))/np.min(np.sqrt(ATAs)))))
# #plot sing vec and sing vals
fig, axs = plt.subplots(1,2,figsize=(12,6))
axs[0].scatter(range(num_meas),tang_heights)
#axs.title('Measurement Setup with Conditonnumber for Forward map '+  str(cond_A))
axs[0].set_xlabel('Number of Measurements')
axs[0].set_ylabel('Height in km')
axs[1].scatter(range(0,num_layers),As)
fig.suptitle('Singular values of A with Conditionnumber ' + str(np.around(cond_A)))
axs[1].set_yscale('log')
axs[1].set_ylabel('Value')
axs[1].set_xlabel('Index')
plt.savefig('ExpScalExp.png')
plt.show()


#find best configuration of layers and num_meas
#so that cond(A) is not inf
num_meas = 100
num_layers = 45
layers = np.linspace(min_h, max_h,num_layers+1)
#meas_ang = min_ang + ((max_ang - min_ang) * np.exp(coeff * (np.linspace(0, int(num_meas) - 1, int(num_meas)+1) - (int(num_meas) - 1))))
meas_ang = np.linspace(min_ang, max_ang, num_meas+ 1)

A, tang_heights = gen_forward_map(meas_ang[0:-1],layers,obs_height,R)
Au, As, Avh = np.linalg.svd(A)
ATA = np.matmul(A.T,A)
#condition number for A
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))



#to test that we have the same dr distances
tot_r = np.zeros(num_meas)
#calculate total length
for j in range(0,num_meas):
    tot_r[j] = 2*np.sqrt( (layers[-1] + R)**2 - (tang_heights[j] + R )**2 )
print('Distance trhough layers check: ' + str(np.allclose( sum(A.T), tot_r)))

#dont consider last h_val as we have layers from lowest h_val up to second highest
ATAu, ATAs, ATAvh = np.linalg.svd(ATA)#plot_svd(ATA, layers[0:-1])
print("ATA: " + str(orderOfMagnitude(np.max(np.sqrt(ATAs))/np.min(np.sqrt(ATAs)))))
# #plot sing vec and sing vals
fig, axs = plt.subplots(1,2,figsize=(12,6))
axs[0].scatter(range(num_meas),tang_heights)
#axs.title('Measurement Setup with Conditonnumber for Forward map '+  str(cond_A))
axs[0].set_xlabel('Number of Measurements')
axs[0].set_ylabel('Height in km')
axs[1].scatter(range(0,num_layers),As)
fig.suptitle('Singular values of A with Conditionnumber ' + str(np.around(cond_A)))
axs[1].set_yscale('log')
axs[1].set_ylabel('Value')
axs[1].set_xlabel('Index')
plt.savefig('ExpScalLin.png')
plt.show()


# ATA_inv = np.zeros(np.shape(ATA))
# for i in range(len(ATA)):
#     e = np.zeros(len(ATA))
#     e[i] = 1
#     ATA_inv[:,i] , exitCode = gmres(ATA, e,tol = 1e-3, restart= 25)
#     print(exitCode)
# CheckATA = np.matmul(ATA,ATA_inv)
# print(np.allclose(CheckATA,np.eye(len(ATA)),atol = 1e-3))

#graph Laplacian
neigbours = np.zeros((len(layers)-1,2))
neigbours[0] = np.nan, 1
neigbours[-1] = len(layers)-3, np.nan
for i in range(1,len(layers)-2):
    neigbours[i] = i-1, i+1
L = generate_L(neigbours)


lam = 1e3
#number is approx 130 so 10^2
#B is symmetric and positive semidefinite and normal
B = (ATA -lam* L) #* 1e-14eta/gamma
Bu, Bs, Bvh = np.linalg.svd(B)
fig, axs = plt.subplots(1,1,figsize=(12,6))
axs.set_yscale('log')
plt.scatter(range(0,num_layers),Bs)
plt.savefig('SingVal_B_exp.png')
plt.show()

#condition number for B
cond_B =  np.max(Bs)/np.min(Bs)

print("normal: " + str(orderOfMagnitude(cond_B)))



# B_inv = np.zeros(np.shape(ATA))
# for i in range(len(ATA)):
#     e = np.zeros(len(ATA))
#     e[i] = 1
#     B_inv[:,i] , exitCode = gmres(B, e,tol = 1e-3, restart= 25)
#     print(exitCode)
#
# CheckB_inv = np.matmul(B,B_inv)
# print(np.allclose(CheckB_inv,np.eye(len(ATA)),atol = 1e-3))


#now compute the action of B^-1 L


# B_inv_L = np.zeros(np.shape(ATA))
# for i in range(len(ATA)):
#     B_inv_L[:,i] , exitCode = gmres(B, L[:,i],tol = 1e-5, restart= 25)
#     print(exitCode)
#
# CheckB_inv_L = np.matmul(B,B_inv_L)
# print(np.allclose(CheckB_inv_L,L,atol = 1e-3))

''' taylor expansion for g
'''
#calc trace of B_inv_L with monte carlo estiamtion
#do 4 times as colin
# num_z = 4
# trace_Bs = np.zeros(num_z)
# for k in range(num_z):
#     z = np.random.randint(2, size= len(B))
#     z[z==0] = -1
#     trace_Bs[k] = np.matmul(z.T, np.matmul(B_inv_L, z))
#
#
# trace_B_inv_l = np.mean(trace_Bs)



#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
#/Users/lennart/PycharmProjects/firstModelCheck
#obs_height= 300 #in km
#filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml' #/home/lennartgolks/Python /Users/lennart/PycharmProjects/firstModelCheck/tropical.O3.xml
filename = 'tropical.O3.xml'

VMR_O3, height_values, pressure_values = testReal.get_data(filename, obs_height * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
d_height = (height_values[1::] - height_values[0:-1] )
#height_values = np.linspace(18,93,45)
pressure_values = pressure_values * 1e-1 # in cgs
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

# plt.plot(pressure_values, height_values)
# #plt.plot(VMR_O3, height_values)
# plt.show()

temp_values = get_temp_values(height_values)
#x = VMR_O3 * N_A * pressure_values /(R_gas * temp_values)#* 1e-13

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




#constants in si
h = constants.h * 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1

C2 = h * c_cgs /k_b_cgs
C1 = h * c_cgs**2


mol_M = 48 #g/mol for Ozone
ind = 293

#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
f_0 = c_cgs*v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")

#plancks function
Source = np.array(2 * C1/(lamba**5 * (np.exp(C2/(lamba*temp_values))-1))).reshape((47,1))
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
w_cross = np.ones((len(height_values),1)) * S[ind,0] * 1e17
w_cross[0], w_cross[-1] = 0, 0
#S[ind,0] * C4[i][0]


''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''
num_mole = (pressure_values / (constants.Boltzmann * 1e7  * temp_values))
theta = (num_mole * w_cross * VMR_O3 * Source)
Ax = np.matmul(A, theta[1:-1])
#convolve measurements and add noise
y = add_noise(Ax, 0.01, np.max(Ax))





""" finaly calc f with a linear solver... gmres"""
#B^-1 A^T y
B_inv_A_trans_y = np.zeros(np.shape(ATA))
A_trans_y = np.matmul(A.T,y)
B_inv_A_trans_y , exitCode = gmres(B, A_trans_y[0::,0],tol = 1e-6, restart= 25)
print(exitCode)

CheckB_A_trans_y = np.matmul(B,B_inv_A_trans_y)
print(np.allclose(CheckB_A_trans_y.T,A_trans_y[0::,0],atol = 1e-5))

#already did (B^-1 L)^r look g
# all together (A^T y )^T (B^-1 L)^r B^-1 (A^T y)
# f_1 =  np.matmul(A_trans_y.T, np.matmul(B_inv_L ,B_inv_A_trans_y))
# f_2 =np.matmul( np.matmul(A_trans_y.T,B_inv_L ), np.matmul(B_inv_L ,B_inv_A_trans_y) )

#f = f_1 - f_2

#%%
lam= np.logspace(-4,14,1000)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))
for j in range(len(lam)):
    f_func[j] = f(A, y, L, lam[j])
    g_func[j] = g(A, L, lam[j])

fig,axs = plt.subplots(1,2)
axs[1].plot(lam,g_func)
axs[0].plot(lam,f_func)

#axs.set_yscale('log')
axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[0].set_yscale('log')

axs[0].set_xlabel('$\lambda$')
axs[1].set_xlabel('$\lambda$')

axs[0].set_ylabel('f($\lambda$)')
axs[1].set_ylabel('g($\lambda$)')

plt.show()


apha = 44/2 + 1
#%%

#mean = apha/beta
#Bayesian framework



vari = np.zeros((len(VMR_O3),1))


def func(x, x0, sigma):
    return  - (x - x0) ** 2 / (2 * sigma ** 2)

#x = x * 1e17
for j in range(1,len(x)-1):
    rra = (x[j] + x[j]/2)
    dist = np.linspace(-rra, rra,200)
    y = np.zeros((200,1))

    #plot normal distribution of one data point X_i to find hyperparamter
    for i in range(1,200):
        y[i] = (-0.5 * ((dist[i] - x[j-1])**2 + (dist[i] - x[j+1])**2 ))

    y = y[1:-1]
    dist = dist[1:-1]
    # Executing curve_fit on noisy data
    mean = (sum(dist * y) / sum(y))
    sigma = np.sqrt( sum(y * (dist -  x[j])**2 / sum(y)))
    popt, pcov = curve_fit(func, dist.flatten(), y.flatten(), p0=[  x[j], sigma] )

    vari[j] = popt[1]**2
    # print(j)
    # ym = func(dist, popt[0], popt[1])
    # plt.figure()
    # plt.plot(dist, ym, 'k', linewidth=2)
    # plt.plot(dist, y)
    # plt.show()




    #plt.show()
#gmres(C - lamba * L, np.identity(num_meas), tol = 1e-3,restart = 25, maxiter = 1e4)

#variance is
# ym = func(dist, popt[0], popt[1])
# plt.figure()
# plt.plot(dist, ym, 'k', linewidth=2)
# plt.plot(dist, y)reboot
# plt.show()




#first guesses
gamma = 0.01 * max(Ax)
eta = 1/ (2 * np.mean(vari[2:-3]))







#hyperarameters
number_samples = 100
gammas = np.zeros(number_samples)
etas = np.zeros(number_samples)

#inintialize sample

gammas[0] = 9.5e-12

etas[0] = 9.5e-12

print('bla')


#for assignment
# fig10,(ax1,ax2) = plt.subplots(2)
# t = np.linspace(0,10,11)
# t_fin = np.linspace(0,10,100)
# ax1.set_title('np.sin(2 * pi * v_0 = 0.1 * t)')
# ax1.plot(t,np.sin(2 * np.pi * 0.1 * t))
# ax1.plot(t_fin,np.sin(2 * np.pi * 0.1 * t_fin))
# ax2.set_title('np.sin(2 * pi * (v_0 = 0.1 + 1/ T = 1) * t)')
# ax2.plot(t,np.sin(2 * np.pi * (0.1 + 1) * t))
# ax2.plot(t_fin,np.sin(2 * np.pi * (0.1 + 1) * t_fin))
# plt.show()




