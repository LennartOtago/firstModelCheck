"""           Appendix
    Storage for code to make main.py look nicer
    I put in most the analysis for the forward model """

import numpy as np

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


""" analayse forward map without any real data values"""


#FakeObsHeight = MaxH + 5
''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first

#height_diff = height_values[1:-1] - height_values[0:-2]
# LayersCore = height_values #np.linspace(MinH, MaxH, SpecNumLayers)
# layers = np.zeros(SpecNumLayers + 2)
# layers[1:-1] =  LayersCore
# layers[0]= MinH-3
# layers[-1] = MaxH+5
# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile

#add zero layers
#SpecNumLayers = SpecNumLayers + 2
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




''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
#filename = 'tropical.O3.xml'

#VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
#d_height = layers[1::] - layers[0:-1]
# N_A = constants.Avogadro # in mol^-1
# k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
# R_gas = N_A * k_b_cgs # in ..cm^3


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




""" finaly calc f and g with a linear solver adn certain lambdas
 using the gmres"""

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



''' check taylor series in f(lambda)
around lam0 delta_lam = '''

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

# if exitCode != 0:
#     print(exitCode)
# CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
# print(np.allclose(CheckB_inv_ATy, ATy[0::, 0], rtol=relative_tol_ATy))

# B_inv = np.zeros(np.shape(B))
# for i in range(len(B)):
#     B_inv[:, i], exitCode = gmres(B, e, tol=tol, restart=25)
#     if exitCode != 0:
#         print('B_inv ' + str(exitCode))
#
# B_inv_L = np.matmul(B_inv, L)

# CheckB_inv_L = np.matmul(B, B_inv_L)
# print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)



"""plot results and gorund truth"""


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

# fig5, ax1 = plt.subplots()
# line2 = plt.plot(num_mole[:,0],height_values, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
#
# #plt.plot(theta,layers[0:-1] + d_height/2, color = 'red')np.mean(Results,0)[1:-1]/( num_mole[1:-1,0] * Source[1:-1,0] *)
# #line1 = plt.plot(np.mean(Results,0)[1:-1]/(S[ind,0] * f_broad * 1e-4 * scalingConst*Source[1:-1,0]  ),layers[1:-2] + d_height[1:-1]/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
# ax1.set_xlabel('Ozone Source Value')
# ax1.set_ylabel('Height in km')
# plt.show()
#fig3.savefig('TrueProfile.png')

# fig5, ax1 = plt.subplots()
# line2 = plt.errorbar(np.mean(Results,0 ).reshape((SpecNumLayers+1,1)) / (num_mole * S[ind,0] * f_broad * 1e-4 * Source * scalingConst),height_values,capsize=4,yerr = np.zeros(len(height_values)),color = 'red', label = 'MC estimate')
# #plt.plot(theta,layers[0:-1] + d_height/2, color = 'red')np.mean(Results,0)[1:-1]/( num_mole[1:-1,0] * Source[1:-1,0] *)
# #line1 = plt.plot(np.mean(Results,0)[1:-1]/(S[ind,0] * f_broad * 1e-4 * scalingConst*Source[1:-1,0]  ),layers[1:-2] + d_height[1:-1]/2, color = [0,0.5,0.5], linewidth = 5, label = 'true parameter value')
# ax1.set_xlabel('Ozone Source Value')
# ax1.set_ylabel('Height in km')
# plt.show()
#cause too sensitive to noise when close to zero
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


'''mtc sampling using taylor expansions 
is slow for small matrices '''

# number_samples = 10000
# gammas = np.zeros(number_samples)
# deltas = np.zeros(number_samples)
# lambdas = np.zeros(number_samples)
#
# #inintialize sample
# gammas[0] = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
# deltas[0] =  minimum[1] #0.275#1/(2*np.mean(vari))0.1#
# lambdas[0] = deltas[0]/gammas[0]
#
# ATy = np.matmul(A.T, y)
#
# B = (ATA + lambdas[0] * L)
#
# B_inv_L = np.zeros(np.shape(B))
# for i in range(len(B)):
#     B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
# B_inv = np.zeros(np.shape(B))
# for i in range(len(B)):
#     e = np.zeros(len(B))
#     e[i] = 1
#     B_inv[:, i], exitCode = gmres(B, e, tol=tol, restart=25)
#     if exitCode != 0:
#         print('B_inv ' + str(exitCode))
#
# B_inv_L = np.matmul(B_inv,L)
#
# B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
# if exitCode != 0:
#     print(exitCode)
#
# B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
# B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
# B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
# B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)
#
#
# Bu, Bs, Bvh = np.linalg.svd(B)
# cond_B =  np.max(Bs)/np.min(Bs)
# print("normal: " + str(orderOfMagnitude(cond_B)))


# k = 0
# wLam = 20
# #wgam = 1e-5
# #wdelt = 1e-1
# betaG = 1e-4
# betaD = 1e-4
# alphaG = 1
# alphaD = 1
# rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambdas[0]
# # draw gamma with a gibs step
# shape = SpecNumLayers/2 + alphaD + alphaG
#
# startTime = time.time()
# for t in range(number_samples-1):
#     #print(t)
#
#     # # draw new lambda
#     lam_p = normal(lambdas[t], wLam)
#
#     while lam_p < 0:
#             lam_p = normal(lambdas[t], wLam)
#
#     delta_lam = lam_p - lambdas[t]
#     delta_f = f_tayl(delta_lam, B_inv_A_trans_y, ATy[0::, 0], B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4,B_inv_L_5)
#     delta_g = g_tayl(delta_lam, B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4, B_inv_L_5)
#
#     log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam
#
#     #accept or rejeict new lam_p
#     u = uniform()
#     if np.log(u) <= log_MH_ratio:
#     #accept
#         k = k + 1
#         lambdas[t + 1] = lam_p
#         #only calc when lambda is updated
#         #B = (ATA_lin + lambdas[t+1] * L)
#         B = (ATA + lam_p * L)
#         B_inv = np.zeros(np.shape(B))
#         for i in range(len(B)):
#             e = np.zeros(len(B))
#             e[i] = 1
#             B_inv[:, i], exitCode = gmres(B, e, tol=tol, restart=25)
#             if exitCode != 0:
#                 print('B_inv ' + str(exitCode))
#
#         B_inv_L = np.matmul(B_inv, L)
#         B_inv_A_trans_y = np.matmul(B_inv, ATy[0::, 0])
#
#         B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
#         B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
#         B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
#         B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)
#
#         rate = f(ATy, y, B_inv_A_trans_y)/2 + betaG + betaD * lam_p#lambdas[t+1]
#
#     else:
#         #rejcet
#         lambdas[t + 1] = np.copy(lambdas[t])
#
#
#
#
#     gammas[t+1] = np.random.gamma(shape, scale = 1/rate)
#
#     deltas[t+1] = lambdas[t+1] * gammas[t+1]
#

