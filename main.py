import numpy as np
import scipy as sc
from scipy import constants
import plotly.graph_objects as go

import testTheo
import testReal
import matplotlib.pyplot as plt
import glob
import pandas as pd

# h_tang = 15000
# ind_limb = 100
# h_inf = 50000

#testTheo.make_figure(h_tang, ind_limb, h_inf)



# filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml'
# VMR_O3, height_values, pressure_values,  temp_values= test1.get_data(filename)
#
# #VMR_O3 = VMR_O3 #* 1e6 #get rid of ppm convert to molecule/m^3
#
# filedir = glob.glob('/home/lennartgolks/Python/firstModelCheck/HITRAN_o3_data/*.xsc')
# frequency = 117.389 #in GHz 100
#
# absorption_coeff , max_absorption, max_frequency, temp = test1.get_absorption(frequency, filedir)
# #in cm^2/molecule
#
#
#
#
# test1.make_figure(height_values, ind_limb, h_inf, VMR_O3, pressure_values,
#                   temp_values, absorption_coeff)
#
#



# ##################################################
#
# ##check absoprtion coeff in different heights and different freqencies
#
# filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml'
# VMR_O3, height_values, pressure_values, temp_values = testReal.get_data(filename)
# #[ppm], [m], [Pa] = [kg / (m s^2) ]
#
# filedir = glob.glob('/home/lennartgolks/Python/firstModelCheck/HITRAN_o3_data/*.xsc')
#
# absorption_coeff, temp, frequencies = testReal.get_absorption( filedir)
# # [cm^2/molecule]
#
# testReal.make_absorp_fig(frequencies, VMR_O3, absorption_coeff, temp_values, height_values, pressure_values)
#
# ########################################################
#
# files = '/home/lennartgolks/Python/firstModelCheck/634f1dc4.par'
#
# my_data = pd.read_csv(files, header=None)
# data_set = my_data.values
#
# size = data_set.shape
# wvnmbr = np.zeros((size[0],1))
# S = np.zeros((size[0],1))
# A = np.zeros((size[0],1))
# g_air = np.zeros((size[0],1))
# g_self = np.zeros((size[0],1))
# E = np.zeros((size[0],1))
# n_air = np.zeros((size[0],1))
#
# print(data_set[0])
# #current = list(data_set[0].split(" "))
#
# for i, lines in enumerate(data_set):
#     wvnmbr[i] = float(lines[0][5:15])
#     S[i] = float(lines[0][16:25])
#     A[i] = float(lines[0][26:35])
#     g_air[i] = float(lines[0][35:40])
#     g_self[i] = float(lines[0][40:45])
#     E[i] = float(lines[0][46:55])
#     n_air[i] = float(lines[0][55:59])








#get absoption coefficent for selected frequency


#get data
filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml'
VMR_O3, height_values, pressure_values,  temp_values= testReal.get_data(filename)

filedir = glob.glob('/home/lennartgolks/Python/firstModelCheck/HITRAN_o3_data/*.xsc')

absorption_coeff, temp, frequencies = testReal.get_absorption( filedir)
#in cm^2/molecule

freq_ind = 50 # ind 50 868 tHz
#print(frequencies[:,freq_ind])
#print(frequencies[:,0])
#print(frequencies[:,-1])
frequency = int(np.mean(frequencies[:,freq_ind])) #868 tHz
abs_coeff = int(np.mean(absorption_coeff[:,freq_ind])*1e24)*1e-28 #in m^2/molecule

# # extend to observer height at 600 km
# ext_numb = 1000
# height_values = np.append(height_values, np.linspace(94000, 600000, num=ext_numb))
# VMR_O3 = np.append(VMR_O3, [0.01] * ext_numb)
# temp_values = np.append(temp_values, [473.15] * ext_numb )
# pressure_values = np.append(pressure_values, [1000] * ext_numb)

#calculate weighted absorption crosssection

w_cross = VMR_O3 * abs_coeff

#source funciton
h = constants.h
c = constants.c
T = temp_values[0:-1]
k_b = constants.Stefan_Boltzmann

S = 2 * h * frequency**3 / c**2 * 1/(np.exp(h * frequency/(k_b *T) ) - 1)

#transmission starting from ground layer
# length is one shorter than heitght values
R = 6371 #earth radius
tang_ind = 0
d_height = height_values[1::] - height_values[0:-1]
trans_per_h = w_cross[0:-1] * (height_values[0:-1] + R)/np.sqrt((height_values[0:-1]+ R)**2 + (height_values[tang_ind]+ R)**2 )* d_height




#pre integration
#kernel = S * d_height * w_cross[0:-1] *pressure_values[0:-1] / (k_b *T * np.sqrt((height_values[0:-1]+ R)**2 + (height_values[tang_ind]+ R)**2 ) )

kernel = np.log(S) + np.log(d_height) + np.log(w_cross[0:-1]) + np.log(pressure_values[0:-1]) - np.log(k_b *T * np.sqrt((height_values[0:-1]+ R)**2 + (height_values[tang_ind]+ R)**2 ) ) +  np.log(height_values[0:-1] + R)

trans_pre = [ sum(trans_per_h[i::]) for i in range(0,len(trans_per_h)) ]

trans_after  = [ sum(trans_per_h[0:i]) for i in range(0,len(trans_per_h)) ]

tot_trans_pre = trans_pre[0]

measurements_before = np.array([ sum(kernel[tang_ind::] - trans_pre[tang_ind::] ) for tang_ind in range(0, len(height_values[0:-1]))])

measurements_after = np.array([ sum(kernel[tang_ind::] - trans_after[tang_ind::] - tot_trans_pre) for tang_ind in range(0, len(height_values[0:-1]))])


measurements = measurements_after + measurements_before

A = np.matmul(measurements.reshape(len(measurements), 1), measurements.reshape(1, len(measurements)))

#set how many layers
#A = np.around(A * 1e-6)

w,v = np.linalg.eig(A)
#u,s,vh = np.linalg.svd(A)

sort_w = np.sort(np.sqrt(abs(w)), axis=None)[::-1]


print(sort_w.real)



print('bla')






# B = [[-149, -50, -154],
#      [-50, 180, -9],
#      [-154, -9, -25]]
# print(B)
#
# w,v = sc.linalg.eig(B)
# u,s,vh = np.linalg.svd(B)
#
# #sort_w = np.sort(w, axis=None)[::-1]
#
# print(w.real)
#
# print(s)



