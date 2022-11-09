import numpy as np
import scipy as sc
from scipy import constants
import plotly.graph_objects as go
from decimal import Decimal
import math
import testTheo
import testReal
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.special import wofz

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
##check absoprtion coeff in different heights and different freqencies

filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml'
VMR_O3, height_values, pressure_values, temp_values = testReal.get_data(filename)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
height_values = height_values * 1e-3#in km 1e2 # in cm
height_values = np.linspace(18,93,45)

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
files = '/home/lennartgolks/Python/firstModelCheck/634f1dc4.par'

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
A = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))
#sig_air = np.zeros((size[0],1))
#print(data_set[0])
#current = list(data_set[0].split(" "))

for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    A[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])





#calculate voigt function
ind = 293

#pick wavenumber in cm^-1
v_0 = wvnmbr[ind]
print("Frequency " + str(np.around(v_0[0]*3e1,2)) + " in GHz")
#calc pressure HWHM
#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K
p_ref = pressure_values[0]
gamma = [ (T_ref/ temp)**n_air[ind] * (p/p_ref) * 2 * g_self[ind] for (p,temp) in zip(pressure_values, temp_values)]

#calc Dopple HWHM
mol_M = 48 #g/mol
alpha = [7.17e-7 * v_0 * np.sqrt(temp/mol_M) for temp in temp_values] # in cm^-1
sigma = alpha / np.sqrt(2 * np.log(2))


#voigt function as real part of Faddeeva function
def V(x, sigma, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    #sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / (sigma * np.sqrt(2*np.pi))


def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha * np.exp(-(x / alpha)**2 * np.log(2))

wvnmbrs = np.linspace(4,5.3,1000)
 #wvnmbrs = np.linspace(-0.8,0.8,1000)
norm_Voigt = V(wvnmbrs - v_0,sigma[20],gamma[20])/sum(V(wvnmbrs - v_0,sigma[20],gamma[20]))


#plt.plot(wvnmbrs - v_0, V(wvnmbrs - v_0,sigma[20],gamma[20])/sum(V(wvnmbrs - v_0,sigma[20],gamma[20])))
#plt.plot(wvnmbrs - v_0, G(wvnmbrs - v_0,alpha[20]))
#plt.show()


#get absoption for selected frequency



#calculate weighted absorption crosssection
d_height = height_values[1::] - height_values[0:-1]
R_gas = constants.R * 1e6 # in ..cm^3
N_A = constants.Avogadro
VV = V(0,sigma[20],gamma[20])[0]
#w_cross = [ (1e-6  * pressure_values[i] / (R_gas* temp_values[i]) )**(1/3) * S[ind][0] *N_A* norm_Voigt[0] * 1e11 for i, vmr in enumerate(VMR_O3)]
#w_cross = [ (vmr  * N_A * pressure_values[i] / (R_gas* temp_values[i]) )**(1/3) * S[ind][0]/N_A * norm_Voigt[0] for i, vmr in enumerate(VMR_O3)]
w_cross = [ 1 for i, vmr in enumerate(VMR_O3)]

#w_cross = [ (vmr)**(1/3) * S[ind][0] * norm_Voigt[0] for i, vmr in enumerate(VMR_O3)]


#source funciton in cm ..
h = constants.h* 1e4
c = constants.c * 1e2
k_b = constants.Boltzmann * 1e4
T = temp_values[0:-1]


Source = 2 * h * c**2 * v_0**3 * 1/(np.exp(h * c * v_0/(k_b *T) ) - 1)
Source = np.ones(len(T))
#transmission starting from ground layer
# length is one shorter than heitght values
R = 63#earth radius
#tang_ind = 20
measurements = [None] * (len(height_values)-1)
measurements = np.zeros((len(height_values)-1,1))
for tang_ind in range(0,len(height_values)-1):

    trans_per_h = w_cross[tang_ind:-1] * (height_values[tang_ind:-1] + R)/np.sqrt(
            (height_values[tang_ind:-1]+ R)**2 + (height_values[tang_ind]+ R)**2 ) * d_height[tang_ind::]

    kernel = Source[tang_ind::]  * w_cross[tang_ind:-1] * pressure_values[tang_ind:-1] * d_height[tang_ind::]/\
                (k_b *T[tang_ind::] * np.sqrt((height_values[tang_ind:-1]+ R)**2 + (height_values[tang_ind]+ R)**2 ) ) * (height_values[tang_ind:-1] + R)
    trans_pre = [ Decimal(np.exp(1))**int(-sum(trans_per_h[i::])) for i in range(0,len(trans_per_h)) ]
    print(-sum(trans_per_h))
    trans_after = [ Decimal(np.exp(1))**int(-sum(trans_per_h[0:i])) for i in range(1,len(trans_per_h)+1) ]
    #trans_after = np.ones(44)
    #trans_pre = np.ones(44)
    L = [float(Decimal(kernels) * trans_pre[i])  for i,kernels in enumerate(kernel)]
    L_after= [float(Decimal(kernels) * trans_after[i] * trans_pre[0])  for i,kernels in enumerate(kernel)]
    measurements[tang_ind] = sum(L) + sum(L_after) * 1e-4 #to go back to m units



plt.plot(np.linspace(0,44,44), measurements)
plt.show()

A = np.matmul(measurements.reshape(len(measurements), 1), measurements.reshape(1, len(measurements)))



w,v = np.linalg.eig(A)
#u,s,vh = np.linalg.svd(A)

sort_w = np.sort(np.sqrt(abs(w)), axis=None)[::-1]


print(sort_w.real)



print('bla')







