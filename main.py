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



# ##################################################
#
##check absoprtion coeff in different heights and different freqencies

filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml'
VMR_O3, height_values, pressure_values, temp_values = testReal.get_data(filename)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
height_values = height_values * 1e-3#in km 1e2 # in cm
height_values = np.linspace(18,93,45)
pressure_values = pressure_values * 1e-1 # in cgs
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

x = VMR_O3 * N_A * pressure_values /(R_gas * temp_values)#* 1e-13
fig1,ax1 = plt.subplots()
ax1.plot(VMR_O3, height_values)
ax1.set_ylabel('height in km')
ax1.set_xlabel('Ozone number density / cm^3')
plt.show()

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




#constants in si
h = constants.h * 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1

C2 = h * c_cgs /k_b_cgs
C1 = h * c_cgs**2
#calculate voigt function
ind = 293

#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")

Source = 2 * C1/(lamba**5 * (np.exp(C2/(lamba*temp_values))-1))
#T_bright =
#print(2 * C1/(lamba**5 * (np.exp(C2/(lamba*216))-1)))

#calc pressure HWHM
#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K
p_ref = pressure_values[0]

#gamma = [ (T_ref/ temp)**n_air[ind] * (p/p_ref) * 2 * g_self[ind] for (p,temp) in zip(pressure_values, temp_values)]
gamma = [ (T_ref/ temp)**n_air[ind] * 2 * g_self[ind] for (p,temp) in zip(pressure_values, temp_values)]

#calc Dopple HWHM for different temperatures
mol_M = 48 #g/mol
C3 = np.sqrt( N_A  * k_b_cgs * 2* np.log(2))/c_cgs #[7.17e-7
alpha =  np.array([ C3*v_0 * np.sqrt(temp/mol_M) for temp in temp_values])
sigma = alpha / np.sqrt(2 * np.log(2))

sigma_gauss = v_0 * np.sqrt(k_b_cgs * 260/ mol_M)

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
    return np.sqrt(1/ (2* np.pi)) / alpha * np.exp(-(x / alpha)**2) # * np.log(2))


x_wvnmbrs = np.linspace(-1*v_0,1*v_0,1000)

print(np.exp(-(2 / alpha[20])**2 * np.log(2)))

norm_Voigt = V(x_wvnmbrs,sigma[20],gamma[20])/sum(V(x_wvnmbrs,sigma[20],gamma[20]))
print(alpha[20] / np.sqrt( np.log(2)))
norm_Gauss = G(x_wvnmbrs,sigma_gauss)# / sum(G(x_wvnmbrs,alpha[20]))
plt.plot(x_wvnmbrs, norm_Voigt, 'red')
plt.plot(x_wvnmbrs, norm_Gauss, 'blue')
plt.show()


#get absoption for selected frequency



#calculate weighted absorption crosssection
d_height = (height_values[1::] - height_values[0:-1] )
R_gas = N_A * k_b_cgs # in ..cm^3


#w_cross = [ (1e-6  * pressure_values[i] / (R_gas* temp_values[i]) )**(1/3) * S[ind][0] *N_A* norm_Voigt[0] * 1e11 for i, vmr in enumerate(VMR_O3)]
#w_cross = [ (vmr  * N_A * pressure_values[i] / (R_gas* temp_values[i]) )**(1/3) * S[ind][0]/N_A * norm_Voigt[0] for i, vmr in enumerate(VMR_O3)]

# caculate number of molecules in one cm^3
num_mole = [ (  pressure_values[i]/ (k_b_cgs * temp_values[i]) ) for i in range(0, len(pressure_values))]
#num_mole = [ ( 1e6 * pressure_values[i]  ) for i in range(0, len(pressure_values))]

# vmr might not have correct units
C4 =  [V(x_wvnmbrs[500],sigma[temp],gamma[temp])/sum(V(x_wvnmbrs,sigma[temp],gamma[temp])) for temp in range(0, len(temp_values)) ] #scaling trhough dopller/voigt profile
w_cross = [ S[ind] * vmr * C4[i] for i, vmr in enumerate(VMR_O3)]

#w_cross = [ (vmr)**(1/3) * S[ind][0] * norm_Voigt[0] for i, vmr in enumerate(VMR_O3)]

meas_per_h = [ w_cross[i]  * num_mole[i]  * Source[i] for i, p in enumerate(pressure_values)]

#transmission starting from ground layer
# length is one shorter than heitght values

R = 63#earth radius
measurements = np.zeros((len(height_values)-1,1))
for tang_ind in range(0,len(height_values)-1):

    trans_per_h = [ meas_per_h[i] * (height_values[i] + R)/np.sqrt(
            (height_values[i]+ R)**2 + (height_values[tang_ind]+ R)**2 ) * d_height[i]  * 1e5
                    for i in range(tang_ind,len(height_values)-1)]

    measurements[tang_ind] = sum(trans_per_h) + sum(trans_per_h)

#
fig2, ax2 = plt.subplots()
ax2.plot(np.linspace(0,43,44), measurements*1e18)
ax2.set_xlabel('tangent index layer')
ax2.set_ylabel('measurement')
ax2.set_yscale('log')
fig2.savefig('measurements.png')
plt.show()

# A = np.matmul(measurements.reshape(len(measurements), 1), measurements.reshape(1, len(measurements)))
#
#
#
# w,v = np.linalg.eig(A)
# #u,s,vh = np.linalg.svd(A)
#
# sort_w = np.sort(np.sqrt(abs(w)), axis=None)[::-1]
#
#
# fig3, ax3 = plt.subplots()
# ax3.scatter(np.linspace(0,44,44),sort_w.real)
# ax3.set_xlabel('index')
# ax3.set_ylabel('singular values (log scale)')
# ax3.set_yscale('log')
# fig3.savefig('svd.png')
# plt.show()


#print(sort_w.real)



print('bla')







