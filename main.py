import numpy as np
from functions import *
from scipy import constants
import testReal
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rd
from numpy.fft import fft, ifft, fft2



''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
#/Users/lennart/PycharmProjects/firstModelCheck
obs_height= 300 #in km
filename = '/Users/lennart/PycharmProjects/firstModelCheck/tropical.O3.xml' #/home/lennartgolks/Python
VMR_O3, height_values, pressure_values = testReal.get_data(filename, obs_height * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
d_height = (height_values[1::] - height_values[0:-1] )
#height_values = np.linspace(18,93,45)
pressure_values = pressure_values * 1e-1 # in cgs
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3


#plt.plot(VMR_O3, height_values)
#plt.show()

temp_values = get_temp_values(height_values)
x = VMR_O3 * N_A * pressure_values /(R_gas * temp_values)#* 1e-13

files = '/Users/lennart/PycharmProjects/firstModelCheck/634f1dc4.par'

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
w_cross = np.ones((len(height_values),1)) * S[ind,0]
w_cross[0], w_cross[-1] = 0, 0
#S[ind,0] * C4[i][0]


''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''


R = 6371 # earth radiusin km
#obs_height = 300 # in km
#fing minimum and max angle in radians
max_ang = np.arcsin( (height_values[-2] + R) / (R + obs_height) )
min_ang = np.arcsin( R / (R + obs_height) )

#specify where measurements are taken
num_meas = 20
meas_ang = np.linspace(min_ang+(max_ang-min_ang)/4, max_ang-(max_ang-min_ang)/4, num_meas)
meas_ang = np.linspace(min_ang, max_ang, num_meas)

Ax = gen_measurement(meas_ang, height_values, w_cross, VMR_O3, pressure_values ,temp_values, Source)
#get tangent height for each measurement
tang_height = np.around((np.sin(meas_ang) * (obs_height + R) ) -R,2)


#to test that we have the same dr distances
tot_r = np.zeros(num_meas)
#calculate total length
for j in range(0,num_meas):
    # np.sqrt( (obs_height + R)**2 - (tang_height[j] + R )**2 ) -
    tot_r[j] = np.sqrt( (height_values[-1] + R)**2 - (tang_height[j] + R )**2 )





fig2, ax2 = plt.subplots()
ax2.plot( Ax,np.linspace(1,num_meas,num_meas,dtype = int))
ax2.set_ylabel('measurement')
ax2.set_xlabel('measurement1')
ax2.set_xscale('log')
fig2.savefig('measurements.png')
plt.show()


#convolve measurements and add noise


sigma_noise = 0.01 * np.max(Ax)


#Y2 = sec_meas + rd.normal(0, np.sqrt(sigma_noise ), (44,1))


#beta = np.linalg.norm(sec_meas - Y2)**2/2 + 1e-4
apha = 44/2 + 1

#mean = apha/beta
#Bayesian framework

#graph Laplacian
neigbours = np.zeros((len(height_values)-1,2))

neigbours[0] = np.nan, 1
neigbours[-1] = len(height_values)-3, np.nan
for i in range(1,len(height_values)-2):
    neigbours[i] = i-1, i+1


L = generate_L(neigbours)


# f and g



#f = np.matmul(Y2.T,np.matmul(np.matmul(np.matmul(A,  np.linalg.inv(np.matmul(A.T,A) + L)),A.T),Y2))

#g = np.linalg.det(np.matmul(A.T,A) + L)

#hyperarameters
number_samples = 100
gammas = np.zeros(number_samples)
etas = np.zeros(number_samples)

#inintialize sample

gammas[0] = 9.5e-12

etas[0] = 9.5e-12

print('bla')







