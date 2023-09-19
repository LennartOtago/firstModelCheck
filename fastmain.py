import numpy as np
from importetFunctions import *
import time
import pickle as pl
import matlab.engine
from functions import *
from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)

tol = 1e-6

df = pd.read_excel('ExampleOzoneProfiles.xlsx')

#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpp [ascal pr millibars]
O3 = df['Ozone (VMR)'].values
pressure_values = press[7:44]
VMR_O3 = O3[7:44]
scalingConstkm = 1e-3
height_values = 145366.45 * (1 - ( press[7:44] /1013.25)**0.190284 ) * 0.3048 * scalingConstkm


""" analayse forward map without any real data values"""

MinH = height_values[0]
MaxH = height_values[-1]
R = 6371 # earth radiusin km
ObsHeight = 500 # in km

''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas = 105
SpecNumLayers = len(height_values)

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile
MaxAng = np.arcsin((height_values[-1]+ R) / (R + ObsHeight))
MinAng = np.arcsin((height_values[0] + R) / (R + ObsHeight))

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#meas_ang = min_ang + ((max_ang - min_ang) * np.exp(coeff * (np.linspace(0, int(num_meas) - 1, int(num_meas)+1) - (int(num_meas) - 1))))
meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
A_lin, tang_heights_lin, extraHeight = gen_forward_map(meas_ang,height_values,ObsHeight,R)

ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
A_lin = A_lin
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros(SpecNumMeas)
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = 2*(np.sqrt( ( extraHeight + R)**2 - (tang_heights_lin[j] + R )**2) )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T), tot_r)))












#graph Laplacian
neigbours = np.zeros((len(height_values),2))
neigbours[0] = np.nan, 1
neigbours[-1] = len(height_values)-2, np.nan
for i in range(1,len(height_values)-1):
    neigbours[i] = i-1, i+1
L = generate_L(neigbours)
np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')



#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
filename = 'tropical.O3.xml'

#VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
#d_height = layers[1::] - layers[0:-1]
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3


temp_values = get_temp_values(height_values)
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
#T = temp_values[0:-1] #in K
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
Source = np.array(C1 /(np.exp(C2) - 1) ).reshape((SpecNumLayers,1))

#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K usually
p_ref = pressure_values[0]




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
A_scal = pressure_values.reshape((SpecNumLayers,1)) / ( temp_values)
num_mole = 1 / (scy.constants.Boltzmann )#* temp_values)
scalingConst = 1e16
theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst )




A = A_lin * A_scal.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(A_lins)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))

Ax = np.matmul(A, theta)

#convolve measurements and add noise
y = add_noise(Ax, 0.01)

ATy = np.matmul(A.T, y)

np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')




"""start the mtc algo with first guesses of noise and lumping const delta"""


vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MargPost(params):#, coeff):

    gamma = params[0]
    delta = params[1]
    if delta < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers

    Bp = ATA + delta/gamma * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  delta/gamma)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(delta) + 0.5 * G + 0.5 * gamma * F + 1e-4 * ( delta + gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MargPost, [1/np.var(y),1/(2*np.mean(vari))])

#minimum = optimize.minimize(MargPost, [1/np.var(y), 1/(2*np.mean(vari))], args = [ATA_lin, L])
print(minimum)
print(minimum[1]/minimum[0])


'''do the sampling'''
number_samples = 10000
gammas = np.zeros(number_samples)
deltas = np.zeros(number_samples)
lambdas = np.zeros(number_samples)

#inintialize sample
gammas[0] = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
deltas[0] =  minimum[1] #0.275#1/(2*np.mean(vari))0.1#
lambdas[0] = deltas[0]/gammas[0]

ATy = np.matmul(A.T, y)

B = (ATA + lambdas[0] * L)

B_inv_L = np.zeros(np.shape(B))
for i in range(len(B)):
    B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

B_inv = np.zeros(np.shape(B))
for i in range(len(B)):
    e = np.zeros(len(B))
    e[i] = 1
    B_inv[:, i], exitCode = gmres(B, e, tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv ' + str(exitCode))

B_inv_L = np.matmul(B_inv,L)

B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)


Bu, Bs, Bvh = np.linalg.svd(B)
cond_B =  np.max(Bs)/np.min(Bs)
print("normal: " + str(orderOfMagnitude(cond_B)))

k = 0
wLam = 20
#wgam = 1e-5
#wdelt = 1e-1
betaG = 1e-4
betaD = 1e-4
alphaG = 1
alphaD = 1
rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambdas[0]
# draw gamma with a gibs step
shape = SpecNumLayers/2 + alphaD + alphaG

startTime = time.time()
for t in range(number_samples-1):
    #print(t)

    # # draw new lambda
    lam_p = normal(lambdas[t], wLam)

    while lam_p < 0:
            lam_p = normal(lambdas[t], wLam)

    delta_lam = lam_p - lambdas[t]
    B = (ATA + lam_p * L)
    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)


    f_new = f(ATy, y,  B_inv_A_trans_y)
    g_new = g(A, L,  lam_p)

    delta_f = f_new - f_old
    delta_g = g_new - g_old

    log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

    #accept or rejeict new lam_p
    u = uniform()
    if np.log(u) <= log_MH_ratio:
    #accept
        k = k + 1
        lambdas[t + 1] = lam_p
        #only calc when lambda is updated

        B = (ATA + lam_p * L)
        f_old = f(ATy, y,  B_inv_A_trans_y)
        g_old = g(A, L,  lambdas[t + 1])
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
