import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from functions import *
import scipy as scy
from scipy import optimize
import matplotlib.pyplot as plt
from functools import partial
from jax import jit
n_bins = 20
burnIn = 50
#number_samples = 1000

tol = 1e-6

df = pd.read_excel('ExampleOzoneProfiles.xlsx')

#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
minInd = 7
maxInd = 44
pressure_values = press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd]
scalingConstkm = 1e-3
# https://en.wikipedia.org/wiki/Pressure_altitude
# https://www.weather.gov/epz/wxcalc_pressurealtitude
heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm

height_values = heights[minInd:maxInd]

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












# graph Laplacian
# direchlet boundary condition
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

# https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
temperature = get_temp_values(heights)
temp_values = temperature[minInd:maxInd]
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
g_doub_prime= np.zeros((size[0],1))


for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][155:160])


#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
#T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1
R = constants.gas_constant


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
w_cross =  VMR_O3 * f_broad * 1e-4
#w_cross[0], w_cross[-1] = 0, 0

#from : https://hitran.org/docs/definitions-and-units/
HitrConst2 = 1.4387769 # in cm K

# internal partition sum
Q = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ temp_values)
Q_ref = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ 296)
LineInt = S[ind,0] * Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))
LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))

#fig, axs = plt.subplots(tight_layout=True)
#plt.plot(LineInt,height_values)
#plt.show()

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''

#take linear
num_mole = 1 / (scy.constants.Boltzmann )#* temp_values)

AscalConstKmToCm = 1e3
#1e2 for pressure values from hPa to Pa
A_scal = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm/ ( temp_values)
scalingConst = 1e11
#theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst )
theta = num_mole* w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]
A = A_lin * A_scal.T
ATA = np.matmul(A.T,A)
Ax = np.matmul(A, theta)

y = add_noise(Ax, 0.01)

ATy = np.matmul(A.T, y)

"""start the mtc algo with first guesses of noise and lumping const delta"""


vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb = params[1]
    if lamb < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = ATA + lamb * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    # if exitCode != 0:
    #     print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F + 1e-4 * ( lamb * gamma + gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [1/(max(Ax) * 0.01)[0],(2*np.mean(vari))*1/(max(Ax) * 0.01)[0]])

lambda0 = minimum[1]
gamma0 = minimum[0]
print(minimum)




B = (ATA + lambda0 * L)
Bu, Bs, Bvh = np.linalg.svd(B)
cond_B =  np.max(Bs)/np.min(Bs)
print("B: " + str(orderOfMagnitude(cond_B)))
B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)

# B_inv_A_trans_y, exitCode = jax.scipy.sparse.linalg.gmres(B, ATy.reshape(-1), tol=tol,  x0 = B_inv_A_trans_y_real,atol = tol, restart=25, solve_method='batched')
# B_inv_A_trans_y2, exitCode = jax.scipy.sparse.linalg.gmres(B,  ATy[0::, 0], tol=tol, atol = tol, x0 = B_inv_A_trans_y_real,  restart=25, solve_method='incremental')
#
# relative_tol_L = tol
# CheckB_inv_A_trans_y = np.matmul(B, B_inv_A_trans_y2)
# print(np.linalg.norm(ATy.reshape(-1)- CheckB_inv_A_trans_y)/np.linalg.norm(ATy.reshape(-1))<relative_tol_L)
#
# relative_tol_L = tol
# CheckB_inv_A_trans_y = np.matmul(B, B_inv_A_trans_y_real)
# print(np.linalg.norm(ATy.reshape(-1)- CheckB_inv_A_trans_y)/np.linalg.norm(ATy.reshape(-1))<relative_tol_L)
#

#
# print(B_inv_A_trans_y_real)
# print(B_inv_A_trans_y)
# print(B_inv_A_trans_y2)
B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))


B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)

f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y0)
f_0_2 = -2 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
f_0_3 = 6 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y0)

g_0_1 = np.trace(B_inv_L)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)

##
B = (ATA + lambda0 * L)

B_inv_A_trans_y, info = jax.scipy.sparse.linalg.gmres(B, ATy[0::, 0], tol=tol, x0=B_inv_A_trans_y0, atol=tol, restart=25, solve_method='batched')

f_new = y.T @ y - ATy.T @ B_inv_A_trans_y


from jax.experimental.host_callback import call

def nb_MHwG(n, buIn, lam0, gam0, ATdata, Lapl, ATA_for, x0, tol, data, keyF):
    wLam = 2e2
    betaG = 1e-4
    betaD = 1e-4
    alphaG = 1
    alphaD = 1
    k = 0
    #key = jax.random.PRNGKey(137)
    gammas = jnp.zeros( n + buIn)
    # deltas = np.zeros(n+ buIn)
    lambdas = jnp.zeros(n + buIn)

    gammas = gammas.at[0].set(gam0)
    lambdas = lambdas.at[0].set(lam0)


    B_inv = x0
    f_new = data.T @ data - ATdata.T @ B_inv_A_trans_y
    #call(lambda f_new: print(f"{f_new}"), f_new)
    rate = f_new / 2 + betaG + betaD * lambda0

    shape = SpecNumMeas / 2 + alphaD + alphaG
    #call(lambda shape : print(f"{shape }"), shape )
    def forloop_fun(i, x):
        ATdata, lambdas, gammas, keyF, wLam, ATA_for, Lapl, k, rate, x0 , data, tol, f_new, B_inv = x
        #call(lambda f_new: print(f"{f_new}"), f_new)
        new_key, subkey = jax.random.split(keyF)
        del keyF
        keyF = new_key
        lam_p = lambdas[i] + jax.random.normal(key=subkey) * wLam
        del subkey
        del new_key
        def cond_fun(x):
            lam_p, lam, key, wLam = x
            return lam_p < 0


        def body_fun(x):
            lam_p, lam,  keyT, wLam = x
            new_key, subkey = jax.random.split(keyT)
            del keyT
            lam_p =  lam + jax.random.normal(key=subkey) * wLam
            del subkey
            return lam_p, lam, new_key, wLam

        def draw_lam(x):
            return jax.lax.while_loop(cond_fun, body_fun, x)


        def false_func(x):
            lam_p, lam,  keyF, wLam = x
            new_key, subkey = jax.random.split(keyF)
            del subkey
            del keyF
            return lam_p, lam, new_key, wLam

        new_key, subkey = jax.random.split(keyF)
        del keyF
        lam = jnp.copy(lambdas[i])
        operand = (lam_p, lam, subkey, wLam)

        lam_p, lam, keyF, wLam =  jax.lax.cond(lam_p < 0, draw_lam , false_func, operand)
        del subkey
        keyF = new_key
        del new_key
        #call(lambda lam_p: print(f"{lam_p}"), lam_p)

        delta_lam = lam_p - lambdas[i]
        delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3
        delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3

        log_MH_ratio = ((SpecNumLayers)/ 2) * (jnp.log(lam_p) - jnp.log(lambdas[i])) - 0.5 * (delta_g + gammas[i] * delta_f) - betaD * gammas[i] * delta_lam

        def accept_func(x):
            lam_p, lambdas, i, B_inv , bol  = x
            #k += 1
            lambdas = lambdas.at[i + 1].set(lam_p)

            bol = True
            #call(lambda f_new: print(f"inside accept before {f_new}"), f_new)
            #B = (ATA + lam_p * L)
            #B_inv, exitCode = jax.scipy.sparse.linalg.gmres(B, ATdata, tol=tol,
            #                                                          x0=x0, atol=tol, restart=25,
            #                                                        solve_method='batched')

            #f_new = data.T @ data - ATdata.T @ B_inv_A_trans_y
            #call(lambda B_inv_A_trans_y: print(f"{B_inv_A_trans_y}"), B_inv_A_trans_y)
            #call(lambda f_new: print(f"just after {f_new}"), f_new)
            #call(lambda exitCode: print(f"exitcode {exitCode}"), exitCode)
            #check if converges
            # op = ATdata, B, tol, x0, B_inv, exitCode
            # def inverse_batch(x):
            #     ATdata, B,  tol, x0, B_inv, exitCode = x
            #     #call(lambda f_new: print(f"{f_new}"), f_new)
            #     #call(lambda exitCode: print(f"{exitCode}"), exitCode)
            #     B_inv, exitCode = jax.scipy.sparse.linalg.gmres(B, ATdata, tol=tol,
            #                                                               x0=x0, atol=tol, restart=25,
            #                                                               solve_method='batched')
            #     return ATdata,B , tol, x0, B_inv, exitCode
            # def all_good(x):
            #     ATdata, B, tol, x0, B_inv, exitCode = x
            #     return ATdata,B , tol, x0, B_inv, exitCode
            # ATdata, B, tol, x0, B_inv, exitCode = jax.lax.select(jnp.not_equal(exitCode,0),  all_good, inverse_batch, op)
            #
            #call(lambda B_inv_A_trans_y: print(f"{B_inv_A_trans_y}"), B_inv_A_trans_y)

            #call(lambda x: print(f"data {data}"), data)
            #call(lambda x: print(f"ATdata {ATdata}"), ATdata)
            #f_new = data.T @ data - ATdata.T @ B_inv
            #call(lambda f_new: print(f"inside accept{f_new}"), f_new)
            #rate = f_new / 2 + betaG + betaD * lam_p

            return lam_p, lambdas, i, B_inv, bol

        def reject_func(x):
            lam_p, lambdas, i , B_inv, bol = x
            temp_lam = jnp.copy(lambdas[i])
            lambdas = lambdas.at[i + 1].set(temp_lam)
            bol = False
            return lam_p, lambdas, i, B_inv, bol

        new_key, subkey = jax.random.split(keyF)
        del keyF
        #accept or rejeict new lam_p
        u = jax.random.uniform(key=subkey)
        del subkey
        keyF = new_key
        del new_key
        #call(lambda f_new: print(f"outside{f_new}"), f_new)
        #call(lambda x: print(f"data {data}"), data)
        #call(lambda x: print(f"ATdata {ATdata}"), ATdata)
        bol = True
        operand = (lam_p, lambdas, i , B_inv, bol)
        lam_p, lambdas, i, B_inv, bol =  jax.lax.cond(jnp.log(u) <= log_MH_ratio, accept_func , reject_func, operand)

        new_key, subkey = jax.random.split(keyF)
        del keyF
        #call(lambda x: print(f"{x}"), bol)
        #@partial(jit, static_argnames=["ATdata"])
        def inverse( x ):
            ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new = x
            B = (ATA_for + lam_p * Lapl)
            B_inv, exitCode = jax.scipy.sparse.linalg.gmres(B, ATdata, tol=tol, x0=x0, atol=tol, restart=25,solve_method='batched')
            f_new = data.T @ data - ATdata.T @ B_inv
            #call(lambda f_new: print(f"inverse {f_new}"), f_new)
            #print(f"inverse")
            rate = f_new / 2 + betaG + betaD * lam_p
            return ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new

        #@partial(jit, static_argnames=["ATdata"])
        def false( x):
            ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new = x
            return ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new
        exitCode = 0
        #op = (ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new)
        #ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new = jax.lax.cond(bol, partial(inverse, ATdata) , partial(false, ATdata), op)

        op = (ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new)
        ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new = jax.lax.cond(bol, inverse , false, op)



        #call(lambda exitCode: print(f"exitcode {exitCode}"), exitCode)
        #@partial(jit, static_argnames=["ATdata"])
        def inverse_batch( x ):
            ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new = x
            B = (ATA_for + lam_p * Lapl)
            B_inv, exitCode = jax.scipy.sparse.linalg.gmres(B, ATdata, tol=tol, x0=x0, atol=tol, restart=25, solve_method='incremental')
            f_new = data.T @ data - ATdata.T @ B_inv
            #call(lambda f_new: print(f"batch {f_new}"), f_new)
            #print(f"inverse batch")
            rate = f_new / 2 + betaG + betaD * lam_p
            #call(lambda exitCode: print(f"{exitCode}"), exitCode)
            return ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new

        #@partial(jit, static_argnames=["ATdata"])
        def true_func(x):
            ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode,rate, f_new = x
            return ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new
        #op = ( ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new)
        #ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new = jax.lax.cond(jnp.equal(exitCode,0),   partial(true_func, ATdata),partial(inverse_batch, ATdata), op)
        op = (   ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new)
        ATdata, ATA_for, Lapl, lam_p, tol, x0, B_inv, exitCode, rate, f_new = jax.lax.cond(jnp.equal(exitCode,0),   true_func ,inverse_batch, op)


        # f_new = data.T @ data - ATdata.T @ B_inv
        # #call(lambda f_new: print(f"{f_new}"), f_new)
        # rate = f_new / 2 + betaG + betaD * lambdas[i+1]
        propGam = jax.random.gamma(key=subkey,a = shape)/rate
        #call(lambda propGam: print(f"{propGam[0,0]}"), propGam)
        gammas = gammas.at[i+1].set(propGam[0,0])
        del subkey
        keyF = new_key
        del new_key

        return ATdata, lambdas, gammas, keyF, wLam, ATA_for, Lapl, k, rate, x0,  data, tol, f_new, B_inv

    key, subkey = jax.random.split(keyF)
    ForInputs = (ATdata, lambdas, gammas, subkey, wLam,ATA_for, Lapl, k, rate, x0, data, tol, f_new, B_inv)
    ATdata, lambdas, gammas, subkey, wLam, ATA_for, Lapl, k, rate, x0, data, tol, f_new, B_inv = jax.lax.fori_loop(0, n + buIn-1, forloop_fun, ForInputs)



    return lambdas, gammas, k


number_sam = 10000
key = jax.random.PRNGKey(137)
new_key, subkey = jax.random.split(key)
del key
MHwG_jit = jax.jit(nb_MHwG, static_argnames =['n', 'buIn', 'lam0', 'gam0'])
del subkey
key = new_key
del new_key
#print("Compiling function:")
new_key, subkey = jax.random.split(key)
del key
key = new_key
del new_key
#key, *subkeyms = jax.random.split(key, number_sam+burnIn)
startTime = time.time()
lambdas, gammas, k = MHwG_jit(number_sam, burnIn, lambda0, gamma0, ATy[0::, 0], L, ATA, B_inv_A_trans_y0, tol,y, subkey)
elapsed = time.time() - startTime
print('First compile Done in ' + str(elapsed) + ' s')
del subkey


new_key, subkey = jax.random.split(key)
del key


startTime = time.time()
lambdas, gammas, k = MHwG_jit(number_sam, burnIn, lambda0, gamma0, ATy[0::, 0], L,  ATA, B_inv_A_trans_y0, tol, y, subkey)
elapsed = time.time() - startTime
print('First proper run done in ' + str(elapsed) + ' s')
del subkey
key = new_key

new_key, subkey = jax.random.split(key)
del key
startTime = time.time()
lambdas, gammas, k = MHwG_jit(number_sam, burnIn, lambda0, gamma0, ATy[0::, 0], L, ATA, B_inv_A_trans_y0, tol, y, subkey)
elapsed = time.time() - startTime
print('First proper run done in  ' + str(elapsed) + ' s')
del subkey
key = new_key

print('bla')


#fig, axs = plt.subplots(2, 1, tight_layout=True)
plt.figure()
#plt.hist(lambdas,bins=n_bins)
plt.plot(range(number_sam+burnIn),lambdas)
plt.plot(range(number_sam+burnIn),gammas)
#axs[1].hist(gammas,bins=n_bins)
plt.show()



print('bla')

#%time jit_matrix_product(B, B).block_until_ready()