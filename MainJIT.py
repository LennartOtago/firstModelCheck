
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from functions import *
import scipy as scy
from scipy import optimize



n_bins = 20
burnIn = 50
number_samples = 1000

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


""" plot forward model values """


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


B = (ATA + lambda0* L)

B_inv_A_trans_y, exitCode = jax.scipy.sparse.linalg.gmres(B, ATy[0::, 0], tol=tol, restart=25)


B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))


B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)

f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y)
f_0_2 = -2 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y)
f_0_3 = 6 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y)

g_0_1 = np.trace(B_inv_L)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)


def matrix_product(matrix1, matrix2):
    print(matrix1)
    result = matrix1 @ matrix2
    return result
key = jax.random.PRNGKey(137)
B = jax.random.normal(key = jax.random.PRNGKey(137), shape=(3, 3))
jit_matrix_product = jax.jit(matrix_product)
print("First call to jit_multipy(): ", jit_matrix_product(B, B))
gammas = jnp.zeros(number_samples + burnIn)
# deltas = np.zeros(number_samples + burnIn)
lambdas = jnp.zeros(number_samples + burnIn)

gammas = gammas.at[0].set(gamma0)
lambdas = lambdas.at[0].set(lambda0)
delta_lam = 1 - lambdas[0]
print(len(lambdas))
wLam = 2e2
key = jax.random.PRNGKey(1)
#key, subkey = jax.random.split(key)
lam_p = lambdas[0] + jax.random.normal(key=key) * wLam
print(lam_p)
#lam_p = 10 + jax.random.normal(key=key) * wLam
#lam_p = -10
def body_fun(x):
    lam, key, wLam = x
    new_key, subkey = jax.random.split(key)
    del key
    sample = lam + jax.random.normal(key=subkey) * wLam
    del subkey
    return (sample, new_key, wLam)

def cond_fun(x):
    lam , key, j = x
    new_key, subkey = jax.random.split(key)
    del key
    del subkey
    return lam < 0
u=lam_p
output = jax.lax.while_loop(cond_fun, body_fun, (u,  key, wLam))
print('xla', output)
print('xla', u)
def true_func(x):
    u, key, wLam = x
    return u, key, wLam

operand = (u,  key, wLam)
u = jax.lax.cond(opera, true_func , true_func, operand)

print('xla', u)

##
print("old key", key)
new_key, subkey = jax.random.split(key)
del key  # The old key is discarded -- we must never use it again.
normal_sample = jax.random.normal(subkey)
print(r"    \---SPLIT --> new key   ", new_key)
print(r"             \--> new subkey", subkey, "--> normal", normal_sample)
del subkey  # The subkey is also discarded after use.

# Note: you don't actually need to `del` keys -- that's just for emphasis.
# Not reusing the same values is enough.

key = new_key


##


def nb_MHwG(n, buIn, lam0, gam0, data, Lapl, keyF):
    wLam = 2e2
    betaG = 1e-4
    betaD = 1e-4
    alphaG = 1
    alphaD = 1
    k = 0
    #key = jax.random.PRNGKey(137)
    gammas = jnp.zeros( n + buIn)
    # deltas = np.zeros(n+ buIn)
    lambdas = jnp.zeros(n + buIn )

    gammas = gammas.at[0].set(gam0)
    lambdas = lambdas.at[0].set(lam0)

    B = (ATA + lam0 * Lapl)

    B_inv_A_trans_y, info = jax.scipy.sparse.linalg.gmres(B, ATy[0::, 0], tol=1e-8, restart=25)

    # if exitCode != 0:
    #     print(exitCode)

    shape = SpecNumMeas / 2 + alphaD + alphaG
    f_new = y.T @ y - ATy.T @ B_inv_A_trans_y
    #f_new = y[0::, 0].T @ y[0::, 0] - ATy[0::, 0].T @ B_inv_A_trans_y

    rate = f_new / 2 + betaG + betaD * lambda0


    for t in range(number_samples + burnIn-1):
        #print(t)
        # # draw new lambda
        new_key, subkey = jax.random.split(keyF)
        lam_p = lambdas[t] + jax.random.normal(key=subkey) *  wLam

        del subkey
        del keyF
        keyF = new_key


        def body_fun(x):
            lam, keyF, wLam = x
            new_key, subkey = jax.random.split(keyF)
            del keyF
            sample = lam + jax.random.normal(key=subkey) * wLam
            del subkey
            return (sample, new_key, wLam)

        def cond_fun(x):
            lam, keyF, wLam = x
            return lam < 0
        def false_fun(x):
            lam, keyF, wLam = x
            return lam
        new_key, subkey = jax.random.split(keyF)

        #lam_p, subkey, wLam = jax.lax.cond(lam_p < 0, jax.lax.while_loop(cond_fun, body_fun, (lam_p, subkey, wLam)), lambda x: x ,(lam_p, subkey, wLam))

        lam_p, subkey, wLam = jax.lax.cond(lam_p < 0, jax.lax.while_loop(cond_fun, body_fun, (lam_p, subkey, wLam)), lambda x: x ,(lam_p, subkey, wLam))


            #lam_p, subkey, wLam = jax.lax.while_loop(cond_fun, body_fun, (lam_p, subkey, wLam))
        del subkey
        del keyF
        keyF = new_key






        delta_lam = lam_p - lambdas[t]
        # B = (ATA + lam_p * L)
        # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        # if exitCode != 0:
        #     print(exitCode)


        # f_new = f(ATy, y,  B_inv_A_trans_y)
        # g_new = g(A, L,  lam_p)
        #
        # delta_f = f_new - f_old
        # delta_g = g_new - g_old

        delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3
        delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3

        log_MH_ratio = ((SpecNumLayers)/ 2) * (jnp.log(lam_p) - jnp.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        new_key, subkey = jax.random.split(keyF)
        del keyF
        #accept or rejeict new lam_p
        u = jax.random.uniform(key=subkey)
        del subkey
        keyF = new_key
        if jnp.log(u) <= log_MH_ratio:
        #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            #only calc when lambda is updated

            B = (ATA + lam_p * L)
           # B_inv_A_trans_y = GMRES(B, b = ATy[0::, 0], x0 = np.zeros(len(ATy)), e=tol, restart=25, nmax_iter = 500)
            # if exitCode != 0:
            #     print(exitCode)

            f_new = y.T @ y - ATy.T @ B_inv_A_trans_y
            #f_new = y[0::, 0].T @ y[0::, 0] - ATy[0::, 0].T @ B_inv_A_trans_y
            #g_old = np.copy(g_new)
            rate = f_new/2 + betaG + betaD * lam_p#lambdas[t+1]

        else:
            #rejcet
            lambdas[t + 1] =lambdas[t]# np.copy(lambdas[t])

        new_key, subkey = jax.random.split(keyF)
        del keyF
        gammas[t+1] = jax.random.gamma(key=subkey,a = shape)/rate
        del subkey
        keyF = new_key

        #deltas[t+1] = lambdas[t+1] * gammas[t+1]

    return lambdas, gammas, k


print('bla')


MHwG_jit = jax.jit(nb_MHwG, static_argnames =['n', 'buIn'])
#print("Compiling function:")
MHwG_jit(number_samples, burnIn, lambda0, gamma0, y[0::, 0], L, key)



#%time jit_matrix_product(B, B).block_until_ready()