import numpy as np
import matplotlib as mpl
import kneed
from importetFunctions import *
import time
import pickle as pl
import matlab.engine
from functions import *
from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
#import tikzplotlib
plt.rcParams.update({'font.size': 18})
import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import tikzplotlib
#mpl.rc('text.latex', preamble=r"\boldmath")

""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)
# pgf_params = { "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'font.size' : 11,
#     'text.usetex': True,
#     'pgf.rcfonts': False}
# 'font.size': 12,
# 'axes.labelsize': 12,  # -> axis labels
# 'legend.fontsize': 12,
fraction = 1.5
pgf_params = { "pgf.texsystem": "pdflatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
'axes.labelsize': 12,  # -> axis labels
'legend.fontsize': 12}

dpi = 300

PgWidthPt = 245
n_bins = 20
burnIn = 50
betaG = 1e-4
betaD = 1e-10  # 1e-4
#Colors
#pyTCol = [230/255,159/255, 0/255]
pyTCol = [213/255,94/255, 0/255]
#pyTCol = [240/255, 228/255, 66/255]
MTCCol = 'k'
dataCol = [225/255, 190/255, 106/255]
#dataCol =[230/255,159/255, 0/255]
regCol = [212/255, 17/255, 89/255]
#MargCol = [86/255, 180/255, 233/255]
MargCol = [255/255, 194/255, 10/255]
defBack = mpl.get_backend()


ResCol = "#1E88E5"#"#0072B2"
MeanCol = 'k'#"#FFC107"#"#d62728"
RegCol = "#D81B60"#"#D55E00"
TrueCol = 'green' # "#004D40" #'k'
DatCol =  'gray' # 'k'"#332288"#"#009E73"


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
#coeff = 1/(SpecNumMeas)
#meas_ang = (MinAng) + (MaxAng - MinAng) * np.exp(- coeff * 1* np.linspace(0, int(SpecNumMeas) -1 , SpecNumMeas ))
# coeff = 1/np.log(SpecNumMeas)
# meas_ang = (MinAng) + (MaxAng - MinAng) * coeff * 0.9 * np.log( np.linspace(1, int(SpecNumMeas) , SpecNumMeas ))

# fig, axs = plt.subplots(tight_layout=True)
# plt.scatter(range(len(meas_ang )),meas_ang )
# plt.show()
meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
A_lin, tang_heights_lin, extraHeight = gen_forward_map(meas_ang,height_values,ObsHeight,R)


# fig, axs = plt.subplots(tight_layout=True)
# plt.scatter(range(len(tang_heights_lin)),tang_heights_lin)
# #plt.show()

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









##


# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))
# neigbours[0] = np.nan, np.nan, 1, 2
# neigbours[-1] = len(height_values)-2, len(height_values)-3, np.nan, np.nan
# neigbours[0] = np.nan, 1
# neigbours[-1] = len(height_values)-2, np.nan
for i in range(0,len(height_values)):
    neigbours[i] = i-1, i+1
    #neigbours[i] = i-2, i-1, i+1, i+2#, i+3 i-3,


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)
startInd = 23
L[startInd::, startInd::] = L[startInd::, startInd::] * 5
L[startInd, startInd] = -L[startInd, startInd-1] - L[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]

#cholesky decomposition of L for W1 and v1
lowC_L = scy.linalg.cholesky(L, lower = True)

#L[startInd+1, startInd+1] = -L[startInd+1, startInd+1-1] - L[startInd+1,startInd+1+1] -L[startInd+1, startInd+1-2] - L[startInd+1, startInd+1+2]
# L[16, 16] = 13

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
#ind = 293
ind = 623
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
LineIntScal =  Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))
#
# fig, axs = plt.subplots(tight_layout=True)
# plt.plot(LineInt,height_values)
# plt.show()

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''

#take linear
num_mole = 1 / ( scy.constants.Boltzmann )#* temp_values)

AscalConstKmToCm = 1e3
#1e2 for pressure values from hPa to Pa
A_scal = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm/ ( temp_values)
scalingConst = 1e11
#theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst )
theta = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]

# A_scal = pressure_values.reshape((SpecNumLayers,1)) / ( temp_values)
# scalingConst_old = 1e16
# theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst_old )
#
#num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst

""" plot forward model values """
numDensO3 =  N_A * press * 1e2 * O3 / (R * temp_values[0,:]) * 1e-6
# fig, axs = plt.subplots(tight_layout=True)
# plt.plot(press ,heights,color = [0, 205/255, 127/255])
# #plt.plot((1/ temp_values) ,heights,color ='k')
# axs.set_ylabel('Height in km')
# axs.set_xlabel('Number density of Ozone in cm$^{-3}$')
# plt.savefig('theta.png')
# plt.show()



fig, axs = plt.subplots(tight_layout=True, figsize=set_size(PgWidthPt, fraction=fraction))
#plt.plot(press/1013.25,heights, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
#plt.plot(Source/max(Source),height_values, label = r'Source in $\frac{W}{m^2 sr}\frac{1}{\frac{1}{cm}}$/' + str(np.around(max(Source[0]),5)) )
plt.plot(temperature,heights, color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
#plt.plot(LineInt,heights[minInd:maxInd], color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
#axs.legend()
axs.tick_params(axis = 'x', labelcolor="darkred")
ax2 = axs.twiny() # ax1 and ax2 share y-axis
line3 = ax2.plot(press[minInd:maxInd],heights[minInd:maxInd], color = 'blue') #, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
ax2.spines['top'].set_color('blue')
ax2.tick_params(labelcolor="blue")
ax2.set_xlabel('Pressure in hPa')
axs.set_ylabel('Height in km')
axs.set_xlabel('Temperature in K')
#axs.set_xlabel('Line intensity in cm / molecule')
#axs.set_title()
plt.savefig('PandQ.png')
plt.show()


A = A_lin * A_scal.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))
#theta[0] = 0
#theta[-1] = 0
Ax = np.matmul(A, theta)

#convolve measurements and add noise
#y = add_noise(Ax, 0.01)
#y[y<=0] = 0

#y, gamma = add_noise(Ax, 60)

y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))


ATy = np.matmul(A.T, y)


np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
np.savetxt('ForWardMatrix.txt', A, header = 'Forward Matrix A', fmt = '%.15f', delimiter= '\t')
np.savetxt('height_values.txt', height_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('tan_height_values.txt', tang_heights_lin, fmt = '%.15f', delimiter= '\t')

np.savetxt('pressure_values.txt', pressure_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('VMR_O3.txt', VMR_O3, fmt = '%.15f', delimiter= '\t')




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


    # B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], rtol=tol, restart=25)
    # if exitCode != 0:
    #     print(exitCode)
    LowTri = np.linalg.cholesky(Bp)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb * gamma + betaG *gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
# minimum = optimize.fmin(MinLogMargPost, [1/(np.max(Ax) * 0.01)**2,1/(np.mean(vari))*(np.max(Ax) * 0.01)**2], maxiter = 25)
#
# lam0 = minimum[1]




##
""" finally calc f and g with a linear solver adn certain lambdas
 using the gmres"""

lam= np.logspace(-5,15,500)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))



for j in range(len(lam)):

    B = (ATA + lam[j] * L)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
    #print(exitCode)
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
    if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol:
        f_func[j] = f(ATy, y, B_inv_A_trans_y)
    else:
        f_func[j] = np.nan

    g_func[j] = g(A, L, lam[j])


np.savetxt('f_func.txt', f_func, fmt = '%.15f')
np.savetxt('g_func.txt', g_func, fmt = '%.15f')
np.savetxt('lam.txt', lam, fmt = '%.15f')

##
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''
startTime = time.time()
minimum = optimize.fmin(MinLogMargPost, [1/(np.max(Ax) * 0.01)**2,1/(np.mean(vari))*(np.max(Ax) * 0.01)**2], maxiter = 25)

lam0 = minimum[1]
print(minimum)
# taylor series arounf lam_0

B = (ATA + lam0 * L)

#B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
#print(exitCode)

#CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y0)


LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
# check if L L.H = B
B_inv_A_trans_y0 = lu_solve(LowTri, UpTri,  ATy[0::, 0])

B_inv_L = np.zeros(np.shape(B))
# startTime = time.time()
# for i in range(len(B)):
#     B_inv_L[:, i], exitCode = gmres(B, L[:, i], rtol=1e-10, restart=25)
#     # if exitCode != 0:
#     #     print('B_inv_L ' + str(exitCode))
# print('time for B inverse ' + str(time.time()-startTime))
LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
for i in range(len(B)):
    B_inv_L[:, i] = lu_solve(LowTri, UpTri,  L[:, i])
print('time for B inverse ' + str(time.time()-startTime))

# relative_tol_L = tol
# CheckB_inv_L = np.matmul(B, B_inv_L)
# print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)
# print(np.allclose(CheckB_inv_L, L))


B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)
B_inv_L_6 = np.matmul(B_inv_L_4, B_inv_L_2)


f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y0)
f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y0)
f_0_4 = 0#-1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)
#f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)


g_0_1 = np.trace(B_inv_L)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)
g_0_4 = 0#-1 /24 * np.trace(B_inv_L_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)




##
#startTime = time.time()
'''do the sampling'''
number_samples = 10000


#inintialize sample
gamma0 = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
#0.275#1/(2*np.mean(vari))0.1#
lambda0 = minimum[1]#deltas[0]/gammas[0]
#deltas[0] =  minimum[1] * minimum[0]
#ATy = np.matmul(A.T, y)
#B = (ATA + lambda0 * L)

# B_inv_A_trans_y0_gmres, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
# # if exitCode != 0:
# #     print(exitCode)
# print('time for gmres: ' + str( time.time() - startTime))

#startTime = time.time()


# print('time for cholesky: ' + str( time.time()- startTime))
# print('Cholesky: ' + str(np.allclose(B, LowTri @ UpTri )))
# print('subSolve compare: ' + str(np.allclose(B @ B_inv_A_trans_y0, B @ B_inv_A_trans_y0_gmres, rtol=0.05)))
# print('subSolve : ' + str(np.allclose(ATy[0::, 0], B @ B_inv_A_trans_y0)))
# print('subSolve gmres : ' + str(np.allclose(ATy[0::, 0], B @ B_inv_A_trans_y0_gmres)))

#
# Bu, Bs, Bvh = np.linalg.svd(B)
# cond_B =  np.max(Bs)/np.min(Bs)
# print("Condition number B: " + str(orderOfMagnitude(cond_B)))

##
#wLam = 2e2#5.5e2
#wgam = 1e-5
#wdelt = 1e-1

alphaG = 1
alphaD = 1
f_0 = f(ATy, y,  B_inv_A_trans_y0)
rate = f_0 / 2 + betaG + betaD * lambda0
# draw gamma with a gibs step
shape = SpecNumMeas/2 + alphaD + alphaG


#g_old = g(A, L,  lambdas[0])

def MHwG(number_samples, burnIn, lambda0, gamma0, f_0):
    wLam = 1e3#7e1

    alphaG = 1
    alphaD = 1
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lambda0

    # B = (ATA + lambda0 * L)
    # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0=B_inv_A_trans_y0, rtol=tol)
    #
    # #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    # if exitCode != 0:
    #     print(exitCode)

    shape = SpecNumMeas / 2 + alphaD + alphaG
    rate = f_0 / 2 + betaG + betaD * lambda0


    for t in range(number_samples + burnIn-1):
        #print(t)

        # # draw new lambda
        lam_p = normal(lambdas[t], wLam)

        while lam_p < 0:
                lam_p = normal(lambdas[t], wLam)

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

        log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = uniform()
        if np.log(u) <= log_MH_ratio:
        #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            #only calc when lambda is updated

            #B = (ATA + lam_p * L)
            # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0= B_inv_A_trans_y0, rtol=tol, restart=25)
            #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)

            # if exitCode != 0:
            #         print(exitCode)

            #f_new = f(ATy, y,  B_inv_A_trans_y)
            f_new = f_0 + delta_f
            #g_old = np.copy(g_new)
            rate = f_new/2 + betaG + betaD * lam_p#lambdas[t+1]

        else:
            #rejcet
            lambdas[t + 1] = np.copy(lambdas[t])




        gammas[t+1] = np.random.gamma(shape = shape, scale = 1/rate)

        #deltas[t+1] = lambdas[t+1] * gammas[t+1]

    return lambdas, gammas,k



#startTime = time.time()
lambdas ,gammas, k = MHwG(number_samples, burnIn, lambda0, gamma0, f_0)
elapsedMWGH = time.time() - startTime
print('MWGH Done in ' + str(elapsedMWGH) + ' s')



print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('samples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsedMWGH), fmt = '%.15f \t %.15f \t %.15f')

#delt_aav, delt_diff, delt_ddiff, delt_itau, delt_itau_diff, delt_itau_aav, delt_acorrn = uWerr(deltas, acorr=None, s_tau=1.5, fast_threshold=5000)

# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.Run_Autocorr_Ana_MTC(nargout=0)
# eng.quit()
#
#
# AutoCorrData = np.loadtxt("auto_corr_dat.txt", skiprows=3, dtype='float')
# #IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'
#
# with open("auto_corr_dat.txt") as fID:
#     for n, line in enumerate(fID):
#        if n == 1:
#             IntAutoDelt, IntAutoGam, IntAutoLam = [float(IAuto) for IAuto in line.split()]
#             break
#
#
#
# #refine according to autocorrelation time
# new_lamb = lambdas[burnIn::math.ceil(IntAutoLam)]
# #SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
# new_gam = gammas[burnIn::math.ceil(IntAutoGam)]
# #SetGamma = new_gam[np.random.randint(low = 0,high =len(new_gam),size =1)]
# new_delt = deltas[burnIn::math.ceil(IntAutoDelt)]
# #SetDelta = new_delt[np.random.randint(low = 0,high =len(new_delt),size =1)]
#
# MTCMargSamp = np.vstack((gammas[burnIn::], lambdas[burnIn::])).T
# TrMTC = np.zeros(number_samples)
# for i,sam in enumerate(MTCMargSamp):
#     TrMTC[i] = MinLogMargPost(sam)


##
#draw paramter samples
paraSamp = 2000#n_bins
Results = np.zeros((paraSamp,len(theta)))
NormRes = np.zeros(paraSamp)
xTLxRes = np.zeros(paraSamp)
#SetGammas = new_gam[np.random.randint(low=0, high=len(new_gam), size=paraSamp)]
#SetDeltas  = new_delt[np.random.randint(low=0, high=len(new_delt), size=paraSamp)]
SetGammas = gammas[np.random.randint(low=0, high=len(gammas), size=paraSamp)]
SetDeltas  = deltas[np.random.randint(low=0, high=len(deltas), size=paraSamp)]

startTimeX = time.time()
for p in range(paraSamp):
    # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
    SetGamma = SetGammas[p] #minimum[0]
    SetDelta  = SetDeltas[p] #minimum[1]
    #W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    W = np.random.normal(loc=0.0, scale=1.0, size=len(A))#.reshape((len(A),1))
    v_1 = np.sqrt(SetGamma) *  A.T @ W
    # W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    # W2 = lowC_L @ np.random.multivariate_normal(np.zeros(len(L)), np.eye(len(L)) )
    W2 = lowC_L @ np.random.normal(loc=0.0, scale=1.0, size=len(L))#.reshape((len(L),1))
    v_2 = np.sqrt(SetDelta) * W2

    SetB = SetGamma * ATA + SetDelta * L
    RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)


    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, RandX)

    #CheckB_inv = np.matmul(SetB, SetB_inv)
    #print(np.linalg.norm(np.eye(len(SetB)) - CheckB_inv) / np.linalg.norm(np.eye(len(SetB))) < tol)

    Results[p, :] = B_inv_A_trans_y

    NormRes[p] = np.linalg.norm( np.matmul(A,B_inv_A_trans_y) - y[0::,0])
    xTLxRes[p] = np.sqrt(np.matmul(np.matmul(B_inv_A_trans_y.T, L), B_inv_A_trans_y))

elapsedX = time.time() - startTimeX
print('Time to solve for x ' + str(elapsedX/paraSamp))


NormLTest = np.linalg.norm( np.matmul(A,np.mean(Results,0 )) - y[0::,0])
xTLxCurveTest = np.sqrt(np.matmul(np.matmul(np.mean(Results,0 ).T, L), np.mean(Results,0 )))


np.savetxt('O3Res.txt', Results/(num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')



##

def postMeanAndVar(margPDF, Grid, ATy, ATA, L, Var):
    gridLen = len(margPDF[0])
    MargResults = np.zeros((gridLen,len(L)))
    gamInt = np.zeros(gridLen)
    VarB = np.zeros((gridLen, len(L), len(L)))
    B_inv = np.zeros((gridLen, len(L), len(L)))
    if len(Grid[0]) != len(margPDF[0]):
        print('Grid not the same lenght as Marg PDF')

    IDiag = np.eye(len(L))
    for p in range(gridLen):
        SetGamma = Grid[0, p]
        SetLambda = Grid[1,p]

        SetB = ATA + SetLambda * L

        LowTri = np.linalg.cholesky(SetB)
        UpTri = LowTri.T
        B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

        MargResults[p, :] = B_inv_A_trans_y * margPDF[1,p]

        if Var == True:
            LowTri = np.linalg.cholesky(SetB)
            UpTri = LowTri.T
            for i in range(len(SetB)):
                B_inv[p, :, i] = lu_solve(LowTri, UpTri,IDiag[:, i])
            VarB[p] = B_inv[p] *  margPDF[1,p]
            gamInt[p] = 1/SetGamma *  margPDF[0,p]

    postMean = np.sum(MargResults,0)
    postVar = np.sum(gamInt) * np.sum(VarB,0)
    return postMean, postVar


##

BinHistStart = 3
print(BinHistStart)
oldpostMean = 0
for PostMeanBinHist in range(BinHistStart+1,100):

    lambHist, lambBinEdges = np.histogram(lambdas[burnIn:], bins= PostMeanBinHist, density =True)
    gamHist, gamBinEdges = np.histogram(gammas[burnIn:], bins= PostMeanBinHist, density =True)
    margPDF = np.array([gamHist/np.sum(gamHist) , lambHist/np.sum(lambHist)])
    Grid = np.array([ gamBinEdges[:-1] + (gamBinEdges[1:] - gamBinEdges[:-1])/2, lambBinEdges[:-1] + (lambBinEdges[1:] - lambBinEdges[:-1])/2])

    startTime = time.time()
    newPostMean, postVar = postMeanAndVar(margPDF, Grid, ATy, ATA, L, True)
    MargTime = time.time() - startTime

    newRelErr = np.linalg.norm(oldpostMean - newPostMean) / np.linalg.norm(newPostMean) * 100
    print(newRelErr)
    if  0.5 > newRelErr:
        print(f'break at {PostMeanBinHist}')
        break
    oldpostMean = np.copy(newPostMean)
    oldRelErr = np.copy(newRelErr)


MargX =  newPostMean / (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
MargVar = postVar / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst) ** 2

NormMargRes = np.linalg.norm(np.matmul(A, newPostMean) - y[0::, 0])
xTLxMargRes = np.sqrt(np.matmul(np.matmul(newPostMean.T, L), newPostMean))


print('Post Mean in ' + str(MargTime) + ' s')

print('MTC Done in ' + str(elapsedMWGH +  MargTime) + ' s')

##
"Fitting prob distr to hyperparameter histogram"

def skew_norm_pdf(x,mean=0,w=1,skewP=0, scale = 0.1):
    # adapated from:
    # http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    t = (x-mean) / w
    return 2.0 * w * scy.stats.norm.pdf(t) * scy.stats.norm.cdf(skewP*t) * scale




##
'''make figure for f and g including the best lambdas and taylor series'''

##
"""f und g for  paper"""


f_mode = f(ATy, y, B_inv_A_trans_y0)



B_MTC = ATA + np.mean(lambdas) * L
# B_MTC_inv_A_trans_y, exitCode = gmres(B_MTC, ATy[0::, 0], rtol=tol)
# if exitCode != 0:
#     print(exitCode)
LowTri = np.linalg.cholesky(B_MTC)
UpTri = LowTri.T
B_MTC_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

f_MTC = f(ATy, y, B_MTC_inv_A_trans_y)




B_MTC_min = ATA + (np.mean(lambdas) - np.sqrt(np.var(lambdas))/2) * L
# B_MTC_min_inv_A_trans_y, exitCode = gmres(B_MTC_min, ATy[0::, 0], rtol=tol)
# if exitCode != 0:
#     print(exitCode)

LowTri = np.linalg.cholesky(B_MTC_min)
UpTri = LowTri.T
B_MTC_min_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])
f_MTC_min = f(ATy, y, B_MTC_min_inv_A_trans_y)

B_MTC_max = ATA + (np.mean(lambdas) + np.sqrt(np.var(lambdas))/2) * L
# B_MTC_max_inv_A_trans_y, exitCode = gmres(B_MTC_max, ATy[0::, 0], rtol=tol)
# if exitCode != 0:
#     print(exitCode)
LowTri = np.linalg.cholesky(B_MTC_max)
UpTri = LowTri.T
B_MTC_max_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])
f_MTC_max = f(ATy, y, B_MTC_max_inv_A_trans_y)

xMTC = np.mean(lambdas) - np.sqrt(np.var(lambdas))/2


B_min = ATA + (np.mean(lambdas) - np.sqrt(np.var(lambdas)) ) * L
# B_min_inv_A_trans_y, exitCode = gmres(B_min, ATy[0::, 0], rtol=tol)
# if exitCode != 0:
#     print(exitCode)
LowTri = np.linalg.cholesky(B_min)
UpTri = LowTri.T
B_pyT_max_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])
f_min = f(ATy, y, B_pyT_max_inv_A_trans_y)

B_max = ATA + (np.mean(lambdas) + np.sqrt(np.var(lambdas)) ) * L
# B_max_inv_A_trans_y, exitCode = gmres(B_max, ATy[0::, 0], rtol=tol)
# if exitCode != 0:
#     print(exitCode)
LowTri = np.linalg.cholesky(B_max)
UpTri = LowTri.T
B_max_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])
f_max = f(ATy, y, B_max_inv_A_trans_y)



##
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12})
fCol = [0, 144/255, 178/255]
gCol = [230/255, 159/255, 0]
#gCol = [240/255, 228/255, 66/255]
#gCol = [86/255, 180/255, 233/255]
gmresCol = [204/255, 121/255, 167/255]

delta_lam = lambBinEdges - minimum[1]

fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)

axs.plot(lam,f_func, color = fCol, zorder = 2, linestyle=  'dotted')

#axs.scatter(minimum[1],f_mode, color = gmresCol, zorder=0, marker = 's')#

#fig.text(minimum[1], 0.1, 'lambda', fontsize = 10, transform=axs.transData)#r'$\lambda_R$')


axs.set_yscale('log')
axs.set_xlabel('$\lambda$')
axs.set_ylabel('$f(\lambda)$')#, color = fCol)
axs.tick_params(axis = 'y',  colors=fCol, which = 'both')

ax2 = axs.twinx() # ax1 and ax2 share y-axis
ax2.plot(lam,g_func, color = gCol, zorder = 2, linestyle=  'dashed')
#ax2.scatter(minimum[1],g(A, L, minimum[1]), color = gmresCol, zorder=0, marker = 's')
ax2.axvline(minimum[1], ymax= 0.497, zorder = 0,color = gmresCol)
ax2.axvline(minimum[1], ymin= 0.95, zorder = 0,color = gmresCol)
ax2.set_ylabel('$g(\lambda)$')#,color = gCol)
ax2.tick_params(axis = 'y', colors= gCol)
axs.set_xscale('log')
axs.plot(lambBinEdges,f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3, f_0_4), color = 'k', linewidth = 1, zorder = 2, label = 'Taylor series' )
ax2.plot(lambBinEdges, g_tayl(delta_lam, g(A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, g_0_4,g_0_5, g_0_6), color = 'k', linewidth = 1, zorder = 1)

#axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
#axs.spines['left'].set_color(fCol)
axs.spines['left'].set_color('k')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color('k')
#ax2.spines['right'].set_color(gCol)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)





axins = axs.inset_axes([0.05,0.5,0.4,0.45], zorder = 3)
axins.set_facecolor('white')
axins.axvline(minimum[1], zorder = 0,color = gmresCol)
axins.plot(lam,f_func, color = fCol, zorder=3, linestyle=  'dotted', linewidth = 3)

axins.plot(lambBinEdges,f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3, f_0_4), color = 'k', linewidth = 1, zorder = 1, label = 'Taylor series' )


#axins.scatter(minimum[1],f_mode, color = gmresCol, s= 95, zorder=0, marker = 's', label = r'\texttt{optimize.fmin()}')
axins.set_xlim(min(lambdas),max(lambdas))
axins.set_ylim(4.5e8,9e8)
axins.set_xlabel('$\lambda$')
axins.set_xlim([np.mean(lambdas) -np.sqrt(np.var(lambdas)), 1.5*np.mean(lambdas) + np.sqrt(np.var(lambdas))])# apply the x-limits
axins.set_yscale('log')
axins.set_xscale('log')


axins.tick_params(axis='y', which='both',  left=False, labelleft=False)
axins.tick_params(axis='x', which='both',  bottom=False, labelbottom=False)
#axins.tick_params(axis='y', which='both', length=0)
axin2 = axins.twinx()
axin2.spines['top'].set_visible(False)
axin2.spines['right'].set_visible(False)
axin2.spines['bottom'].set_visible(False)
axin2.spines['left'].set_visible(False)
lines, lab0 = axins.get_legend_handles_labels()

axs.legend(lines, lab0, loc = 'lower right')


axin2.tick_params(axis = 'y', which = 'both',labelright=False, right=False)
axin2.tick_params(axis='y', which='both', length=0)
# #axin2.set_xticks([np.mean(lambdas) -np.sqrt(np.var(lambdas)) , np.mean(lambdas), np.mean(lambdas) + np.sqrt(np.var(lambdas)) ] )
axin2.plot(lam,g_func, color = gCol, zorder=3, linestyle=  'dashed', linewidth = 3)

axin2.plot(lambBinEdges, g_tayl(delta_lam, g(A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, g_0_4,g_0_5, g_0_6), color = 'k', linewidth = 1, zorder = 2 )

#axin2.scatter(minimum[1],g(A, L, minimum[1]), color = gmresCol, s=95, zorder=0, marker = 's')

axin2.set_ylim(360,460)
axin2.set_xlim(min(lambdas),max(lambdas))
axin2.set_xscale('log')


xlabels = [item.get_text() for item in axs.get_xticklabels()]
xticks = axs.get_xticks()
xlabels = np.append(xlabels, [r'$\lambda_0$'])
xticks=np.append(xticks,minimum[1])
axs.set_xticks( xticks, xlabels)
axs.set_xlim(min(lam),max(lam))

# axs.set_xticks(minimum[1])
# axs.set_xticklabels('lambda')
#baseTicks=list(axs.get_xticks()[0]) #for better control, replace with a range or arange
#axs.set_xticks(baseTicks+[minimum[1]])
#axs.set_xticklabels(baseTicks+['$\lambda$'])


#fig.savefig('f_and_g_paper.pgf', bbox_inches='tight')
plt.savefig('f_and_g_paper.png',bbox_inches='tight')
plt.show()
#for legend
# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save("f_and_g_paper.pgf")
##
fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)
axs.plot(lambBinEdges, g_tayl(delta_lam, g(A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, g_0_4,g_0_5, g_0_6), color = 'k')
axs.plot(lam,g_func, color = gCol, linestyle=  'dashed')
#axs.set_xscale('log')
plt.show()
##
plt.close()
mpl.use('pgf')
mpl.rcParams.update(pgf_params)
fig.savefig('f_and_g_paper.pgf', bbox_inches='tight', dpi = 300)

print('bla')

##
'''L-curve refularoization
'''

lamLCurve = np.logspace(-1,10,200)
#lamLCurve = np.linspace(1e-15,1e3,200)

NormLCurve = np.zeros(len(lamLCurve))
xTLxCurve = np.zeros(len(lamLCurve))
xTLxCurve2 = np.zeros(len(lamLCurve))
for i in range(len(lamLCurve)):
    B = (ATA + lamLCurve[i] * L)

    #x, exitCode = gmres(B, ATy[0::, 0], rtol=tol)
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    x = lu_solve(LowTri, UpTri, ATy[0::, 0])

    if np.allclose(B @ x , ATy[0::, 0]) == False :
        print(False)
        NormLCurve[i] = np.nan
        xTLxCurve[i] = np.nan

    else:
        NormLCurve[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
        xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))

startTime  = time.time()
lamLCurveZoom = np.logspace(1,6,200)
NormLCurveZoom = np.zeros(len(lamLCurve))
xTLxCurveZoom = np.zeros(len(lamLCurve))
for i in range(len(lamLCurveZoom)):
    B = (ATA + lamLCurveZoom[i] * L)

    #x, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    x = lu_solve(LowTri, UpTri, ATy[0::, 0])
    if np.allclose(B @ x , ATy[0::, 0]) == False:
        print(False)
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    B_max_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    NormLCurveZoom[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
    xTLxCurveZoom[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))



# calculate and show knee/elbow
kneedle = kneed.KneeLocator(NormLCurveZoom, xTLxCurveZoom, curve='convex', direction='decreasing', online = True, S = 0, interp_method="interp1d")
knee_point = kneedle.knee

elapsedtRegTime = time.time() - startTime
print('Total Elapsed Time to find optimal Reg Para: ' + str(elapsedtRegTime))
print('Elapsed Time to find oprimal Reg Para per solve: ' + str(elapsedtRegTime/200))
#knee_point = kneedle.knee_y #

lam_opt = lamLCurveZoom[ np.where(NormLCurveZoom == knee_point)[0][0]]
print('Knee: ', lam_opt) #print('Elbow: ', elbow_point)

elbow_point = kneedle.elbow

lam_opt_elbow = lamLCurveZoom[ np.where(NormLCurveZoom == knee_point)[0][0]]

print('Elbow: ', lam_opt_elbow)

B = (ATA + lam_opt * L)
#x_opt, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
x_opt = lu_solve(LowTri, UpTri, ATy[0::, 0])
LNormOpt = np.linalg.norm( np.matmul(A,x_opt) - y[0::,0])#, ord = 2)
xTLxOpt = np.sqrt(np.matmul(np.matmul(x_opt.T, L), x_opt))





# B = (ATA + minimum[1]/minimum[0] * L)
# x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
# if exitCode != 0:
#     print(exitCode)
SampleNorm = np.linalg.norm( np.matmul(A,np.mean(Results,0 )) - y[0::,0])
SamplexTLx = np.sqrt(np.matmul(np.matmul(np.mean(Results,0 ).T, L), np.mean(Results,0 )))

plt.close('all')
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 12})#,
# {'text.usetex': True})
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.it'] = 'STIXGeneral:italic'
# mpl.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'[0, 114/255, 178/255]
fig, axs = plt.subplots( tight_layout=True,figsize=set_size(245, fraction=fraction))
axs.scatter(NormLCurve,xTLxCurve, zorder = 0, color =  DatCol, s = 1, marker ='s')
#axs.scatter(LNormOpt ,xTLxOpt, zorder = 10, color = 'red', label = 'Opt. Tikh. regularization ')
#axs.scatter(opt_norm ,opt_regNorm, zorder = 10, color = 'red')
axs.scatter(NormRes, xTLxRes, color = ResCol, s = 1.5, marker = "+")# ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')
#axs.scatter(NewNormRes, NewxTLxRes, color = 'red', label = 'MTC RTO method')#, marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')

#axs.scatter(SampleNorm, SamplexTLx, color = 'green', marker = 's', s= 100)

#E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[\mathbf{x}_{\lambda}]$
#axs.axvline(x = knee_point)
axs.scatter(knee_point, kneedle.knee_y, color = regCol, marker = 'v',label = 'max. curvature', s= 25)
axs.scatter(NormMargRes, xTLxMargRes, color = MeanCol, marker = '.', s= 25, label = 'posterior mean' )
#zoom in
x1, x2, y1, y2 = NormLCurveZoom[0], NormLCurveZoom[-31], xTLxCurveZoom[0], xTLxCurveZoom[-1] # specify the limits
axins = axs.inset_axes([0.1,0.05,0.55,0.45])
#axins.scatter(LNormOpt ,xTLxOpt, zorder = 10, color = regCol)

axins.scatter(NormRes, xTLxRes, color = ResCol, label = r'posterior samples ',marker = '+')#,$\mathbf{x} \sim \pi (\mathbf{x}| \mathbf{y}, \mathbf{\theta})$ s = 15)
axins.scatter(NormLCurve,xTLxCurve, color =  DatCol,marker = 's', s= 10, label = 'regularized solutions')

# axins.scatter(LNormOpt, xTLxOpt, color = 'crimson', marker = "s", s =80)[240/255,228/255,66/255]
#axins.annotate(r'E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[\lambda]$ = ' + str('{:.2f}'.format(lam_opt)), (LNormOpt+0.05,xTLxOpt))
#axins.scatter(NewNormRes, NewxTLxRes, color = 'red', label = 'MTC RTO method', s = 10)#, marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')
axins.scatter(knee_point, kneedle.knee_y, color = RegCol, marker = 'v', s = 120)
axins.scatter(NormMargRes, xTLxMargRes, color = MeanCol, marker = '.', s= 100)
axins.set_xlim(x1-0.01, x2-1) # apply the x-limits
#axins.set_ylim(y2,y1)
axins.set_ylim(y2,max(xTLxRes)+0.001) # apply the y-limits (negative gradient)
axins.tick_params(axis = 'x', which = 'both', labelbottom=False, bottom = False)
axins.tick_params(axis = 'y', which = 'both', labelleft=False, left = False)
axins.set_xscale('log')
axins.set_yscale('log')
handles2, labels2 = axins.get_legend_handles_labels()
axs.indicate_inset_zoom(axins, edgecolor="none")


axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel(r'$ \sqrt{\mathbf{x}^T \mathbf{L}\mathbf{x}}$', style='italic')
axs.set_xlabel(r'$|| \mathbf{Ax} - \mathbf{y}||$')
#axs.set_title('L-curve for m=' + str(SpecNumMeas))
mark_inset(axs, axins, loc1=1, loc2=3, fc="none", ec="0.5")

handles, labels = axs.get_legend_handles_labels()

axs.legend(handles = [handles[0],handles[1],handles2[0],handles2[1]],loc = 'upper right',  frameon =True)
plt.savefig('LCurve.png')
#tikzplotlib.save("LCurve.tex")
plt.show()


#tikzplotlib_fix_ncols(fig)
#tikzplotlib.save("LCurve.pgf")
print('bla')

np.savetxt('RegSol.txt',x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')
##


mpl.use('pgf')
mpl.rcParams.update(pgf_params)

fig.savefig('LCurve.pgf', bbox_inches='tight')


## make scatter plot for results
def fitFunc(x,loc, a, scale):
    y = (x - loc) / scale
    return scy.stats.skewnorm.pdf(y, a) / scale


#BinHist = 30#n_bins
lambHist, lambBinEdges = np.histogram(lambdas, bins= PostMeanBinHist, density= True)
#paramsSkew, covs = scy.optimize.curve_fit(skew_norm_pdf,lambBinEdges[1::], lambHist/ np.sum(lambHist), p0 = [np.mean(lambBinEdges[1::]),np.sqrt(np.var(lambdas)),0.01, 1] )#np.mean(new_lamb)+1e3
paramsSkew, covs = scy.optimize.curve_fit(fitFunc,lambBinEdges[1:], lambHist, p0 = [np.mean(lambdas), 1, np.sqrt(np.var(lambdas))], bounds=(0, np.inf))

mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), gridspec_kw={'height_ratios': [3, 1]} )#, dpi = dpi)

#axs[0].scatter(gammas[burnIn::math.ceil(IntAutoLam)+5],deltas[burnIn::math.ceil(IntAutoLam)+5], marker = '.', color = MTCCol)
axs[0].scatter(gammas[burnIn:],deltas[burnIn:], marker = '.', color = MTCCol)

axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[0].set_ylabel(r'the smoothnes parameter $\delta$')
#axs[1].hist(new_lamb,bins=BinHist, color = MTCCol, zorder = 0, density = True)#10)
axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)

#axs[1].plot(lambBinEdges[1::],  skew_norm_pdf(lambBinEdges[1::], *paramsSkew )/np.sum(skew_norm_pdf(lambBinEdges[1::], *paramsSkew )), zorder = 1, color =  gmresCol)#"#009E73")
axs[1].plot(lambBinEdges[1::], fitFunc(lambBinEdges[1::], *paramsSkew )/np.sum(fitFunc(lambBinEdges[1::], *paramsSkew )), zorder = 1, color = 'limegreen')#"#009E73")


axs[1].axvline( lam_opt, color = RegCol,linewidth=2)
axs[1].axvline(minimum[1], zorder = 1,color = gmresCol)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
xlabels = [item.get_text() for item in axs[1].get_xticklabels()]
xticks = axs[1].get_xticks()
xlabels = np.append(xlabels[:-1], [r'$\lambda_R$', r'$\lambda_0$'])
xticks = np.append(xticks[:-1],[lam_opt, minimum[1]])
axs[1].set_xticks( xticks, xlabels)
axs[1].set_xlim(0)
plt.savefig('ScatterplusHisto.png')
plt.show()



#tikzplotlib.save("ScatterplusHisto.pgf")
##


mpl.use('pgf')
mpl.rcParams.update(pgf_params)
fig.savefig('ScatterplusHisto.pgf', bbox_inches='tight')
# mpl.use(defBack)
# mpl.rcParams.update(mpl.rcParamsDefault)




###
plt.close('all')

TrueCol = [50/255,220/255, 0/255]#'#02ab2e'
#Sol = B_inv_Res/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
#Sol= Results[2,:]/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
#x = np.mean(Results,0 )/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
#xerr = np.sqrt(np.var(Results / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst), 0)) / 2
xerr = np.sqrt(np.var(Results,0)/(num_mole *S[ind,0]  * f_broad * 1e-4 * scalingConst)**2)/2
XOPT = x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
#MargX = MargInteg/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
mpl.use(defBack)
#mpl.use("png") bbox_inches='tight'
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.serif"] = "cmr"
fig3, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
 # ax1 and ax2 share y-axis
line3 = ax2.scatter(y, tang_heights_lin, label = r'data', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =7)

# edgecolor = [0, 158/255, 115/255]
#line1 = ax1.plot(VMR_O3,height_values, color = [0, 158/255, 115/255], linewidth = 10, zorder=0)
for n in range(0,paraSamp,1):
#for n in range(0, PostMeanBinHist-1):
    #Sol =  B_inv_Res[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)
    Sol =  Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

    ax1.plot(Sol,height_values,marker= '+',color = ResCol,label = 'posterior samples ', zorder = 1, linewidth = 0.5, markersize = 5)
    with open('Samp' + str(n) +'.txt', 'w') as f:
        for k in range(0, len(Sol)):
            f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
            f.write('\n')

# ax1.plot(Sol, height_values, marker='+', color=ResCol, label='posterior samples ', zorder=4, linewidth=0.5,
# markersize=2, linestyle = 'none')
#$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $' , markerfacecolor = 'none'
ax1.plot(XOPT, height_values, markerfacecolor = 'none', markeredgecolor = RegCol, color = RegCol ,marker='v', zorder=1, label='regularized sol. ', markersize =8, linewidth = 2 )# color="#D55E00"
#line2 = ax1.errorbar(x,height_values,capsize=5, yerr = np.zeros(len(height_values)) ,color = MTCCol,zorder=5,markersize = 5, fmt = 'o',label = r'$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $')#, label = 'MC estimate')
#line3 = ax1.plot(MargX,height_values, markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.', label = 'posterior mean ', markersize =3, linewidth =1)#, markerfacecolor = 'none'
line3 = ax1.errorbar(MargX,height_values,  xerr = 3*np.sqrt(np.diag(MargVar)), markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.', label = 'posterior mean ', markersize =3, linewidth =1)#, markerfacecolor = 'none'

#E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[h(\mathbf{x})]$
# markersize = 6
#line4 = ax1.errorbar(x, height_values,capsize=5, xerr = xerr,color = MTCCol, fmt = 'o', markersize = 5,zorder=5)#, label = 'MC estimate')
#line5 = ax1.errorbar(MargX,height_values, color = MargCol, markeredgecolor= MargCol, capsize=5,  markersize = 6, zorder=3, fmt = 's')
#xerr =MargXErr/2,yerr = np.zeros(len(height_values))



#line5 = ax1.plot(x_opt/(num_mole * S[ind,0] * f_broad * 1e-4 * scalingConst),height_values, color = 'crimson', linewidth = 7, label = 'reg. sol.', zorder=1)

ax1.set_xlabel(r'Ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax2.set_ylabel('(Tangent) Height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Handles = [handles[0], handles[1], handles[2]]
# Labels =  [labels[0], labels[1], labels[2]]
# LegendVertical(ax1, Handles, Labels, 90, XPad=-45, YPad=12)

#legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0],handles[-2],handles[-1]], loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

#plt.ylabel('Height in km')
ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])
#ax2.set_xlim([min(y),max(y)])
#ax1.set_xlim([min(x)-max(xerr)/2,max(x)+max(xerr)/2]) Ozone


#ax2.set_xlabel(r'Spectral radiance in $\frac{\text{W } \text{cm}}{\text{m}^2 \text{ sr}} $',labelpad=10)# color =dataCol,
ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)

plt.savefig('FirstRecRes.png')#, dpi = dpi)
plt.show()

#import tikzplotlib
#tikzplotlib.save("FirstRecRes.pgf")

Samp = Results[::15,:] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

np.savetxt('14Samples.txt', Samp, fmt = '%.15f', delimiter= '\t')
#np.savetxt('GroundTruth.txt',VMR_O3, fmt = '%.15f', delimiter= '\t')
#np.savetxt('RegSolution.txt', XOPT, fmt = '%.15f', delimiter= '\t')
#np.savetxt('PosteriorMean.txt', MargX, fmt = '%.15f', delimiter= '\t')
#np.savetxt('HeigthValues.txt', height_values, fmt = '%.15f', delimiter= '\t')

##
with open('RegSolution.txt', 'w') as f:
    for n in range(0,len(XOPT)):
        f.write('(' + str(XOPT[n]) + ' , ' + str(height_values[n]) + ')')
        f.write('\n')

with open('GroundTruth.txt', 'w') as f:
    for n in range(0,len(VMR_O3)):
        f.write('(' + str(VMR_O3[n]) + ' , ' + str(height_values[n]) + ')')
        f.write('\n')

with open('PosteriorMean.txt', 'w') as f:
    for n in range(0,len(MargX)):
        f.write('(' + str(MargX[n]) + ' , ' + str(height_values[n]) + ')')
        f.write('\n')

with open('SimData.txt', 'w') as f:
    for n in range(0,len(y)):
        f.write('(' + str(y[n,0]) + ' , ' + str(tang_heights_lin[n]) + ')')
        f.write('\n')

##
#fig3.savefig('FirstRecRes.png')#, dpi = dpi)

##

# pgf_params = { "pgf.texsystem": "pdflatex",
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# 'axes.labelsize': 12,  # -> axis labels
# 'legend.fontsize': 12,
# "font.serif": "cmr"}





##

mpl.use('pgf')
mpl.rcParams.update(pgf_params)
fig3.savefig('FirstRecRes.pgf')#, dpi = dpi)




print('bla')