from importetFunctions import *
import time
import math
from functions import *
from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres, minres
import testReal
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import pandas as pd
from numpy.random import uniform, normal, gamma

# import pickle as pl
# #to open figure
# fig_handle = pl.load(open('/Users/lennart/Downloads/TraceMTCPara.pickle','rb'))
# fig_handle.show()



from matplotlib.ticker import FuncFormatter
def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)

""" analayse forward map without any real data values"""

MinH = 5
MaxH = 95
R = 6371 # earth radiusin km
ObsHeight = 300 # in km
#FakeObsHeight = MaxH + 5







''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas =  105
SpecNumLayers = 46
LayersCore = np.linspace(MinH, MaxH, SpecNumLayers)
layers = np.zeros(SpecNumLayers + 2)
layers[1:-1] =  LayersCore
layers[0]= MinH-3
layers[-1] = MaxH+5
#fing minimum and max angle in radians
MaxAng = np.arcsin((layers[-1]+ R) / (R + ObsHeight))
MinAng = np.arcsin((layers[0] + R) / (R + ObsHeight))
#add zero layers
SpecNumLayers = SpecNumLayers + 2
#meas_ang = MinAng + ((MaxAng - MinAng) * np.exp(coeff * (np.linspace(0, int(spec_num_meas) - 1, int(spec_num_meas) + 1) - (int(spec_num_meas) - 1))))
#meas_ang = np.linspace(min_ang, max_ang, num_meas+ 1)



#find best configuration of layers and num_meas
#so that cond(A) is not inf
#meas_ang = min_ang + ((max_ang - min_ang) * np.exp(coeff * (np.linspace(0, int(num_meas) - 1, int(num_meas)+1) - (int(num_meas) - 1))))
meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas + 1)
A_lin, tang_heights_lin = gen_forward_map(meas_ang[0:-1],layers,ObsHeight,R)
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros(SpecNumMeas)
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = 2*np.sqrt( (layers[-1] + R)**2 - (tang_heights_lin[j] + R )**2 )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T), tot_r)))







#graph Laplacian
neigbours = np.zeros((len(layers)-1,2))
neigbours[0] = np.nan, 1
neigbours[-1] = len(layers)-3, np.nan
for i in range(1,len(layers)-2):
    neigbours[i] = i-1, i+1
L = generate_L(neigbours)


#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
#/Users/lennart/PycharmProjects/firstModelCheck
#obs_height= 300 #in km
#filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml' #/home/lennartgolks/Python /Users/lennart/PycharmProjects/firstModelCheck/tropical.O3.xml
filename = 'tropical.O3.xml'

VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
d_height = layers[1::] - layers[0:-1]

pressure_values = pressure_values * 1e-1 # in cgs
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

# plt.plot(pressure_values, height_values)
# plt.plot(VMR_O3, layers)
# plt.show()

temp_values = get_temp_values(layers[0:-1] + d_height/2 )
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
#take linear
num_mole = (pressure_values / (constants.Boltzmann * 1e7  * temp_values))
theta = (num_mole * w_cross * VMR_O3 * Source)
Ax = np.matmul(A_lin, theta)
#convolve measurements and add noise
y = add_noise(Ax, 0.01)
ATy = np.matmul(A_lin.T, y)

np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')


"""start the mtc algo with first guesses of noise and lumping const delta"""

tol = 1e-4
vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MargPost(params):#, coeff):

    gamma = params[0]
    delta = params[1]
    # ATA_lin = coeff[0]
    # L = coeff[1]
    if delta < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers-1

    Bp= ATA_lin + delta/gamma * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A_lin, L,  delta/gamma)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(delta) + 0.5 * G + 0.5 * gamma * F + 1e-4 * ( delta + gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MargPost, [1/np.var(y),1/(2*np.mean(vari))])

#minimum = optimize.minimize(MargPost, [1/np.var(y), 1/(2*np.mean(vari))], args = [ATA_lin, L])
print(minimum)
print(minimum[1]/minimum[0])

'''L-curve refularoization
'''
tol = 1e-4
lamLCurve = np.logspace(-20,20,500)
#lamLCurve = np.linspace(1e-1,1e4,300)

NormLCurve = np.zeros(len(lamLCurve))
xTLxCurve = np.zeros(len(lamLCurve))
for i in range(len(lamLCurve)):
    B = (ATA_lin + lamLCurve[i] * L)

    x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    NormLCurve[i] = np.linalg.norm( np.matmul(A_lin,x) - y[0::,0])
    #NormLCurve[i] =np.linalg.norm( np.matmul(A_lin,x))
    #NormLCurve[i] = np.sqrt(np.sum((np.matmul(A_lin, x) - y)**2))
    xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))
    #xTLxCurve[i] = np.linalg.norm(np.matmul(L,x))

A_linu, A_lins, A_linvh = csvd(A_lin)
#reg_c = l_cuve(A_linu, A_lins, y[0::,0], plotit=True)
#reg_c = l_corner(NormLCurve,xTLxCurve,lamLCurve,A_linu,A_lins,y[0::,0])
#B = (ATA_lin + reg_c * L)
B = (ATA_lin + minimum[1]/minimum[0] * L)

x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

lamLCurveZoom = np.logspace(4,10,500)
NormLCurveZoom = np.zeros(len(lamLCurve))
xTLxCurveZoom = np.zeros(len(lamLCurve))
for i in range(len(lamLCurveZoom)):
    B = (ATA_lin + lamLCurveZoom[i] * L)

    x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    NormLCurveZoom[i] = np.linalg.norm( np.matmul(A_lin,x) - y[0::,0])
    xTLxCurveZoom[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))

A_linu, A_lins, A_linvh = csvd(A_lin)
B = (ATA_lin + minimum[1]/minimum[0] * L)
x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

fig, axs = plt.subplots( 1,1, tight_layout=True)
axs.scatter(NormLCurve,xTLxCurve, zorder = 0)
axs.scatter(np.linalg.norm(np.matmul(A_lin, x) - y[0::, 0]),np.sqrt(np.matmul(np.matmul(x.T, L), x)))
#axs.annotate('$\lambda_0$ = ' + str(math.ceil(minimum[1]/minimum[0])), (np.linalg.norm(np.matmul(A_lin, x) - y[0::, 0]),np.sqrt(np.matmul(np.matmul(x.T, L), x))))
#axs.annotate('$\lambda$ = 1e' + str(orderOfMagnitude(lamLCurve[0])), (NormLCurve[0],xTLxCurve[0]))
#axs.annotate('$\lambda$ = 1e' + str(orderOfMagnitude(lamLCurve[-1])), (NormLCurve[-1],xTLxCurve[-1]))
import mpl_toolkits as mplT
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes, inset_axes
#axins = zoomed_inset_axes(axs,zoom = 1,loc='lower left')

#axins.scatter(NormLCurveZoom,xTLxCurveZoom)#,'o', color='black')
#axins.plot(MTCnorms[:,0], MTCnorms[:,1], marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')
#axins.scatter(norm_data, norm_f)
x1, x2, y1, y2 = NormLCurveZoom[0], NormLCurveZoom[-1], xTLxCurveZoom[0], xTLxCurveZoom[-1] # specify the limits
#axins = mplT.axes_grid1.inset_locator.inset_axes( parent_axes = axs,  bbox_transform=axs.transAxes, bbox_to_anchor =(0.05,0.05,0.75,0.75) , width = '100%' , height = '100%')#,  loc= 'lower left')
axins = axs.inset_axes(  [0.01,0.05,0.75,0.75])#,  loc= 'lower left')

axins.scatter(NormLCurveZoom,xTLxCurveZoom)#,'o', color='black')
axins.set_xscale('log')
axins.set_yscale('log')
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y2, y1) # apply the y-limits (negative gradient)
axins.set_xticklabels([])
axins.set_yticklabels([])
axs.indicate_inset_zoom(axins, edgecolor="black")
#axins.tick_params( bottom=False, left=False, labelbottom = False, labelleft = False)
#mark_inset(axs, axins, loc1=2, loc2=4 ,fc="none", ec="0.5")

axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel(r'$\sqrt{x^T L x}$')
axs.set_xlabel(r'$|| Ax - y ||$')
axs.set_title('L-curve for m=' + str(SpecNumMeas))
plt.savefig('LCurve.png')
plt.show()


np.savetxt('LCurve.txt', np.vstack((NormLCurve,xTLxCurve, lamLCurve)).T, header = 'Norm ||Ax - y|| \t sqrt(x.T L x) \t lambdas', fmt = '%.15f \t %.15f \t %.15f')

np.savetxt('A_lin.txt', A_lin, header = 'linear forward model A', fmt = '%.15f', delimiter= '\t')






'''do the sampling'''

number_samples = 10000
gammas = np.zeros(number_samples)
deltas = np.zeros(number_samples)
lambdas = np.zeros(number_samples)

#inintialize sample
gammas[0] = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
deltas[0] =  minimum[1] #0.275#1/(2*np.mean(vari))0.1#
lambdas[0] = deltas[0]/gammas[0]

ATy = np.matmul(A_lin.T, y)

B = (ATA_lin + lambdas[0] * L)

B_inv_L = np.zeros(np.shape(B))
for i in range(len(B)):
    B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)



k = 0
wLam = 2e4
wgam = 1e-5
wdelt = 1e-1
betaG = 1e-4
betaD = 1e-4
alphaG = 1
alphaD = 1
rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambdas[0]

startTime = time.time()
for t in range(number_samples-1):


    # # draw new lambda
    lam_p = normal(lambdas[t], wLam)

    while lam_p < 0:
            lam_p = normal(lambdas[t], wLam)

    delta_lam = lam_p - lambdas[t]
    delta_f = f_tayl(delta_lam, B_inv_A_trans_y, ATy[0::, 0], B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4,B_inv_L_5)
    delta_g = g_tayl(delta_lam, B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4, B_inv_L_5)

    log_MH_ratio = ((SpecNumLayers - 1)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

    #accept or rejeict new lam_p
    u = uniform()
    if np.log(u) <= log_MH_ratio:
    #accept
        k = k + 1
        lambdas[t + 1] = lam_p
        #only calc when lambda is updated
        #B = (ATA_lin + lambdas[t+1] * L)
        B = (ATA_lin + lam_p * L)
        B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        # if exitCode != 0:
        #     print(exitCode)
        # CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
        # print(np.allclose(CheckB_inv_ATy, ATy[0::, 0], rtol=relative_tol_ATy))

        B_inv_L = np.zeros(np.shape(B))
        for i in range(len(B)):
            B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
            # if exitCode != 0:
            #    print(exitCode)
        #CheckB_inv_L = np.matmul(B, B_inv_L)
        #print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

        B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
        B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
        B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
        B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)

        rate = f(ATy, y, B_inv_A_trans_y)/2 + betaG + betaD * lam_p#lambdas[t+1]

    else:
        #rejcet
        lambdas[t + 1] = np.copy(lambdas[t])

    #draw gamma with a gibs step
    shape =  (SpecNumLayers - 1)/ 2 + alphaD + alphaG


    gammas[t+1] = np.random.gamma(shape, scale = 1/rate)

    deltas[t+1] = lambdas[t+1] * gammas[t+1]



elapsed = time.time() - startTime
print('acceptance ratio: ' + str(k/number_samples))
print(np.mean(gammas[30::]))
print(np.mean(deltas[30::]))
print(np.mean(lambdas[30::]))
np.savetxt('samples.txt', np.vstack((gammas, deltas, lambdas)).T, header = 'Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed) + ' \n gammas \t deltas \t lambdas \n ', fmt = '%.15f \t %.15f \t %.15f')




print('bla')