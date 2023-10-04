import matplotlib as mpl
import numpy as np
from scipy.sparse.linalg import gmres

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from matplotlib.ticker import FuncFormatter


def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)


def f(ATy, y, B_inv_A_trans_y):

    return np.matmul(y[0::,0].T, y[0::,0]) - np.matmul(ATy[0::,0].T,B_inv_A_trans_y)


def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T,A) + l * L
    Bu, Bs, Bvh = np.linalg.svd(B)
    # np.log(np.prod(Bs))
    return np.sum(np.log(Bs))


A = np.loadtxt('ForWardMatrix.txt')
L = np.loadtxt('GraphLaplacian.txt')
y = np.loadtxt('dataY.txt').reshape((105,1))
f_func = np.loadtxt('f_func.txt')
g_func = np.loadtxt('g_func.txt')
lam = np.loadtxt('lam.txt')


ATA = A.T @ A
ATy = A.T @ y
lam_mean = 530
lam_std = 252
tol = 1e-6
lamPyT = 552
varPyT = 58293
lam0 = 527


B_0 = ATA + lam0 * L
B_0_inv_A_trans_y, exitCode = gmres(B_0, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_lam0 = f(ATy, y, B_0_inv_A_trans_y)


B_tw = ATA + lamPyT * L
B_tw_inv_A_trans_y, exitCode = gmres(B_tw, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_tW = f(ATy, y, B_tw_inv_A_trans_y)

B_MTC = ATA + lam_mean * L
B_MTC_inv_A_trans_y, exitCode = gmres(B_MTC, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_MTC = f(ATy, y, B_MTC_inv_A_trans_y)

B_MTC_min = ATA + (lam_mean - lam_std/2 ) * L
B_MTC_min_inv_A_trans_y, exitCode = gmres(B_MTC_min, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_MTC_min = f(ATy, y, B_MTC_min_inv_A_trans_y)

B_MTC_max = ATA + (lam_mean + lam_std/2) * L
B_MTC_max_inv_A_trans_y, exitCode = gmres(B_MTC_max, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_MTC_max = f(ATy, y, B_MTC_max_inv_A_trans_y)

xMTC = lam_mean - lam_std/2

B_pyT_min = ATA + (lamPyT - np.sqrt(varPyT)/2) * L
B_pyT_min_inv_A_trans_y, exitCode = gmres(B_pyT_min, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_pyT_min = f(ATy, y, B_pyT_min_inv_A_trans_y)

B_pyT_max = ATA + (lamPyT + np.sqrt(varPyT)/2) * L
B_pyT_max_inv_A_trans_y, exitCode = gmres(B_pyT_max, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_pyT_max = f(ATy, y, B_pyT_max_inv_A_trans_y)

xpyT = lamPyT - np.sqrt(varPyT)/2

B_min = ATA + (lam_mean - lam_std ) * L
B_min_inv_A_trans_y, exitCode = gmres(B_min, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_min = f(ATy, y, B_min_inv_A_trans_y)

B_max = ATA + (lam_mean + lam_std ) * L
B_max_inv_A_trans_y, exitCode = gmres(B_max, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)
f_max = f(ATy, y, B_max_inv_A_trans_y)




fig,axins = plt.subplots()#sharex=True)#tight_layout =  True)
axins.tick_params(labelleft=False, labelright=False, labelbottom=False, bottom = False)
#axs.indicate_inset_zoom(axins, edgecolor="black")
axins.plot(lam,f_func, color = 'blue')
axins.set_ylim(f_min,f_max)
axins.set_xlim( lam_mean - lam_std, lam_mean  + lam_std )# apply the x-limits

axins.scatter(lam0,f_lam0, color = 'green', s= 70, zorder=4)
axins.errorbar(lam_mean ,f_MTC, color = 'red', zorder=5,xerr=lam_std/2, fmt='o')
axins.errorbar(lamPyT,f_tW, xerr=np.sqrt(varPyT)/2, color = 'k', zorder=5,fmt='o')
axins.add_patch(mpl.patches.Rectangle( (xpyT, f_pyT_min), np.sqrt(varPyT), f_pyT_max- f_pyT_min,color="black", alpha = 0.5))
axins.add_patch(mpl.patches.Rectangle((xMTC, f_MTC_min), lam_std, f_MTC_max- f_MTC_min,color="red", alpha = 0.5))
axins.set_yscale('log')
axins.tick_params(labelbottom='off')
#axins.add_patch(mpl.patches.Rectangle( (xpyT, f_pyT_min), np.sqrt(varPyT), f_pyT_max- f_pyT_min,color="black", alpha = 0.5))
axins.set_xlim(lam_mean - lam_std, lam_mean  + lam_std)# apply the x-limits
axins.set_xscale('log')
axin2 = axins.twinx()
axin2.tick_params(labelleft=False, labelright=False, labelbottom=False, bottom = False)
axin2.plot(lam,g_func, color = 'darkred')
axin2.set_ylim(g(A, L, lam_mean - lam_std), g(A, L, lam_mean + lam_std))
axin2.scatter(lam0,g(A, L, lam0 ), color = 'green', s=70, zorder=4)
axin2.errorbar(lamPyT,g(A, L, lamPyT) , xerr=np.sqrt(varPyT)/2, color = 'k', zorder=5, fmt='o')
axin2.errorbar(lam_mean ,g(A, L, lam_mean), xerr=lam_std/2, color = 'red', zorder=5, fmt='o')
axin2.set_xscale('log')
axin2.set_xlim(lam_mean  - lam_std, lam_mean + lam_std )# apply the x-limits
axin2.set_yticklabels([])
axin2.tick_params(labelbottom='off')
#axins.tick_params(left=False, labelleft=False, top=False, labeltop=False,right=False, labelright=False, bottom=False, labelbottom=False)
plt.show()