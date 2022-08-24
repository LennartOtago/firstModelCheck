import numpy as np

import test2
import test1
import matplotlib.pyplot as plt
import glob

#test2.make_figure()



filename = '/home/lennartgolks/Python/firstModelCheck/tropical.O3.xml'
VMR_O3, height_values, pressure_values = test1.get_data(filename)

#VMR_O3 = VMR_O3 #* 1e6 #get rid of ppm convert to molecule/m^3

filedir = glob.glob('/home/lennartgolks/Python/firstModelCheck/HITRAN_o3_data/*.xsc')
frequency = 117.389 #in GHz

absorption_coeff , max_absorption, max_frequency = test1.get_absorption(frequency, filedir)
#in cm^2/molecule
sigma =  np.mean(absorption_coeff) * 1e-4

tangent_ind = 0
print(type(VMR_O3))
#print(VMR_O3[3])
#VMR_O3[3] = 0
#[ print(i) for i in range(0, tangent_ind) ]
for i in range(0, tangent_ind):
    VMR_O3[i] = 0
R = 6371
h_tangent = height_values[tangent_ind] - 0.1
h_max = height_values[-1]

r_t = np.sqrt((h_max + R) ** 2 - (h_tangent + R) ** 2)

v_transf = [np.sqrt((heights + R) ** 2 - (h_tangent + R) ** 2) for heights in height_values ]
v_transf= np.round(v_transf)
x = height_values

k = [  ( VMRs  * 1e6 * sigma * ( heights + R )  / v_s ) for (heights, VMRs, v_s) in zip(height_values, VMR_O3, v_transf) ]

k_sum = [ sum(k[0:i]) for i in range(0, len(k) ) ]

before = [ np.exp(k_s ) * ( heights + R ) / v_s for (heights, v_s, k_s) in zip(height_values, v_transf, k_sum) ]
after = [ np.exp( (k_sum[-1] -k_s) ) * ( heights + R ) / v_s  for (heights, v_s, k_s) in zip(height_values, v_transf, k_sum) ]

res = [y_1 + y_2 for (y_1, y_2) in zip(before, after)]

x = height_values

plt.plot(x[0:10], res[0:10])
plt.show()