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
frequency = 117.389 #in GHz 100

absorption_coeff , max_absorption, max_frequency, temp = test1.get_absorption(frequency, filedir)
#in cm^2/molecule
sigma =  np.mean(absorption_coeff) * 1e-4

test1.make_figure(height_values, VMR_O3, pressure_values, sigma)