from scipy.special import wofz
import numpy as np
from scipy import constants
import plotly.graph_objects as go

#voigt function as real part of Faddeeva function
def V(x, sigma, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    #sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / (sigma * np.sqrt(2*np.pi))

def Lorenz(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))

def generate_L(neigbours):
    siz = int(np.size(neigbours,0))
    neig = np.size(neigbours,1)
    L = np.zeros((siz,siz))

    for i in range(0,siz):
        L[i,i] = 2
        for j in range(0,neig):
            if ~np.isnan(neigbours[i,j]):
                L[i,int(neigbours[i,j])] = -1
    return L

def get_temp_values(height_values):
    """ based on the ISA model see omnicalculator.com/physics/altitude-temperature"""
    temp_values2 = np.zeros(len(height_values))
    temp_values2[0] = 15 - (height_values[0] - 0) * 6.5  + 273.15
    ###calculate temp values
    for i in range(1,len(height_values)):
        if 0 < height_values[i] < 11:
            temp_values2[i] = temp_values2[i - 1] - (height_values[i] - height_values[i - 1]) * 6.5
        if 11 < height_values[i] < 13:
            temp_values2[i] = -55 + 273.15
        if 13 < height_values[i] < 48:
            temp_values2[i] = temp_values2[i-1] + (height_values[i] - height_values[i-1]) * 1.6
        if 48 < height_values[i] < 51:
            temp_values2[i] = -1 + 273.15
        if 51 < height_values[i] < 86:
            temp_values2[i] = temp_values2[i - 1] - (height_values[i] - height_values[i - 1]) * 2.5
        if 85 < height_values[i]:
            temp_values2[i] = -87  + 273.15


    return temp_values2.reshape((len(height_values),1))

def gen_measurement(meas_ang, layers, w_cross, VMR_O3, P ,T, Source, obs_height = 300):
    '''generates Measurement given the input measurement angels and depending on the model layers in km
    obs_height is given in km
    '''


    R = 6371
    # get tangent height for each measurement layers[0:-1] #
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    # get dr's for measurements of different layers
    A_height = np.zeros((num_meas, len(layers) - 1))
    t = 1
    for m in range(0, num_meas):

        while (layers[t-1] <= tang_height[m] < layers[t]) == 0:
            t += 1
        # first dr
        A_height[m, t - 1] = np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            # A_height[j,i] =  (height_values[j+i+1] + R)/np.sqrt((height_values[j+i+1]+ R)**2 - (height_values[j]+ R)**2 ) * d_height[j+i]
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]
    #calc mearuements

    R_gas = constants.Avogadro  * constants.Boltzmann * 1e7  # in ..cm^3
    # caculate number of molecules in one cm^3
    num_mole = (P / (constants.Boltzmann * 1e7  * T))

    THETA = (num_mole * w_cross * VMR_O3 * Source)
    #2 * A_height * 1e5....2 * np.matmul(A_height*1e5, THETA[1::]) A_height in km
    #* 1e5 converts to cm
    return  2 * np.matmul(A_height, THETA[1::]), 2*A_height, THETA[1::] , tang_height

def add_noise(Ax, percent, max_value):
    return Ax + np.random.normal(0, percent * max_value, (len(Ax),1))

'''
we plot left singular vectors wighted with the singular value
for symmetric sqaure matrix
'''
def plot_svd(A):
    Au, As, Avh = np.linalg.svd(A)

    # Create figure
    fig = go.Figure()
    #k_values = int(np.linspace(0, len(As)-1, len(As)))

    # Add traces, one for each slider step
    for k in range(0,len(As)):


        fig.add_trace(
            go.Scatter(
                visible=False,
                line= dict(color="#00CED1", width=6),
                name= f"index = {k}",
                x= np.linspace(0,len(Au[:,k])-1,len(Au[:,k])),
                y= As[k] * Au[:,k]
            ))


    # Make 10th trace visible
    fig.data[10].visible = True
    k = np.linspace(0,len(As)-1,len(As))

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to index: " + str(k[i]) + "/m"}],
            label=str(k[i]),  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "k= ", "suffix": ""},
        pad={"b": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="Left singlar Vectors weighted with singular values",
        xaxis_title = "sU"
    )

    fig.show()

    fig.write_html('SVD.html')