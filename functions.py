from scipy.special import wofz
import numpy as np
from scipy import constants
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math
from scipy.sparse.linalg import gmres


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


# voigt function as real part of Faddeeva function
def V(x, sigma, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    # sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / (sigma * np.sqrt(2 * np.pi))


def Lorenz(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x ** 2 + gamma ** 2)


def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha \
           * np.exp(-(x / alpha) ** 2 * np.log(2))


def generate_L(neigbours):
    siz = int(np.size(neigbours, 0))
    neig = np.size(neigbours, 1)
    L = np.zeros((siz, siz))

    for i in range(0, siz):
        L[i, i] = 2
        for j in range(0, neig):
            if ~np.isnan(neigbours[i, j]):
                L[i, int(neigbours[i, j])] = -1
    return L


def get_temp_values(height_values):
    """ based on the ISA model see omnicalculator.com/physics/altitude-temperature"""
    temp_values2 = np.zeros(len(height_values))
    temp_values2[0] = 15 - (height_values[0] - 0) * 6.5 + 273.15
    ###calculate temp values
    for i in range(1, len(height_values)):
        if 0 < height_values[i] < 11:
            temp_values2[i] = temp_values2[i - 1] - (height_values[i] - height_values[i - 1]) * 6.5
        if 11 < height_values[i] < 13:
            temp_values2[i] = -55 + 273.15
        if 13 < height_values[i] < 48:
            temp_values2[i] = temp_values2[i - 1] + (height_values[i] - height_values[i - 1]) * 1.6
        if 48 < height_values[i] < 51:
            temp_values2[i] = -1 + 273.15
        if 51 < height_values[i] < 86:
            temp_values2[i] = temp_values2[i - 1] - (height_values[i] - height_values[i - 1]) * 2.5
        if 85 < height_values[i]:
            temp_values2[i] = -87 + 273.15

    return temp_values2.reshape((len(height_values), 1))


def gen_measurement(meas_ang, layers, w_cross, VMR_O3, P, T, Source, obs_height=300):
    '''generates Measurement given the input measurement angels and depending on the model layers in km
    obs_height is given in km
    '''

    # exclude first layer at h = 0  and
    # last layer at h = Observer
    min_ind = 1
    max_ind = -1
    layers = layers[min_ind: max_ind]

    w_cross = w_cross[min_ind: max_ind - 1]
    VMR_O3 = VMR_O3[min_ind: max_ind - 1]
    Source = Source[min_ind: max_ind - 1]
    P = P[min_ind: max_ind - 1]
    T = T[min_ind: max_ind - 1]

    R = 6371
    # get tangent height for each measurement layers[0:-1] #
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    # get dr's for measurements of different layers
    A_height = np.zeros((num_meas, len(layers) - 1))
    t = 0
    for m in range(0, num_meas):

        while layers[t] <= tang_height[m]:
            t += 1
        print(t)
        # first dr
        A_height[m, t - 1] = np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            # A_height[j,i] =  (height_values[j+i+1] + R)/np.sqrt((height_values[j+i+1]+ R)**2 - (height_values[j]+ R)**2 ) * d_height[j+i]
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]
    # calc mearuements

    R_gas = constants.Avogadro * constants.Boltzmann * 1e7  # in ..cm^3
    # caculate number of molecules in one cm^3
    num_mole = (P / (constants.Boltzmann * 1e7 * T))

    THETA = (num_mole * w_cross * VMR_O3 * Source)
    # 2 * A_height * 1e5....2 * np.matmul(A_height*1e5, THETA[1::]) A_height in km
    # * 1e5 converts to cm
    return 2 * np.matmul(A_height, THETA), 2 * A_height, THETA, tang_height


def add_noise(Ax, percent, max_value):
    return Ax + np.random.normal(0, percent * max_value, (len(Ax), 1))


def plot_svd(ATA, height_values):
    '''
    we plot left singular vectors wighted with the singular value
    for symmetric sqaure matrix
    '''
    ATAu, ATAs, ATAvh = np.linalg.svd(ATA)

    # Create figure
    fig = go.Figure()
    # k_values = int(np.linspace(0, len(As)-1, len(As)))

    # Add traces, one for each slider step
    for k in range(0, len(ATAs)):
        x = height_values  # np.linspace(0, len(Au[:, k]) - 1, len(Au[:, k]))
        y = ATAu[:, k]  # *As[k]
        df = pd.DataFrame(dict(x=x, y=y))

        fig.add_trace(
            go.Scatter(
                x=df['x'],
                y=df['y'],
                visible=False,
                line=dict(color="#00CED1", width=6),
                name=f"index = {k}"
            )
        )

    # Make 10th trace visible
    fig.data[10].visible = True
    k = np.linspace(0, len(ATAs) - 1, len(ATAs))

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider at tangent model layer: " + str(height_values[i]) + " in m"}],
            label=str(height_values[i]),  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "index= ", "suffix": ""},
        pad={"b": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="Left Singlar Vectors weighted with Singular Values",
        xaxis_title="height values"
    )
    fig.update_yaxes(range=[np.min(ATAu), np.max(ATAu)])

    fig.show()

    fig.write_html('SVD.html')
    return ATAu, ATAs, ATAvh


def gen_forward_map(meas_ang, layers, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)

    num_meas = len(tang_height)

    A_height = np.zeros((num_meas, len(layers) - 1))
    t = 0
    for m in range(0, num_meas):

        while layers[t] <= tang_height[m]:
            t += 1
        # first dr
        A_height[m, t - 1] = np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]

    return 2 * A_height, tang_height


def f(A, y, L, l):
    """ calclulate taylor series of f"""
    # B^-1  A^T y
    B = np.matmul(A.T,A) + l * L

    A_trans_y = np.matmul(A.T, y)
    B_inv_A_trans_y, exitCode = gmres(B, A_trans_y[0::, 0], tol=1e-6, restart=25)
    #print(exitCode)

    CheckB_A_trans_y = np.matmul(B, B_inv_A_trans_y)
    print(np.allclose(CheckB_A_trans_y.T, A_trans_y[0::, 0], atol=1e-3))

    return np.matmul(y.T, y)- np.matmul(np.matmul(y.T,A),B_inv_A_trans_y)


def g(A, L, l):
    """ calclulate taylor series of g"""
    B = np.matmul(A.T,A) + l * L
    B_inv_L = np.zeros(np.shape(B))
    for i in range(len(B)):
        B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=1e-5, restart=25)
        #print(exitCode)

    CheckB_inv_L = np.matmul(B, B_inv_L)
    print(np.allclose(CheckB_inv_L, L, atol=1e-3))

    num_z = 4
    trace_Bs = np.zeros(num_z)
    for k in range(num_z):
        z = np.random.randint(2, size=len(B))
        z[z == 0] = -1
        trace_Bs[k] = np.matmul(z.T, np.matmul(B_inv_L, z))

    return  np.mean(trace_Bs)
