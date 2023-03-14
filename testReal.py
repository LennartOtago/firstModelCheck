import plotly.graph_objects as go
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def get_data(filename: str, obs_height):
    tree = ET.parse(filename)

    root = tree.getroot()

    pressure_values = root[0][0].text
    pressure_values = list(pressure_values.split("\n"))
    pressure_values = np.array( pressure_values[1:-1], dtype= 'float')

    stand_temp_values = list([288, 216, 216, 228, 270, 270, 214])
    # ref pressure in Pa
    ref_pressure = list([101325, 22632, 5474, 868, 110, 66, 3])
    # ref_height in m
    ref_height = list([0, 11000, 20000, 32000, 47000, 51000, 71000])
    R_const = 8.3
    gravity = 9.8
    # molar mas of air (reference?)
    molar_mass = 0.03
    height_values = [None] * len(pressure_values)
    # get height
    b = 0
    for idx, pressure in enumerate(pressure_values):
        if pressure > ref_pressure[b]:

            height_values[idx] = -(np.log(pressure / ref_pressure[b]) * R_const * stand_temp_values[b]) / (
                    gravity * molar_mass) + ref_height[b]
        else:
            b = b + 1
            if b > 6:
                b = b - 1

            height_values[idx] = -(np.log(pressure / ref_pressure[b]) * R_const * stand_temp_values[b]) / (
                    gravity * molar_mass) + ref_height[b]

    height_values = np.array(height_values, dtype= 'float')


    vmr_o3 = root[0][3].text
    vmr_o3 = list(vmr_o3.split("\n"))
    vmr_o3 = np.array(vmr_o3[1:-1], dtype= 'float')

    #append so that full model from seelevel to observer
    vmr_o3 =  np.append(0, np.append(vmr_o3,0))
    height_values =  np.append(0, np.append(height_values,obs_height))
    pressure_values = np.append(ref_pressure[0], np.append(pressure_values,0 ))
    return vmr_o3.reshape((len(height_values),1)), height_values, pressure_values.reshape((len(height_values),1));


# frequnecy in GHz
# returns absorptions coeff not dependend on Temperature
def get_absorption(filedir: list):
    data = []
    wavenumbers = []
    temp = []
    for files in filedir:
        #print(files)
        my_data = pd.read_csv(files)
        # print(len(my_data.values))

        #print(my_data.axes[1][0][49:55])
        temp.append(my_data.axes[1][0][49:55])
        current_data = []
        for data_sets in my_data.values[1:]:
            current = list(data_sets[0].split(" "))
            current_data.extend(list(map(float, current[1:])))

        data.append(current_data)

        wavenumbers.append( np.round( np.linspace(2890100, 4099900, len(current_data) ) * 3 * 1e8,9) )

    #index = np.where(wavenumbers[1] == frequency)
    #absorption_coeff = [data[i][index[0][0]] for i in range(0, len(wavenumbers))]
    #max_absorption = [ max(data[i]) for i in range(0, len(wavenumbers)) ]
    #max_index = [ data[i].index(max_absorption[i]) for i in range(0, len(wavenumbers))]
    #max_frequency = [ wavenumbers[i][ max_index[i] ] for i in range(0, len(wavenumbers)) ]

    #[46:53]
    temp = list( map(float, temp) )

    absorption_coeff = []
    for set in data:
        absorption_coeff.append(list(map(float, set)))

    return np.array(absorption_coeff), np.array(temp), np.array(wavenumbers);


def calc_kernel(height_values, ind_limb, h_inf, VMR_O3, pressure_values, temp_values, absorption_coeff):
    R = 6371
    X = list( map( int, np.round(np.linspace( height_values[-1], h_inf, 100)) ) )
    height_values.extend( X )
    #h_limb = height_values[ind_limb]
    h_tangent = height_values[0] - 0.1

    pressure_values.extend([0.0] * len(X) )
    absorption_coeff.extend([0.0] * len(X) )
    VMR_O3.extend([0.0] * len(X) )

    T_ref = 288.15  # in kelvin
    k_b = 1.4 * 1e-23
    numb_dens = [p_s / k_b * T_ref for p_s in pressure_values]


    v_transf = [np.sqrt((heights + R) ** 2 - (h_tangent + R) ** 2) for heights in height_values]
    v_transf = list(np.round(v_transf))

    absorption_coeff = list(map(float, absorption_coeff))
    sigma = [0] * len(pressure_values)
    ind = 0
    for i in range(0, len(temp_values)):
        if temp_values[i] == 216:
            sigma[i] = absorption_coeff[0] * 1e-4
        elif temp_values[i] == 228:
            sigma[i] = absorption_coeff[1] * 1e-4
        elif temp_values[i] == 270:
            sigma[i] = absorption_coeff[2] * 1e-4
        elif temp_values[i] == 214:
            sigma[i] = absorption_coeff[3] * 1e-4

    #1e-6
    k = [(VMRs  * sigmas * n_s * (heights + R) / v_s) for (heights, sigmas, VMRs, v_s, n_s) in
         zip(height_values, sigma, VMR_O3, v_transf, numb_dens)]

    #k_sum = [sum(k[0:i]) for i in range(0, len(k))]
    k[0] = 1
    k_sum_bef = [sum(k[i:]) for i in range(0, len(k))]

    k_sum_after = [sum(k[0:i]) for i in range(0, len(k))]

    v_transf_1 = [np.log((heights + R) / v_s) for (v_s, heights) in zip(v_transf, height_values)]

    # v_transf_sum = [ sum( v_transf_1[0:i] ) for i in range(0, len(v_transf_1) ) ]

    # np.log(1+ np.exp(- k_sum[-1])) aprrox 0
    kernel_before = [ np.exp( - k_s ) * ( (heights + R) / v_s) for (heights, v_s, k_s) in zip(height_values, v_transf , k_sum_bef)]

    kernel_after = [  np.exp( - k_sum_bef[1] ) * np.exp( - k_s ) * ( (heights + R) / v_s) for (heights, v_s, k_s) in zip(height_values, v_transf, k_sum_after)]

    kernel_before[ind_limb + 1:] = 0

    kernel = kernel_before + kernel_after

    return kernel;


def make_figure(h_tang, ind_limb, h_inf, VMR_O3, pressure_values, temp_values, absorption_coeff):

    height_values = np.linspace(h_tang, h_inf, 1000)
    # Create figure
    fig = go.Figure()
    change_tags = height_values
    change_values = range(0, len(height_values) )
    # Add traces, one for each slider step
    for ind in change_values:
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(height_values[ind]),
                x=height_values[ind:],
                y=calc_kernel(h_tang, ind_limb, h_inf, VMR_O3, pressure_values, temp_values, absorption_coeff)
            ))

    # Make 10th trace visible
    fig.data[10].visible = True
    k = np.round(change_values, 3)

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to tangent height: " + str(change_tags[i]) + "km"}],
            label=str(change_tags[i]),  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "h_t= ", "suffix": ""},
        pad={"b": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        yaxis_title="log of kernel",
        xaxis_title="height in km",
        title="tangent height in km"
    )

    fig.show()

    fig.write_html('test1.html')



def make_absorp_fig(frequencies, VMR_O3, absorption_coeff, temp_values, height_values, pressure_values):
    sigma = [None] * len(temp_values)
    sigma = [None] * len(temp_values)
    T_ref = 288.15  # in kelvin
    k_b = 1.4 * 1e-23  # [m^2 kg / ( s^2 K )]
    numb_dens = [p_s / k_b * T_ref for p_s in pressure_values]
    # [1/m^3]
    # Create figure
    fig = go.Figure()
    f_values = frequencies[1][::10]

    # Add traces, one for each slider step
    for ind in range(0, len(f_values)):

        coeff = []
        for i in range(0, len(absorption_coeff)):
            coeff.append(absorption_coeff[i][ind])

        for i in range(0, len(temp_values)):
            if temp_values[i] == 216:
                sigma[i] = coeff[0] * 1e-4
            elif temp_values[i] == 228:
                sigma[i] = coeff[1] * 1e-4
            elif temp_values[i] == 270:
                sigma[i] = coeff[2] * 1e-4
            elif temp_values[i] == 214:
                sigma[i] = coeff[3] * 1e-4

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(f_values[ind]),
                x=[VMR_s * sigmas * numbs for sigmas, numbs, VMR_s in zip(sigma, numb_dens, VMR_O3)],
                y=height_values
            ))

    # Make 10th trace visible
    fig.data[10].visible = True
    freqs = np.round(f_values, 1)

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to frequency: " + str(freqs[i]) + "Hz"}],
            label=str(freqs[i]),  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "f= ", "suffix": ""},
        pad={"b": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        xaxis_title="absorption per m",
        yaxis_title="height in km",
        title="f in Hz"
    )

    fig.show()
