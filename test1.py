import plotly.graph_objects as go
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def get_data(filename: str):
    tree = ET.parse(filename)

    root = tree.getroot()

    pressure_values = root[0][0].text
    pressure_values = list(pressure_values.split("\n"))
    pressure_values = list(map(float, pressure_values[1:-1]))

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

    height_values = list(map(int, height_values))

    vmr_o3 = root[0][3].text
    vmr_o3 = list(vmr_o3.split("\n"))
    vmr_o3 = list(map(float, vmr_o3[1:-1]))

    return vmr_o3, height_values, pressure_values;


# frequnecy in GHz
# returns absorptions coeff not dependend on Temperature
def get_absorption(frequency: float, filedir: list):
    data = []
    wavenumbers = []
    temp = []
    for files in filedir:
        #print(files)
        my_data = pd.read_csv(files)
        # print(len(my_data.values))

        print(my_data.axes[1][0][49:55])
        temp.append(my_data.axes[1][0][49:55])
        current_data = []
        for data_sets in my_data.values[1:]:
            current = list(data_sets[0].split(" "))
            current_data.extend(list(map(float, current[1:])))

        data.append(current_data)

        wavenumbers.append( np.round( np.linspace(2890100, 4099900, len(current_data)) * 29.9792458 * 1e-6,3) )

    index = np.where(wavenumbers[1] == frequency)
    absorption_coeff = [data[i][index[0][0]] for i in range(0, len(wavenumbers))]
    max_absorption = [ max(data[i]) for i in range(0, len(wavenumbers)) ]
    max_index = [ data[i].index(max_absorption[i]) for i in range(0, len(wavenumbers))]
    max_frequency = [ wavenumbers[i][ max_index[i] ] for i in range(0, len(wavenumbers)) ]

    #[46:53]

    return absorption_coeff, max_absorption, max_frequency, temp;


def calc_kernel(height_values, tangent_ind, VMR_O3, pressure_values, sigma):
    R = 6371
    h_tangent = height_values[tangent_ind] - 0.1

    T_ref = 288.15  # in kelvin
    k_b = 1.4 * 1e-23
    numb_dens = [p_s / k_b * T_ref for p_s in pressure_values]
    height_values = height_values[tangent_ind:]

    v_transf = [np.sqrt((heights + R) ** 2 - (h_tangent + R) ** 2) for heights in height_values]
    v_transf = list(np.round(v_transf))

    VMR_O3 = VMR_O3[tangent_ind:]
    numb_dens = numb_dens[tangent_ind:]

    k = [(VMRs * 1e6 * sigma * n_s * (heights + R) / v_s) for (heights, VMRs, v_s, n_s) in
         zip(height_values, VMR_O3, v_transf, numb_dens)]

    k_sum = [sum(k[0:i]) for i in range(0, len(k))]

    k_sum_bef = [sum(k[i:]) for i in range(0, len(k))]

    k_sum_total = [sum(k_sum[0:i]) for i in range(0, len(k_sum))]

    v_transf_1 = [np.log((heights + R) / v_s) for (v_s, heights) in zip(v_transf, height_values)]

    # v_transf_sum = [ sum( v_transf_1[0:i] ) for i in range(0, len(v_transf_1) ) ]

    # np.log(1+ np.exp(- k_sum[-1])) aprrox 0
    kernel = [np.log(1 + np.exp(- k_sum[-1])) - k_s + v_s for (v_s, k_s) in zip(v_transf_1, k_sum)]

    return kernel;


def make_figure(height_values, VMR_O3, pressure_values, sigma):


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
                y=calc_kernel(height_values, ind, VMR_O3, pressure_values, sigma)
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
