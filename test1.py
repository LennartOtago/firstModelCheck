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
    for files in filedir:
        #print(files)
        my_data = pd.read_csv(files)
        # print(len(my_data.values))
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

    return absorption_coeff, max_absorption, max_frequency;


def calc_kernel(height_values, tangent_height, vmr_o3):
    # get all important values above tangent height

    # calculate kernel for specific species and absoprption cross section and tangent height

    kernel = 0
    return kernel;


def make_figure():
    R = 6371
    h_t = R + 15
    h_max = R + 90

    r_t = np.sqrt((h_max + R) ** 2 - (h_t + R) ** 2)
    v_max = (h_max + R) ** 2 - (h_t + R) ** 2

    h_values = np.linspace(h_t + .01, h_max, 1000)

    # Create figure
    fig = go.Figure()
    k_values = np.linspace(0, 0.1, 100)
    # Add traces, one for each slider step
    for k in k_values:
        # h = np.sqrt(k/2 * ( ( h_values - R)**2 - (h_t - R)**2 ) )
        v = np.sqrt((h_values + R) ** 2 - (h_t + R) ** 2)
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(k),
                x=h_values - R,
                # k = ,
                y=np.exp(-k * r_t) * (h_values + R) / v * (np.exp(- k * v) + np.exp(k * v))
            ))

    # Make 10th trace visible
    fig.data[10].visible = True
    k = np.round(k_values, 3)

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to k: " + str(k[i]) + "/km"}],
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
        xaxis_title="height in km",
        title="k in 1/km"
    )

    fig.show()

    fig.write_html('test1.html')
