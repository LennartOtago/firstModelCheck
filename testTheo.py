import plotly.graph_objects as go
import numpy as np

def make_figure(h_tang, ind_limb, h_inf):

    R = 6371000 #in m

    h_values = np.linspace(h_tang, h_inf, 1000)
    h_values[0] = h_values[0]+0.1
    h_limb = h_values[ind_limb]
    r_t = np.sqrt((h_limb + R) ** 2 - (h_tang + R) ** 2)




    # Create figure
    fig = go.Figure()
    k_values = np.linspace(0, 1e-5, 100)

    # Add traces, one for each slider step
    for k in k_values:
        # h = np.sqrt(k/2 * ( ( h_values - R)**2 - (h_t - R)**2 ) )
        v = np.sqrt((h_values + R) ** 2 - (h_tang + R) ** 2)
        #print(v)
        before = np.exp(-k * ( r_t - v ) ) * (h_values + R) / v
        after = np.exp(-k * ( r_t + v ) ) * (h_values + R) / v
        before[ind_limb+1:] = 0

        fig.add_trace(
            go.Scatter(
                visible=False,
                line= dict(color="#00CED1", width=6),
                name= "𝜈 = " + str(k),
                x= h_values,
                y= before + after
            ))

    # Make 10th trace visible
    fig.data[10].visible = True
    k = np.round(k_values, 7)

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to k: " + str(k[i]) + "/m"}],
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
        xaxis_title="height in m",
        title="k in 1/m"
    )

    fig.show()

    fig.write_html('test2.html')



