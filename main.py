# import random
#
# def generateRandom(upper):
#     """
#
#     :param upper: >= 0
#     :return: int
#     """
#     r = random.randint(1, upper)
#     return r
#
# def main():
#     run = True
#     num1 = generateRandom(10)
#     num2 = generateRandom(10)
#     result = num1 * num2
#     while run:
#         ans = input("What is " + str(num1) + " x " + str(num2) + "? ")
#
#         if ans.isdigit():
#             if int(ans) == result:
#                 print("Correct!")
#                 run = False
#             else:
#                 print("Incorrect! Try again.")
#
#         else:
#             print("Answer must be a positive number, try again.")
#
#
# # global vars
# times = 10
#
# for x in range(times):
#     main()
#


#import cProfile
#from cProfile import label
import plotly.graph_objects as go
import numpy as np


R = 6371
h_t = R + 15
h_max = R + 90

r_t = np.sqrt( (h_max + R)**2 - (h_t + R)**2 )
v_max = (h_max + R)**2 - (h_t + R)**2


h_values = np.linspace(h_t, h_max, 1000)

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for k in np.linspace(0, 0.1, 100):
    #h = np.sqrt(k/2 * ( ( h_values - R)**2 - (h_t - R)**2 ) )
    v =  np.sqrt( ( h_values + R)**2 - (h_t + R)**2 )
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(k),
            x = h_values-R,
            y= np.exp(-k*r_t) * ( np.exp( -k * v ) + np.exp( k* v ) )
            ) )

# Make 10th trace visible
fig.data[10].visible = True
k= np.round(np.linspace(0, 0.1, 100), 3)


# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to k: " + str(k[i]) + "/km" }],
              label = str( k[i] ),  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "k= ", "suffix": ""},
    pad= {"b": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    xaxis_title="height in km",
    title="k in 1/km"
)

fig.show()
fig.show()
fig.write_html('test.html')



