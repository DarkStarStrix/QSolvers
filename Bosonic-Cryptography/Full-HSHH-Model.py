# make a graph of the full HSHH model have grids axis and the number of photons axis and plot vectors in the graph

import plotly.graph_objects as go

# Define the gridlines and vectors
gridlines = [([-10, 10], [0, 0], [0, 0]), ([0, 0], [-10, 10], [0, 0]), ([0, 0], [0, 0], [-10, 10])]
vectors = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

# Create a 3D figure and add the gridlines and vectors
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color, width=2)) for (x, y, z), color in zip(gridlines, ['blue', 'red', 'green'])] +
                     [go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode='lines', line=dict(color='black', width=2)) for x, y, z in vectors] +
                     [go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], sizemode="scaled", sizeref=0.2) for x, y, z in vectors])

# Update layout of the figure
fig.update_layout(scene=dict(
    xaxis=dict(backgroundcolor="rgb(200, 200, 230)", gridcolor="white", showbackground=True, zerolinecolor="white", nticks=10, range=[-10, 10]),
    yaxis=dict(backgroundcolor="rgb(230, 200,230)", gridcolor="white", showbackground=True, zerolinecolor="white", nticks=10, range=[-10, 10]),
    zaxis=dict(backgroundcolor="rgb(230, 230,200)", gridcolor="white", showbackground=True, zerolinecolor="white", nticks=10, range=[-10, 10]),
    aspectmode='cube'
))

# Show the figure
fig.show()
