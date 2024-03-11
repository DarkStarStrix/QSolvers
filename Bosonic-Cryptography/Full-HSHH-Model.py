# make a graph of the full HSHH model have grids axis and the number of photons axis and plot vectors in the graph

import plotly.graph_objects as go
import numpy as np

# Create a 3D figure
fig = go.Figure()

# Add the gridlines
fig.add_trace(go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-10, 10], mode='lines', line=dict(color='green', width=2)))

# Add some vectors for demonstration
vectors = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for vector in vectors:
    fig.add_trace(go.Scatter3d(x=[0, vector[0]], y=[0, vector[1]], z=[0, vector[2]], mode='lines', line=dict(color='black', width=2)))
    fig.add_trace(go.Cone(x=[vector[0]], y=[vector[1]], z=[vector[2]], u=[vector[0]], v=[vector[1]], w=[vector[2]], sizemode="scaled", sizeref=0.2))

# Update layout of the figure
fig.update_layout(scene=dict(
    xaxis=dict(
        backgroundcolor="rgb(200, 200, 230)",
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white",
        nticks=10,
        range=[-10, 10],
    ),
    yaxis=dict(
        backgroundcolor="rgb(230, 200,230)",
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white",
        nticks=10,
        range=[-10, 10],
    ),
    zaxis=dict(
        backgroundcolor="rgb(230, 230,200)",
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white",
        nticks=10,
        range=[-10, 10],
    ),
    aspectmode='cube'
))

# Show the figure
fig.show()
