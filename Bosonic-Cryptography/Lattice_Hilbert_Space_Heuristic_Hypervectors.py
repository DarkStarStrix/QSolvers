import numpy as np
import plotly.graph_objects as go

# Define the gridlines and vectors
gridlines = [([-10, 10], [0, 0], [0, 0]), ([0, 0], [-10, 10], [0, 0]), ([0, 0], [0, 0], [-10, 10])]
vectors = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]  # Modify these coordinates to match the lattice points

# Create a 3D figure and add the gridlines and vectors
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color, width=2), showlegend=False) for (x, y, z), color in zip(gridlines, ['blue', 'red', 'green'])] +
                     [go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode='lines', line=dict(color='black', width=2), name=f'Hypervector {i+1}') for i, (x, y, z) in enumerate(vectors)])

# Define the lattice points
lattice_points = [(x, y, z) for x in np.arange(-10, 11) for y in np.arange(-10, 11) for z in np.arange(-10, 11)]

# Add the lattice points to the figure
fig.add_trace(go.Scatter3d(x=[x for x, _, _ in lattice_points], y=[y for _, y, _ in lattice_points], z=[z for _, _, z in lattice_points], mode='markers', marker=dict(size=2, color='purple'), showlegend=False))

# Update layout of the figure
fig.update_layout(scene=dict(
    xaxis=dict(backgroundcolor="rgb(200, 200, 230)", gridcolor="white", showbackground=True, zerolinecolor="white", nticks=10, range=[-10, 10], title='X Axis'),
    yaxis=dict(backgroundcolor="rgb(230, 200,230)", gridcolor="white", showbackground=True, zerolinecolor="white", nticks=10, range=[-10, 10], title='Y Axis'),
    zaxis=dict(backgroundcolor="rgb(230, 230,200)", gridcolor="white", showbackground=True, zerolinecolor="white", nticks=10, range=[-10, 10], title='Z Axis'),
    aspectmode='cube'
),
legend=dict(
    x=0.1,
    y=1,
    xanchor="left",
    yanchor="top",
    traceorder="normal",
    font=dict(
        family="sans-serif",
        size=12,
        color="black"
    ),
    bgcolor="LightSteelBlue",
    bordercolor="Black",
    borderwidth=2
),
showlegend=True,
title='3D Plot of Hypervectors',
title_font=dict(
    family="Times New Roman",
    size=18,
    color="black"
))

# Show the figure
fig.show()
