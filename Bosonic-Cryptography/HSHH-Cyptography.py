import numpy as np
import plotly.graph_objects as go
from qutip import basis, sigmax


# Define a function for the XOR operation
def xor_operation(vector_a, vector_b):
    return np.logical_xor (vector_a, vector_b)


# Define a function for the quantum-inspired transformation
def quantum_transformation(input_vector, input_scalar):
    transformed_vector = np.array ([input_scalar * element for element in input_vector])
    return transformed_vector


# Generate more random 3D vectors to represent hypervectors for visualization
vectors = [np.random.randint (2, size=3) for _ in range (5)]  # List of binary vectors

# Perform XOR operation and quantum-inspired transformation on the vectors
xor_vectors = [xor_operation (vectors [i], vectors [(i + 1) % 5]) for i in range (5)]
quantum_vectors = [quantum_transformation (vector, 0.5) for vector in vectors]  # Example scalar is 0.5

# Central point coordinates
central_point = [0, 0, 0]

# Create a 3D line for the x-axis
x_axis = go.Scatter3d (
    x=[-1, 1], y=[0, 0], z=[0, 0],
    marker=dict (size=4, color='red'),
    line=dict (color='red', width=2),
    name='X-axis'
)

# Create a 3D line for the y-axis
y_axis = go.Scatter3d (
    x=[0, 0], y=[-1, 1], z=[0, 0],
    marker=dict (size=4, color='green'),
    line=dict (color='green', width=2),
    name='Y-axis'
)

# Create a 3D line for the z-axis
z_axis = go.Scatter3d (
    x=[0, 0], y=[0, 0], z=[-1, 1],
    marker=dict (size=4, color='blue'),
    line=dict (color='blue', width=2),
    name='Z-axis'
)

# Create a 3D scatter plot for the vectors and axes
fig = go.Figure (data=[x_axis, y_axis, z_axis] + [go.Scatter3d (
    x=[central_point [0], vector [0]],
    y=[central_point [1], vector [1]],
    z=[central_point [2], vector [2]],
    mode='lines+markers',
    marker=dict (size=6, color=np.random.rand (3, )),  # Random color for each vector
    name=f"Vector {i + 1}",
    hovertext=f"Vector {i + 1}"
) for i, vector in enumerate (quantum_vectors)])

# Set the layout of the figure
fig.update_layout (
    title="Enhanced Visualization of HSHH Cryptography",
    scene=dict (
        xaxis=dict (title="X", range=[-1, 1]),
        yaxis=dict (title="Y", range=[-1, 1]),
        zaxis=dict (title="Z", range=[-1, 1]),
        aspectmode="cube"
    ),
    margin=dict (l=0, r=0, b=0, t=50),
    showlegend=True
)

# Show the figure
fig.show ()
