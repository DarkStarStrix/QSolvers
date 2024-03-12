import numpy as np
import plotly.graph_objects as go


# Define a function for the XOR operation
def xor_operation(vector_a, vector_b):
    return np.logical_xor (vector_a, vector_b)


# Define a function for the quantum-inspired transformation
def quantum_transformation(input_vector, input_scalar):
    return np.array ([input_scalar * element for element in input_vector])


# Generate more random 3D vectors to represent hypervectors for visualization
vectors = [np.random.randint (2, size=3) for _ in range (5)]  # List of binary vectors

# Perform XOR operation and quantum-inspired transformation on the vectors
xor_vectors = [xor_operation (vectors [i], vectors [(i + 1) % 5]) for i in range (5)]
quantum_vectors = [quantum_transformation (vector, 0.5) for vector in vectors]  # Example scalar is 0.5

# Create a 3D scatter plot for the vectors and axes
fig = go.Figure (data=[go.Scatter3d (x=[0, vector [0]], y=[0, vector [1]], z=[0, vector [2]], mode='lines+markers',
                                     marker=dict (size=6, color=np.random.rand (3)), name=f"Vector {i + 1}")
                       for i, vector in enumerate (quantum_vectors)])

# Add axes to the figure
for axis, color, name in zip ([[[-1, 1], [0, 0], [0, 0]], [[0, 0], [-1, 1], [0, 0]], [[0, 0], [0, 0], [-1, 1]]],
                              ['red', 'green', 'blue'], ['X-axis', 'Y-axis', 'Z-axis']):
    fig.add_trace (go.Scatter3d (x=axis [0], y=axis [1], z=axis [2], marker=dict (size=4, color=color),
                                 line=dict (color=color, width=2), name=name))

# Set the layout of the figure
fig.update_layout (title="Enhanced Visualization of HSHH Cryptography",
                   scene=dict (xaxis=dict (title="X", range=[-1, 1]),
                               yaxis=dict (title="Y", range=[-1, 1]),
                               zaxis=dict (title="Z", range=[-1, 1]),
                               aspectmode="cube"),
                   margin=dict (l=0, r=0, b=0, t=50), showlegend=True)

# Show the figure
fig.show ()
