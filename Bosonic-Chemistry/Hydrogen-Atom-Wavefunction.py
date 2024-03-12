import numpy as np
from qutip import sesolve, basis, destroy
import plotly.graph_objects as go
import plotly.offline as pyo
import os

# Parameters
N, a0, Z, r = 100, 1.0, 1, np.linspace(0, 20, 100)

# Define the radial part of the Hamiltonian
a = destroy(N)
H = -0.5 * a.dag() * a + Z / a0 * (a + a.dag())

# Solve the Schrödinger equation and calculate the wavefunction
result = sesolve(H, basis(N, 0), r)
wavefunctions = [np.abs(state.full().flatten()[0]) ** 2 for state in result.states]

# Create a 3D plot using Plotly
fig = go.Figure(data=[go.Surface(z=np.reshape(wavefunctions, (10, 10)))])

fig.update_layout(title='Hydrogen Atom Wavefunction', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# Use plotly offline to create an HTML file and show the figure
pyo.plot(fig, filename='hydrogen_wavefunction.html')
fig.show()

# Print the current directory
print("Current directory:", os.getcwd())
