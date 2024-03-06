import numpy as np
from qutip import sesolve, Qobj, basis, destroy
import plotly.graph_objects as go
import plotly.offline as pyo
import os

# Parameters
N = 100  # Number of Fock states
a0 = 1.0  # Bohr radius
Z = 1  # Atomic number of hydrogen
r = np.linspace(0, 20, 100)  # Radial distance array

# Define the radial part of the Hamiltonian
a = destroy(N)
H = -0.5 * a.dag() * a + Z / a0 * (a + a.dag())

# Define the initial state
psi0 = basis(N, 0)  # Ground state

# Solve the Schrödinger equation
result = sesolve(H, psi0, r)

# Calculate the wavefunction at each point in space
wavefunctions = np.zeros(len(r))
for i in range(len(r)):
    psi = result.states[i]
    wavefunctions[i] = np.abs(psi.full().flatten()[0]) ** 2

# Reshape wavefunctions into a 2D array
wavefunctions_2d = np.reshape(wavefunctions, (10, 10))

# Create a 3D plot using Plotly
fig = go.Figure(data=[go.Surface(z=wavefunctions_2d)])

fig.update_layout(title='Hydrogen Atom Wavefunction', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# Use plotly offline to create an HTML file
pyo.plot(fig, filename='hydrogen_wavefunction.html')

# Print the current directory
print("Current directory:", os.getcwd())

fig.show()
