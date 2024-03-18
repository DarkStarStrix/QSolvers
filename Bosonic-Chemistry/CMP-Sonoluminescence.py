import numpy as np
import plotly.graph_objects as go
from qutip import *

# Define system parameters
N = 50  # number of states in the Hilbert space
omega = 1.0  # frequency of the harmonic trap

# Define the Hamiltonian
a = destroy(N)  # annihilation operator
H = omega * (a.dag() * a + 0.5)  # Hamiltonian

# Define the initial state
psi0 = basis(N, 0)  # ground state

# Define the times over which to evolve the system
times = np.linspace(0.0, 10.0, 100)

# Evolve the system
result = sesolve(H, psi0, times, [])

# The result object contains the state of the system at each time
final_state = result.states[-1]

# Calculate the wavefunction from the final state
wavefunction = np.array(final_state.full()).flatten()

# Create a grid of x values
x = np.linspace(-5, 5, N)

# Use the real and imaginary parts of the wavefunction as y values
y_real = np.real(wavefunction)
y_imag = np.imag(wavefunction)

# Create a 2D plot for the real part
fig_real = go.Figure(data=go.Scatter(x=x, y=y_real, mode='lines', name='Real'))
fig_real.update_layout(title='Real Part of the Wavefunction',
                  autosize=False, width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# Create a 2D plot for the imaginary part
fig_imag = go.Figure(data=go.Scatter(x=x, y=y_imag, mode='lines', name='Imaginary'))
fig_imag.update_layout(title='Imaginary Part of the Wavefunction',
                  autosize=False, width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig_real.show()
fig_imag.show()
