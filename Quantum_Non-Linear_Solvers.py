# Add quantum elements to the non-linear solvers  make a library of quantum non-linear solvers

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Define the parameters for the non-linear Schrödinger equation
N = 100
L = 10.0
x = np.linspace (-L / 2, L / 2, N)

# Define the creation and annihilation operators
a = destroy (N)
adag = a.dag ()

# Define the Hamiltonian for the non-linear Schrödinger equation
H = -1.0 * (adag * a + 0.5 * adag * adag * a * a)

# Solve the Schrödinger equation
psi0 = basis (N, N // 2)  # initial state
t = np.linspace (0, 10.0, 100)  # time
result = sesolve (H, psi0, t, [])

# calculate the Wigner function of the final state
xvec = np.linspace (-5, 5, 200)
W = wigner (result.states [-1], xvec, xvec)

# Plot the results using plot_wigner using the last state in the result
fig, ax = plt.subplots (1, 1, figsize=(10, 10))
xvec = np.linspace (-5, 5, 200)
W = wigner (result.states [-1], xvec, xvec)
cont = ax.contourf (xvec, xvec, W, 100, cmap="bwr")
plt.show ()

# Plot the results using plot_wigner using the last state in the result in 3D
fig = plt.figure (figsize=(10, 10))
ax = fig.add_subplot (111, projection='3d')
xvec = np.linspace (-5, 5, 200)
W = wigner (result.states [-1], xvec, xvec)
X, Y = np.meshgrid (xvec, xvec)
ax.plot_surface (X, Y, W, rstride=1, cstride=1, cmap="bwr")
plt.show ()

# Plot the results using plot_wigner using the last state in the result in 3D using plotly
fig = go.Figure (data=[go.Surface (z=W)])
fig.update_layout (title='Wigner function', autosize=False,
                   width=500, height=500,
                   margin=dict (l=65, r=50, b=65, t=90))
fig.show ()
