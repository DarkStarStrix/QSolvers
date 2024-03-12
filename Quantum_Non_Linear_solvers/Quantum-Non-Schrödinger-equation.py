# Add quantum elements to the non-linear solvers  make a library of quantum non-linear solvers

from qutip import destroy, basis, sesolve, wigner
import numpy as np
import matplotlib.pyplot as plt

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
cont = ax.contourf (xvec, xvec, W, 100, cmap="bwr")
plt.show ()
