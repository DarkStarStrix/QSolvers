import numpy as np
import plotly.figure_factory as ff
from qutip import *
import plotly.io as pio


class NonLinearSchrodingerSolver:
    def __init__(self, hamiltonian, initial_state, time_array):
        self.hamiltonian = hamiltonian
        self.initial_state = initial_state
        self.time_array = time_array
        self.solution = None

    @staticmethod
    def nonlinear_term(self, state, t, x, y):
        Naiver_Stokes = 1j * (sigmax () * state - state * sigmax ())
        u = np.sin (np.pi * x) * np.cos (np.pi * y)
        v = -np.cos (np.pi * x) * np.sin (np.pi * y)
        p = -0.25 * (np.cos (2 * np.pi * x) + np.cos (2 * np.pi * y))
        return u, v, p

    def solve(self, x, y):
        self.solution = sesolve (self.hamiltonian, self.initial_state, self.time_array,
                                 e_ops=[lambda state, t: self.nonlinear_term (state, t, x, y)])
        if not self.solution.states:
            raise ValueError ("No states were found. Please check the inputs.")

    def visualize(self):
        abs_states = [np.abs (state.full ()) for state in self.solution.states]
        real_states = [np.real (state) for state in abs_states]
        imag_states = [np.imag (state) for state in abs_states]
        if real_states and imag_states:  # Check if real_states and imag_states are not empty
            real_states_flat = np.concatenate (real_states).ravel ()
            imag_states_flat = np.concatenate (imag_states).ravel ()
            quiver_plot = ff.create_quiver (real_states_flat, imag_states_flat, real_states_flat, imag_states_flat,
                                            scale=.2, line=dict (width=1), marker=dict (size=1),
                                            name='Schrödinger Equation')
            pio.write_html (quiver_plot, 'quiver_plot.html')


# Define the Hamiltonian
hamiltonian = sigmaz ()

# Define the initial state
initial_state = basis (2, 0)

# Define the time array
time_array = np.linspace (0, 10, 100)

# Define x and y
x = np.linspace (0, 1, 100)
y = np.linspace (0, 1, 100)

# Check the inputs
assert hamiltonian is not None, "Hamiltonian is not defined"
assert initial_state is not None, "Initial state is not defined"
assert len (time_array) > 0, "Time array is empty"

# Create an instance of the NonLinearSchrodingerSolver class
solver = NonLinearSchrodingerSolver (hamiltonian, initial_state, time_array)

# Solve the Schrödinger equation
try:
    solver.solve (x, y)
except ValueError as e:
    pass

# Visualize the results
solver.visualize ()
