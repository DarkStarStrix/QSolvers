import numpy as np
from qutip import sesolve, sigmax, basis
import plotly.figure_factory as ff
import plotly.io as pio


class NonLinearSchrodingerSolver:
    def __init__(self, hamiltonian, initial_state, time_array):
        self.hamiltonian = hamiltonian
        self.initial_state = initial_state
        self.time_array = time_array
        self.solution = None

    def nonlinear_term(self, state, t, x, y):
        u = np.sin (np.pi * x) * np.cos (np.pi * y)
        v = -np.cos (np.pi * x) * np.sin (np.pi * y)
        p = -0.25 * (np.cos (2 * np.pi * x) + np.cos (2 * np.pi * y))
        return u, v, p

    def solve(self, x, y):
        self.solution = sesolve (self.hamiltonian, self.initial_state, self.time_array,
                                 e_ops=[lambda state, t: self.nonlinear_term (state, t, x, y)])

    def visualize(self):
        abs_states = [np.abs (state.full ()) for state in self.solution.states]
        real_states = [state.real for state in abs_states]
        imag_states = [state.imag for state in abs_states]
        real_states_flat = np.concatenate (real_states).ravel ()
        imag_states_flat = np.concatenate (imag_states).ravel ()
        quiver_plot = ff.create_quiver (real_states_flat, imag_states_flat, real_states_flat, imag_states_flat,
                                        scale=.2, line=dict (width=1), marker=dict (size=1),
                                        name='Schrödinger Equation')
        pio.write_html (quiver_plot, 'quiver_plot.html')


hamiltonian = sigmax ()
initial_state = basis (2, 0)
time_array = np.linspace (0, 10, 100)
x = np.linspace (0, 1, 100)
y = np.linspace (0, 1, 100)

solver = NonLinearSchrodingerSolver (hamiltonian, initial_state, time_array)
solver.solve (x, y)
solver.visualize ()
