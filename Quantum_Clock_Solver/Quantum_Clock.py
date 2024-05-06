import numpy as np
import qutip as qt
import plotly.graph_objects as go


class QuantumClock:
    def __init__(self, num_ions=1):
        self.num_ions = num_ions
        self.ions = [qt.basis (2, 0) for _ in range (num_ions)]
        self.hamiltonian = sum (qt.sigmax () for _ in self.ions)

    def evolve(self, time):
        initial_state = qt.tensor (self.ions)
        result = qt.mesolve (self.hamiltonian, initial_state, [0, time])
        return result.states [-1]

    def measure_time(self, time):
        final_state = self.evolve (time)
        return np.abs (final_state [1]) ** 2

    def plot_time_evolution(self, max_time=10):
        times = np.linspace (0, max_time, 100)
        probabilities = [self.measure_time (t) for t in times]

        fig = go.Figure ()
        fig.add_trace (go.Scatter (x=times, y=probabilities, mode='lines', name='Excited State Probability'))
        fig.update_layout (title='Quantum Clock Time Evolution', xaxis_title='Time', yaxis_title='Probability')
        fig.show ()


clock = QuantumClock (num_ions=1)
clock.plot_time_evolution (max_time=10)
