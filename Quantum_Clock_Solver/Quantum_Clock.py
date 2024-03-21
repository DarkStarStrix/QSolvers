import numpy as np
import qutip as qt
import plotly.graph_objects as go


class QuantumClock:
    def __init__(self, num_ions=1):
        if not isinstance (num_ions, int) or num_ions <= 0:
            raise ValueError ("Number of ions must be a positive integer.")
        self.num_ions = num_ions
        self.ions = [qt.basis (2, 0) for _ in range (num_ions)]  # Initialize ions in ground state
        self.hamiltonian = self.create_hamiltonian ()

    def create_hamiltonian(self):
        # Define the Hamiltonian for the quantum clock
        H = 0
        for _ in self.ions:
            H += qt.sigmax ()  # Example: Interaction with Pauli-X operator
        return H

    def evolve(self, time):
        if not isinstance (time, (int, float)) or time < 0:
            raise ValueError ("Time must be a non-negative number.")
        # Evolve the system using the Hamiltonian
        initial_state = qt.tensor (self.ions)  # Combine all ion states into a single state
        result = qt.mesolve (self.hamiltonian, initial_state, [0, time])
        return result.states [-1]  # Final state after evolution

    def measure_time(self, time):
        if not isinstance (time, (int, float)) or time < 0:
            raise ValueError ("Time must be a non-negative number.")
        final_state = self.evolve (time)
        probability = np.abs (final_state [1]) ** 2  # Probability of being in excited state
        return probability

    def plot_time_evolution(self, max_time=10):
        if not isinstance (max_time, (int, float)) or max_time <= 0:
            raise ValueError ("Max time must be a positive number.")
        times = np.linspace (0, max_time, 100)
        probabilities = [self.measure_time (t) for t in times]

        fig = go.Figure ()
        fig.add_trace (go.Scatter (x=times, y=probabilities, mode='lines', name='Excited State Probability'))
        fig.update_layout (title='Quantum Clock Time Evolution', xaxis_title='Time', yaxis_title='Probability')
        fig.show ()  # This will block the execution until the plot window is closed


# Example usage
clock = QuantumClock (num_ions=1)
clock.plot_time_evolution (max_time=10)
