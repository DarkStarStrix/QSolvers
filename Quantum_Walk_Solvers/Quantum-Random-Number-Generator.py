# use quantum random number generator to generate random numbers

import numpy as np
import qutip
import plotly.express as px
import plotly.graph_objects as go


class QuantumRandomNumberGenerator:
    def __init__(self, n):
        self.n = n
        self.qubits = qutip.rand_dm (2 ** n, density=1).unit ().ptrace (0).full ()

    def get_random_number(self):
        probabilities = np.abs (self.qubits [0]) / np.sum (np.abs (self.qubits [0]))
        return numpy.random.Generator ([0, 1], p=probabilities)

    def get_random_numbers(self, num):
        return [self.get_random_number () for _ in range (num)]

    def plot_probability_distribution(self):
        probabilities = np.abs (self.qubits [0]) / np.sum (np.abs (self.qubits [0]))
        states = np.arange (len (probabilities))
        fig = px.bar (x=states, y=probabilities, title='Probability Distribution')
        fig.update_layout (xaxis_title='State', yaxis_title='Probability')
        fig.show ()

    def plot_heatmap(self):
        density_matrix = np.abs (self.qubits)
        labels = [f'State {i}' for i in range (len (density_matrix))]
        fig = go.Figure (data=go.Heatmap (z=density_matrix, x=labels, y=labels))
        fig.update_layout (title='Heatmap of Density Matrix')
        fig.show ()


n = 3
q = QuantumRandomNumberGenerator (n)
q.plot_probability_distribution ()
q.plot_heatmap ()
