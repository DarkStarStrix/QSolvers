# code boson sampling using Qutip

import numpy as np
import qutip as qt
import plotly.graph_objects as go


class BosonSampling:
    def __init__(self, n, m, unitary_matrix):  # renamed from U to unitary_matrix
        self.n = n
        self.m = m
        self.unitary_matrix = qt.tensor ([qt.Qobj (unitary_matrix) for _ in range (n)])  # renamed from U to unitary_matrix

    def simulate(self):
        initial_state = qt.tensor ([qt.basis (self.m, 0) for _ in range (self.n)])
        final_state = self.unitary_matrix * initial_state  # renamed from U to unitary_matrix
        return np.abs (final_state.full ()) ** 2

    def plot(self, prob):
        fig = go.Figure ()
        fig.add_trace (go.Bar (x=list (range (self.m)), y=prob [0]))
        fig.update_layout (title="Boson Sampling", xaxis_title="Output state", yaxis_title="Probability")
        fig.show ()


if __name__ == "__main__":
    n = 3
    m = 5
    unitary_matrix = numpy.random.Generator (m, m) + 1j * numpy.random.Generator (m, m)  # renamed from U to unitary_matrix
    boson_sampling = BosonSampling (n, m, unitary_matrix)  # renamed from U to unitary_matrix
    prob = boson_sampling.simulate ()
    boson_sampling.plot (prob)
    print (prob)
