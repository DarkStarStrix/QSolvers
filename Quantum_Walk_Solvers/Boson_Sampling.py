# code boson sampling using Qutip

import numpy as np
import qutip as qt
import plotly.graph_objects as go


class BosonSampling:
    def __init__(self, n, m, U):
        self.n = n
        self.m = m
        self.U = qt.tensor ([qt.Qobj (U) for _ in range (n)])

    def simulate(self):
        initial_state = qt.tensor ([qt.basis (self.m, 0) for _ in range (self.n)])
        final_state = self.U * initial_state
        return np.abs (final_state.full ()) ** 2

    def plot(self, prob):
        fig = go.Figure ()
        fig.add_trace (go.Bar (x=list (range (self.m)), y=prob [0]))
        fig.update_layout (title="Boson Sampling", xaxis_title="Output state", yaxis_title="Probability")
        fig.show ()


if __name__ == "__main__":
    n = 3
    m = 5
    U = np.random.rand (m, m) + 1j * np.random.rand (m, m)
    boson_sampling = BosonSampling (n, m, U)
    prob = boson_sampling.simulate ()
    boson_sampling.plot (prob)
    print (prob)
