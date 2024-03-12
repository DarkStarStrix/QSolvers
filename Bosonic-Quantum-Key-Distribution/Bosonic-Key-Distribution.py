# implement the bosonic quantum key distribution protocol to generate a secret key between Alice and Bob use qutip

import numpy as np
import qutip as qt
import plotly.graph_objects as go


def create_circuit(hadamard=False, measure=False):
    state = qt.basis (2, 0)
    if hadamard:
        state = qt.hadamard_transform () * state
    if measure:
        state = qt.sigmaz () * state
    return state


def quantum_channel(alice):
    return (alice + 0.3 * qt.rand_ket (2)).unit ()


def run():
    alice = create_circuit (hadamard=True, measure=True)
    bob = quantum_channel (alice)
    return qt.fidelity (alice, bob)


def plot_fidelity(fidelity):
    go.Figure (data=go.Bar (y=[fidelity]), layout=go.Layout (title_text='Fidelity of Quantum States')).show ()


if __name__ == "__main__":
    plot_fidelity (run ())
