# implement the bosonic quantum key distribution protocol to generate a secret key between Alice and Bob use qutip

import numpy as np
import qutip as qt
import plotly.graph_objects as go


def create_circuit(hadamard=False, measure=False):
    state = qt.basis (2, 0)
    if hadamard:
        hadamard_gate = qt.Qobj ([[1, 1], [1, -1]]) / np.sqrt (2)
        state = hadamard_gate * state
    if measure:
        state = qt.sigmaz () * state
    return state


def create_circuit_bob():
    state = qt.basis (2, 0)
    return state


def quantum_channel(alice):
    noise = qt.rand_ket(2)  # Create a random quantum state
    bob_state = alice + 0.3 * noise  # Add the noise to Alice's state
    bob_state = bob_state.unit()  # Normalize the state
    return bob_state


def execute_circuit(state):
    return state


def run():
    alice = create_circuit (hadamard=True, measure=True)
    bob = quantum_channel (alice)
    alice_state = execute_circuit (alice)
    bob_state = execute_circuit (bob)
    fidelity = qt.fidelity (alice_state, bob_state)
    return fidelity


def plot_fidelity(fidelity):
    fig = go.Figure (data=go.Bar (y=[fidelity]))
    fig.update_layout (title_text='Fidelity of Quantum States')
    fig.show ()


if __name__ == "__main__":
    fidelity = run ()
    plot_fidelity (fidelity)
