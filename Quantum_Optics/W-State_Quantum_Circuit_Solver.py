import numpy as np
import qutip as qt
from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import gate_sequence_product
import plotly.graph_objects as go


# Define the W-state circuit
def w_state_circuit():
    # Create a 3-qubit quantum circuit
    qc = QubitCircuit (3)

    # Apply Hadamard gate to qubit 0
    qc.add_gate ("SNOT", targets=[0])

    # Apply CNOT gates to create the W-state
    qc.add_gate ("CNOT", controls=[0], targets=[1])
    qc.add_gate ("CNOT", controls=[1], targets=[2])

    # Define the initial state |000⟩
    zero_state = qt.basis (2, 0)  # |0⟩
    initial_state = qt.tensor (zero_state, zero_state, zero_state)

    # Run the circuit
    U = gate_sequence_product (qc.propagators ())
    final_state = U * initial_state

    return final_state


# Simulate the W-state circuit
w_state = w_state_circuit ()

# Calculate probabilities for each basis state
probabilities = np.abs(w_state.full()) ** 2

# Create an interactive line chart using Plotly
fig = go.Figure ()

# Add traces for probabilities
for i, basis_state in enumerate (["000", "001", "010", "100"]):
    fig.add_trace (go.Scatter (x=[0], y=[probabilities [i] [0]],
                               mode="lines+markers",
                               name=f"|{basis_state}⟩",
                               line=dict (width=2)))

# Customize the chart
fig.update_layout (title="W-State Probabilities",
                   xaxis_title="Basis State",
                   yaxis_title="Probability",
                   xaxis=dict (tickvals=[0], ticktext=["|000⟩", "|001⟩", "|010⟩", "|100⟩"]),
                   yaxis=dict (range=[0, 1]),
                   showlegend=True)

# Show the interactive chart
fig.show ()
