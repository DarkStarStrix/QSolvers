import numpy as np
import qutip as qt
from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import gate_sequence_product
import plotly.graph_objects as go


def w_state_circuit():
    qc = QubitCircuit (3)
    qc.add_gate ("SNOT", targets=[0])
    qc.add_gate ("CNOT", controls=[0], targets=[1])
    qc.add_gate ("CNOT", controls=[1], targets=[2])
    initial_state = qt.tensor (qt.basis (2, 0), qt.basis (2, 0), qt.basis (2, 0))
    U = gate_sequence_product (qc.propagators ())
    return U * initial_state


w_state = w_state_circuit ()
probabilities = np.abs (w_state.full ()) ** 2

fig = go.Figure ()
for i, basis_state in enumerate (["000", "001", "010", "100"]):
    fig.add_trace (go.Scatter (x=[0], y=[probabilities [i] [0]], mode="lines+markers", name=f"|{basis_state}⟩"))
fig.update_layout (title="W-State Probabilities", xaxis_title="Basis State", yaxis_title="Probability",
                   xaxis=dict (tickvals=[0], ticktext=["|000⟩", "|001⟩", "|010⟩", "|100⟩"]), yaxis=dict (range=[0, 1]))
fig.show ()
