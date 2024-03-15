# hard code the variational quantum eigensolver for 3 qubits

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.visualization import plot_histogram
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.operators import WeightedPauliOperator
import numpy as np

# Load IBM Q account and get the least busy backend device
provider = IBMQ.load_account()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                       not x.configuration().simulator and x.status().operational))

# Create a Quantum Circuit acting on the q register
qc = QuantumCircuit(3, 3)

# Apply Hadamard gate to all qubits
qc.h(range(3))

# Apply Oracle
qc.x(range(3))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(3))

# Apply variational quantum eigensolver
# Define the Hamiltonian
pauli_dict = {
    'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
              {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "ZI"},
              {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "IZ"},
              {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
              {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}]
}
qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

# Define the variational form
var_form = RY(qubit_op.num_qubits, depth=3, entanglement='linear')

# Define the optimizer
optimizer = COBYLA(maxiter=1000)

# Define the VQE algorithm
vqe = VQE(qubit_op, var_form, optimizer)

# Execute the circuit on the least busy backend. Monitor the job
job = execute(qc, backend, shots=1024)
result = job.result()

# Get the results from the computation
counts = result.get_counts(qc)

# Plot the results
plot_histogram(counts)

# Execute the VQE algorithm
result = vqe.run(backend)
