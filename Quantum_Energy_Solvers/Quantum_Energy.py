from qiskit.opflow import Z, I
from qiskit.algorithms import VQE
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import SPSA


class QuantumSolver:
    @staticmethod
    def create_hamiltonian(loss_coefficients):
        return sum (coeff * (I ^ (Z - I)) for coeff in loss_coefficients)

    def __init__(self, loss_coefficients, num_qubits=3, reps=1, maxiter=100):
        self.loss_coefficients = loss_coefficients
        self.h_loss = self.create_hamiltonian (self.loss_coefficients)  # renamed from H_loss to h_loss
        self.quantum_instance = QuantumInstance (Aer.get_backend ('aer_simulator_statevector'))
        self.ansatz = EfficientSU2 (num_qubits=num_qubits, reps=reps)
        self.optimizer = SPSA (maxiter=maxiter)
        self.vqe = VQE (self.ansatz, self.optimizer, quantum_instance=self.quantum_instance)

    def compute_minimum_eigenvalue(self):
        result = self.vqe.compute_minimum_eigenvalue (operator=self.h_loss)  # renamed from H_loss to h_loss
        return result.eigenvalue.real


# Usage
solver = QuantumSolver ([1.2, 0.9, 1.5])
print (f"Minimum Loss Configuration Energy: {solver.compute_minimum_eigenvalue ()}")
