# In qiskit code a quantum adiabatic algorithm

import numpy as np
from qiskit import Aer
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import VQE
from qiskit.aqua import QuantumInstance


class QuantumAdiabatic:
    def __init__(self, n, h, J, backend, shots):
        self.n = n
        self.h = h
        self.J = J
        self.backend = backend
        self.shots = shots
        self.hamiltonian = self.create_hamiltonian()
        self.optimizer = COBYLA()
        self.vqe = VQE(operator=self.hamiltonian, optimizer=self.optimizer,
                       quantum_instance=QuantumInstance(backend=Aer.get_backend(backend), shots=shots))
        self.result = self.vqe.run()
        self.print_result()

    def create_hamiltonian(self):
        pauli_dict = {str(i) + str(j) + str(k) + str(l): self.J
                      for i in range(self.n) for j in range(self.n) if i != j
                      for k in range(self.n) if k != i and k != j
                      for l in range(self.n) if l != i and l != j and l != k}
        pauli_dict.update({str(i) * 4: self.h for i in range(self.n)})
        return WeightedPauliOperator(paulis=pauli_dict)

    def print_result(self):
        print("Result: ", self.result)
        print("Energy: ", self.result.eigenvalue.real)
        print("Optimal parameters: ", self.result.optimal_point)
        print("Optimal value: ", self.result.optimal_value)
        print("Cost function evaluations: ", self.result.cost_function_evals)
        print("Time taken: ", self.result.optimizer_time)
        print("Number of evaluations: ", self.result.optimizer_evals)
        print("Backend: ", self.backend)


if __name__ == "__main__":
    n = 4
    h = 1
    J = 1
    backend = "qasm_simulator"
    shots = 1024
    quantum_adiabatic = QuantumAdiabatic(n, h, J, backend, shots)
