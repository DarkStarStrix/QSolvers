# code the TSP problem and solve it using a quantum convex approach in oop style

# Define TSP problem (e.g., cities and distances)
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.algorithms.optimizers import COBYLA
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer


class TSP:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances


def create_circuit(distances):
    n = len (distances)
    q = QuantumRegister (n, 'q')
    c = ClassicalRegister (n, 'c')
    qc = QuantumCircuit (q, c)
    qc.h (q)
    qc.barrier ()
    for i in range (n):
        for j in range (n):
            if i != j:
                qc.cp (distances [i] [j], q [i], q [j])
    qc.barrier ()
    qc.h (q)
    qc.barrier ()
    qc.measure (q, c)
    return qc


optimizer = COBYLA (maxiter=1000)
backend = Aer.get_backend ('qasm_simulator')
quantum_instance = QuantumInstance (backend, shots=1000)
quadratic_program = QuadraticProgram ()
vqe = VQE (quantum_instance=quantum_instance)
minimum_eigen_optimizer = MinimumEigenOptimizer (vqe)


def main():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    TSP (cities, distances)
    create_circuit (distances)
    result = minimum_eigen_optimizer.solve (quadratic_program)
    x = result.x


if __name__ == '__main__':
    main ()
