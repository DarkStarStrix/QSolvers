from qiskit import QuantumCircuit, execute, Aer
import numpy as np


class QuantumAnt:
    def __init__(self, num_qubits):
        self.circuit = QuantumCircuit (num_qubits)
        self.circuit.h (range (num_qubits))
        self.circuit.measure_all ()

    def run(self):
        return execute (self.circuit, Aer.get_backend ('qasm_simulator'), shots=1).result ().get_counts ()


class QuantumAntColony:
    def __init__(self, num_ants, num_qubits):
        self.ants = [QuantumAnt (num_qubits) for _ in range (num_ants)]

    def run(self):
        return [ant.run () for ant in self.ants]


num_cities = 5
distances = np.random.rand (num_cities, num_cities)
np.fill_diagonal (distances, 0)
distances = (distances + distances.T) / 2

colony = QuantumAntColony (num_cities, num_cities)
print (colony.run ())
