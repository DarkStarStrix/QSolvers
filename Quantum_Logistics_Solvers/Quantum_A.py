from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram


class TSP:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances


class QuantumAStar:
    def __init__(self, tsp):
        self.tsp = tsp
        self.qc = self.make_qc ()

    def make_qc(self):
        qc = QuantumCircuit (len (self.tsp.cities), len (self.tsp.cities))
        qc.h (range (len (self.tsp.cities)))
        qc.measure (range (len (self.tsp.cities)), range (len (self.tsp.cities)))
        return qc

    def run_qc(self):
        backend = Aer.get_backend ('qasm_simulator')
        job = backend.run (self.qc, shots=1024)
        result = job.result ()
        counts = result.get_counts ()
        return counts


def main():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    quantum_a_star = QuantumAStar (tsp)
    counts = quantum_a_star.run_qc ()
    print (counts)

    plot_histogram (counts)


if __name__ == '__main__':
    main ()
