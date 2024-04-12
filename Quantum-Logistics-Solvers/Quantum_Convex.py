# code the TSP problem and solve it using a quantum convex approach in oop style

# Define TSP problem (e.g., cities and distances)
import numpy as np
from qiskit import QuantumCircuit


class TSP:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances
        self.qc = self.create_circuit ()

    def create_circuit(self):
        n = len (self.distances)
        qc = QuantumCircuit (n, n)
        qc.h (range (n))
        qc.barrier ()
        for i in range (n):
            for j in range (n):
                if i != j:
                    qc.cp (self.distances [i] [j], i, j)
        qc.barrier ()
        qc.h (range (n))
        qc.barrier ()
        qc.measure (range (n), range (n))
        return qc


def main():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    x = tsp.qc.draw ()
    print (x)


if __name__ == '__main__':
    main ()
