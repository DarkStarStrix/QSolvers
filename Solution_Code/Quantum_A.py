import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor


class TSP:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances

    def get_cities(self):
        return self.cities

    def get_distances(self):
        return self.distances

    def get_distance(self, city1, city2):
        return self.distances [city1] [city2]

    def get_distance_matrix(self):
        return self.distances

    def get_number_of_cities(self):
        return len (self.cities)

    def get_cities_list(self):
        return list (self.cities.keys ())

    def get_cities_list_without_start(self):
        return list (self.cities.keys ()) [1:]

    def get_start_city(self):
        return list (self.cities.keys ()) [0]


class QuantumAStar:
    def __init__(self, tsp):
        self.qc = None
        self.tsp = tsp
        self.start_city = tsp.get_start_city ()
        self.cities_list = tsp.get_cities_list ()
        self.cities_list_without_start = tsp.get_cities_list_without_start ()
        self.number_of_cities = tsp.get_number_of_cities ()
        self.distance_matrix = tsp.get_distance_matrix ()
        self.distance_matrix_without_start = [row [1:] for row in self.distance_matrix [1:]]
        self.make_qc ()
        self.qc.draw (output='mpl')
        plt.show ()

    def make_qc(self):
        self.qc = QuantumCircuit (self.number_of_cities, self.number_of_cities)
        self.qc.h (range (self.number_of_cities))
        self.qc.barrier ()
        for i in range (self.number_of_cities):
            for j in range (self.number_of_cities):
                if i != j:
                    self.qc.cp (2 * np.arcsin (np.sqrt (self.distance_matrix_without_start [i] [j] / 10)), i, j)

    self.qc.barrier ()
    self.qc.h (range (self.number_of_cities))
    self.qc.barrier ()
    self.qc.measure (range (self.number_of_cities), range (self.number_of_cities))

    def run_qc(self):
        backend = Aer.get_backend ('qasm_simulator')
        job = execute (self.qc, backend, shots=1000)
        result = job.result ()
        counts = result.get_counts ()
        return counts

    def get_best_path(self, counts):
        best_path = None
        best_path_cost = None
        for path in counts:
            cost = self.get_cost (path)
            if best_path_cost is None or cost < best_path_cost:
                best_path = path
                best_path_cost = cost
        return best_path, best_path_cost

    def get_cost(self, path):
        cost = 0
        for i in range (self.number_of_cities):
            if path [i] == '1':
                cost += self.distance_matrix [self.start_city] [i]
        return cost

    def get_path(self, path):
        path_list = []
        for i in range (self.number_of_cities):
            if path [i] == '1':
                path_list.append (self.cities_list [i])
        return path_list


def main():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    quantum_a_star = QuantumAStar (tsp)
    counts = quantum_a_star.run_qc ()
    best_path, best_path_cost = quantum_a_star.get_best_path (counts)
    print ('best path:', quantum_a_star.get_path (best_path))
    print ('best path cost:', best_path_cost)


if __name__ == '__main__':
    main ()
