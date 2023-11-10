# code the traveling salesman problem using a quantum infused A* algorithm use oop

import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor


# make Traveling Salesman Problem class
def get_start_city_index():
    return 0


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


# make a class for the quantum A* algorithm
class QuantumAStar:
    def __init__(self, tsp):
        self.qc = None
        self.tsp = tsp
        self.start_city = tsp.get_start_city ()
        self.cities_list = tsp.get_cities_list ()
        self.cities_list_without_start = tsp.get_cities_list_without_start ()
        self.number_of_cities = tsp.get_number_of_cities ()
        self.distance_matrix = tsp.get_distance_matrix ()
        self.distance_matrix_without_start = self.distance_matrix [1:] [:, 1:]
        self.make_qc ()
        self.qc.draw (output='mpl')
        plt.show ()

    def make_qc(self):
        # make a quantum circuit
        qc = QuantumCircuit (self.number_of_cities, self.number_of_cities)
        # put the start city in the first position
        qc.x (0)
        qc.barrier ()
        # put all the other cities in superposition
        for i in range (1, self.number_of_cities):
            qc.h (i)
        qc.barrier ()
        # make a controlled swap gate
        for i in range (1, self.number_of_cities):
            qc.cswap (0, i, i)
        qc.barrier ()

        # make a controlled rotation gate
        for i in range (1, self.number_of_cities):
            qc.cu3 (self.distance_matrix_without_start [0] [i - 1], 0, 0, 0, i)
        qc.barrier ()

        # make main loop
        for i in range (1, self.number_of_cities):
            # make a controlled swap gate
            for j in range (1, self.number_of_cities):
                qc.cswap (i, j, j)
            qc.barrier ()
            # make a controlled rotation gate
            for j in range (1, self.number_of_cities):
                qc.cu3 (self.distance_matrix_without_start [i] [j - 1], 0, 0, i, j)
            qc.barrier ()

        # make main function
        qc.measure (range (self.number_of_cities), range (self.number_of_cities))
        self.qc = qc

    def run_qc(self):
        # run the quantum circuit
        backend = Aer.get_backend ('qasm_simulator')
        job = execute (self.qc, backend, shots=1000)
        result = job.result ()
        counts = result.get_counts ()
        return counts

    def get_best_path(self, counts):
        # get the best path from the counts
        best_path = None
        best_path_cost = None
        for path in counts:
            cost = self.get_cost (path)
            if best_path_cost is None or cost < best_path_cost:
                best_path = path
                best_path_cost = cost
        return best_path, best_path_cost

    def get_cost(self, path):
        # get the cost of a path
        cost = 0
        for i in range (self.number_of_cities):
            if path [i] == '1':
                cost += self.distance_matrix [self.start_city] [i]
        return cost

    def get_path(self, path):
        # get the path
        path_list = []
        for i in range (self.number_of_cities):
            if path [i] == '1':
                path_list.append (self.cities_list [i])
        return path_list

    def get_path_cost(self, path):
        # get the cost of a path
        cost = 0
        for i in range (self.number_of_cities):
            if path [i] == '1':
                cost += self.distance_matrix [self.start_city] [i]
        return cost

    def get_path_distance(self, path):
        # get the distance of a path
        distance = 0
        for i in range (self.number_of_cities - 1):
            distance += self.distance_matrix [path [i]] [path [i + 1]]
        return distance

    def get_path_distance_matrix(self, path):
        # get the distance matrix of a path
        distance_matrix = np.zeros ((self.number_of_cities, self.number_of_cities))
        for i in range (self.number_of_cities - 1):
            distance_matrix [path [i]] [path [i + 1]] = 1
        return distance_matrix

    def get_path_distance_matrix_without_start(self, path):
        # get the distance matrix of a path
        distance_matrix = np.zeros ((self.number_of_cities - 1, self.number_of_cities - 1))
        for i in range (self.number_of_cities - 2):
            distance_matrix [path [i + 1]] [path [i + 2]] = 1
        return distance_matrix


# make a class for the classical A* algorithm
def main():
    # make a list of cities
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # make a distance matrix
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    # make a tsp
    tsp = TSP (cities, distances)
    # make a quantum A* algorithm
    quantum_a_star = QuantumAStar (tsp)
    # run the quantum A* algorithm
    counts = quantum_a_star.run_qc ()
    # get the best path
    best_path, best_path_cost = quantum_a_star.get_best_path (counts)
    # print the best path
    print ('best path:', quantum_a_star.get_path (best_path))
    # print the best path cost
    print ('best path cost:', best_path_cost)
    # make a classical A* algorithm
    classical_a_star = ClassicalAStar (tsp)
    # run the classical A* algorithm
    counts = classical_a_star.run_qc ()
    # get the best path
    best_path, best_path_cost = classical_a_star.get_best_path (counts)
    # print the best path
    print ('best path:', classical_a_star.get_path (best_path))
    # print the best path cost
    print ('best path cost:', best_path_cost)

    # run the main


if __name__ == '__main__':
    main()
