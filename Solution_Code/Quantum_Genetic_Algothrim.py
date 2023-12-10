# Code the Traveling Salesman Problem using a quantum genetic algorithm and use variable neighborhood search to improve the solution.
# Code it oop style with plots and all. Use qiskit for the quantum part.

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class QuantumTSP:
    def __init__(self, cities, pop_size, generations, mutation_rate, elite_size, num_qubits):
        self.cities = cities
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.num_qubits = num_qubits
        self.population = [np.random.permutation (len (cities)) for _ in range (pop_size)]
        self.fitness = []
        self.best_fitness = []
        self.best_individual = []

    def calculate_fitness(self):
        self.fitness = [self.calculate_route_length (individual) for individual in self.population]

    def calculate_route_length(self, route):
        return sum (
            np.linalg.norm (np.array (self.cities [route [i]]) - np.array (self.cities [route [i + 1]])) for i in
            range (len (self.cities) - 1)) + np.linalg.norm (
            np.array (self.cities [route [-1]]) - np.array (self.cities [route [0]]))

    def select_parents(self):
        parents = []
        for _ in range (min (self.elite_size, len (self.population))):
            parents.append (self.population.pop (np.argmin (self.fitness)))
        return parents

    def crossover(self, parents):
        return [self.create_child (parents) for _ in range (self.pop_size - self.elite_size)]

    def create_child(self, parents):
        parent1, parent2 = np.random.choice (len (parents), 2, replace=False)
        parent1, parent2 = parents [parent1], parents [parent2]
        start, end = sorted (np.random.choice (len (self.cities), 2, replace=False))
        child = [-1] * len (self.cities)
        child [start:end] = parent1 [start:end]
        remaining_cities = [city for city in parent2 if city not in child]
        child = [remaining_cities.pop (0) if city == -1 else city for city in child]
        return child

    def mutate(self, children):
        for child in children:
            if np.random.rand () < self.mutation_rate:
                index1, index2 = np.random.choice (len (self.cities), 2, replace=False)
                child [index1], child [index2] = child [index2], child [index1]
        return children

    def variable_neighborhood_search(self, children):
        for child in children:
            if np.random.rand () < self.mutation_rate:
                self.local_search (child)
        return children

    def local_search(self, child):
        improved = True
        while improved:
            improved = False
            for index1 in range (len (self.cities) - 1):
                for index2 in range (index1 + 2, len (self.cities) + int (index1 > 0)):
                    new_route = child [:index1] + child [index1:index2] [::-1] + child [index2:]
                    if self.calculate_route_length (new_route) < self.calculate_route_length (child):
                        child [:] = new_route
                        improved = True

    def run(self):
        self.calculate_fitness ()
        for _ in range (self.generations):
            parents = self.select_parents ()
            children = self.crossover (parents)
            children = self.mutate (children)
            children = self.variable_neighborhood_search (children)
            self.population = parents + children
            self.calculate_fitness ()
            index = np.argmin (self.fitness)
            self.best_fitness.append (self.fitness [index])
            self.best_individual.append (self.population [index])
        self.plot ()

    def plot(self):
        plt.figure (figsize=(8, 6))
        x, y = zip (*[self.cities [i] for i in self.best_individual [np.argmin (self.best_fitness)]])
        plt.plot (x, y, marker='o', linestyle='-')
        plt.xlabel ('X Coordinate')
        plt.ylabel ('Y Coordinate')
        plt.title ('TSP Route')
        plt.grid (True)
        plt.show ()

    def plot_fitness(self):
        plt.figure (figsize=(8, 6))
        plt.plot (self.best_fitness)
        plt.xlabel ('Generation')
        plt.ylabel ('Fitness')
        plt.title ('Fitness over Generations')
        plt.grid (True)
        plt.show ()

    def plot_circuit(self):
        qr = QuantumRegister (self.num_qubits)
        cr = ClassicalRegister (self.num_qubits)
        qc = QuantumCircuit (qr, cr)
        for i in range (self.num_qubits):
            qc.h (qr [i])
        qc.measure (qr, cr)
        qc.draw (output='mpl')
        plt.show ()

    def plot_histogram(self):
        qr = QuantumRegister (self.num_qubits)
        cr = ClassicalRegister (self.num_qubits)
        qc = QuantumCircuit (qr, cr)
        for i in range (self.num_qubits):
            qc.h (qr [i])
        qc.measure (qr, cr)
        qc.draw (output='mpl', style='iqp')
        plt.show ()

    def plot_cities(self):
        plt.figure (figsize=(8, 6))
        x, y = zip (*self.cities)
        plt.plot (x, y, marker='o', linestyle='')
        plt.xlabel ('X Coordinate')
        plt.ylabel ('Y Coordinate')
        plt.title ('Cities')
        plt.grid (True)
        plt.show ()

    def plot_route(self):
        plt.figure (figsize=(8, 6))
        x, y = zip (*[self.cities [i] for i in self.best_individual [np.argmin (self.best_fitness)]])
        plt.plot (x, y, marker='o', linestyle='-')
        plt.xlabel ('X Coordinate')
        plt.ylabel ('Y Coordinate')
        plt.title ('TSP Route')
        plt.grid (True)
        plt.show ()


if __name__ == "__main__":
    cities = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160), (100, 160), (200, 160), (140, 140), (40, 120),
              (100, 120), (180, 100), (60, 80), (120, 80), (180, 60), (20, 40), (100, 40), (200, 40), (20, 20),
              (60, 20), (160, 20)]
    tsp = QuantumTSP (cities, pop_size=100, generations=100, mutation_rate=0.01, elite_size=20, num_qubits=10)
    tsp.run ()
    tsp.plot_fitness ()
    tsp.plot_circuit ()
    tsp.plot_histogram ()
    tsp.plot_cities ()
    tsp.plot_route ()
