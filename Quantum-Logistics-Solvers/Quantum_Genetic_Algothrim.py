# Code the Traveling Salesman Problem using a quantum genetic algorithm and use variable neighborhood search to improve the solution.
# Code it oop style with plots and all. Use qiskit for the quantum part.

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, execute, Aer


class QuantumTSP:
    def __init__(self, cities, pop_size, generations, mutation_rate, elite_size):
        self.cities = cities
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = [np.random.permutation (len (cities)) for _ in range (pop_size)]
        self.fitness = []
        self.best_fitness = []
        self.best_individual = []
        self.calculate_fitness ()

    def calculate_fitness(self):
        self.fitness = [self.calculate_route_length (individual) for individual in self.population]

    def calculate_route_length(self, route):
        return sum (
            np.linalg.norm (np.array (self.cities [route [i]]) - np.array (self.cities [route [i + 1]])) for i in
            range (len (self.cities) - 1)) + np.linalg.norm (
            np.array (self.cities [route [-1]]) - np.array (self.cities [route [0]]))

    def select_parents(self):
        parents = []
        for _ in range (self.elite_size):
            index = np.argmin (self.fitness)
            parents.append (self.population.pop (index))
            self.fitness.pop (index)
        return parents

    def crossover(self, parents):
        return [self.create_child (parents) for _ in range (len (parents))]

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

    def run(self):
        for _ in range (self.generations):
            parents = self.select_parents ()
            children = self.crossover (parents)
            children = self.mutate (children)
            self.population = parents + children
            self.calculate_fitness ()
            self.best_fitness.append (np.min (self.fitness))
            self.best_individual.append (self.population [np.argmin (self.fitness)])

    def plot(self):
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
    tsp = QuantumTSP (cities, pop_size=100, generations=100, mutation_rate=0.01, elite_size=20)
    tsp.run ()
    tsp.plot ()
