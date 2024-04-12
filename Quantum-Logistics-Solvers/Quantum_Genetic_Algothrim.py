# Code the Traveling Salesman Problem using a quantum genetic algorithm and use variable neighborhood search to improve the solution.
# Code it oop style with plots and all. Use qiskit for the quantum part.

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit


class QuantumTSP:
    def __init__(self, cities, pop_size, generations, mutation_rate, elite_size):
        self.cities = cities
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = [np.random.permutation (len (cities)) for _ in range (pop_size)]
        self.fitness = self.calculate_fitness ()
        self.best_individual = self.population [np.argmin (self.fitness)]
        self.best_fitness = np.min (self.fitness)

    def calculate_fitness(self):
        fitness = []
        for individual in self.population:
            distance = 0
            for i in range (len (individual) - 1):
                distance += np.linalg.norm (self.cities [individual [i]] - self.cities [individual [i + 1]])
            distance += np.linalg.norm (self.cities [individual [-1]] - self.cities [individual [0]])
            fitness.append (distance)
        return fitness

    def select_parents(self):
        fitness = 1 / np.array (self.fitness)
        fitness /= np.sum (fitness)
        parents = [self.population [i] for i in
                   np.random.choice (len (self.population), self.elite_size, p=fitness, replace=False)]
        return parents

    def crossover(self, parents):
        children = []
        for i in range (self.pop_size - self.elite_size):
            parent1 = parents [np.random.randint (len (parents))]
            parent2 = parents [np.random.randint (len (parents))]
            child = np.copy (parent1)
            for j in range (len (child)):
                if np.random.rand () < 0.5:
                    child [j] = parent2 [j]
            children.append (child)
        return children

    def mutate(self, children):
        for i in range (len (children)):
            if np.random.rand () < self.mutation_rate:
                index1, index2 = np.random.choice (len (children [i]), 2, replace=False)
                children [i] [index1], children [i] [index2] = children [i] [index2], children [i] [index1]
        return children

    def create_circuit(self):
        n = len (self.cities)
        qc = QuantumCircuit (n, n)
        qc.h (range (n))
        qc.barrier ()
        for i in range (n):
            for j in range (n):
                if i != j:
                    qc.cp (np.linalg.norm (self.cities [i] - self.cities [j]), i, j)
        qc.barrier ()
        qc.h (range (n))
        qc.barrier ()
        qc.measure (range (n), range (n))
        return qc

    def __str__(self):
        return f'{self.best_individual} {self.best_fitness}'


if __name__ == "__main__":
    cities = np.random.rand (10, 2)
    tsp = QuantumTSP (cities, 100, 100, 0.01, 10)
    for _ in range (tsp.generations):
        parents = tsp.select_parents ()
        children = tsp.crossover (parents)
        children = tsp.mutate (children)
        tsp.population = parents + children
        tsp.fitness = tsp.calculate_fitness ()
        best_index = np.argmin (tsp.fitness)
        if tsp.fitness [best_index] < np.min (tsp.best_fitness):
            tsp.best_individual = tsp.population [best_index]
            tsp.best_fitness = tsp.fitness [best_index]
    print (tsp.best_individual, tsp.best_fitness)
