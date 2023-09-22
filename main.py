# Code the Traveling Salesman Problem using a quantum genetic algorithm and use variable neighborhood search to improve the solution.
# Code it oop style with plots and all. Use qiskit for the quantum part.

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class QuantumTSP:
    def __init__(self, num_cities, pop_size, generations, mutation_rate, crossover_rate, elite_size, num_qubits):
        self.cities = None
        self.num_cities = num_cities
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.num_qubits = num_qubits
        self.population = []
        self.fitness = []
        self.best_fitness = []
        self.best_individual = []

    def initialize_population(self):
        for _ in range(self.pop_size):
            individual = np.random.permutation(self.num_cities)
            self.population.append(individual)

    def calculate_fitness(self):
        self.fitness = []
        for i in range(self.pop_size):
            individual = self.population[i]
            fitness = 0
            for j in range(self.num_cities - 1):
                # Assuming cities are represented as (x, y) coordinates
                city1 = self.cities[individual[j]]
                city2 = self.cities[individual[j + 1]]
                fitness += np.linalg.norm(np.array(city1) - np.array(city2))
            # Add distance from the last city back to the starting city
            first_city = self.cities[individual[0]]
            last_city = self.cities[individual[-1]]
            fitness += np.linalg.norm(np.array(last_city) - np.array(first_city))
            self.fitness.append(fitness)

    def select_parents(self):
        parents = []
        for _ in range(self.elite_size):
            index = np.argmin(self.fitness)
            parents.append(self.population[index])
            self.fitness[index] = np.inf  # Mark as selected
        return parents

    def crossover(self, parents):
        children = []
        for _ in range(self.pop_size - self.elite_size):
            parent1 = parents[np.random.randint(0, self.elite_size)]
            parent2 = parents[np.random.randint(0, self.elite_size)]
            start = np.random.randint(0, self.num_cities)
            end = np.random.randint(start + 1, self.num_cities + 1)
            child = [-1] * self.num_cities
            child[start:end] = parent1[start:end]
            remaining_cities = [x for x in parent2 if x not in child]
            for i in range(self.num_cities):
                if child[i] == -1:
                    child[i] = remaining_cities.pop(0)
            children.append(child)
        return children

    def mutate(self, children):
        for i in range(self.pop_size - self.elite_size):
            if np.random.rand() < self.mutation_rate:
                index1, index2 = np.random.choice(self.num_cities, 2, replace=False)
                children[i][index1], children[i][index2] = children[i][index2], children[i][index1]
        return children

    def variable_neighborhood_search(self, children):
        for i in range(self.pop_size - self.elite_size):
            if np.random.rand() < self.mutation_rate:
                index1, index2 = np.random.choice(self.num_cities, 2, replace=False)
                children[i][index1], children[i][index2] = children[i][index2], children[i][index1]
            if np.random.rand() < self.mutation_rate:
                index1, index2 = np.random.choice(self.num_cities, 2, replace=False)
                children[i][index1], children[i][index2] = children[i][index2], children[i][index1]
        return children

    def quantum_circuit(self, children):
        q = QuantumRegister(self.num_qubits, 'q')
        c = ClassicalRegister(self.num_qubits, 'c')
        circuit = QuantumCircuit(q, c)

        for i in range(self.num_qubits):
            circuit.h(q[i])

        for i in range(self.num_cities):
            circuit.cswap(q[children[0][i]], q[children[1][i]], q[children[2][i]])

        for i in range(self.num_qubits):
            circuit.h(q[i])
            circuit.x(q[i])

        circuit.h(q[self.num_qubits - 1])
        circuit.mct(q[0:self.num_qubits - 1], q[self.num_qubits - 1], None, mode='noancilla')
        circuit.h(q[self.num_qubits - 1])

        for i in range(self.num_qubits):
            circuit.x(q[i])
            circuit.h(q[i])

        for i in range(self.num_qubits):
            circuit.measure(q[i], c[i])

        return circuit

    def run(self):
        self.initialize_population()
        self.calculate_fitness()
        for _ in range(self.generations):
            parents = self.select_parents()
            children = self.crossover(parents)
            children = self.mutate(children)
            children = self.variable_neighborhood_search(children)
            self.population = parents + children
            self.calculate_fitness()
            index = np.argmin(self.fitness)
            self.best_fitness.append(self.fitness[index])
            self.best_individual.append(self.population[index])
            self.population[index] = np.inf
            self.fitness[index] = np.inf
        self.best_individual = self.best_individual[np.argmin(self.best_fitness)]
        self.plot()

    def plot(self):
        plt.figure(figsize=(8, 6))
        x = [self.cities[i][0] for i in self.best_individual]
        y = [self.cities[i][1] for i in self.best_individual]
        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('TSP Route')
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    num_cities = 10
    pop_size = 100
    generations = 100
    mutation_rate = 0.01
    crossover_rate = 0.8
    elite_size = 10
    num_qubits = 10

    # Generate random city coordinates
    cities = np.random.rand(num_cities, 2)

    tsp = QuantumTSP(num_cities, pop_size, generations, mutation_rate, crossover_rate, elite_size, num_qubits)
    tsp.cities = cities  # Set the city coordinates
    tsp.run()
