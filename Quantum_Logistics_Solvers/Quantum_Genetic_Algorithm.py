# Code the Traveling Salesman Problem using a quantum genetic algorithm and use variable neighborhood search to improve the solution.
# Code it oop style with plots and all. Use qiskit for the quantum part.

import numpy as np
from qiskit import QuantumCircuit
import plotly.graph_objects as go


class QuantumTSP:
    def __init__(self, cities, pop_size, generations, mutation_rate, elite_size):
        self.cities = cities
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = [np.random.permutation(len(cities)) for _ in range(pop_size)]
        self.fitness = self.calculate_fitness()
        self.best_individual = self.population[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def calculate_fitness(self):
        fitness = []
        for individual in self.population:
            distance = 0
            for i in range(len(individual) - 1):
                distance += np.linalg.norm(self.cities[individual[i]] - self.cities[individual[i + 1]])
            distance += np.linalg.norm(self.cities[individual[-1]] - self.cities[individual[0]])
            fitness.append(distance)
        return fitness

    def select_parents(self):
        fitness = 1 / np.array(self.fitness)
        fitness /= np.sum(fitness)
        rng = np.random.default_rng()
        parents = [self.population[i] for i in rng.choice(len(self.population), self.elite_size, p=fitness, replace=False)]
        return parents

    def crossover(self, parents):
        children = []
        for i in range(self.pop_size - self.elite_size):
            parent1 = parents[rng.choice(len(parents))]
            parent2 = parents[rng.choice(len(parents))]
            child = np.copy(parent1)
            for j in range(len(child)):
                if np.random.rand() < 0.5:
                    child[j] = parent2[j]
            children.append(child)
        return children

    def mutate(self, children):
        for i in range(len(children)):
            if np.random.rand() < self.mutation_rate:
                index1, index2 = np.random.choice(len(children[i]), 2, replace=False)
                children[i][index1], children[i][index2] = children[i][index2], children[i][index1]
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

    # Implement two opt inversion and QAOA for the quantum part
    def two_opt_inversion(self):
        QuantumTSP (self.cities, self.pop_size, self.generations, self.mutation_rate, self.elite_size)

    def qaoa(self):
        tsp_problem = tsp.TspData ('TSP', len (self.cities), np.array (self.cities), self.distances)
        tsp_qubit_op, _ = tsp_get_operator (tsp_problem)

        # Create the QAOA circuit
        qaoa = QAOA (optimizer=COBYLA (), reps=1, quantum_instance=Aer.get_backend ('qasm_simulator'))
        qaoa_result = qaoa.compute_minimum_eigenvalue (tsp_qubit_op)

        # Get the solution
        x = tsp.sample_most_likely (qaoa_result.eigenstate)
        return tsp.get_tsp_solution (x)

    @staticmethod
    def plot_route(cities, individual):
        # Create a trace for the cities
        city_trace = go.Scatter (
            x=cities [:, 0],
            y=cities [:, 1],
            mode='markers',
            name='Cities',
            marker=dict (size=10, color='rgba(255, 0, 0, .8)')
        )

        # Create a trace for the route
        route_trace = go.Scatter (
            x=np.append (cities [individual, 0], cities [individual [0], 0]),
            y=np.append (cities [individual, 1], cities [individual [0], 1]),
            mode='lines+markers',
            name='Route',
            line=dict (color='rgba(0, 0, 255, .8)')
        )

        # Create the layout
        layout = go.Layout (
            title='Best Route',
            xaxis=dict (title='X'),
            yaxis=dict (title='Y'),
            showlegend=True
        )

        # Create the figure and add the traces
        fig = go.Figure (data=[city_trace, route_trace], layout=layout)

        # Show the figure
        fig.show ()

    def __str__(self):
        return f'{self.best_individual} {self.best_fitness}'


if __name__ == "__main__":
    rng = np.random.default_rng()
    cities = rng.random((10, 2))
    tsp = QuantumTSP(cities, 100, 100, 0.01, 10)
    for _ in range(tsp.generations):
        parents = tsp.select_parents()
        children = tsp.crossover(parents)
        children = tsp.mutate(children)
        tsp.population = parents + children
        tsp.fitness = tsp.calculate_fitness()
        best_index = np.argmin(tsp.fitness)
        if tsp.fitness[best_index] < tsp.best_fitness:
            tsp.best_individual = tsp.population[best_index]
            tsp.best_fitness = tsp.fitness[best_index]
    print(tsp.best_individual, tsp.best_fitness)
    QuantumTSP.plot_route(cities, tsp.best_individual)  # visualize the best route
