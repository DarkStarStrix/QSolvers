import pytest
import numpy as np
from Solution_Code.Quantum_Genetic_Algothrim import QuantumTSP
import matplotlib.pyplot as plt


def test_quantum_genetic_algorithm_process():
    # Initialize the QuantumTSP instance with a list of cities
    cities = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)]
    tsp = QuantumTSP(cities, 10, 10, 0.01, 8, 2)

    # Test population initialization
    tsp.initialize_population()
    assert len(tsp.population) == 10
    for individual in tsp.population:
        assert len(individual) == 5
        assert len(set(individual)) == 5

    # Test fitness calculation
    tsp.cities = np.array([(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)])
    tsp.calculate_fitness()
    assert len(tsp.fitness) == 10

    # Test parent selection
    parents = tsp.select_parents()
    assert len(parents) == 8

    # Test crossover
    children = tsp.crossover(parents)  # Fixed typo here
    assert len(children) == len(parents)  # The number of children should be equal to the number of parents
    for child in children:
        assert len(set(child)) == 5

    # Test mutation
    mutated_children = tsp.mutate(children)
    for child in mutated_children:
        assert len(set(child)) == 5

    # Test variable neighborhood search
    vns_children = tsp.variable_neighborhood_search(children)
    for child in vns_children:
        assert len(set(child)) == 5

    # plot the test results over the generations
    plt.plot(tsp.best_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over the generations')
    plt.show()  # Display the plot
