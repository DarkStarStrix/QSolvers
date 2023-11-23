import pytest
from Quantum_Genetic_Algothrim import QuantumTSP
import numpy as np


def test_initial_population_creation():
    tsp = QuantumTSP (5, 10, 10, 0.01, 0.8, 2, 5)
    tsp.initialize_population ()
    assert len (tsp.population) == 10
    for individual in tsp.population:
        assert len (individual) == 5
        assert len (set (individual)) == 5


def test_fitness_calculation():
    tsp = QuantumTSP (3, 10, 10, 0.01, 0.8, 2, 3)
    tsp.cities = np.array ([(0, 0), (1, 0), (0, 1)])
    tsp.population = [np.array ([0, 1, 2]) for _ in range (10)]
    tsp.calculate_fitness ()
    assert len (tsp.fitness) == 10
    for fitness in tsp.fitness:
        assert fitness == 2 + np.sqrt (2)


def test_parent_selection():
    tsp = QuantumTSP (3, 10, 10, 0.01, 0.8, 2, 3)
    tsp.population = [np.array ([i, (i + 1) % 3, (i + 2) % 3]) for i in range (10)]
    tsp.fitness = list (range (10))
    parents = tsp.select_parents ()
    assert len (parents) == 2
    assert np.array_equal (parents [0], np.array ([0, 1, 2]))
    assert np.array_equal (parents [1], np.array ([1, 2, 0]))


def test_crossover():
    tsp = QuantumTSP (3, 10, 10, 0.01, 0.8, 2, 3)
    parents = [np.array ([0, 1, 2]), np.array ([2, 0, 1])]
    children = tsp.crossover (parents)
    assert len (children) == 8
    for child in children:
        assert len (set (child)) == 3


def test_mutation():
    tsp = QuantumTSP (3, 10, 10, 1.0, 0.8, 2, 3)
    children = [np.array ([0, 1, 2]) for _ in range (10)]
    mutated_children = tsp.mutate (children)
    for child in mutated_children:
        assert len (set (child)) == 3


def test_variable_neighborhood_search():
    tsp = QuantumTSP (3, 10, 10, 1.0, 0.8, 2, 3)
    children = [np.array ([0, 1, 2]) for _ in range (10)]
    vns_children = tsp.variable_neighborhood_search (children)
    for child in vns_children:
        assert len (set (child)) == 3


def test_quantum_circuit():
    tsp = QuantumTSP (3, 10, 10, 0.01, 0.8, 2, 3)
    children = [np.array ([0, 1, 2]), np.array ([2, 0, 1]), np.array ([1, 2, 0])]
    circuit = tsp.quantum_circuit (children)
    assert circuit.count_ops () ['h'] == 6
    assert circuit.count_ops () ['cswap'] == 3
    assert circuit.count_ops () ['x'] == 6
    assert circuit.count_ops () ['measure'] == 3
