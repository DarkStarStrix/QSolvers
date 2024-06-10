from setuptools import setup, find_packages

setup (
    name="QSolvers",
    version="0.1",
    packages=find_packages (),
)
# Path: QSolvers/QSolvers-API.py

# Methods for the API

# When the user installs the QSolvers package, they can use the API to run the algorithms. The API provides the following methods:
# For the quantum logistics algorithms:
# they would first do from QSolvers import QuantumLogisticSolvers.quantumgenetic as qg
# then they would do qg.run_algorithm('genetic', 10) to run the genetic algorithm with 10 cities.
# then they would get the output of the algorithm.plot quantum circuit and counts of the algorithm.


