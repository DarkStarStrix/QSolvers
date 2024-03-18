# How the Solvers work

## Grover's algorithm 

using the Qiskit library in Python. Grover's algorithm is a quantum algorithm that finds with high probability the unique input to a black box function that produces a particular output value, using just O(sqrt(N)) evaluations of the function, where N is the size of the function's domain.
The first part of the code imports necessary libraries from Qiskit, loads the IBM Q account, and selects the least busy backend device that has at least 3 qubits and is operational.

```python
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram

provider = IBMQ.load_account()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                   not x.configuration().simulator and x.status().operational))
```

Next, a quantum circuit is created with 3 qubits and 3 classical bits. The Hadamard gate is applied to all qubits to create a superposition of all possible states.

```python
qc = QuantumCircuit(3, 3)
qc.h(range(3))
```

The Oracle is then applied. This part of the code flips the sign of the state we are searching for. In this case, the Oracle is hard-coded to flip the sign of the state `|111>`.

```python
qc.x(range(3))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(3))
```

The Grover diffusion operator is applied next. This part of the code amplifies the probability of the state we are searching for.

```python
qc.h(range(3))
qc.x(range(3))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(3))
qc.h(range(3))
```

The qubits are then measured and the results are stored in the classical bits.

```python
qc.measure(range(3), range(3))
```

Finally, the quantum circuit is executed on the selected backend, the results are retrieved, and a histogram of the results is plotted.

```python
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts(qc)
plot_histogram(counts)
```

In summary, this code is a complete implementation of Grover's algorithm for 3 qubits using the Qiskit library. It creates a quantum circuit, applies the Oracle and Grover diffusion operator, measures the qubits, and plots the results.

##  Shor's algorithm 

Shor's algorithm is a quantum algorithm for integer factorization, which underlies the security of many cryptographic systems.
The first part of the code imports necessary libraries from Qiskit, loads the IBM Q account, and selects the least busy backend device that has at least 3 qubits and is operational.

```python
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
import numpy as np

provider = IBMQ.load_account()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                                           not x.configuration().simulator and x.status().operational))
```

Next, a quantum circuit is created with 3 qubits and 3 classical bits. The Hadamard gate is applied to all qubits to create a superposition of all possible states.

```python
qc = QuantumCircuit(3, 3)
qc.h(range(3))
```

The Oracle is then applied. This part of the code flips the sign of the state we are searching for. In this case, the Oracle is hard-coded to flip the sign of the state `|111>`.

```python
qc.x(range(3))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(2))
```

The Shor's algorithm is applied next. This part of the code amplifies the probability of the state we are searching for.

```python
qc.h(range(3))
qc.measure(range(3), range(3))
```

Finally, the quantum circuit is executed on the selected backend, the results are retrieved, and a histogram of the results is plotted.

```python
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts(qc)
plot_histogram(counts)
```

In summary, this code is a complete implementation of Shor's algorithm for 3 qubits using the Qiskit library. It creates a quantum circuit, applies the Oracle and Shor's algorithm, measures the qubits, and plots the results.

##  Variational Quantum Eigensolver (VQE) 

The first part of the code imports necessary libraries from Qiskit and numpy. It also loads the IBM Q account and selects the least busy backend device that has at least 3 qubits and is operational.

```python
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.visualization import plot_histogram
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.operators import WeightedPauliOperator
import numpy as np

provider = IBMQ.load_account()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                       not x.configuration().simulator and x.status().operational))
```

Next, a quantum circuit is created with 3 qubits and 3 classical bits. The Hadamard gate is applied to all qubits to create a superposition of all possible states. An Oracle is then applied which flips the sign of the state we are searching for.

```python
qc = QuantumCircuit(3, 3)
qc.h(range(3))
qc.x(range(3))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(3))
```

The VQE algorithm is then applied. This part of the code defines the Hamiltonian, the variational form, and the optimizer. The Hamiltonian is defined using a dictionary of Pauli operators, the variational form is defined using the RY gate, and the optimizer is defined using the COBYLA method.

```python
pauli_dict = {
    'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
              {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "ZI"},
              {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "IZ"},
              {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
              {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}]
}
qubit_op = WeightedPauliOperator.from_dict(pauli_dict)
var_form = RY(qubit_op.num_qubits, depth=3, entanglement='linear')
optimizer = COBYLA(maxiter=1000)
vqe = VQE(qubit_op, var_form, optimizer)
```

Finally, the quantum circuit is executed on the selected backend, the results are retrieved, and a histogram of the results is plotted. The VQE algorithm is then executed on the backend.

```python
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts(qc)
plot_histogram(counts)
result = vqe.run(backend)
```

In summary, this code is a complete implementation of the VQE algorithm for 3 qubits using the Qiskit library. It creates a quantum circuit, applies the Oracle and VQE algorithm, measures the qubits, plots the results, and executes the VQE algorithm.

##  Boson Sampling 

Boson Sampling is a type of quantum computing algorithm that is used to simulate the behavior of photons in a linear optical system.
The code begins by importing the necessary libraries: numpy for numerical computations, QuTiP (Quantum Toolbox in Python) for quantum computations, and plotly for data visualization.

```python
import numpy as np
import qutip as qt
import plotly.graph_objects as go
```
A class named `BosonSampling` is defined with three methods: `__init__`, `simulate`, and `plot`. The `__init__` method initializes the class with three parameters: `n` (the number of photons), `m` (the number of modes), and `U` (the unitary matrix representing the linear optical system). The unitary matrix `U` is converted into a tensor product of quantum objects using the `qt.tensor` and `qt.Qobj` functions from the QuTiP library.

```python
class BosonSampling:
    def __init__(self, n, m, U):
        self.n = n
        self.m = m
        self.U = qt.tensor ([qt.Qobj (U) for _ in range (self.n)])
```
The `simulate` method creates an initial state as a tensor product of basis states, applies the unitary transformation `U` to this state, and returns the absolute square of the final state, which represents the probability distribution of the output states.

```python
def simulate(self):
    initial_state = qt.tensor ([qt.basis (self.m, 0) for _ in range (self.n)])
    final_state = self.U * initial_state
    return abs (final_state.full ()) ** 2
```
The `plot` method creates a bar plot of the output state probabilities using the plotly library.

```python
def plot(self, prob):
    fig = go.Figure ()
    fig.add_trace (go.Bar (x=list (range (self.m)), y=prob [0]))
    fig.update_layout (title="Boson Sampling", xaxis_title="Output state", yaxis_title="Probability")
    fig.show ()
```
In the main part of the code, an instance of the `BosonSampling` class is created with `n=3` photons, `m=5` modes, and a random unitary matrix `U`. The `simulate` method is called to compute the output state probabilities, and the `plot` method is called to visualize these probabilities.

```python
if __name__ == "__main__":
    n = 3
    m = 5
    U = np.random.rand (m, m) + 1j * np.random.rand (m, m)
    boson_sampling = BosonSampling (n, m, U)
    prob = boson_sampling.simulate ()
    boson_sampling.plot (prob)
    print (prob)
```

In summary, this code is a complete implementation of Boson Sampling for a given number of photons and modes using the QuTiP library in Python. It creates a BosonSampling object, simulates the behavior of photons in a linear optical system, and visualizes the output state probabilities.

## BosonSampling 2

The first part of the code imports the necessary libraries and modules. The BosonSampling class is imported from the Quantum_Walk_Solvers library, numpy is imported for numerical computations, and matplotlib.pyplot is imported for data visualization.

```python
from Quantum_Walk_Solvers.Boson_Sampling import BosonSampling
import numpy as np
import matplotlib.pyplot as plt
```
Next, an instance of the BosonSampling class is created with `n=3` photons, `m=5` modes, and a random unitary matrix `U`. The `simulate` method is then called on this instance to compute the output state probabilities.
```python
n = 3
m = 5
U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
boson_sampling = BosonSampling(n, m, U)
prob = boson_sampling.simulate()
```

The code then analyzes the output state probabilities by computing the mean, median, standard deviation, and variance using numpy's built-in functions. These statistics are printed to the console.

```python
mean = np.mean(prob)
median = np.median(prob)
std_dev = np.std(prob)
variance = np.var(prob)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
```

Finally, the code creates a histogram of the output state probabilities using matplotlib. The histogram has 20 bins, a title, labels for the x and y axes, and a grid.

```python
plt.figure(figsize=(10, 6))
plt.hist(prob, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

In summary, this code is a complete implementation of Boson Sampling simulation, analysis, and visualization in Python. It creates a BosonSampling object, simulates the Boson Sampling, analyzes the output state probabilities, and visualizes these probabilities as a histogram.

## Quantum Random Number Generator (QRNG). 

The QRNG is encapsulated in a class named `QuantumRandomNumberGenerator`. this quantum random number generator uses a quantum state to generate random numbers.
The class is initialized with a single parameter `n`, which represents the number of qubits. In the `__init__` method, a density matrix of a quantum state is created using the `qutip.rand_dm` function. This density matrix is then normalized and converted into a full matrix and a numpy array.

```python
def __init__(self, n):
    self.n = n
    self.qubits = qutip.rand_dm (2 ** n, density=1)
    self.qubits = self.qubits.unit ()
    self.qubits = self.qubits.ptrace (0)
    self.qubits = self.qubits.full ()
    self.qubits = np.array (self.qubits)
```

The `get_random_number` method generates a random number based on the quantum state. It calculates the absolute values of the quantum state, normalizes them to get probabilities, and then uses numpy's `random.choice` function to generate a random number based on these probabilities.

```python
def get_random_number(self):
    abs_qubits = np.absolute (self.qubits [0])
    probabilities = abs_qubits / np.sum (abs_qubits)
    return np.random.choice ([0, 1], p=probabilities)
```

The `get_random_numbers` method generates a specified number of random numbers by calling the `get_random_number` method in a loop.

```python
def get_random_numbers(self, num):
    return [self.get_random_number () for _ in range (num)]
```

The `plot_probability_distribution` method visualizes the probability distribution of the quantum state using a bar plot from the plotly library.

```python
def plot_probability_distribution(self):
    abs_qubits = np.absolute (self.qubits [0])
    probabilities = abs_qubits / np.sum (abs_qubits)
    states = np.arange (len (probabilities))  # create an array of states
    fig = px.bar (x=states, y=probabilities, title='Probability Distribution')
    fig.update_layout (xaxis_title='State', yaxis_title='Probability')
    fig.show ()
```

The `plot_heatmap` method visualizes the density matrix of the quantum state using a heatmap from the plotly library.

```python
def plot_heatmap(self):
    density_matrix = np.absolute(self.qubits)
    # Create labels for the axes
    labels = [f'State {i}' for i in range(len(density_matrix))]
    fig = go.Figure(data=go.Heatmap(z=density_matrix, x=labels, y=labels))
    fig.update_layout(title='Heatmap of Density Matrix')
    fig.show()
```

In the main part of the code, an instance of the `QuantumRandomNumberGenerator` class is created with `n=3` qubits. The probability distribution and the heatmap of the quantum state are then plotted.

```python
n = 3
q = QuantumRandomNumberGenerator (n)
q.plot_probability_distribution ()
q.plot_heatmap ()
```

In summary, this code is a complete implementation of a Quantum Random Number Generator in Python. It creates a QRNG object, generates random numbers based on the quantum state, and visualizes the probability distribution and the density matrix of the quantum state.

## Bernstein-Vazirani algorithm 

The Bernstein-Vazirani algorithm is a quantum algorithm that is used to find a hidden integer from a black box function. The algorithm is encapsulated in a class named `BernsteinVaziraniAlgorithm`.
The first part of the code imports necessary libraries from Qiskit, which is a Python library for quantum computing.

```python
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer
```

Next, a quantum circuit is created with 3 qubits and 3 classical bits. 

```python
q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q, c)
```

The Hadamard gate is applied to each qubit to create a superposition of all possible states.

```python
for qubit in q:
    qc.h(qubit)
```

A barrier is then applied. This is used to separate different parts of the circuit and does not affect the results.

```python
qc.barrier()
```

The inner-product oracle is then applied. This part of the code flips the sign of the state we are searching for.

```python
qc.cx(q[0], q[2])
qc.cx(q[1], q[2])
```

Another barrier is applied before the Hadamard gate is applied to each qubit again.

```python
qc.barrier()
for qubit in q:
    qc.h(qubit)
```

The qubits are then measured and the results are stored in the classical bits.

```python
for i in range(3):
    qc.measure(q[i], c[i])
```

Finally, the quantum circuit is executed on a simulator backend, the results are retrieved, and printed.

```python
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend_sim)
result_sim = job_sim.result()
print("simulation: ", result_sim)
print(result_sim.get_counts(qc))
```

In summary, this code is a complete implementation of the Bernstein-Vazirani algorithm for 3 qubits using the Qiskit library. It creates a quantum circuit, applies the Hadamard gate and the inner-product oracle, measures the qubits, and retrieves and prints the results.

## Quantum Fourier Transform (QFT) 

its inverse, and a method called Quantum Fourier Fishing for a given state vector. It also includes a function to check the equality of two quantum states.
The `qft` function creates a Quantum Fourier Transform circuit on `n` qubits. It starts by initializing a quantum circuit with `n` qubits. Then, it applies a series of Hadamard gates and controlled phase rotation gates to the qubits.

```python
def qft(n):
    """Create a Quantum Fourier Transform circuit on n qubits."""
    circuit = QuantumCircuit (n)
    for j in range (n):
        for k in range (j):
            circuit.cu1 (np.pi / float (2 ** (j - k)), k, j)
        circuit.h (j)
    return circuit
```

The `inverse_qft` function creates an inverse Quantum Fourier Transform circuit on `n` qubits. It also starts by initializing a quantum circuit with `n` qubits. Then, it applies a series of Hadamard gates and controlled phase rotation gates with negative angles to the qubits.

```python
def inverse_qft(n):
    """Create an inverse Quantum Fourier Transform circuit on n qubits."""
    circuit = QuantumCircuit (n)
    for j in range (n):
        circuit.h (j)
        for k in range (j):
            circuit.cu1 (-np.pi / float (2 ** (j - k)), k, j)
    return circuit
```

The `quantum_fourier_fishing` function performs Quantum Fourier Fishing on a state vector. It creates a quantum circuit, initializes the circuit with the state vector, applies the QFT, performs a measurement on all qubits, and executes the circuit on a simulator backend. The function returns the counts of the measurement results.

```python
def quantum_fourier_fishing(n, state_vector):
    """Perform Quantum Fourier Fishing on a state vector."""
    # Create a quantum circuit
    circuit = QuantumCircuit (n)
    # Initialize the circuit with the state vector
    circuit.initialize (state_vector, range (n))
    # Apply the QFT
    circuit.append (qft (n), range (n))
    # Perform a measurement
    circuit.measure_all ()
    # Execute the circuit
    result = execute (circuit, Aer.get_backend ('qasm_simulator')).result ()
    counts = result.get_counts (circuit)
    return counts
```

The `fourier_checking` function checks if the actual counts match the expected counts. It simply compares the two input dictionaries and returns `True` if they are equal and `False` otherwise.

```python
def fourier_checking(expected_counts, actual_counts):
    """Check if the actual counts match the expected counts."""
    return expected_counts == actual_counts
```

In summary, this code is a complete implementation of the Quantum Fourier Transform and its inverse, Quantum Fourier Fishing, and Fourier Checking in Python. It creates quantum circuits, applies the QFT and its inverse, performs Quantum Fourier Fishing on a state vector, and checks the equality of two quantum states.

## Deutsch-Jozsa algorithm for 3 qubits. 

The Deutsch-Jozsa algorithm is a quantum algorithm that can determine whether a function is constant or balanced with only one query, which is a significant improvement over classical algorithms.
The script begins by importing necessary libraries from Qiskit, which is a Python library for quantum computing.

```python
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
```

Next, it loads an IBM Q account and gets the least busy backend device that has at least 3 qubits, is not a simulator, and is operational.

```python
provider = IBMQ.load_account()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                       not x.configuration().simulator and x.status().operational))
```

A quantum circuit is then created with 3 qubits and 3 classical bits.

```python
qc = QuantumCircuit(3, 3)
```

The Hadamard gate is applied to all qubits to create a superposition of all possible states.

```python
qc.h(range(3))
```

An oracle is then applied to the qubits. This oracle is a black box function that flips the sign of the state we are searching for.

```python
qc.x(range(3))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(3))
```

The qubits are then measured and the results are stored in the classical bits.

```python
qc.measure(range(3), range(3))
```

The quantum circuit is then executed on the least busy backend, the results are retrieved, and a histogram of the results is plotted.

```python
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts(qc)
plot_histogram(counts)
```

Finally, the quantum circuit is printed to the console.

```python
print(qc)
```

In summary, this code is a complete implementation of the Deutsch-Jozsa algorithm for 3 qubits using the Qiskit library. It creates a quantum circuit, applies the Hadamard gate and the oracle, measures the qubits, and retrieves and visualizes the results.

## Simon's algorithm. 

Simon's algorithm is a quantum algorithm used to solve a specific black box problem, known as Simon's problem, with a quadratic speedup over the best known classical algorithms.
The script begins by importing necessary libraries from Qiskit, which is a Python library for quantum computing, and numpy for numerical computations.

```python
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.visualization import plot_histogram
from qiskit.aqua.algorithms import Simon
from qiskit.aqua.components.oracles import TruthTableOracle
```

Next, it loads an IBM Q account and gets the backend device 'ibmq_qasm_simulator'.

```python
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_qasm_simulator')
```

The length of the input string is set to 4.

```python
n = 4
```

An oracle is then set up using the `TruthTableOracle` class from Qiskit. Simon's algorithm is a black-box algorithm, so we don't need to know the details of the oracle. The oracle is set to '0011'.

```python
oracle = TruthTableOracle('0011')
```

A Simon algorithm instance is then created with the oracle.

```python
simon = Simon(oracle)
```

The algorithm is then run on the backend and the result is retrieved.

```python
result = simon.run(backend)
```

The result is then printed to the console.

```python
print(result['result'])
```

A histogram of the result is then plotted.

```python
plot_histogram(result['measurement'])
```

Finally, the quantum circuit used in the Simon algorithm is printed to the console.

```python
print(simon.construct_circuit(measurement=True))
```

In summary, this code is a complete implementation of Simon's algorithm using the Qiskit library. It creates a Simon algorithm instance, runs the algorithm on a backend, retrieves and prints the results, and visualizes the results as a histogram.
