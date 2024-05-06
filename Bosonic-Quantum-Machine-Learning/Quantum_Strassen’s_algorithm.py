# code strassen algorithm for matrix multiplication in and quantize it

import numpy as np
import random
from qiskit import QuantumCircuit


class QuantumStrassen:
    def __init__(self, matrix_a, matrix_b, threshold):
        self.matrix_a = np.pad (matrix_a, ((0, 2 ** int (np.log2 (len (matrix_a))) - len (matrix_a)),
                                           (0, 2 ** int (np.log2 (len (matrix_a))) - len (matrix_a))), 'constant')
        self.matrix_b = np.pad (matrix_b, ((0, 2 ** int (np.log2 (len (matrix_b))) - len (matrix_b)),
                                           (0, 2 ** int (np.log2 (len (matrix_b))) - len (matrix_b))), 'constant')
        self.threshold = threshold
        self.matrix_c = self.strassen (self.matrix_a, self.matrix_b) [:len (matrix_a), :len (matrix_a)]

    def strassen(self, matrix_a, matrix_b):
        if len (matrix_a) <= self.threshold:
            return np.dot (matrix_a, matrix_b)
        else:
            n2 = len (matrix_a) // 2
            A11, A12, A21, A22 = matrix_a [:n2, :n2], matrix_a [:n2, n2:], matrix_a [n2:, :n2], matrix_a [n2:, n2:]
            B11, B12, B21, B22 = matrix_b [:n2, :n2], matrix_b [:n2, n2:], matrix_b [n2:, :n2], matrix_b [n2:, n2:]
            P1 = self.strassen (A11, B12 - B22)
            P2 = self.strassen (A11 + A12, B22)
            P3 = self.strassen (A21 + A22, B11)
            P4 = self.strassen (A22, B21 - B11)
            P5 = self.strassen (A11 + A22, B11 + B22)
            P6 = self.strassen (A12 - A22, B21 + B22)
            P7 = self.strassen (A11 - A21, B11 + B12)
            C = np.zeros ((2 * len (A11), 2 * len (A11)))
            C [:len (A11), :len (A11)] = P5 + P4 - P2 + P6
            C [:len (A11), len (A11):] = P1 + P2
            C [len (A11):, :len (A11)] = P3 + P4
            C [len (A11):, len (A11):] = P5 + P1 - P3 - P7
            return C

    def get_matrix_c(self):
        return self.matrix_c


if __name__ == '__main__':
    random.seed (0)
    np.random.seed (0)
    n = 4
    matrix_a = np.random.rand (n, n)
    matrix_b = np.random.rand (n, n)
    threshold = 2
    quantum_strassen = QuantumStrassen (matrix_a, matrix_b, threshold)
    print (quantum_strassen.get_matrix_c ())
