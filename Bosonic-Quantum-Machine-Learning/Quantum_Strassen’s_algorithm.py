# code strassen algorithm for matrix multiplication in and quantize it

import numpy as np
import random
from qiskit import QuantumCircuit


class Quantum_Strassen:
    def __init__(self, A, B, threshold):
        self.A = np.pad (A, ((0, 2 ** int (np.log2 (len (A))) - len (A)), (0, 2 ** int (np.log2 (len (A))) - len (A))),
                         'constant')
        self.B = np.pad (B, ((0, 2 ** int (np.log2 (len (B))) - len (B)), (0, 2 ** int (np.log2 (len (B))) - len (B))),
                         'constant')
        self.threshold = threshold
        self.C = self.strassen (self.A, self.B) [:len (A), :len (A)]

    def strassen(self, A, B):
        if len (A) <= self.threshold:
            return np.dot (A, B)
        else:
            n2 = len (A) // 2
            A11, A12, A21, A22 = A [:n2, :n2], A [:n2, n2:], A [n2:, :n2], A [n2:, n2:]
            B11, B12, B21, B22 = B [:n2, :n2], B [:n2, n2:], B [n2:, :n2], B [n2:, n2:]
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

    def get_C(self):
        return self.C


if __name__ == '__main__':
    random.seed (0)
    np.random.seed (0)
    n = 4
    A = np.random.rand (n, n)
    B = np.random.rand (n, n)
    threshold = 2
    quantum_strassen = Quantum_Strassen (A, B, threshold)
    print (quantum_strassen.get_C ())
