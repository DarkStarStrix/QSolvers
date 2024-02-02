import unittest
from unittest.mock import patch, MagicMock
from QAOA_Test import QAOASolver
import networkx as nx


class TestQAOASolver (unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph ()
        self.G.add_edge (0, 1, weight=10)
        self.G.add_edge (0, 2, weight=15)
        self.G.add_edge (0, 3, weight=20)
        self.G.add_edge (1, 2, weight=35)
        self.G.add_edge (1, 3, weight=25)
        self.G.add_edge (2, 3, weight=30)
        self.solver = QAOASolver (self.G, p=1, gamma=0.5, beta=0.5)

    def test_qaoa_circuit_creation(self):
        qc = self.solver.qaoa_circuit ()
        self.assertEqual (len (qc.qubits), self.G.number_of_nodes ())
        self.assertEqual (len (qc.clbits), self.G.number_of_nodes ())

    @patch ('QAOA.execute')
    def test_run_qaoa(self, mock_execute):
        mock_result = MagicMock ()
        mock_result.get_counts.return_value = {'1010': 500, '0101': 500}
        mock_execute.return_value.result.return_value = mock_result
        counts = self.solver.run_qaoa ()
        self.assertEqual (counts, {'1010': 500, '0101': 500})

    @patch ('QAOA.MinimumEigenOptimizer')
    @patch ('QAOA.QAOA')
    @patch ('QAOA.tsp.get_operator')
    def test_solve(self, mock_get_operator, mock_qaoa, mock_meo):
        mock_result = MagicMock ()
        mock_result.eigenstate = '1010'
        mock_meo_instance = mock_meo.return_value
        mock_meo_instance.solve.return_value = mock_result
        result, most_likely = self.solver.solve ()
        self.assertEqual (result, mock_result)
        self.assertEqual (most_likely, '1010')


if __name__ == '__main__':
    unittest.main ()
