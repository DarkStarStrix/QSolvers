import unittest
from unittest.mock import patch, MagicMock
from Quantum_Ant_Colony import QuantumAnt, QuantumAntColony


class QuantumAntColonyOptimizationTests (unittest.TestCase):
    @patch ('Quantum_Ant_Colony.execute')
    def quantum_ant_run_returns_expected_result(self, mock_execute):
        mock_result = MagicMock ()
        mock_result.get_counts.return_value = {'1010': 500, '0101': 500}
        mock_execute.return_value.result.return_value = mock_result
        ant = QuantumAnt (4)
        counts = ant.run ()
        self.assertEqual (counts, {'1010': 500, '0101': 500})

    def quantum_ant_colony_initializes_correct_number_of_ants(self):
        colony = QuantumAntColony (5, 5)
        self.assertEqual (len (colony.ants), 5)

    @patch ('Quantum_Ant_Colony.QuantumAnt.run')
    def quantum_ant_colony_run_calls_ant_run(self, mock_run):
        colony = QuantumAntColony (5, 5)
        colony.run ()
        self.assertEqual (mock_run.call_count, 5)


if __name__ == '__main__':
    unittest.main ()
