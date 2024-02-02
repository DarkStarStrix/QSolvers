import unittest
from unittest.mock import patch, MagicMock
from QPO_Test import QuantumParticle, QuantumSwarm


class QuantumParticleSwarmOptimizationTests (unittest.TestCase):
    @patch ('QPO.execute')
    def quantum_particle_run_returns_expected_result(self, mock_execute):
        mock_result = MagicMock ()
        mock_result.get_counts.return_value = {'1010': 500, '0101': 500}
        mock_execute.return_value.result.return_value = mock_result
        particle = QuantumParticle (4)
        counts = particle.run ()
        self.assertEqual (counts, {'1010': 500, '0101': 500})

    def quantum_swarm_initializes_correct_number_of_particles(self):
        swarm = QuantumSwarm (5, 5)
        self.assertEqual (len (swarm.particles), 5)

    @patch ('QPO.QuantumParticle.run')
    def quantum_swarm_run_calls_particle_run(self, mock_run):
        swarm = QuantumSwarm (5, 5)
        swarm.run ()
        self.assertEqual (mock_run.call_count, 5)


if __name__ == '__main__':
    unittest.main ()
