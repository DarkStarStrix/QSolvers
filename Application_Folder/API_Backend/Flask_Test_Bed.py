import unittest
from flask import Flask
from Flask_app import app, load_user, execute_algorithm
from unittest.mock import patch, MagicMock


class TestFlaskApp (unittest.TestCase):
    def setUp(self):
        self.app = app.test_client ()
        self.app.testing = True

    def test_successful_login(self):
        with patch ('Flask_app.login_user') as mock_login:
            response = self.app.post ('/login', data={'username': 'test_user'})
            mock_login.assert_called_once ()
            self.assertEqual (response.status_code, 302)

    def test_logout(self):
        with patch ('Flask_app.logout_user') as mock_logout:
            response = self.app.get ('/logout')
            mock_logout.assert_called_once ()
            self.assertEqual (response.status_code, 302)

    def test_load_user(self):
        user = load_user ('test_user')
        self.assertEqual (user.id, 'test_user')

    def test_run_algorithm_with_invalid_algorithm(self):
        response = self.app.post ('/run_algorithm', json={'algorithm': 'Invalid'})
        self.assertEqual (response.status_code, 400)

    def test_run_algorithm_with_valid_algorithm(self):
        with patch ('Flask_app.execute_algorithm.delay') as mock_execute:
            mock_execute.return_value = MagicMock (id='test_task_id')
            response = self.app.post ('/run_algorithm', json={'algorithm': 'Quantum Genetic Algorithm'})
            mock_execute.assert_called_once ()
            self.assertEqual (response.status_code, 202)
            self.assertEqual (response.get_json (), {'task_id': 'test_task_id'})

    def test_check_task_with_pending_state(self):
        with patch ('Flask_app.execute_algorithm.AsyncResult') as mock_result:
            mock_result.return_value = MagicMock (state='PENDING')
            response = self.app.get ('/check_task/test_task_id')
            self.assertEqual (response.status_code, 200)
            self.assertEqual (response.get_json (), {'state': 'PENDING', 'status': 'Pending...'})

    def test_check_task_with_failure_state(self):
        with patch ('Flask_app.execute_algorithm.AsyncResult') as mock_result:
            mock_result.return_value = MagicMock (state='FAILURE', info='Test error')
            response = self.app.get ('/check_task/test_task_id')
            self.assertEqual (response.status_code, 200)
            self.assertEqual (response.get_json (), {'state': 'FAILURE', 'status': 'Test error'})

    def test_execute_algorithm(self):
        with patch ('Flask_app.AmazonBraketClient') as mock_client:
            mock_client.return_value.create_quantum_task.return_value = {'taskId': 'test_task_id'}
            mock_client.return_value.get_quantum_task.return_value = {'status': 'COMPLETED'}
            result = execute_algorithm (None, lambda: QuantumCircuit (2))
            self.assertEqual (result, {'status': 'COMPLETED'})


if __name__ == '__main__':
    unittest.main ()
