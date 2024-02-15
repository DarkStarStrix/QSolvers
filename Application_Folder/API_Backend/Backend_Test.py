import unittest
from flask import Flask
from Backend_API import app, User, db
from unittest.mock import patch, MagicMock


class TestBackendAPI (unittest.TestCase):
    def setUp(self):
        self.app = app.test_client ()
        self.app.testing = True

    def registration_with_user_category(self):
        with patch ('Backend_API.db.session.add') as mock_add, patch ('Backend_API.db.session.commit') as mock_commit:
            response = self.app.post ('/register', data={'email': 'test_user@test.com', 'user': True})
            mock_add.assert_called_once ()
            mock_commit.assert_called_once ()
            self.assertEqual (response.status_code, 302)

    def registration_with_business_category(self):
        with patch ('Backend_API.db.session.add') as mock_add, patch ('Backend_API.db.session.commit') as mock_commit:
            response = self.app.post ('/register', data={'email': 'test_business@test.com', 'business': True})
            mock_add.assert_called_once ()
            mock_commit.assert_called_once ()
            self.assertEqual (response.status_code, 302)

    def registration_with_no_category(self):
        response = self.app.post ('/register', data={'email': 'test_no_category@test.com'})
        self.assertEqual (response.status_code, 200)

    def user_route_access(self):
        response = self.app.get ('/user/test_user@test.com')
        self.assertEqual (response.status_code, 200)

    def business_route_access(self):
        response = self.app.get ('/business/test_business@test.com')
        self.assertEqual (response.status_code, 200)

    def pay_route_with_success_status(self):
        response = self.app.get ('/pay/test_user@test.com', query_string={'status': 'success'})
        self.assertEqual (response.status_code, 200)

    def pay_route_with_failure_status(self):
        response = self.app.get ('/pay/test_user@test.com', query_string={'status': 'failure'})
        self.assertEqual (response.status_code, 200)

    def pay_route_with_no_status(self):
        response = self.app.get ('/pay/test_user@test.com')
        self.assertEqual (response.status_code, 200)


if __name__ == '__main__':
    unittest.main ()
