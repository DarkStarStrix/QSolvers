import unittest
from Backend_API import app
from unittest.mock import patch


class TestBackendAPI (unittest.TestCase):
    def setUp(self):
        self.app = app.test_client ()
        self.app.testing = True

    def test_registration(self):
        for data in [{'email': 'test_user@test.com', 'user': True},
                     {'email': 'test_business@test.com', 'business': True}, {'email': 'test_no_category@test.com'}]:
            with patch ('Backend_API.db.session.add'), patch ('Backend_API.db.session.commit'):
                response = self.app.post ('/register', data=data)
                self.assertIn (response.status_code, [200, 302])

    def test_route_access(self):
        for route in ['/user/test_user@test.com', '/business/test_business@test.com', '/pay/test_user@test.com']:
            response = self.app.get (route)
            self.assertEqual (response.status_code, 200)

    def test_pay_route_with_status(self):
        for status in ['success', 'failure', None]:
            response = self.app.get ('/pay/test_user@test.com', query_string={'status': status} if status else {})
            self.assertEqual (response.status_code, 200)


if __name__ == '__main__':
    unittest.main ()
