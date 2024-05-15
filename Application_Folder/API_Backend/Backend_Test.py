import unittest
from Backend_API import app, db, User, Feedback


class FlaskTestCase (unittest.TestCase):

    def setUp(self):
        self.app = app.test_client ()
        self.app_context = app.app_context ()
        self.app_context.push ()
        self.app.testing = True
        db.create_all ()

    def tearDown(self):
        db.session.remove ()
        db.drop_all ()
        self.app_context.pop ()

    def test_index(self):
        response = self.app.get ('/')
        self.assertEqual (response.status_code, 200)

    def test_register(self):
        response = self.app.post ('/register', data=dict (email="test@test.com", user=True), follow_redirects=True)
        user = User.query.filter_by (email="test@test.com").first ()
        self.assertIsNotNone (user)
        self.assertEqual (response.status_code, 200)

    def test_feedback(self):
        response = self.app.post ('/feedback', data=dict (email="test@test.com", feedback="This is a test feedback"),
                                  follow_redirects=True)
        feedback = Feedback.query.filter_by (user_email="test@test.com").first ()
        self.assertIsNotNone (feedback)
        self.assertEqual (response.status_code, 200)

    def test_run_algorithm(self):
        response = self.app.post ('/run_algorithm', json=dict (algorithm="Quantum Genetic Algorithm", num_cities=5),
                                  follow_redirects=True)
        self.assertEqual (response.status_code, 200)


if __name__ == '__main__':
    unittest.main ()
