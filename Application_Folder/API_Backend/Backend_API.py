from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Email
from flask_sqlalchemy import SQLAlchemy
from textblob import TextBlob
from Quantum_Logistics_Solvers.Quantum_Genetic_Algorithm import QuantumTSP

app = Flask (__name__)
app.config ['SECRET_KEY'] = 'your-secret-key'
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy (app)


class User (db.Model):
    id = db.Column (db.Integer, primary_key=True)
    email = db.Column (db.String (120), unique=True, nullable=False)
    category = db.Column (db.String (120), nullable=False)


class Feedback (db.Model):
    id = db.Column (db.Integer, primary_key=True)
    user_email = db.Column (db.String (120), unique=True, nullable=False)
    feedback = db.Column (db.Text, nullable=False)


class RegistrationForm (FlaskForm):
    email = StringField ('Email', validators=[DataRequired (), Email ()])
    user = BooleanField ('User')
    business = BooleanField ('Business')
    submit = SubmitField ('Register')


class FeedbackForm (FlaskForm):
    email = StringField ('Email', validators=[DataRequired (), Email ()])
    feedback = TextAreaField ('Feedback', validators=[DataRequired ()])
    submit = SubmitField ('Submit')


@app.route ('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm ()
    if form.validate_on_submit ():
        user = User (email=form.email.data, category='user' if form.user.data else 'business')
        db.session.add (user)
        db.session.commit ()
        return redirect (url_for (user.category, email=user.email))
    return render_template ('register.html', form=form)


@app.route ('/<category>/<email>', methods=['GET', 'POST'])
def user(category, email):
    return render_template ('user.html', email=email)


@app.route ('/feedback', methods=['GET', 'POST'])
def feedback():
    form = FeedbackForm ()
    if form.validate_on_submit ():
        feedback = Feedback (user_email=form.email.data, feedback=form.feedback.data)
        db.session.add (feedback)
        db.session.commit ()
        sentiment = TextBlob (feedback.feedback).sentiment
        return {"message": "Feedback received", "sentiment": sentiment}, 201
    return render_template ('feedback.html', form=form)


@app.route ('/')
def index():
    return render_template ('index.html')


@app.route ('/run_algorithm', methods=['POST'])
def run_algorithm():
    # Check if the request is JSON
    if not request.is_json:
        return {"error": "Missing JSON in request"}, 400

    # Extract the algorithm name and the number of cities from the request
    algorithm_name = request.json.get ('algorithm')
    num_cities = request.json.get ('num_cities')

    # Check if algorithm_name or num_cities is None
    if algorithm_name is None:
        return {"error": "The algorithm value is required"}, 400
    if num_cities is None:
        return {"error": "The num_cities value is required"}, 400

    # Convert num_cities to an integer
    try:
        num_cities = int(num_cities)
    except ValueError:
        return {"error": "The num_cities value must be an integer"}, 400

    print (f"Running {algorithm_name} with {num_cities} cities")  # Log the number of cities

    # Check if the algorithm is in the Algorithm dictionary
    if algorithm_name in Algorithm:
        # Get the algorithm class
        algorithm_class = Algorithm [algorithm_name]

        # Create an instance of the algorithm
        algorithm = algorithm_class (num_cities, pop_size=100, generations=500, mutation_rate=0.01, elite_size=20)

        # Run the algorithm with the provided number of cities
        try:
            result = algorithm.execute ()  # Call the execute method
        except Exception as e:
            return {"error": str(e)}, 500

        # Print "Plot printed" in the console
        print ("Plot printed")

        # Return the result and the plot as a base64 string
        return {"result": result, "plot": result ['plot']}, 200
    else:
        # Return an error message if the algorithm is not found
        return {"error": "Algorithm not found"}


Algorithm = {
    'Quantum Genetic Algorithm': QuantumTSP
}


if __name__ == '__main__':
    app.run (debug=True)
