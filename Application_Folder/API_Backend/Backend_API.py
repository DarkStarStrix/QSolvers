from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Email
from flask_sqlalchemy import SQLAlchemy
import gzip
from textblob import TextBlob

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
        user = User (email=form.email.data)
        if form.user.data:
            user.category = 'user'
        elif form.business.data:
            user.category = 'business'
        db.session.add (user)
        db.session.commit ()
        if user.category == 'user':
            return redirect (url_for ('user', email=user.email))
        else:
            return redirect (url_for ('business', email=user.email))
    return render_template ('register.html', form=form)


@app.route ('/user/<email>', methods=['GET', 'POST'])
def user(email):
    # Handle form submission and redirect to payment page
    return render_template ('user.html', email=email)


@app.route ('/business/<email>', methods=['GET', 'POST'])
def business(email):
    # Handle form submission and redirect to payment page
    return render_template ('business.html', email=email)


@app.route ('/pay/<email>', methods=['GET', 'POST'])
def pay(email):
    status = request.args.get ('status')
    if status == 'success':
        return render_template ('success.html')
    elif status == 'failure':
        return render_template ('failure.html')
    return render_template ('pay.html', email=email)


@app.route ('/feedback', methods=['GET', 'POST'])
def feedback():
    form = FeedbackForm ()
    if form.validate_on_submit ():
        feedback = Feedback (user_email=form.email.data, feedback=form.feedback.data)
        db.session.add (feedback)
        db.session.commit ()
        return redirect (url_for ('index'))
    return render_template ('feedback.html', form=form)


@app.route ('/feedback', methods=['GET', 'POST'])
def feedback():
    form = FeedbackForm ()
    if form.validate_on_submit ():
        feedback = Feedback (user_email=form.email.data, feedback=form.feedback.data)
        db.session.add (feedback)
        db.session.commit ()

        # Save feedback to a compressed text file
        with gzip.open ('feedback.txt.gz', 'wt') as f:
            f.write (feedback.feedback)

        # Run sentiment analysis on the feedback
        blob = TextBlob (feedback.feedback)
        sentiment = blob.sentiment

        return {"message": "Feedback received", "sentiment": sentiment}, 201

    return render_template ('feedback.html', form=form)


@app.route ('/')
def index():
    return render_template ('index.html')


if __name__ == '__main__':
    app.run (debug=True)
