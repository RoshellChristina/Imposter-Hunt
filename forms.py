from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Email
from models import User

class RegistrationForm(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired(), Length(1, 50)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(5, 120)])
    username = StringField('Username', validators=[DataRequired(), Length(2, 20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(6, 128)])
    confirm = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already exists.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data.lower()).first()
        if user:
            raise ValidationError('Email already registered.')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class PredictForm(FlaskForm):
    text1 = TextAreaField('Text 1', validators=[DataRequired(), Length(max=20000)])
    text2 = TextAreaField('Text 2', validators=[DataRequired(), Length(max=20000)])
    submit = SubmitField('Predict')






