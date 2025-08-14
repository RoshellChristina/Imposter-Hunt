from flask import Flask, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import config
from models import db, User, Prediction
from forms import RegistrationForm, LoginForm, PredictForm
from utils import predict_real_text_id
import pymysql

pymysql.install_as_MySQLdb()
app = Flask(__name__)
app.config.from_object(config.Config)
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET','POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        user = User(
            name=form.name.data.strip(),
            email=form.email.data.strip().lower(),
            username=form.username.data.strip(),
            password=hashed
        )
        db.session.add(user)
        db.session.commit()
        flash('Account created!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        u = User.query.filter_by(username=form.username.data).first()
        if u and check_password_hash(u.password, form.password.data):
            login_user(u)
            return redirect(url_for('dashboard'))
        flash('Login failed', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard', methods=['GET','POST'])
@login_required
def dashboard():
    form = PredictForm()
    if form.validate_on_submit():
        t1, t2 = form.text1.data, form.text2.data
        res = predict_real_text_id(t1, t2)
        p = Prediction(text1=t1, text2=t2, result=res, user_id=current_user.id)
        db.session.add(p); db.session.commit()
        return render_template('result.html', result=res, text1=t1, text2=t2)
    return render_template('dashboard.html', form=form)

@app.route('/history')
@login_required
def history():
    preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.id.desc()).all()
    return render_template('history.html', predictions=preds)

if __name__ == '__main__':
    app.run(debug=True)
