from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from geopy.distance import geodesic
import pandas as pd
import os
import json
from datetime import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mines.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

class Mine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'active' or 'defused'
    file_source = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()
        # Create predefined users if they don't exist
        for i in range(1, 6):
            username = f'user{i}'
            if not User.query.filter_by(username=username).first():
                user = User(
                    username=username,
                    password_hash=generate_password_hash(username)  # Password same as username
                )
                db.session.add(user)
        db.session.commit()

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    files = [f for f in os.listdir('data') if f.startswith('mine_locations_')]
    return render_template('dashboard.html', files=files)

@app.route('/api/mines', methods=['GET'])
@login_required
def get_mines():
    file_name = request.args.get('file')
    if not file_name:
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        mines = Mine.query.filter_by(file_source=file_name).all()
        mines_data = [{
            'id': mine.id,
            'latitude': mine.latitude,
            'longitude': mine.longitude,
            'status': mine.status
        } for mine in mines]
        return jsonify(mines_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-file', methods=['POST'])
@login_required
def load_file():
    file_name = request.json.get('file')
    if not file_name:
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Clear existing mines for this file
        Mine.query.filter_by(file_source=file_name).delete()
        
        # Load new mines
        df = pd.read_csv(os.path.join('data', file_name))
        for _, row in df.iterrows():
            mine = Mine(
                latitude=row['latitude'],
                longitude=row['longitude'],
                status=row.get('status', 'active'),  # Default to active if status not provided
                file_source=file_name
            )
            db.session.add(mine)
        db.session.commit()
        
        return jsonify({'message': 'File loaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/defuse-mine', methods=['POST'])
@login_required
def defuse_mine():
    mine_id = request.json.get('mine_id')
    if not mine_id:
        return jsonify({'error': 'No mine ID provided'}), 400
    
    try:
        mine = Mine.query.get(mine_id)
        if not mine:
            return jsonify({'error': 'Mine not found'}), 404
        
        mine.status = 'defused'
        db.session.commit()
        
        # Notify all clients about the update
        socketio.emit('mine_defused', {
            'mine_id': mine_id,
            'latitude': mine.latitude,
            'longitude': mine.longitude
        })
        
        return jsonify({'message': 'Mine defused successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nearest-mine', methods=['POST'])
@login_required
def find_nearest_mine():
    user_lat = request.json.get('latitude')
    user_lon = request.json.get('longitude')
    
    if not user_lat or not user_lon:
        return jsonify({'error': 'User location not provided'}), 400
    
    try:
        active_mines = Mine.query.filter_by(status='active').all()
        if not active_mines:
            return jsonify({'message': 'No active mines found'}), 404
        
        user_location = (user_lat, user_lon)
        nearest_mine = min(
            active_mines,
            key=lambda m: geodesic(user_location, (m.latitude, m.longitude)).meters
        )
        
        return jsonify({
            'id': nearest_mine.id,
            'latitude': nearest_mine.latitude,
            'longitude': nearest_mine.longitude
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    # socketio.run(app, debug=True)
    app.run(host='0.0.0.0', port=5001, ssl_context=('ssl.crt', 'ssl.key'))

