from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from hmmlearn import hmm
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import folium
from PIL import Image, ImageTk
import math
import os
import pickle
from datetime import datetime
from flask_cors import CORS
import sqlite3
from functools import wraps
from werkzeug.utils import secure_filename
import csv

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create required directories
for directory in [UPLOAD_FOLDER, 'models', 'data']:
    if not os.path.exists(directory):
        os.makedirs(directory)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class MultiModelPredictor:
    def __init__(self):
        self.mine_locations = []
        self.predicted_locations = []
        self.success_count = 0
        self.failure_count = 0
        self.failure_attempts = 0
        self.max_failure_attempts = 3
        self.pattern_type = "Unknown"
        self.pattern_data = []
        
        # Initialize models
        self.arima_model = None
        self.rf_model = None
        self.hmm_model = None
        self.prophet_model = None
        
        # Model weights
        self.weights = {
            'arima': 0.90,
            'rf': 0.025,
            'hmm': 0.025,
            'prophet': 0.05
        }
        
        # Model accuracy tracking
        self.model_accuracy = {
            'arima': {'correct': 0, 'total': 0, 'accuracy': 0.0, 'mae': 0.0},
            'rf': {'correct': 0, 'total': 0, 'accuracy': 0.0, 'mae': 0.0},
            'hmm': {'correct': 0, 'total': 0, 'accuracy': 0.0, 'mae': 0.0},
            'prophet': {'correct': 0, 'total': 0, 'accuracy': 0.0, 'mae': 0.0}
        }
        
        self.last_predictions = {}
        self.load_model()

    def prepare_data(self, pattern_data):
        if len(pattern_data) < 2:
            return None, None, None, None, None
            
        distances = np.array([p['distance'] for p in pattern_data])
        bearings = np.array([p['bearing'] for p in pattern_data])
        
        X = np.column_stack([
            distances[:-1],
            bearings[:-1],
            np.diff(distances),
            np.diff(bearings)
        ])
        y_distance = distances[1:]
        y_bearing = bearings[1:]
        
        return X, y_distance, y_bearing, distances, bearings

    def train_models(self, pattern_data):
        if len(pattern_data) < 9:
            return {"error": "Need at least 10 mine locations to train the model."}
            
        try:
            X, y_distance, y_bearing, distances, bearings = self.prepare_data(pattern_data)
            
            # Train ARIMA
            self.arima_distance = ARIMA(distances, order=(3, 1, 3))
            self.arima_bearing = ARIMA(bearings, order=(3, 1, 3))
            self.arima_distance_fit = self.arima_distance.fit()
            self.arima_bearing_fit = self.arima_bearing.fit()
            
            # Train Random Forest
            self.rf_distance = RandomForestRegressor(n_estimators=100)
            self.rf_bearing = RandomForestRegressor(n_estimators=100)
            self.rf_distance.fit(X, y_distance)
            self.rf_bearing.fit(X, y_bearing)
            
            # Train HMM
            self.hmm_model = hmm.GaussianHMM(n_components=3, n_iter=100)
            self.hmm_model.fit(np.column_stack([distances, bearings]))
            
            # Validate and normalize the transition matrix
            transmat = self.hmm_model.transmat_
            for i, row in enumerate(transmat):
                row_sum = np.sum(row)
                if not np.isclose(row_sum, 1.0):
                    print(f"Row {i} of transition matrix does not sum to 1. Normalizing...")
                    self.hmm_model.transmat_[i] = row / row_sum
            
            # Train Prophet
            df_prophet = pd.DataFrame({
                'ds': pd.date_range(start='2023-01-01', periods=len(distances)),
                'y_distance': distances,
                'y_bearing': bearings
            })
            
            self.prophet_distance = Prophet()
            self.prophet_bearing = Prophet()
            self.prophet_distance.fit(df_prophet[['ds', 'y_distance']].rename(columns={'y_distance': 'y'}))
            self.prophet_bearing.fit(df_prophet[['ds', 'y_bearing']].rename(columns={'y_bearing': 'y'}))
            
            self.save_model()
            return {"message": "Models trained successfully"}
        except Exception as e:
            return {"error": str(e)}

    def predict_next(self):
        if len(self.mine_locations) < 10:
            return {"error": "Need at least 10 mine locations to make predictions."}
            
        try:
            predictions = {}
            
            # ARIMA prediction
            arima_distance = self.arima_distance_fit.forecast(steps=1)[0]
            arima_bearing = self.arima_bearing_fit.forecast(steps=1)[0]
            predictions['arima'] = (arima_distance, arima_bearing)
            
            # Random Forest prediction
            X, _, _, distances, bearings = self.prepare_data(self.pattern_data)
            rf_distance = self.rf_distance.predict(X[-1:])[-1]
            rf_bearing = self.rf_bearing.predict(X[-1:])[-1]
            predictions['rf'] = (rf_distance, rf_bearing)
            
            # HMM prediction
            hmm_pred = self.hmm_model.sample(1)[0][0]
            predictions['hmm'] = (hmm_pred[0], hmm_pred[1])
            
            # Prophet prediction
            future_dates = pd.DataFrame({
                'ds': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=len(self.pattern_data))]
            })
            prophet_distance = self.prophet_distance.predict(future_dates)['yhat'].iloc[-1]
            prophet_bearing = self.prophet_bearing.predict(future_dates)['yhat'].iloc[-1]
            predictions['prophet'] = (prophet_distance, prophet_bearing)
            
            # Store predictions for accuracy calculation
            self.last_predictions = predictions
            
            # Weighted ensemble prediction
            final_distance = sum(self.weights[model] * pred[0] for model, pred in predictions.items())
            final_bearing = sum(self.weights[model] * pred[1] for model, pred in predictions.items())
            
            last_mine = self.mine_locations[-1]
            new_lat, new_lon = self.calculate_new_position(
                last_mine[0], last_mine[1], final_distance, final_bearing)
            
            self.predicted_locations.append((new_lat, new_lon))
            
            confidence = 0
            if len(self.pattern_data) >= 3:
                distance_std = np.std([p['distance'] for p in self.pattern_data[-3:]])
                bearing_std = np.std([p['bearing'] for p in self.pattern_data[-3:]])
                distance_confidence = max(0, min(100, 100 - distance_std / 5))
                bearing_confidence = max(0, min(100, 100 - bearing_std / 3))
                confidence = (distance_confidence + bearing_confidence) / 2
            
            return {
                "latitude": new_lat,
                "longitude": new_lon,
                "confidence": confidence,
                "mine_locations": self.mine_locations
            }
        except Exception as e:
            return {"error": str(e)}

    def calculate_distance(self, point1, point2):
        try:
            lat1, lon1 = point1
            lat2, lon2 = point2
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            return c * r * 1000  # Return distance in meters
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 0

    def calculate_bearing(self, point1, point2):
        try:
            lat1, lon1 = point1
            lat2, lon2 = point2
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            bearing = math.atan2(y, x)
            bearing = math.degrees(bearing)
            bearing = (bearing + 360) % 360
            return bearing
        except Exception as e:
            print(f"Error calculating bearing: {e}")
            return 0

    def calculate_new_position(self, lat, lon, distance, bearing):
        try:
            lat, lon, bearing = map(math.radians, [lat, lon, bearing])
            R = 6371000
            lat2 = math.asin(math.sin(lat) * math.cos(distance/R) + 
                            math.cos(lat) * math.sin(distance/R) * math.cos(bearing))
            lon2 = lon + math.atan2(math.sin(bearing) * math.sin(distance/R) * math.cos(lat),
                                   math.cos(distance/R) - math.sin(lat) * math.sin(lat2))
            lat2, lon2 = map(math.degrees, [lat2, lon2])
            return lat2, lon2
        except Exception as e:
            print(f"Error calculating new position: {e}")
            return lat, lon

    def update_pattern_data(self):
        if len(self.mine_locations) < 2:
            return
            
        prev_mine = self.mine_locations[-2]
        curr_mine = self.mine_locations[-1]
        
        distance = self.calculate_distance(prev_mine, curr_mine)
        bearing = self.calculate_bearing(prev_mine, curr_mine)
        
        self.pattern_data.append({
            'prev_lat': prev_mine[0],
            'prev_lon': prev_mine[1],
            'curr_lat': curr_mine[0],
            'curr_lon': curr_mine[1],
            'distance': distance,
            'bearing': bearing
        })

    def detect_pattern(self):
        if len(self.mine_locations) < 5:
            return {"error": "Need at least 5 mine locations to detect pattern."}
            
        try:
            df = pd.DataFrame(self.pattern_data)
            distances = df['distance'].values
            bearings = df['bearing'].values
            
            bearing_std = np.std(bearings)
            bearing_diffs = np.abs(np.diff(bearings))
            bearing_diffs = np.mod(bearing_diffs, 180)
            grid_score = np.std(bearing_diffs)
            zigzag_score = np.mean(np.abs(np.diff(np.diff(bearings))))
            
            if bearing_std < 15:
                self.pattern_type = "Linear"
                confidence = max(0, min(100, 100 - bearing_std * 3))
            elif grid_score < 20 and np.mean(bearing_diffs) > 70 and np.mean(bearing_diffs) < 110:
                self.pattern_type = "Grid"
                confidence = max(0, min(100, 100 - grid_score * 2))
            elif zigzag_score < 30:
                self.pattern_type = "Zigzag"
                confidence = max(0, min(100, 100 - zigzag_score * 1.5))
            else:
                self.pattern_type = "Complex/Random"
                confidence = 50
            
            return {
                "pattern": self.pattern_type,
                "confidence": confidence
            }
        except Exception as e:
            return {"error": str(e)}

    def save_model(self):
        try:
            if not os.path.exists("models"):
                os.makedirs("models")
            
            model_data = {
                'arima_distance_fit': self.arima_distance_fit,
                'arima_bearing_fit': self.arima_bearing_fit,
                'rf_distance': self.rf_distance,
                'rf_bearing': self.rf_bearing,
                'hmm_model': self.hmm_model,
                'prophet_distance': self.prophet_distance,
                'prophet_bearing': self.prophet_bearing,
                'model_accuracy': self.model_accuracy,
                'pattern_type': self.pattern_type,
                'success_count': self.success_count,
                'failure_count': self.failure_count
            }
            
            with open("models/multimodel.pkl", "wb") as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self):
        try:
            if os.path.exists("models/multimodel.pkl"):
                with open("models/multimodel.pkl", "rb") as f:
                    model_data = pickle.load(f)
                
                self.arima_distance_fit = model_data['arima_distance_fit']
                self.arima_bearing_fit = model_data['arima_bearing_fit']
                self.rf_distance = model_data['rf_distance']
                self.rf_bearing = model_data['rf_bearing']
                self.hmm_model = model_data['hmm_model']
                self.prophet_distance = model_data['prophet_distance']
                self.prophet_bearing = model_data['prophet_bearing']
                self.model_accuracy = model_data['model_accuracy']
                self.pattern_type = model_data['pattern_type']
                self.success_count = model_data['success_count']
                self.failure_count = model_data['failure_count']
                
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Initialize predictor
predictor = MultiModelPredictor()

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or not session.get('is_admin'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Database initialization
def init_db():
    try:
        conn = sqlite3.connect('instance/mines.db')
        c = conn.cursor()
        
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE NOT NULL,
                     password TEXT NOT NULL,
                     is_admin INTEGER DEFAULT 0,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Add default admin user if not exists
        c.execute("SELECT * FROM users WHERE username = 'admin'")
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ('admin', 'admin', 1))
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

init_db()

# Routes
@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            conn = sqlite3.connect('instance/mines.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                     (username, password))
            user = c.fetchone()
            conn.close()
            
            if user:
                session['username'] = username
                session['is_admin'] = bool(user[3])
                return redirect(url_for('admin_dashboard' if user[3] else 'dashboard'))
            else:
                return render_template('login.html', error="Invalid credentials")
        except Exception as e:
            return render_template('login.html', error=str(e))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html')
# home.html has defuse feature in build not working perfect
#  index.html have no in build defuse feature



@app.route('/admin/admin_dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

# @app.route('/dashboard')
# @login_required
# def dashboard():
#     data_folder = 'data'
#     files = []
    
#     if os.path.exists(data_folder):
#         all_files = os.listdir(data_folder)
#         files = [f for f in all_files if f.endswith('.csv')]
    
#     return render_template('dashboard.html', files=files)

@app.route('/dashboard')
@login_required
def dashboard():
    data_folder = 'data'
    files = []
    
    if os.path.exists(data_folder):
        all_files = os.listdir(data_folder)
        files = [f for f in all_files if f.endswith('.csv')]
    
    return render_template('dashboard.html', files=files)

@app.route('/admin/about')
def about():
    return render_template('about.html')

@app.route('/admin/users')
@admin_required
def manage_users():
    try:
        conn = sqlite3.connect('instance/mines.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users")
        users = c.fetchall()
        conn.close()
        return render_template('manage_users.html', users=users)
    except Exception as e:
        return render_template('manage_users.html', error=str(e))

@app.route('/admin/add_user', methods=['POST'])
@admin_required
def admin_add_user():
    try:
        username = request.form['username']
        password = request.form['password']
        is_admin = bool(int(request.form.get('is_admin', 0)))
        
        conn = sqlite3.connect('instance/mines.db')
        c = conn.cursor()
        
        # Check if username already exists
        c.execute("SELECT username FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            return render_template('manage_users.html', error="Username already exists")
        
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                 (username, password, is_admin))
        conn.commit()
        conn.close()
        
        return redirect(url_for('manage_users'))
    except Exception as e:
        return render_template('manage_users.html', error=str(e))
@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    try:
        conn = sqlite3.connect('instance/mines.db')
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id = ? AND username != 'admin'", (user_id,))
        conn.commit()
        conn.close()
        return redirect(url_for('manage_users'))
    except Exception as e:
        return render_template('manage_users.html', error=str(e))


# API Routes
@app.route('/api/add_location', methods=['POST'])
@login_required
def add_location():
    data = request.get_json()
    try:
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({
                "error": "Invalid coordinates. Latitude must be between -90 and 90, and Longitude between -180 and 180."
            }), 400
        
        predictor.mine_locations.append((lat, lon))
        if len(predictor.mine_locations) >= 2:
            predictor.update_pattern_data()
        
        if len(predictor.mine_locations) >= 10:
            predictor.train_models(predictor.pattern_data)
            pattern_result = predictor.detect_pattern()
        else:
            pattern_result = None
        
        return jsonify({
            "message": "Location added successfully",
            "total_locations": len(predictor.mine_locations),
            "pattern": pattern_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/admin/navigate')
@login_required
def navigate():
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['GET'])
@login_required
def predict():
    result = predictor.predict_next()
    return jsonify(result)

@app.route('/api/detect_pattern', methods=['GET'])
@login_required
def detect_pattern():
    result = predictor.detect_pattern()
    return jsonify(result)

@app.route('/api/feedback', methods=['POST'])
@login_required
def feedback():
    data = request.get_json()
    success = data.get('success', False)
    
    if not predictor.predicted_locations:
        return jsonify({"error": "No prediction to provide feedback for"}), 400
    
    try:
        if not success:
            predictor.failure_attempts += 1
            if predictor.failure_attempts >= predictor.max_failure_attempts:
                predictor.predicted_locations = []
                predictor.failure_count += 1
                return jsonify({
                    "message": "Maximum failure attempts reached",
                    "success_count": predictor.success_count,
                    "failure_count": predictor.failure_count
                })
            
            # Get the last predicted location
            failed_location = predictor.predicted_locations.pop()
            predictor.failure_count += 1
            
            # Create a circle around the failed location
            failed_lat, failed_lon = failed_location
            failure_circle = {
                "latitude": failed_lat,
                "longitude": failed_lon,
                "radius": 50,  # 3 meters
                "color": "red",  # Red color for failure
                "opacity": 0.5  # Low opacity
            }
            
            return jsonify({
                "message": "Failure marked and circle created",
                "failure_circle": failure_circle,
                "success_count": predictor.success_count,
                "failure_count": predictor.failure_count
            })
        else:
            predictor.success_count += 1
            predictor.failure_attempts = 0
            predicted = predictor.predicted_locations[-1]
            predictor.mine_locations.append(predicted)
            predictor.update_pattern_data()
            predictor.train_models(predictor.pattern_data)
            predictor.predicted_locations.pop()
        
        total = predictor.success_count + predictor.failure_count
        accuracy = (predictor.success_count / total * 100) if total > 0 else 0
        
        return jsonify({
            "message": "Feedback recorded successfully",
            "success_count": predictor.success_count,
            "failure_count": predictor.failure_count,
            "accuracy": accuracy
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400  
@app.route('/api/statistics', methods=['GET'])
@login_required
def statistics():
    total = predictor.success_count + predictor.failure_count
    accuracy = (predictor.success_count / total * 100) if total > 0 else 0
    
    return jsonify({
        "success_count": predictor.success_count,
        "failure_count": predictor.failure_count,
        "accuracy": accuracy,
        "pattern_type": predictor.pattern_type,
        "total_locations": len(predictor.mine_locations)
    })

@app.route('/api/locations', methods=['GET'])
@login_required
def get_locations():
    return jsonify({
        "mine_locations": predictor.mine_locations,
        "predicted_locations": predictor.predicted_locations
    })

@app.route('/api/clear_locations', methods=['POST'])
@login_required
def clear_locations():
    try:
        predictor.mine_locations = []
        predictor.predicted_locations = []
        predictor.pattern_data = []
        predictor.pattern_type = "Unknown"
        predictor.success_count = 0
        predictor.failure_count = 0
        predictor.failure_attempts = 0
        
        return jsonify({"message": "All locations cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_locations', methods=['POST'])
@login_required
def save_locations():
    try:
        data = request.get_json()
        locations = data.get('locations', [])
        
        if not locations:
            return jsonify({"error": "No locations provided"}), 400
        
        if not os.path.exists('data'):
            os.makedirs('data')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"mine_locations_{timestamp}.csv"
        file_path = os.path.join('data', file_name)
        
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['latitude', 'longitude'])
            writer.writerows(locations)
        
        return jsonify({
            "message": "Locations saved successfully",
            "file_path": file_path,
            "file_name": file_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_csv', methods=['POST'])
@login_required
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            df = pd.read_csv(filepath)
            required_columns = ['latitude', 'longitude']
            
            if not all(col in df.columns for col in required_columns):
                os.remove(filepath)
                return jsonify({"error": "CSV must contain 'latitude' and 'longitude' columns"}), 400
            
            predictor.mine_locations = []
            predictor.predicted_locations = []
            predictor.pattern_data = []
            
            for _, row in df.iterrows():
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                
                predictor.mine_locations.append((lat, lon))
                if len(predictor.mine_locations) >= 2:
                    predictor.update_pattern_data()
            
            os.remove(filepath)
            
            if len(predictor.mine_locations) >= 10:
                predictor.train_models(predictor.pattern_data)
                pattern_result = predictor.detect_pattern()
            else:
                pattern_result = None
            
            return jsonify({
                "message": f"Successfully loaded {len(predictor.mine_locations)} locations",
                "total_locations": len(predictor.mine_locations),
                "pattern": pattern_result
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400


@app.route('/api/clear_data', methods=['POST'])
@login_required
def clear_data():
    try:
        # Connect to the database
        conn = sqlite3.connect('instance/mines.db')
        c = conn.cursor()

        # Clear all data from relevant tables
        c.execute("DELETE FROM mine_locations")
        c.execute("DELETE FROM predictions")
        c.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()

        return jsonify({"message": "All data has been cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/api/load-file', methods=['POST'])
# @login_required
# def load_file():
#     try:
#         data = request.get_json()
#         file_name = data.get('file')
        
#         if not file_name:
#             return jsonify({"error": "No file specified"}), 400
        
#         file_path = os.path.join('data', file_name)
        
#         if not os.path.exists(file_path):
#             return jsonify({"error": "File not found"}), 404
        
#         df = pd.read_csv(file_path)
#         predictor.mine_locations = []
#         predictor.predicted_locations = []
#         predictor.pattern_data = []
        
#         for _, row in df.iterrows():
#             lat = float(row['latitude'])
#             lon = float(row['longitude'])
#             predictor.mine_locations.append((lat, lon))
#             if len(predictor.mine_locations) >= 2:
#                 predictor.update_pattern_data()
        
#         if len(predictor.mine_locations) >= 10:
#             predictor.train_models(predictor.pattern_data)
#             pattern_result = predictor.detect_pattern()
#         else:
#             pattern_result = None
        
#         return jsonify({
#             "message": f"File '{file_name}' loaded successfully",
#             "total_locations": len(predictor.mine_locations),
#             "pattern": pattern_result
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route('/api/load-file', methods=['POST'])
@login_required
def load_file():
    try:
        # Get the file name from the request
        data = request.get_json()
        file_name = data.get('file')
        
        if not file_name:
            return jsonify({"error": "No file specified"}), 400
        
        # Construct the file path
        file_path = os.path.join('data', file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        predictor.mine_locations = []
        predictor.predicted_locations = []
        predictor.pattern_data = []
        
        # Process the file data
        for _, row in df.iterrows():
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            predictor.mine_locations.append((lat, lon))
            if len(predictor.mine_locations) >= 2:
                predictor.update_pattern_data()
        
        # Train models if enough data is available
        if len(predictor.mine_locations) >= 10:
            predictor.train_models(predictor.pattern_data)
            pattern_result = predictor.detect_pattern()
        else:
            pattern_result = None
        
        return jsonify({
            "message": f"File '{file_name}' loaded successfully",
            "total_locations": len(predictor.mine_locations),
            "pattern": pattern_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Check if SSL certificate and key exist, if not generate them

    
    context = ('ssl.crt', 'ssl.key')
    app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=True)