import os
import pickle
from flask import Flask, request, jsonify, render_template

# 1. Get the directory containing index.py (/api)
API_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Get the root project directory (one level above /api)
BASE_DIR = os.path.dirname(API_DIR)

# 3. Explicitly construct absolute paths to templates, static, and models
template_dir = os.path.join(BASE_DIR, 'templates')
model_path = os.path.join(BASE_DIR, 'Models', 'ridge.pkl')
scaler_path = os.path.join(BASE_DIR, 'Models', 'scaler.pkl')

app = Flask(__name__, template_folder=template_dir)

# 4. Safely load model and scaler files
ridge_model = None
scaler_model = None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, 'rb') as f:
        ridge_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        new_data = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        prediction = ridge_model.predict(new_data)

        return render_template('home.html', result=prediction[0])
    else:
        return render_template('home.html')