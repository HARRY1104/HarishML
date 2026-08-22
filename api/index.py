import os
import pickle
from flask import Flask, request, render_template

API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(API_DIR)

template_dir = os.path.join(BASE_DIR, 'templates')
model_path = os.path.join(BASE_DIR, 'models', 'ridge.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

app = Flask(__name__, template_folder=template_dir)

# Initialize global holders
ridge_model = None
scaler_model = None

# Safely attempt to load models and print exact errors to Vercel Logs
try:
    print(f"Checking BASE_DIR: {BASE_DIR}")
    print(f"Looking for model at: {model_path}")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            ridge_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler_model = pickle.load(f)
        print("Models loaded successfully!")
    else:
        print(f"FILE NOT FOUND ERROR: Check folder/file casing. Model exists: {os.path.exists(model_path)}, Scaler exists: {os.path.exists(scaler_path)}")
except Exception as e:
    print(f"CRITICAL PICKLE ERROR: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if ridge_model is None or scaler_model is None:
            return "Error: Model files failed to load on server startup.", 500

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