import os
import pickle
import traceback
from flask import Flask, request, render_template

API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(API_DIR)

# Locate templates directory
template_candidates = [
    os.path.join(API_DIR, 'templates'),
    os.path.join(BASE_DIR, 'templates'),
    os.path.join(os.getcwd(), 'api', 'templates'),
    os.path.join(os.getcwd(), 'templates'),
    '/var/task/api/templates',
    '/var/task/templates'
]
template_dir = next((p for p in template_candidates if os.path.exists(p)), os.path.join(API_DIR, 'templates'))

app = Flask(__name__, template_folder=template_dir)

# Helper function to find files across possible paths in Vercel Serverless environment
def find_file(filename, subfolder='models'):
    candidates = [
        os.path.join(API_DIR, subfolder, filename),
        os.path.join(BASE_DIR, subfolder, filename),
        os.path.join(os.getcwd(), 'api', subfolder, filename),
        os.path.join(os.getcwd(), subfolder, filename),
        os.path.join('/var/task', 'api', subfolder, filename),
        os.path.join('/var/task', subfolder, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

ridge_model = None
scaler_model = None
load_error_message = None

def load_models():
    global ridge_model, scaler_model, load_error_message
    if ridge_model is not None and scaler_model is not None:
        return True, ""
    
    model_path = find_file('ridge.pkl')
    scaler_path = find_file('scaler.pkl')

    if not model_path or not scaler_path:
        load_error_message = f"Model files not found. Searched in: API_DIR={API_DIR}, BASE_DIR={BASE_DIR}, cwd={os.getcwd()}."
        print(load_error_message)
        return False, load_error_message

    try:
        with open(model_path, 'rb') as f:
            ridge_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler_model = pickle.load(f)
        print(f"Models loaded successfully from: {model_path}, {scaler_path}")
        return True, ""
    except Exception as e:
        load_error_message = f"Failed to unpickle models: {str(e)}\n{traceback.format_exc()}"
        print(load_error_message)
        return False, load_error_message

# Initial attempt to load at startup
load_models()

@app.route('/')
@app.route('/api/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@app.route('/api/predict', methods=['GET', 'POST'])
@app.route('/api/index/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        success, err = load_models()
        if not success or ridge_model is None or scaler_model is None:
            return f"Error: Model files failed to load on server startup.<br><pre>{err}</pre>", 500

        try:
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

            return render_template('home.html', result=round(float(prediction[0]), 2))
        except Exception as e:
            return f"Prediction Error: {str(e)}", 500
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)