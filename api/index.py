import os
import pickle
from flask import Flask, request, render_template

API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(API_DIR)

# Locate templates directory across possible runtime directories
template_dir = os.path.join(BASE_DIR, 'templates')
if not os.path.exists(template_dir):
    template_dir = os.path.join(API_DIR, 'templates')
if not os.path.exists(template_dir):
    template_dir = os.path.join(os.getcwd(), 'templates')

app = Flask(__name__, template_folder=template_dir)

# Helper function to find files across possible base paths in Vercel Serverless
def find_file(relative_path):
    candidates = [
        os.path.join(BASE_DIR, relative_path),
        os.path.join(API_DIR, relative_path),
        os.path.join(os.getcwd(), relative_path),
        os.path.join('/var/task', relative_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

model_path = find_file(os.path.join('models', 'ridge.pkl'))
scaler_path = find_file(os.path.join('models', 'scaler.pkl'))

ridge_model = None
scaler_model = None

# Safely attempt to load models and log detailed info
try:
    if model_path and scaler_path:
        with open(model_path, 'rb') as f:
            ridge_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler_model = pickle.load(f)
        print("Models loaded successfully!")
    else:
        print(f"FILE NOT FOUND ERROR: model_path={model_path}, scaler_path={scaler_path}")
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

if __name__ == "__main__":
    app.run(debug=True)