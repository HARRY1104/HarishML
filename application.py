import os
import pickle
from flask import Flask,request, jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(BASE_DIR, 'templates')

application = Flask(__name__, template_folder=template_dir)
app = application

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'Models', 'ridge.pkl')
scaler_path = os.path.join(BASE_DIR, 'Models', 'scaler.pkl')

ridge_model = pickle.load(open(model_path, 'rb'))
scaler_model = pickle.load(open(scaler_path, 'rb'))

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

        scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        prediction = ridge_model.predict([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        return render_template('home.html', result=prediction[0])

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)