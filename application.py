import pickle
from flask import Flask,request, jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('Models/scaler.pkl', 'rb'))

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