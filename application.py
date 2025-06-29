from flask import Flask,render_template,request
import joblib
import numpy as np
import os
from config.path_config import *

app = Flask(__name__)

model_path = os.path.join(MODEL_DIR,"model.pkl")
scaler_path =SCALER

model = joblib.load(model_path)
scaler=joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template("index.html",prediction=None)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        healthcare_cost =float(request.form['Healthcare_Costs'])
        Tumor_Size_mm = float(request.form['Tumor_Size_mm'])
        Mortality_Rate_per_100K = float(request.form['Mortality_Rate_per_100K'])

        treatment_type =int(request.form['treatment_type'])
        Diabetes = int(request.form['Diabetes'])

        input =np.array([healthcare_cost,Tumor_Size_mm,Mortality_Rate_per_100K,treatment_type,Diabetes]).reshape(1,-1)

        scaled_input =scaler.transform(input)
        prediction =  model.predict(scaled_input)[0]

        return render_template('index.html',prediction=prediction)
    
    except Exception as e:
        return str(e)
    
    
if __name__ =="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)