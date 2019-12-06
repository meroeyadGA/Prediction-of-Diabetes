import numpy as np
import pandas as pd
import pickle
from sklearn.externals.joblib import load
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict')
def submit():
    user_input = request.args
    sc = load('./static/model/std_scaler.bin')
    data = np.array([
            float(user_input['Insulin']),
            float(user_input['SkinThickness']),
            float(user_input['Glucose']),
            int(user_input['Age']),
            float(user_input['BMI'])
        ]).reshape(1,-1)

    data = sc.transform(data)
    model = pickle.load(open('./static/model/diabetes.pkl', 'rb'))
    prediction = model.predict(data)[0]

    if prediction > 0:
       msg = "Success"
    else:
       msg = "Unsuccess"

    return render_template("index.html",msg=msg, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
