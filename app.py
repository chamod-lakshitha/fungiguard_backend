from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = joblib.load('models/FungiGuard_model.joblib') 
scaler = joblib.load('models/minmax_scaler.joblib') 

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Flask API! This is the default endpoint.'
    })

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    array_of_inputs = [1.        , 0.8       , 0.45454545, 1.        , 0.63636364,
       0.66666667, 0.14285714, 0.04577407, 0.21207041, 0.03669543]
    # data = request.get_json()
    # features = np.array(data['features']).reshape(1, -1)
    # scaled_features = scaler.transform(features)
    # prediction = model.predict(scaled_features)
    edibility = round(float(model.predict(np.asarray([array_of_inputs]))[0][0]), 3)
    print(edibility)
    return jsonify({
        "prediction" : edibility
    })

if __name__ == "__main__":
    app.run(debug=True)