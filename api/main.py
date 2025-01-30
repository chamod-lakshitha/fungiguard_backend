from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import random
import joblib
import pandas as pd
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask_cors import CORS

np.random.seed(42)
random.seed(42)

app = Flask(__name__)
CORS(app)
# Load data and model
train_df = pd.read_csv("api/train_data/X_train.csv")

# Load model and scaler
model = joblib.load('api/models/FungiGuard_model.joblib') 
scaler = joblib.load('api/models/minmax_scaler.joblib') 

key_order = [
    "capDiameter", "capShape", "capSurface", "capColor", "doesBruiseOrBleed",
    "gillAttachment", "gillSpacing", "gillColor", "stemHeight", "stemWidth",
    "stemColor", "hasRing", "ringType", "habitat", "season"
]

lime_model = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.asarray(train_df),
    feature_names=train_df.columns,
    class_names=["e", "p"],
    mode="classification",
    random_state = 42
)

def predict_proba(X):
    if X.ndim == 1:
        X = np.asarray(X).reshape(1, -1)
    prob_class_1 = model.predict(X)
    prob_class_0 = 1 - prob_class_1
    return np.hstack((prob_class_0, prob_class_1))

@app.route("/")
def home():
    return jsonify({"message": "Hello from Flask on Vercel!"})

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    input_data = request.json
    array_of_inputs = [input_data[key] for key in key_order]
    scaled_array_of_inputs = scaler.transform(np.array(array_of_inputs).reshape(1, -1))
    required_array_of_inputs = scaled_array_of_inputs[0, [0, 1, 2, 4, 7, 9, 10, 12, 13, 14]]
    edibility = round(float(model.predict(np.asarray([required_array_of_inputs]))[0][0]), 3)
    return jsonify({
        "prediction" : "True",
        "result" : "Edible" if edibility < 0.45 else "Not_Edible",
        "value" : edibility,
        "inputs" :  np.asarray(required_array_of_inputs).tolist()
    })

@app.route('/api/v1/explain', methods=['POST'])
def explain():
    explainer = lime_model.explain_instance(
    np.asarray(request.json),
    predict_proba,               
)
    buf = BytesIO()
    fig = explainer.as_pyplot_figure()
    print(f"Edible Probability: {explainer.predict_proba[0]:.3f}, Non-Edible Probability: {explainer.predict_proba[1]:.3f}")
    plt.title(f"Edible Probability: {explainer.predict_proba[0]:.3f}, Non-Edible Probability: {explainer.predict_proba[1]:.3f}") 
    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free memory
    
    # Encode the image as base64
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    print(explainer.as_list())
    
    return jsonify({"feature_values" : explainer.as_list(), 'image_data': img_base64})


if __name__ == "__main__":
    app.run(debug=True)