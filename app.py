import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load pre-trained model
with open("model/customer_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Customer Churn Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting JSON with {"features": [values]}
        data = request.json
        features = np.array([data["features"]])
        prob = model.predict_proba(features)[0][1]  # Probability of churn
        return jsonify({"churn_probability": round(float(prob), 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
