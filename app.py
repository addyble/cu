from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
# Allow all origins during dev (you can restrict later)
CORS(app)

# Load model and schema
with open("model/customer_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/model_schema.pkl", "rb") as f:
    schema = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Churn Prediction API is running",
        "endpoints": {
            "GET /features-template": "Get example JSON payload structure",
            "POST /predict": "Predict churn probability"
        }
    })

@app.route("/features-template", methods=["GET"])
def features_template():
    """Returns a JSON example with correct types for Postman testing."""
    template = {col: (0 if np.issubdtype(dtype, np.number) else "string_value")
                for col, dtype in schema.items()}
    return jsonify(template)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df_input = pd.DataFrame([data])

        # Reorder columns & enforce schema
        for col, dtype in schema.items():
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(dtype, errors="ignore")
            else:
                df_input[col] = pd.Series([np.nan], dtype=dtype)

        df_input = df_input[list(schema.keys())]

        # Predict
        prob = model.predict_proba(df_input)[0][1]
        label = "Yes" if prob >= 0.5 else "No"

        return jsonify({
            "churn_probability": round(float(prob), 4),
            "churn_prediction": label
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
