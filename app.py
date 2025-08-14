from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("Electricity_cost_prediction.pkl") 

@app.route("/")
def home():
    return render_template("index.html")

import pandas as pd

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "site area": int(request.form["site_area"]),
            "water consumption": float(request.form["water_consumption"]),
            "recycling rate": int(request.form["recycling_rate"]),
            "utilisation rate": int(request.form["utilisation_rate"]),
            "air qality index": int(request.form["air_quality_index"]),
            "issue reolution time": int(request.form["issue_resolution_time"]),
            "resident count": int(request.form["resident_count"]),
            "structure type": request.form["structure_type"]
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        return render_template("index.html", prediction_text=f"Estimated Electricity Cost: {prediction:.2f}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
