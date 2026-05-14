from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ML model
data = pd.read_csv("features.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Convert comma-separated strings into arrays
    vib_x = np.array(
        [float(x) for x in data["vib_x"].split(",")]
    )

    vib_y = np.array(
        [float(y) for y in data["vib_y"].split(",")]
    )

    vib_z = np.array(
        [float(z) for z in data["vib_z"].split(",")]
    )

    # Automatic feature extraction
    sample = pd.DataFrame([{

        "mean_x": np.mean(vib_x),
        "mean_y": np.mean(vib_y),
        "mean_z": np.mean(vib_z),

        "std_x": np.std(vib_x),
        "std_y": np.std(vib_y),
        "std_z": np.std(vib_z),

        "rms_x": np.sqrt(np.mean(vib_x**2)),
        "rms_y": np.sqrt(np.mean(vib_y**2)),
        "rms_z": np.sqrt(np.mean(vib_z**2))

    }])

    # Prediction
    prediction = model.predict(sample)

    return jsonify({
        "prediction": prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True)
