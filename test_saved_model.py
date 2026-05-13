import joblib
import pandas as pd

# Load trained model
model = joblib.load(
    r"C:\Users\Rohan\Desktop\Main EL\Datasets\model.pkl"
)

# Sample input data
sample_data = pd.DataFrame([{
    "mean_x": 0.01,
    "mean_y": -0.02,
    "mean_z": 0.005,

    "std_x": 0.07,
    "std_y": 0.08,
    "std_z": 0.06,

    "rms_x": 0.08,
    "rms_y": 0.09,
    "rms_z": 0.07
}])

# Predict condition
prediction = model.predict(sample_data)

print("Predicted Condition:", prediction[0])