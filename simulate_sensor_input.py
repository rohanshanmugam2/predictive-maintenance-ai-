import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load(
    r"C:\Users\Rohan\Desktop\Main EL\Datasets\model.pkl"
)

# Simulated raw vibration sensor data
vib_x = np.array([0.01, 0.02, 0.015, 0.018, 0.017])
vib_y = np.array([0.02, 0.025, 0.021, 0.023, 0.022])
vib_z = np.array([0.03, 0.028, 0.031, 0.029, 0.032])

# Extract features automatically
sample_data = pd.DataFrame([{
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

# Predict condition
prediction = model.predict(sample_data)

print("Predicted Condition:", prediction[0])
