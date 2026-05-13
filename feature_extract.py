import pandas as pd
import numpy as np

# Load one sensor CSV
data = pd.read_csv(
    r"C:\Users\Rohan\Desktop\Main EL\Datasets\c1\c_1_001.csv",
    header=None
)

# Extract vibration columns
vib_x = data[3]
vib_y = data[4]
vib_z = data[5]

# Calculate features
mean_x = np.mean(vib_x)
std_x = np.std(vib_x)
rms_x = np.sqrt(np.mean(vib_x**2))

print("Mean Vibration X:", mean_x)
print("STD Vibration X:", std_x)
print("RMS Vibration X:", rms_x)
