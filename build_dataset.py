import pandas as pd
import numpy as np
import os

# Folder containing CSV signal files
folder_path = r"C:\Users\Rohan\Desktop\Main EL\Datasets\c1"

# Load wear data
wear_data = pd.read_csv(
    r"C:\Users\Rohan\Desktop\Main EL\Datasets\c1_wear.csv"
)

# Empty list to store extracted features
features_list = []

# Get all CSV files
csv_files = sorted([
    f for f in os.listdir(folder_path)
    if f.endswith(".csv")
])

# Process each CSV file
for i, file in enumerate(csv_files):

    file_path = os.path.join(folder_path, file)

    # Load sensor data
    data = pd.read_csv(file_path, header=None)

    # Extract vibration columns
    vib_x = data[3]
    vib_y = data[4]
    vib_z = data[5]

    # Calculate features
    features = {
        "mean_x": np.mean(vib_x),
        "mean_y": np.mean(vib_y),
        "mean_z": np.mean(vib_z),

        "std_x": np.std(vib_x),
        "std_y": np.std(vib_y),
        "std_z": np.std(vib_z),

        "rms_x": np.sqrt(np.mean(vib_x**2)),
        "rms_y": np.sqrt(np.mean(vib_y**2)),
        "rms_z": np.sqrt(np.mean(vib_z**2)),

        "wear": wear_data.iloc[i, 1]
    }

    features_list.append(features)

# Convert to dataframe
features_df = pd.DataFrame(features_list)

# Save ML-ready dataset
features_df.to_csv(
    r"C:\Users\Rohan\Desktop\Main EL\Datasets\features.csv",
    index=False
)

print(features_df.head())

print("\nDataset successfully created!")
