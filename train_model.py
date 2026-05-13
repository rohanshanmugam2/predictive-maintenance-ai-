import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load feature dataset
data = pd.read_csv(
    r"C:\Users\Rohan\Desktop\Main EL\Datasets\features.csv"
)

# Create labels based on wear
def classify_wear(x):

    if x < 85:
        return "Healthy"

    elif x < 120:
        return "Moderate Wear"

    else:
        return "Worn"

data["label"] = data["wear"].apply(classify_wear)

# Input features
X = data.drop(["wear", "label"], axis=1)

# Output labels
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
# Save trained model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
