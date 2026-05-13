import pandas as pd

data = pd.read_csv(
    r"C:\Users\Rohan\Desktop\Main EL\Datasets\features.csv"
)

print(data["wear"].describe())
