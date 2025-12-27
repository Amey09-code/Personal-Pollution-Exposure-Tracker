import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
Generate Synthetic Dataset
np.random.seed(42)
data_size = 200
data = {
    "AQI": np.random.randint(50, 250, data_size),
    "Time_Outdoors": np.random.uniform(0.5, 6, data_size),
    "Transport": np.random.choice(["Walking", "Bike", "Car", "Bus"], data_size)
}
df = pd.DataFrame(data)
Transport Exposure Factors
transport_factor = {
    "Walking": 1.0,
    "Bike": 1.2,
    "Car": 1.3,
    "Bus": 1.5
}
df["Transport_Factor"] = df["Transport"].map(transport_factor)
# Step 3: Calculate Exposure Score
df["Exposure_Score"] = (
    df["AQI"] * df["Time_Outdoors"] * df["Transport_Factor"]
) / 10
Risk Classification
def classify_risk(score):
    if score < 50:
        return "Low"
    elif score < 100:
        return "Medium"
    else:
        return "High"
df["Risk_Level"] = df["Exposure_Score"].apply(classify_risk)
Encode Target Labels
encoder = LabelEncoder()
df["Risk_Label"] = encoder.fit_transform(df["Risk_Level"])
Feature Selection
X = df[["AQI", "Time_Outdoors", "Transport_Factor"]]
y = df["Risk_Label"]
Train ML Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
print("Personal Pollution Exposure ML model trained successfully.")
