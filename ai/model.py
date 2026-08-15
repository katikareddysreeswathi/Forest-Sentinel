import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Features:
# Temperature, Humidity, Rainfall,
# Vegetation Dryness, Previous Fires, Patrol Coverage

X = np.array([
    [24,75,120,15,0,90],
    [27,65,80,25,1,85],
    [29,60,60,35,1,75],
    [31,55,40,45,2,70],
    [33,48,25,55,3,65],
    [35,40,15,65,4,55],
    [36,35,10,72,5,50],
    [38,30,5,80,6,45],
    [39,25,3,88,7,35],
    [40,20,1,95,9,25],
    [25,70,100,20,0,95],
    [28,62,70,30,1,80],
    [30,58,50,40,2,75],
    [32,50,30,50,3,65],
    [34,45,20,60,4,60],
    [37,32,8,75,5,45],
    [39,27,4,85,7,35],
    [41,18,0,98,10,20],
    [26,72,90,18,0,88],
    [30,57,55,38,2,72],
    [34,44,18,58,4,58],
    [37,31,7,78,6,42],
    [40,22,2,92,8,30]
])

# 0 = Low
# 1 = Moderate
# 2 = High
# 3 = Critical

y = np.array([
    0,0,0,1,1,2,2,3,3,3,
    0,0,1,1,2,2,3,3,
    0,1,2,2,3
])

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

# Train
model.fit(X, y)

# Save trained model
with open("forest_risk_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("===================================")
print("FOREST GUARDIAN AI MODEL")
print("===================================")
print("Model trained successfully!")
print("Training samples:", len(X))
print("Model saved as forest_risk_model.pkl")
print("===================================")