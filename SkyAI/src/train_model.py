from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_anomaly_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "isolation_forest_model.joblib"

FEATURES = [
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_direction"
]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()

model = IsolationForest(
    n_estimators=300,
    contamination=0.001,
    max_samples="auto",
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

model.fit(X)

joblib.dump(model, MODEL_PATH)

predictions = model.predict(X)
anomaly_count = (predictions == -1).sum()

print("MODEL TRAINING COMPLETE")
print()
print(f"Training observations: {len(X)}")
print(f"Features used: {len(FEATURES)}")
print(f"Trees: 300")
print(f"Expected contamination: 0.001")
print(f"Detected training anomalies: {anomaly_count}")
print(f"Model saved to: {MODEL_PATH}")