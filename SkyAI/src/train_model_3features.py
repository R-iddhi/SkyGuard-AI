from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "aws_anomaly_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "isolation_forest_3features.joblib"

FEATURES = [
    "temperature",
    "pressure",
    "humidity"
]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]

model = IsolationForest(
    n_estimators=200,
    contamination=0.001,
    random_state=42
)

model.fit(X)

joblib.dump(model, MODEL_PATH)

print("MODEL TRAINED SUCCESSFULLY")
print(f"Features: {FEATURES}")
print(f"Rows used: {len(X)}")
print(f"Model saved: {MODEL_PATH}")
