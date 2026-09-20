from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "aws_controlled_benchmark.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "isolation_forest_model.joblib"

FEATURES = [
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_direction"
]

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

X = df[FEATURES]

df["ml_prediction"] = model.predict(X)
df["ml_prediction"] = (df["ml_prediction"] == -1).astype(int)

previous = df[FEATURES].shift(1)

temperature_change = (
    df["temperature"] - previous["temperature"]
).abs()

pressure_change = (
    df["pressure"] - previous["pressure"]
).abs()

humidity_change = (
    df["humidity"] - previous["humidity"]
).abs()

physical_anomaly = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100) |
    (df["wind_speed"] < 0) |
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
)

temporal_anomaly = (
    (temperature_change > 10) |
    (pressure_change > 10) |
    (humidity_change > 20)
)

frozen_anomaly = (
    df["temperature"].rolling(8).std().fillna(999) == 0
)

multivariate_anomaly = (
    (
        (df["temperature"].abs() > 45) &
        (df["humidity"] < 40) &
        (df["pressure"] < 960)
    ) |
    (
        (df["temperature"] > 40) &
        (df["humidity"] > 90)
    ) |
    (
        (df["temperature"].abs() > 45) &
        (df["pressure"].between(960, 1030)) &
        (df["humidity"].between(20, 80))
    )
)

df["rule_prediction"] = (
    physical_anomaly |
    temporal_anomaly |
    frozen_anomaly |
    multivariate_anomaly
).astype(int)

df["combined_prediction"] = (
    (df["ml_prediction"] == 1) |
    (df["rule_prediction"] == 1)
).astype(int)

y_true = df["is_anomaly"]


def evaluate(name, prediction):
    matrix = confusion_matrix(y_true, prediction)

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    print("=" * 60)
    print(name)
    print("=" * 60)
    print()
    print("Confusion Matrix:")
    print(matrix)
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()


def event_evaluation(name, prediction):
    anomaly_df = df[df["is_anomaly"] == 1]

    print("=" * 60)
    print(f"{name} - ANOMALY TYPE ANALYSIS")
    print("=" * 60)

    for anomaly_type in anomaly_df["anomaly_type"].unique():
        subset = anomaly_df[
            anomaly_df["anomaly_type"] == anomaly_type
        ]

        total = len(subset)
        detected = int(prediction.loc[subset.index].sum())

        detection_rate = detected / total * 100

        print(
            f"{anomaly_type}: "
            f"{detected}/{total} detected "
            f"({detection_rate:.1f}%)"
        )

    print()


print("CONTROLLED MODEL PERFORMANCE EVALUATION")
print()
print(f"Testing observations: {len(df)}")
print(f"Actual anomalies: {int(y_true.sum())}")
print()

evaluate(
    "ISOLATION FOREST",
    df["ml_prediction"]
)

evaluate(
    "RULE-BASED SYSTEM",
    df["rule_prediction"]
)

evaluate(
    "COMBINED SYSTEM",
    df["combined_prediction"]
)

event_evaluation(
    "ISOLATION FOREST",
    df["ml_prediction"]
)

event_evaluation(
    "RULE-BASED SYSTEM",
    df["rule_prediction"]
)

event_evaluation(
    "COMBINED SYSTEM",
    df["combined_prediction"]
)