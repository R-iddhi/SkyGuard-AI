from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_controlled_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

FEATURES = [
    "temperature",
    "pressure",
    "humidity"
]

WINDOW = 24
Z_THRESHOLD = 4.0

for feature in FEATURES:
    rolling_median = (
        df[feature]
        .rolling(WINDOW, center=True, min_periods=12)
        .median()
    )

    rolling_mad = (
        df[feature]
        .rolling(WINDOW, center=True, min_periods=12)
        .apply(
            lambda x: np.median(np.abs(x - np.median(x))),
            raw=True
        )
    )

    robust_z = (
        0.6745 *
        (df[feature] - rolling_median) /
        rolling_mad.replace(0, np.nan)
    )

    df[f"{feature}_robust_z"] = robust_z.abs()

df["statistical_temporal_anomaly"] = (
    (df["temperature_robust_z"] > Z_THRESHOLD) |
    (df["pressure_robust_z"] > Z_THRESHOLD) |
    (df["humidity_robust_z"] > Z_THRESHOLD)
)

df["statistical_temporal_anomaly"] = (
    df["statistical_temporal_anomaly"]
    .fillna(False)
)

normal = df["is_anomaly"] == 0
actual_anomaly = df["is_anomaly"] == 1
detected = df["statistical_temporal_anomaly"]

true_positive = int((actual_anomaly & detected).sum())
false_positive = int((normal & detected).sum())
false_negative = int((actual_anomaly & ~detected).sum())
true_negative = int((normal & ~detected).sum())

precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0
    else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)

print("STATISTICAL TEMPORAL DETECTOR")
print()
print(f"Window size: {WINDOW} observations")
print(f"Robust Z threshold: {Z_THRESHOLD}")
print()
print(f"True positives: {true_positive}")
print(f"False positives: {false_positive}")
print(f"False negatives: {false_negative}")
print(f"True negatives: {true_negative}")
print()
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print()

print("=" * 60)
print("DETECTION BY ANOMALY TYPE")
print("=" * 60)

anomaly_df = df[df["is_anomaly"] == 1]

for anomaly_type in anomaly_df["anomaly_type"].unique():
    subset = anomaly_df[
        anomaly_df["anomaly_type"] == anomaly_type
    ]

    total = len(subset)
    detected_count = int(
        subset["statistical_temporal_anomaly"].sum()
    )

    rate = detected_count / total * 100

    print(
        f"{anomaly_type}: "
        f"{detected_count}/{total} "
        f"({rate:.1f}%)"
    )