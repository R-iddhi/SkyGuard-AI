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
ROBUST_Z_THRESHOLD = 6.0

for feature in FEATURES:
    change = df[feature].diff().abs()

    baseline = (
        change
        .rolling(
            WINDOW,
            center=True,
            min_periods=12
        )
        .median()
    )

    mad = (
        change
        .rolling(
            WINDOW,
            center=True,
            min_periods=12
        )
        .apply(
            lambda x: np.median(
                np.abs(x - np.median(x))
            ),
            raw=True
        )
    )

    robust_z = (
        0.6745 *
        (change - baseline) /
        mad.replace(0, np.nan)
    )

    df[f"{feature}_change"] = change
    df[f"{feature}_change_z"] = robust_z.abs()

df["temperature_recovery"] = (
    (
        df["temperature"].diff().abs() > 10
    ) &
    (
        df["temperature"].diff().abs().shift(-1) > 10
    )
)

df["pressure_recovery"] = (
    (
        df["pressure"].diff().abs() > 10
    ) &
    (
        df["pressure"].diff().abs().shift(-1) > 10
    )
)

df["humidity_recovery"] = (
    (
        df["humidity"].diff().abs() > 20
    ) &
    (
        df["humidity"].diff().abs().shift(-1) > 20
    )
)

df["temperature_context"] = (
    df["temperature_change_z"] >
    ROBUST_Z_THRESHOLD
)

df["pressure_context"] = (
    df["pressure_change_z"] >
    ROBUST_Z_THRESHOLD
)

df["humidity_context"] = (
    df["humidity_change_z"] >
    ROBUST_Z_THRESHOLD
)

df["context_temporal_anomaly"] = (
    df["temperature_context"] |
    df["pressure_context"] |
    df["humidity_context"] |
    df["temperature_recovery"] |
    df["pressure_recovery"] |
    df["humidity_recovery"]
)

df["context_temporal_anomaly"] = (
    df["context_temporal_anomaly"]
    .fillna(False)
)

actual = df["is_anomaly"] == 1
predicted = df["context_temporal_anomaly"]

true_positive = int(
    (actual & predicted).sum()
)

false_positive = int(
    (~actual & predicted).sum()
)

false_negative = int(
    (actual & ~predicted).sum()
)

true_negative = int(
    (~actual & ~predicted).sum()
)

precision = (
    true_positive /
    (true_positive + false_positive)
    if true_positive + false_positive > 0
    else 0
)

recall = (
    true_positive /
    (true_positive + false_negative)
    if true_positive + false_negative > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)

print("CONTEXT-AWARE TEMPORAL DETECTOR")
print()

print(f"Rolling window: {WINDOW} observations")
print(f"Robust Z threshold: {ROBUST_Z_THRESHOLD}")
print()

print("=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

print(f"True positives:  {true_positive}")
print(f"False positives: {false_positive}")
print(f"False negatives: {false_negative}")
print(f"True negatives:  {true_negative}")

print()

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print()

print("=" * 70)
print("DETECTION BY ANOMALY TYPE")
print("=" * 70)

anomaly_df = df[df["is_anomaly"] == 1]

for anomaly_type in anomaly_df["anomaly_type"].unique():
    subset = anomaly_df[
        anomaly_df["anomaly_type"] == anomaly_type
    ]

    total = len(subset)

    detected = int(
        subset["context_temporal_anomaly"].sum()
    )

    rate = (
        detected / total * 100
        if total > 0
        else 0
    )

    print(
        f"{anomaly_type}: "
        f"{detected}/{total} "
        f"({rate:.1f}%)"
    )

print()

print("=" * 70)
print("FALSE POSITIVE SOURCES")
print("=" * 70)

normal_df = df[df["is_anomaly"] == 0]

print(
    "Temperature context:",
    int(normal_df["temperature_context"].sum())
)

print(
    "Pressure context:",
    int(normal_df["pressure_context"].sum())
)

print(
    "Humidity context:",
    int(normal_df["humidity_context"].sum())
)

print(
    "Temperature recovery:",
    int(normal_df["temperature_recovery"].sum())
)

print(
    "Pressure recovery:",
    int(normal_df["pressure_recovery"].sum())
)

print(
    "Humidity recovery:",
    int(normal_df["humidity_recovery"].sum())
)