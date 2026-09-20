from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

temperature_change_before = df["temperature"].diff().abs()
temperature_change_after = df["temperature"].diff(-1).abs()

pressure_change_before = df["pressure"].diff().abs()
pressure_change_after = df["pressure"].diff(-1).abs()

humidity_change_before = df["humidity"].diff().abs()
humidity_change_after = df["humidity"].diff(-1).abs()

temperature_recovery = (
    (temperature_change_before > 10)
    & (temperature_change_after > 10)
)

pressure_recovery = (
    (pressure_change_before > 10)
    & (pressure_change_after > 10)
)

humidity_recovery = (
    (humidity_change_before > 20)
    & (humidity_change_after > 20)
)

recovery_anomaly = (
    temperature_recovery
    | pressure_recovery
    | humidity_recovery
)

df["recovery_anomaly"] = recovery_anomaly

true_positive = (
    (df["recovery_anomaly"])
    & (df["is_anomaly"] == 1)
).sum()

false_positive = (
    (df["recovery_anomaly"])
    & (df["is_anomaly"] == 0)
).sum()

false_negative = (
    (~df["recovery_anomaly"])
    & (df["is_anomaly"] == 1)
).sum()

true_negative = (
    (~df["recovery_anomaly"])
    & (df["is_anomaly"] == 0)
).sum()

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

print("=" * 70)
print("SKYGUARD AI SPIKE-AND-RECOVERY DETECTION")
print("=" * 70)

print()
print(f"True positives:  {true_positive}")
print(f"False positives: {false_positive}")
print(f"False negatives: {false_negative}")
print(f"True negatives:  {true_negative}")

print()
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print()
print("-" * 70)
print("DETECTION BY ANOMALY TYPE")
print("-" * 70)

anomaly_types = [
    "TEMP_SPIKE_01",
    "TEMP_DROP_01",
    "HUM_HIGH_01",
    "HUM_LOW_01",
    "PRESS_SPIKE_01",
    "PRESS_DROP_01",
    "FROZEN_01",
    "MULTI_01",
    "TEMP_HUM_01"
]

for event in anomaly_types:
    event_rows = df[df["event_id"] == event]

    detected = event_rows["recovery_anomaly"].any()

    print(
        f"{event}: "
        f"{'DETECTED' if detected else 'MISSED'}"
    )

print()
print("-" * 70)
print("FALSE POSITIVE SOURCES")
print("-" * 70)

print(
    f"Temperature recovery: "
    f"{((temperature_recovery) & (df['is_anomaly'] == 0)).sum()}"
)

print(
    f"Pressure recovery:    "
    f"{((pressure_recovery) & (df['is_anomaly'] == 0)).sum()}"
)

print(
    f"Humidity recovery:    "
    f"{((humidity_recovery) & (df['is_anomaly'] == 0)).sum()}"
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)