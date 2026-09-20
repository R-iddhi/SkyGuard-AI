from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

temperature_before = df["temperature"].diff().abs()
temperature_after = df["temperature"].diff(-1).abs()

pressure_before = df["pressure"].diff().abs()
pressure_after = df["pressure"].diff(-1).abs()

humidity_before = df["humidity"].diff().abs()
humidity_after = df["humidity"].diff(-1).abs()

temperature_recovery = (
    (temperature_before > 10)
    & (temperature_after > 10)
    & (
        (
            df["temperature"]
            - (
                df["temperature"].shift(1)
                + df["temperature"].shift(-1)
            ) / 2
        ).abs()
        > 15
    )
)

pressure_recovery = (
    (pressure_before > 10)
    & (pressure_after > 10)
    & (
        (
            df["pressure"]
            - (
                df["pressure"].shift(1)
                + df["pressure"].shift(-1)
            ) / 2
        ).abs()
        > 15
    )
)

humidity_recovery = (
    (humidity_before > 20)
    & (humidity_after > 20)
    & (
        (
            df["humidity"]
            - (
                df["humidity"].shift(1)
                + df["humidity"].shift(-1)
            ) / 2
        ).abs()
        > 25
    )
)

recovery_anomaly = (
    temperature_recovery
    | pressure_recovery
    | humidity_recovery
)

physical_anomaly = (
    (df["humidity"] < 0)
    | (df["humidity"] > 100)
    | (df["wind_speed"] < 0)
    | (df["wind_direction"] < 0)
    | (df["wind_direction"] > 360)
)

frozen_anomaly = (
    df["temperature"]
    .rolling(8)
    .std()
    .fillna(999)
    .eq(0)
)

multivariate_anomaly = (
    (
        (df["temperature"].abs() > 45)
        & (df["humidity"] < 40)
        & (df["pressure"] < 960)
    )
    |
    (
        (df["temperature"] > 40)
        & (df["humidity"] > 90)
    )
    |
    (
        (df["temperature"].abs() > 45)
        & df["pressure"].between(960, 1030)
        & df["humidity"].between(20, 80)
    )
)

df["physical_anomaly"] = physical_anomaly
df["recovery_anomaly"] = recovery_anomaly
df["frozen_anomaly"] = frozen_anomaly
df["multivariate_anomaly"] = multivariate_anomaly

strong_evidence = (
    physical_anomaly
    | frozen_anomaly
    | multivariate_anomaly
)

supporting_evidence = recovery_anomaly

evidence_score = (
    physical_anomaly.astype(int) * 40
    + frozen_anomaly.astype(int) * 40
    + multivariate_anomaly.astype(int) * 40
    + recovery_anomaly.astype(int) * 20
)

evidence_score = evidence_score.clip(upper=100)

status = pd.Series(
    "NORMAL",
    index=df.index
)

status[
    supporting_evidence
    & ~strong_evidence
] = "SUSPICIOUS"

status[
    strong_evidence
] = "ANOMALY"

df["evidence_score"] = evidence_score
df["status"] = status

true_positive = (
    (status == "ANOMALY")
    & (df["is_anomaly"] == 1)
).sum()

false_positive = (
    (status == "ANOMALY")
    & (df["is_anomaly"] == 0)
).sum()

false_negative = (
    (status == "NORMAL")
    & (df["is_anomaly"] == 1)
).sum()

suspicious_anomalies = (
    (status == "SUSPICIOUS")
    & (df["is_anomaly"] == 1)
).sum()

suspicious_normal = (
    (status == "SUSPICIOUS")
    & (df["is_anomaly"] == 0)
).sum()

true_negative = (
    (status == "NORMAL")
    & (df["is_anomaly"] == 0)
).sum()

precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0
    else 0
)

recall = (
    true_positive / (
        true_positive
        + false_negative
        + suspicious_anomalies
    )
    if true_positive
    + false_negative
    + suspicious_anomalies > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)

event_results = {}

events = [
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

for event in events:
    event_rows = df[df["event_id"] == event]

    if (event_rows["status"] == "ANOMALY").any():
        event_results[event] = "ANOMALY"
    elif (event_rows["status"] == "SUSPICIOUS").any():
        event_results[event] = "SUSPICIOUS"
    else:
        event_results[event] = "MISSED"

print("=" * 70)
print("SKYGUARD AI THREE-LEVEL DETECTOR")
print("=" * 70)

print()
print("ANOMALY STATUS DISTRIBUTION")
print("-" * 70)

print(
    status.value_counts()
    .reindex(
        ["NORMAL", "SUSPICIOUS", "ANOMALY"],
        fill_value=0
    )
    .to_string()
)

print()
print("-" * 70)
print("ANOMALY DETECTION METRICS")
print("-" * 70)

print(f"True positives:       {true_positive}")
print(f"False positives:      {false_positive}")
print(f"False negatives:      {false_negative}")
print(f"True negatives:       {true_negative}")
print(f"Suspicious anomalies: {suspicious_anomalies}")
print(f"Suspicious normal:    {suspicious_normal}")

print()
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print()
print("-" * 70)
print("DETECTION BY ANOMALY EVENT")
print("-" * 70)

for event, result in event_results.items():
    print(f"{event}: {result}")

detected_events = sum(
    result != "MISSED"
    for result in event_results.values()
)

anomaly_events = sum(
    result == "ANOMALY"
    for result in event_results.values()
)

suspicious_events = sum(
    result == "SUSPICIOUS"
    for result in event_results.values()
)

print()
print(f"Events detected: {detected_events}/9")
print(f"Events classified as ANOMALY: {anomaly_events}/9")
print(f"Events classified as SUSPICIOUS: {suspicious_events}/9")

print()
print("-" * 70)
print("ANOMALY EVENT DETAILS")
print("-" * 70)

for event in events:
    event_rows = df[df["event_id"] == event]

    print()
    print(event)

    for _, row in event_rows.iterrows():
        print(
            f"{row['timestamp']} | "
            f"Status={row['status']} | "
            f"Score={row['evidence_score']} | "
            f"T={row['temperature']} | "
            f"P={row['pressure']} | "
            f"H={row['humidity']}"
        )

print()
print("-" * 70)
print("FALSE POSITIVE DETAILS")
print("-" * 70)

false_positive_rows = df[
    (df["status"] == "ANOMALY")
    & (df["is_anomaly"] == 0)
]

print(
    f"Physical: "
    f"{false_positive_rows['physical_anomaly'].sum()}"
)

print(
    f"Recovery: "
    f"{false_positive_rows['recovery_anomaly'].sum()}"
)

print(
    f"Frozen: "
    f"{false_positive_rows['frozen_anomaly'].sum()}"
)

print(
    f"Multivariate: "
    f"{false_positive_rows['multivariate_anomaly'].sum()}"
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)